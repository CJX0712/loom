"""工具层：注册表 + 内置工具 + 沙箱。

设计要点
--------
1. **一切文件操作都被限制在 `config.WORKDIR` 内**。`_safe_path()` 解析后必须
   仍位于工作目录之下，否则直接拒绝 —— 这一条是所有安全性的地基。
2. **工具结果是结构化字典**，不是裸字符串：`{"ok": bool, "content": str, "meta": {}}`。
   模型能据此判断成功与否，而不是靠猜。
3. **观测长度有上限**，防止一个 50MB 文件把 16GB 机器的上下文撑爆。
4. **任何异常都转成结构化失败**，绝不把 traceback 抛回智能体循环 —— 模型看到
   traceback 会开始胡编，看到 `{"ok": false, "content": "..."}` 才会自救。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import config

Handler = Callable[..., Awaitable[dict]]


# ---------------------------------------------------------------------------
# 沙箱
# ---------------------------------------------------------------------------
class SandboxError(ValueError):
    """越界访问。"""


def workdir() -> Path:
    r"""规范化后的工作目录。

    **必须两边都规范化再比较。** 否则在 Windows 上会踩这个坑：
    `%TEMP%` 常常是短名形式（`C:\\Users\\ADMINI~1\\...`），而 `Path.resolve()`
    会把候选路径展开成长名（`C:\\Users\\Administrator\\...`）—— 于是同一条路径
    在字符串上前缀对不上，**合法的沙箱内访问会被误判为越界**。
    符号链接、大小写、`\\?\` 前缀同理。
    """
    return config.WORKDIR.resolve()


def safe_path(raw: str) -> Path:
    """把用户/模型给的路径解析到工作目录内，越界即拒绝。

    同时挡住 Windows 的 `C:\\` 绝对路径、UNC 路径、以及 `..` 穿越。
    """
    if raw is None or not str(raw).strip():
        raise SandboxError("路径不能为空")
    work = workdir()
    p = Path(str(raw))
    candidate = p.resolve() if p.is_absolute() else (work / p).resolve()
    if candidate != work and work not in candidate.parents:
        raise SandboxError(
            f"越界访问被拒绝：{candidate} 不在工作目录 {work} 内"
        )
    return candidate


def rel(p: Path) -> str:
    """给模型看的相对路径，避免泄露绝对路径。"""
    try:
        return str(p.resolve().relative_to(workdir())).replace("\\", "/") or "."
    except ValueError:
        return str(p)


def clip(text: str, limit: int | None = None) -> tuple[str, bool]:
    """截断过长的观测。返回 (文本, 是否被截断)。"""
    limit = limit or config.MAX_OBSERVATION_CHARS
    if len(text) <= limit:
        return text, False
    head = text[: limit - 200]
    tail = text[-160:]
    return f"{head}\n… [已截断 {len(text) - limit + 360} 字符] …\n{tail}", True


def ok(content: str, **meta: Any) -> dict:
    text, truncated = clip(content)
    return {"ok": True, "content": text, "meta": {**meta, "truncated": truncated}}


def fail(content: str, **meta: Any) -> dict:
    return {"ok": False, "content": clip(content)[0], "meta": meta}


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler
    dangerous: bool = False
    source: str = "builtin"
    tags: list[str] = field(default_factory=list)

    def schema(self) -> dict:
        """OpenAI / Ollama 通用的 function 描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Registry:
    """工具注册表。MCP 服务器发现到的工具也注册到这里，对模型完全透明。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool

    def unregister_source(self, source: str) -> None:
        for name in [n for n, t in self._tools.items() if t.source == source]:
            del self._tools[name]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in self.names()]

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.all()]

    def describe(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "source": t.source,
                "dangerous": t.dangerous,
                "tags": t.tags,
            }
            for t in self.all()
        ]

    async def dispatch(self, name: str, arguments: dict) -> dict:
        """执行一个工具。**任何异常都被吞掉并转成结构化失败。**"""
        tool = self._tools.get(name)
        if tool is None:
            return fail(
                f"未知工具 {name!r}。可用工具：{', '.join(self.names())}",
                kind="unknown_tool",
            )
        if not isinstance(arguments, dict):
            return fail(f"参数必须是对象，收到 {type(arguments).__name__}", kind="bad_args")
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                tool.handler(**arguments), timeout=config.TOOL_TIMEOUT
            )
        except TimeoutError:
            return fail(f"工具 {name} 超时（>{config.TOOL_TIMEOUT:g}s）", kind="timeout")
        except TypeError as exc:  # 参数名/数量不匹配
            return fail(f"工具 {name} 参数错误: {exc}", kind="bad_args")
        except SandboxError as exc:
            return fail(str(exc), kind="sandbox")
        except Exception as exc:
            return fail(f"工具 {name} 执行失败: {type(exc).__name__}: {exc}", kind="error")
        if not isinstance(result, dict) or "ok" not in result:
            return fail(f"工具 {name} 返回了非法结果类型", kind="bad_result")
        result.setdefault("meta", {})
        result["meta"]["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result


# ---------------------------------------------------------------------------
# 内置工具实现
# ---------------------------------------------------------------------------
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".loom", ".idea", ".vscode"}


async def fs_list(path: str = ".", depth: int = 1) -> dict:
    p = safe_path(path)
    if not p.exists():
        return fail(f"路径不存在: {rel(p)}")
    if p.is_file():
        st = p.stat()
        return ok(f"{rel(p)}  ({st.st_size} 字节)", kind="file", size=st.st_size)
    depth = max(1, min(int(depth), 4))
    lines: list[str] = []

    def walk(d: Path, prefix: str, level: int) -> None:
        if level > depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return
        for e in entries:
            if e.name in SKIP_DIRS:
                continue
            if e.is_dir():
                lines.append(f"{prefix}{e.name}/")
                walk(e, prefix + "  ", level + 1)
            else:
                try:
                    lines.append(f"{prefix}{e.name}  ({e.stat().st_size}B)")
                except OSError:
                    lines.append(f"{prefix}{e.name}")

    walk(p, "", 1)
    body = "\n".join(lines) if lines else "(空目录)"
    return ok(f"{rel(p)}/\n{body}", count=len(lines))


async def fs_read(path: str, start: int = 1, lines: int = 200) -> dict:
    p = safe_path(path)
    if not p.is_file():
        return fail(f"不是文件或不存在: {rel(p)}")
    size = p.stat().st_size
    if size > 8 * 1024 * 1024:
        return fail(f"文件过大（{size} 字节），请用 fs_grep 定位后再分段读")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return fail(f"读取失败: {exc}")
    all_lines = text.splitlines()
    start = max(1, int(start))
    lines = max(1, min(int(lines), 2000))
    chunk = all_lines[start - 1 : start - 1 + lines]
    numbered = "\n".join(f"{start + i:>5}| {ln}" for i, ln in enumerate(chunk))
    return ok(
        numbered or "(空文件)",
        total_lines=len(all_lines),
        shown=f"{start}-{start + len(chunk) - 1}",
        size=size,
    )


async def fs_write(path: str, content: str) -> dict:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    return ok(
        f"{'覆盖' if existed else '新建'} {rel(p)}（{len(content)} 字符）",
        bytes=p.stat().st_size,
    )


async def fs_grep(pattern: str, path: str = ".", glob: str = "*", max_hits: int = 60) -> dict:
    p = safe_path(path)
    if not pattern:
        return fail("pattern 不能为空")
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return fail(f"非法正则: {exc}")
    max_hits = max(1, min(int(max_hits), 300))
    hits: list[str] = []
    files = [p] if p.is_file() else [
        f
        for f in p.rglob(glob or "*")
        if f.is_file() and not any(part in SKIP_DIRS for part in f.parts)
    ]
    for f in files:
        if len(hits) >= max_hits:
            break
        try:
            if f.stat().st_size > 4 * 1024 * 1024:
                continue
            for i, ln in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(ln):
                    hits.append(f"{rel(f)}:{i}: {ln.strip()[:200]}")
                    if len(hits) >= max_hits:
                        break
        except (OSError, UnicodeError):
            continue
    return ok("\n".join(hits) if hits else f"没有匹配 {pattern!r} 的内容",
              hits=len(hits), scanned=len(files))


# -- shell ------------------------------------------------------------------
# 明确的破坏性操作。宁可误伤也不放行 —— 这是沙箱的最后一道闸。
DENY_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf]{1,2}\b.*\s/(\s|$)",
    r"\brm\s+-rf\s+/",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+if=.*of=/dev/",
    r"\bformat\s+[a-zA-Z]:",
    r"\bdiskpart\b",
    r"\bshutdown\b|\breboot\b|\bhalt\b",
    r":\(\)\s*\{.*\};\s*:",
    r"\bdel\s+/[sf]\b.*[a-zA-Z]:\\?(\s|$)",
    r"\brd\s+/s\s+/q\s+[a-zA-Z]:\\?(\s|$)",
    r"\breg\s+delete\b",
    r"\bvssadmin\s+delete\b",
    r"\bbcdedit\b",
    r"\bcipher\s+/w\b",
    r"\bnet\s+user\s+\S+\s+/add\b",
    r"\b(taskkill|Stop-Process).*/(f|IM)\b.*\b(lsass|csrss|wininit|services)\b",
]
_DENY_RX = [re.compile(p, re.IGNORECASE) for p in DENY_PATTERNS]


def check_command(command: str) -> str | None:
    """返回拒绝原因；None 表示放行。"""
    if not command.strip():
        return "命令为空"
    for rx in _DENY_RX:
        if rx.search(command):
            return f"命令被安全策略拒绝（匹配 {rx.pattern[:40]}…）"
    return None


async def shell_run(command: str, timeout: float | None = None) -> dict:
    if not config.SHELL_ENABLED:
        return fail("shell 工具已被 LOOM_SHELL=0 关闭")
    reason = check_command(command)
    if reason:
        return fail(reason, kind="denied")
    timeout = min(float(timeout or config.SHELL_TIMEOUT), 300.0)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "GIT_PAGER": "cat", "PAGER": "cat"}

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            shell=True,
            cwd=str(workdir()),
            capture_output=True,
            timeout=timeout,
            env=env,
        )

    try:
        proc = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return fail(f"命令超时（>{timeout:g}s）: {command[:120]}", kind="timeout")
    except OSError as exc:
        return fail(f"无法执行: {exc}")
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    body = out
    if err.strip():
        body += ("\n" if body else "") + "[stderr]\n" + err
    body = body.strip() or "(无输出)"
    result = ok(f"$ {command}\n{body}", exit_code=proc.returncode)
    if proc.returncode != 0:
        result["ok"] = False
    return result


# -- 网络 --------------------------------------------------------------------
async def web_fetch(url: str, query: str | None = None) -> dict:
    """抓网页转 Markdown。优先走本地 crawl4ai（更强的正文提取 + 去噪）。"""
    if not re.match(r"^https?://", url or ""):
        return fail("url 必须以 http:// 或 https:// 开头")
    if config.CRAWL4AI_ENABLED:
        try:
            async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as c:
                r = await c.post(
                    f"{config.CRAWL4AI_URL}/md",
                    json={"url": url, **({"f": "bm25", "q": query} if query else {})},
                )
                if r.status_code == 200:
                    data = r.json()
                    md = (data.get("markdown") or "").strip()
                    if md:
                        return ok(md, via="crawl4ai", url=data.get("url", url))
                    return fail(f"crawl4ai 未提取到正文: {json.dumps(data)[:300]}")
        except Exception as exc:  # 回退到内置抓取，不因为爬虫没起就瘫掉
            fallback_note = f"crawl4ai 不可用（{type(exc).__name__}），已回退内置抓取。\n"
        else:
            fallback_note = ""
    else:
        fallback_note = ""

    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": "loom/0.1 (+local agent)"})
        html = r.text
    except Exception as exc:
        return fail(f"抓取失败: {type(exc).__name__}: {exc}")
    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>|</(p|div|li|h[1-6])>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return ok(fallback_note + (text or "(空)"), via="builtin", status=r.status_code, url=url)


_DDG_RX = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.S,
)
_DDG_SNIP = re.compile(r'class="result__snippet"[^>]*>(?P<s>.*?)</a>', re.S)


async def web_search(query: str, limit: int = 6) -> dict:
    """无 Key 网页搜索（DuckDuckGo HTML）。失败时明确告诉模型改用 web_fetch。"""
    if not (query or "").strip():
        return fail("query 不能为空")
    limit = max(1, min(int(limit), 12))
    try:
        async with httpx.AsyncClient(
            timeout=config.HTTP_TIMEOUT, follow_redirects=True
        ) as c:
            r = await c.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; loom/0.1)"},
            )
        html = r.text
    except Exception as exc:
        return fail(
            f"搜索失败（{type(exc).__name__}）。可直接用 web_fetch 打开已知 URL。"
        )
    items: list[str] = []
    snips = [re.sub(r"(?s)<[^>]+>", "", m.group("s")).strip() for m in _DDG_SNIP.finditer(html)]
    for i, m in enumerate(_DDG_RX.finditer(html)):
        if len(items) >= limit:
            break
        href = m.group("href")
        m2 = re.search(r"uddg=([^&]+)", href)
        if m2:
            from urllib.parse import unquote

            href = unquote(m2.group(1))
        title = re.sub(r"(?s)<[^>]+>", "", m.group("title")).strip()
        items.append(f"{i + 1}. {title}\n   {href}\n   {snips[i] if i < len(snips) else ''}")
    if not items:
        return fail("搜索无结果（可能被限流）。可改用 web_fetch 打开已知 URL。")
    return ok("\n".join(items), count=len(items), engine="duckduckgo")


async def http_json(url: str, method: str = "GET", body: str | None = None) -> dict:
    """给 API 用的裸 HTTP 调用，返回原始响应文本。"""
    method = (method or "GET").upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        return fail(f"不支持的 method: {method}")
    kwargs: dict[str, Any] = {
        "headers": {"User-Agent": "loom/0.1", "Accept": "application/json, */*"}
    }
    if body:
        try:
            kwargs["json"] = json.loads(body)
        except json.JSONDecodeError as exc:
            return fail(f"body 不是合法 JSON: {exc}")
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT, follow_redirects=True) as c:
            r = await c.request(method, url, **kwargs)
    except Exception as exc:
        return fail(f"请求失败: {type(exc).__name__}: {exc}")
    return ok(
        f"HTTP {r.status_code}\n{r.text}",
        status=r.status_code,
        ok_status=r.status_code < 400,
    )


async def python_run(code: str, timeout: float | None = None) -> dict:
    """在子进程里跑一段 Python，用来做计算/数据加工。产物落在工作目录。"""
    if not (code or "").strip():
        return fail("code 不能为空")
    timeout = min(float(timeout or config.TOOL_TIMEOUT), 120.0)
    tmpdir = workdir() / ".loom-tmp"
    tmpdir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", dir=tmpdir, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        script = Path(fh.name)

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(script)],
            cwd=str(workdir()),
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    try:
        proc = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return fail(f"脚本超时（>{timeout:g}s）", kind="timeout")
    finally:
        script.unlink(missing_ok=True)
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    body = out.strip()
    if err.strip():
        body += ("\n" if body else "") + "[stderr]\n" + err.strip()
    result = ok(body or "(无输出)", exit_code=proc.returncode)
    if proc.returncode != 0:
        result["ok"] = False
    return result


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------
def build_registry() -> Registry:
    reg = Registry()

    reg.register(Tool(
        "fs_list", "列出工作目录内的文件与子目录（递归 depth 层）。",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对工作目录的路径，默认 ."},
            "depth": {"type": "integer", "description": "递归层数 1-4，默认 1"}},
         "required": []},
        fs_list, tags=["fs"]))

    reg.register(Tool(
        "fs_read", "读取工作目录内的文本文件，带行号，可分页。",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "文件相对路径"},
            "start": {"type": "integer", "description": "起始行号，从 1 开始，默认 1"},
            "lines": {"type": "integer", "description": "读取行数，默认 200"}},
         "required": ["path"]},
        fs_read, tags=["fs"]))

    reg.register(Tool(
        "fs_write", "写入（或覆盖）工作目录内的文件，父目录自动创建。",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "文件相对路径"},
            "content": {"type": "string", "description": "完整文件内容"}},
         "required": ["path", "content"]},
        fs_write, dangerous=True, tags=["fs"]))

    reg.register(Tool(
        "fs_grep", "在工作目录内按正则搜索文件内容，返回 文件:行号: 匹配行。",
        {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "正则表达式"},
            "path": {"type": "string", "description": "搜索起点，默认 ."},
            "glob": {"type": "string", "description": "文件名通配，如 *.py，默认 *"},
            "max_hits": {"type": "integer", "description": "最多返回多少条，默认 60"}},
         "required": ["pattern"]},
        fs_grep, tags=["fs"]))

    reg.register(Tool(
        "shell_run", "在工作目录内执行 shell 命令并返回 stdout/stderr。破坏性命令被拒绝。",
        {"type": "object", "properties": {
            "command": {"type": "string", "description": "要执行的命令"},
            "timeout": {"type": "number", "description": "超时秒数，默认取全局配置"}},
         "required": ["command"]},
        shell_run, dangerous=True, tags=["exec"]))

    reg.register(Tool(
        "web_fetch", "抓取网页并转成 Markdown 正文（优先本地 crawl4ai）。",
        {"type": "object", "properties": {
            "url": {"type": "string", "description": "完整 URL"},
            "query": {"type": "string", "description": "可选：按此问题过滤正文"}},
         "required": ["url"]},
        web_fetch, tags=["web"]))

    reg.register(Tool(
        "web_search", "无 Key 网页搜索，返回标题 / 链接 / 摘要列表。",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索词"},
            "limit": {"type": "integer", "description": "返回条数，默认 6"}},
         "required": ["query"]},
        web_search, tags=["web"]))

    reg.register(Tool(
        "http_json", "对任意 HTTP API 发请求（GET/POST/PUT/PATCH/DELETE），body 为 JSON 字符串。",
        {"type": "object", "properties": {
            "url": {"type": "string"},
            "method": {"type": "string", "description": "默认 GET"},
            "body": {"type": "string", "description": "可选，JSON 字符串"}},
         "required": ["url"]},
        http_json, tags=["web"]))

    reg.register(Tool(
        "python_run", "在子进程里运行一段 Python 代码，用于计算与数据加工。",
        {"type": "object", "properties": {
            "code": {"type": "string", "description": "完整 Python 源码"},
            "timeout": {"type": "number", "description": "超时秒数"}},
         "required": ["code"]},
        python_run, dangerous=True, tags=["exec"]))

    return reg

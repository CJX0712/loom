"""MCP（Model Context Protocol）桥接。

**这里一个字节的协议逻辑都没有自己实现。** 走官方 `mcp` Python SDK，
Loom 只负责把 MCP 工具**翻译成模型能看懂的 function schema**，以及把调用
结果翻译成 Loom 的结构化结果。于是整个 MCP 生态（成千上万个现成服务器）
可以零改动插进来。

并发模型
--------
`stdio_client` 的上下文必须在**同一个 asyncio 任务**里进出，否则退出时会炸。
所以每个服务器由一条常驻任务持有，外界通过队列把调用请求投进去、用 Future 取回结果。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .tools import Registry, Tool, fail, ok


@dataclass
class MCPServerConn:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue, init=False, repr=False)
    _ready: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _tools: list[dict] = field(default_factory=list, init=False)
    _error: str | None = field(default=None, init=False)
    _alive: bool = field(default=False, init=False)

    # -- 生命周期 ---------------------------------------------------------
    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30)
        except TimeoutError:
            self._error = self._error or "启动超时（30s）"

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=10)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()

    async def _run(self) -> None:
        try:
            from mcp.client.stdio import stdio_client

            from mcp import ClientSession, StdioServerParameters
        except ImportError as exc:
            self._error = f"未安装官方 MCP SDK（pip install mcp）: {exc}"
            self._ready.set()
            return

        params = StdioServerParameters(
            command=self.command,
            args=list(self.args),
            env={**os.environ, **self.env} if self.env else None,
            cwd=self.cwd,
        )
        try:
            # 这里刻意保持两层嵌套而不是合并上下文：先建连接、再建会话，
            # 层级就是语义本身，合并后的 with 头会把 30 行主体挤到视线外。
            async with stdio_client(params) as (read, write):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self._tools = [
                        {
                            "name": t.name,
                            "description": _field(t, "description", default="") or "",
                            "inputSchema": _field(
                                t, "input_schema", "inputSchema",
                                default={"type": "object", "properties": {}},
                            ),
                            "annotations": _field(t, "annotations"),
                        }
                        for t in listed.tools
                    ]
                    self._alive = True
                    self._ready.set()
                    while True:
                        item = await self._queue.get()
                        if item is None:
                            break
                        call_name, call_args, fut = item
                        if fut.done():
                            continue
                        try:
                            res = await session.call_tool(call_name, call_args)
                            fut.set_result(_normalise(res))
                        except Exception as exc:
                            fut.set_exception(exc)
        except Exception as exc:
            self._error = _describe_exc(exc)
            self._alive = False
            self._ready.set()
        finally:
            self._drain_pending()

    def _drain_pending(self) -> None:
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is None:
                continue
            _, _, fut = item
            if not fut.done():
                fut.set_exception(RuntimeError(f"MCP 服务器 {self.name} 已断开"))

    async def call(self, tool: str, arguments: dict) -> dict:
        if not self._alive:
            return fail(f"MCP 服务器 {self.name} 不可用：{self._error or '未启动'}")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((tool, arguments, fut))
        try:
            return await fut
        except Exception as exc:
            return fail(f"MCP 调用失败: {type(exc).__name__}: {exc}")

    @property
    def tool_defs(self) -> list[dict]:
        return self._tools

    def status(self) -> dict:
        return {
            "name": self.name,
            "command": self.command,
            "alive": self._alive,
            "tools": len(self._tools),
            "error": self._error,
        }


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """按顺序取第一个存在的字段名。

    MCP Python SDK 在 1.x → 2.x 之间把模型字段从驼峰改成了下划线
    （`inputSchema` → `input_schema`、`structuredContent` → `structured_content`、
    `isError` → `is_error`）。我们的桥接层两个版本都要能跑，所以不在字段名上
    赌版本，而是逐个探测。
    """
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return default


def _describe_exc(exc: BaseException) -> str:
    """把异常（可能是 ExceptionGroup）摊平成一句人话。

    asyncio 的 TaskGroup 会把真实错误裹进 `ExceptionGroup` 里，直接 str() 只能
    看到「unhandled errors in a TaskGroup (1 sub-exception)」—— 对排查毫无用处。
    这里递归钻到叶子，去重后拼起来。
    """
    leaves: list[str] = []

    def walk(e: BaseException) -> None:
        subs = getattr(e, "exceptions", None)
        if subs:
            for s in subs:
                walk(s)
        else:
            leaves.append(f"{type(e).__name__}: {e}")

    walk(exc)
    uniq = list(dict.fromkeys(x.strip() for x in leaves if x.strip()))
    return " | ".join(uniq)[:600] or type(exc).__name__


def _normalise(res: Any) -> dict:
    """把 MCP 的 CallToolResult 变成 Loom 的 {ok, content, meta}。"""
    parts: list[str] = []
    for block in _field(res, "content", default=[]) or []:
        btype = _field(block, "type")
        if btype == "text":
            parts.append(_field(block, "text", default="") or "")
        elif btype in ("image", "audio"):
            data = _field(block, "data", default="") or ""
            parts.append(f"[{btype} 内容，{len(data)} 字节 base64]")
        elif btype == "resource":
            res_obj = _field(block, "resource")
            parts.append(
                _field(res_obj, "text", default=None)
                or f"[resource {_field(res_obj, 'uri', default='?')}]"
            )
        else:
            parts.append(json.dumps(block, default=str, ensure_ascii=False))
    structured = _field(res, "structured_content", "structuredContent")
    if not parts and structured:
        parts.append(json.dumps(structured, ensure_ascii=False))
    text = "\n".join(p for p in parts if p) or "(MCP 无文本输出)"
    is_error = bool(_field(res, "is_error", "isError", default=False))
    return fail(text, via="mcp") if is_error else ok(text, via="mcp")


class MCPHub:
    """按 `mcp/servers.json` 起若干 MCP 服务器，并把它们的工具并进注册表。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or config.MCP_CONFIG)
        self.conns: dict[str, MCPServerConn] = {}

    # -- 配置 -------------------------------------------------------------
    @staticmethod
    def _expand(value: Any) -> Any:
        """替换 `{python}` / `{root}` 占位符，让配置文件可以跨机器搬。

        没有这个，`servers.json` 里就得写死 `C:\\Users\\XXX\\...` 或 venv 里的
        python 绝对路径 —— 换个环境整台就废了。
        """
        if isinstance(value, str):
            return (value.replace("{python}", sys.executable)
                         .replace("{root}", str(config.ROOT))
                         .replace("{workdir}", str(config.WORKDIR)))
        if isinstance(value, list):
            return [MCPHub._expand(v) for v in value]
        if isinstance(value, dict):
            return {k: MCPHub._expand(v) for k, v in value.items()}
        return value

    def load_config(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        servers = raw.get("mcpServers", raw) if isinstance(raw, dict) else {}
        out: dict[str, dict] = {}
        for name, spec in (servers or {}).items():
            if not isinstance(spec, dict) or spec.get("disabled"):
                continue
            if not spec.get("command"):
                continue
            out[name] = self._expand(spec)
        return out

    # -- 生命周期 ---------------------------------------------------------
    async def start(self, registry: Registry) -> list[dict]:
        """启动所有配置的服务器并把工具注册进 `registry`。返回状态列表。"""
        specs = self.load_config()
        if not specs:
            return []
        for name, spec in specs.items():
            if shutil.which(spec["command"]) is None:
                self.conns[name] = MCPServerConn(
                    name=name, command=spec["command"], args=[]
                )
                self.conns[name]._error = f"找不到可执行文件: {spec['command']}"
                self.conns[name]._ready.set()
                continue
            conn = MCPServerConn(
                name=name,
                command=spec["command"],
                args=spec.get("args", []),
                env=spec.get("env", {}) or {},
                cwd=spec.get("cwd"),
            )
            await conn.start()
            self.conns[name] = conn
            self._register(registry, conn)
        return self.status()

    def _register(self, registry: Registry, conn: MCPServerConn) -> None:
        registry.unregister_source(f"mcp:{conn.name}")
        for spec in conn.tool_defs:
            name = spec["name"]
            if registry.get(name) is not None:  # 与内置或其他服务器重名 → 加前缀
                name = f"{conn.name}__{name}"
            if registry.get(name) is not None:
                continue

            def _make(conn_ref: MCPServerConn, real: str):
                async def _handler(**kwargs: Any) -> dict:
                    return await conn_ref.call(real, kwargs)
                return _handler

            schema = spec.get("inputSchema") or {}
            if not isinstance(schema, dict) or schema.get("type") != "object":
                schema = {"type": "object", "properties": {}}
            registry.register(Tool(
                name=name,
                description=(spec.get("description") or f"MCP 工具 {name}")[:1024],
                parameters={**schema, "additionalProperties": False},
                handler=_make(conn, spec["name"]),
                source=f"mcp:{conn.name}",
                tags=["mcp", conn.name],
            ))

    async def stop(self) -> None:
        for conn in self.conns.values():
            await conn.stop()
        self.conns.clear()

    def status(self) -> list[dict]:
        return [c.status() for c in self.conns.values()]

    def summary(self) -> dict:
        st = self.status()
        return {
            "servers": st,
            "alive_servers": sum(1 for s in st if s["alive"]),
            "tools": sum(s["tools"] for s in st),
            "configured": len(self.load_config()),
        }

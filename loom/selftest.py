"""不变量自检 —— 离线、确定性、零模型依赖。

**为什么不用真模型**：真模型每次输出都不一样，用来做门禁会偶发失败，
最后所有人都学会忽略红灯。这里用 `ScriptedBackend` 按脚本回放，
把协议不变量钉死；真模型的端到端验证交给 `python -m loom smoke`。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from . import config
from .llm import ScriptedBackend, _normalise_tool_calls
from .loop import SYSTEM_PROMPT, run_agent, trim_context
from .memory import Store, register_memory
from .tools import (
    Registry,
    SandboxError,
    Tool,
    build_registry,
    check_command,
    clip,
    fail,
    ok,
    safe_path,
)

Results = list[tuple[str, bool, str]]


# ---------------------------------------------------------------------------
# 断言小工具
# ---------------------------------------------------------------------------
def add(results: Results, name: str, passed: bool, detail: str = "") -> None:
    results.append((name, bool(passed), detail))


def assert_eq(results: Results, name: str, got, want, detail: str = "") -> None:
    add(results, name, got == want, detail or f"got={got!r} want={want!r}")


# ---------------------------------------------------------------------------
# 驱动器：把 run_agent 的事件流跑干净，同时校验协议
# ---------------------------------------------------------------------------
async def drain(
    backend: ScriptedBackend,
    registry: Registry,
    max_steps: int = 6,
    messages: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "测试"}]
    events: list[dict] = []
    async for ev in run_agent(backend, registry, messages, max_steps=max_steps):
        events.append(ev)
    return events, messages


def protocol_violations(messages: list[dict]) -> list[str]:
    """校验「每个 tool_call 恰好一个 tool 响应」这条铁律。"""
    problems: list[str] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            want = [tc.get("id") for tc in m["tool_calls"]]
            got: list[str] = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                got.append(messages[j].get("tool_call_id"))
                j += 1
            if want != got:
                problems.append(f"step@{i}: tool_calls={want} 但 tool 响应={got}")
            i = j
            continue
        i += 1
    return problems


# ---------------------------------------------------------------------------
# 主套件
# ---------------------------------------------------------------------------
def run() -> Results:
    r: Results = []
    tmp = Path(tempfile.mkdtemp(prefix="loom-selftest-"))
    original_workdir = config.WORKDIR
    config.WORKDIR = tmp

    try:
        reg = build_registry()

        # --- 1. schema 合法 -------------------------------------------------
        bad = [
            t.name for t in reg.all()
            if not t.name or not t.description
            or t.parameters.get("type") != "object"
            or not isinstance(t.parameters.get("properties", {}), dict)
        ]
        add(r, "[schema] 每个工具都有名字/描述/object 参数", not bad, f"违规: {bad}")

        need = [
            t.name for t in reg.all()
            if not set(t.parameters.get("required", [])) <= set(t.parameters["properties"])
        ]
        add(r, "[schema] required 只引用已声明的属性", not need, f"违规: {need}")

        # schema 与实现签名一致（防漂移）
        import inspect
        drift = []
        for t in reg.all():
            try:
                sig = inspect.signature(t.handler)
            except (TypeError, ValueError):
                continue
            params = set(sig.parameters)
            if not set(t.parameters.get("required", [])) <= params:
                drift.append(f"{t.name}: required 不在签名 {sorted(params)} 中")
            if not set(t.parameters.get("properties", {})) <= params:
                drift.append(f"{t.name}: 属性不在签名 {sorted(params)} 中")
        add(r, "[schema] 声明的参数在实现签名里都存在（防漂移）", not drift,
            "; ".join(drift[:3]))

        # --- 2. 沙箱 --------------------------------------------------------
        for probe in ["../../etc/passwd", "..\\..\\windows\\win.ini", "/etc/hosts"]:
            try:
                safe_path(probe)
                add(r, f"[沙箱] 拒绝越界路径 {probe!r}", False, "居然放行了")
            except SandboxError:
                add(r, f"[沙箱] 拒绝越界路径 {probe!r}", True)
        try:
            p = safe_path("sub/dir/file.txt")
            add(r, "[沙箱] 工作目录内的相对路径正常解析",
                p == (tmp / "sub/dir/file.txt").resolve(), str(p))
        except SandboxError as exc:
            add(r, "[沙箱] 工作目录内的相对路径正常解析", False, str(exc))

        # --- 3. shell 闸门 --------------------------------------------------
        dangerous = [
            "rm -rf /", "mkfs.ext4 /dev/sda1", "shutdown /s /t 0",
            "format C:", "reg delete HKLM\\Software /f", ":(){ :|:& };:",
        ]
        leaked = [c for c in dangerous if check_command(c) is None]
        add(r, "[安全] 破坏性命令全部被拒", not leaked, f"漏网: {leaked}")
        allowed = ["ls -la", "python -c 'print(1)'", "git status", "echo hi > a.txt"]
        blocked = [c for c in allowed if check_command(c) is not None]
        add(r, "[安全] 常规命令不被误杀", not blocked, f"误杀: {blocked}")
        add(r, "[安全] 空命令被拒", check_command("   ") is not None)

        # --- 4. 内置工具真跑 ------------------------------------------------
        async def tool_roundtrip() -> Results:
            out: Results = []
            w = await reg.dispatch("fs_write",
                                   {"path": "notes/a.txt", "content": "hello 世界"})
            add(out, "[工具] fs_write 写入成功", w["ok"], w.get("content", ""))
            rd = await reg.dispatch("fs_read", {"path": "notes/a.txt"})
            add(out, "[工具] fs_read 读回内容一致", rd["ok"] and "hello 世界" in rd["content"],
                rd.get("content", "")[:80])
            ls = await reg.dispatch("fs_list", {"path": "."})
            add(out, "[工具] fs_list 深度 1 只列直接子项",
                ls["ok"] and "notes/" in ls["content"] and "a.txt" not in ls["content"],
                ls.get("content", "")[:80])
            ls2 = await reg.dispatch("fs_list", {"path": ".", "depth": 2})
            add(out, "[工具] fs_list 深度 2 递归到子目录内容",
                ls2["ok"] and "a.txt" in ls2["content"],
                ls2.get("content", "")[:80])
            gr = await reg.dispatch("fs_grep", {"pattern": "世界", "glob": "*.txt"})
            add(out, "[工具] fs_grep 命中中文", gr["ok"] and "a.txt" in gr["content"],
                gr.get("content", "")[:80])
            esc = await reg.dispatch("fs_read", {"path": "../../etc/passwd"})
            add(out, "[工具] 越界读取被工具层拦下",
                (not esc["ok"]) and esc["meta"].get("kind") == "sandbox", str(esc["meta"]))
            unknown = await reg.dispatch("no_such_tool", {})
            add(out, "[工具] 未知工具返回结构化失败而非异常",
                (not unknown["ok"]) and unknown["meta"].get("kind") == "unknown_tool")
            badargs = await reg.dispatch("fs_read", {"nosuch": 1})
            add(out, "[工具] 参数不匹配返回结构化失败",
                (not badargs["ok"]) and badargs["meta"].get("kind") == "bad_args",
                str(badargs["meta"]))
            py = await reg.dispatch("python_run", {"code": "print(6*7)"})
            add(out, "[工具] python_run 可执行并回传输出", py["ok"] and "42" in py["content"],
                py.get("content", "")[:80])
            sh = await reg.dispatch("shell_run", {"command": "echo loom"})
            add(out, "[工具] shell_run 正常命令可跑", sh["ok"] and "loom" in sh["content"],
                sh.get("content", "")[:80])
            shd = await reg.dispatch("shell_run", {"command": "rm -rf /"})
            add(out, "[工具] shell_run 拒绝破坏性命令",
                (not shd["ok"]) and shd["meta"].get("kind") == "denied")
            wf = await reg.dispatch("web_fetch", {"url": "ftp://x"})
            add(out, "[工具] web_fetch 拒绝非 http(s) URL", not wf["ok"])
            return out

        r.extend(asyncio.run(tool_roundtrip()))

        # --- 5. 智能体循环：终止性 + 协议 -----------------------------------
        bad_model = ScriptedBackend([
            {"text": "我来查一下。", "calls": [{"name": "fs_list", "arguments": {"path": "."}}]}
            for _ in range(50)
        ])
        events, msgs = asyncio.run(drain(bad_model, reg, max_steps=4))
        finals = [e for e in events if e["type"] == "final"]
        add(r, "[循环] 模型无限要调工具时在 max_steps 处停下",
            len(finals) == 1 and finals[0]["reason"] == "max_steps"
            and finals[0]["steps"] == 4, str(finals[-1] if finals else None))
        # 上限精确生效：max_steps=4 就恰好调用模型 4 次，不多不少。
        # 注意最后一轮的**工具仍然执行**——否则那条带 tool_calls 的 assistant
        # 消息就没有对应的 tool 响应，transcript 会违反协议（见下一条断言）。
        add(r, "[循环] 模型调用次数恰好等于 max_steps（上限精确生效）",
            len(bad_model.calls) == 4, f"模型被调用 {len(bad_model.calls)} 次")

        problems = protocol_violations(msgs)
        add(r, "[协议] 每个 tool_call 恰好一个匹配 id 的 tool 响应", not problems,
            "; ".join(problems[:3]))

        # --- 6. 工具失败不中断循环 -------------------------------------------
        fail_model = ScriptedBackend([
            {"text": "先试一个不存在的工具。",
             "calls": [{"name": "ghost_tool", "arguments": {}}]},
            {"text": "好，改用真实工具。",
             "calls": [{"name": "fs_list", "arguments": {"path": "."}}]},
            {"text": "完成了。"},
        ])
        events, msgs = asyncio.run(drain(fail_model, reg, max_steps=6))
        finals = [e for e in events if e["type"] == "final"]
        tool_results = [e for e in events if e["type"] == "tool_result"]
        add(r, "[循环] 工具失败后循环继续并最终给出答案",
            finals and finals[-1]["reason"] == "complete" and finals[-1]["text"] == "完成了。",
            str(finals[-1] if finals else None))
        add(r, "[循环] 失败的工具调用也有对应结果事件",
            len(tool_results) == 2 and tool_results[0]["ok"] is False
            and tool_results[1]["ok"] is True, str([t["ok"] for t in tool_results]))
        add(r, "[协议] 含失败的会话仍然协议完整", not protocol_violations(msgs))

        # --- 7. 流式事件形状 -------------------------------------------------
        shape_model = ScriptedBackend([{"text": "abc", "calls": []}])
        events, _ = asyncio.run(drain(shape_model, reg, max_steps=3))
        add(r, "[流] 每轮都有 step 事件打头",
            events[0]["type"] == "step" and events[0]["n"] == 1, str(events[0]))
        add(r, "[流] 正文以增量 text 事件产出",
            "".join(e["delta"] for e in events if e["type"] == "text") == "abc")
        add(r, "[流] 事件流以唯一的 final 收尾",
            len([e for e in events if e["type"] == "final"]) == 1
            and events[-1]["type"] == "final")
        add(r, "[流] 所有事件都带 type 字段且可 JSON 序列化",
            all("type" in e for e in events)
            and all(json.dumps(e, ensure_ascii=False) for e in events))

        # --- 8. 上下文裁剪不拆散工具块 ---------------------------------------
        long_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i in range(20):
            long_msgs.append({"role": "user", "content": f"问题 {i}"})
            long_msgs.append({"role": "assistant", "content": "",
                              "tool_calls": [{"id": f"c{i}", "type": "function",
                                              "function": {"name": "fs_list",
                                                           "arguments": {}}}]})
            long_msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                              "name": "fs_list", "content": "{}"})
        trimmed = trim_context(long_msgs, limit=10)
        add(r, "[上下文] 裁剪后不超过上限", len(trimmed) <= 10, f"len={len(trimmed)}")
        add(r, "[上下文] system 消息永远保留",
            trimmed[0]["role"] == "system", str(trimmed[0])[:60])
        add(r, "[上下文] 裁剪没有拆散 tool_call 与 tool 响应",
            not protocol_violations(trimmed),
            "; ".join(protocol_violations(trimmed)[:3]))
        add(r, "[上下文] 未超限时原样返回",
            trim_context(long_msgs[:5], limit=99) == long_msgs[:5])

        # --- 9. 长期记忆 ------------------------------------------------------
        store = Store(tmp / "mem.sqlite3")
        mem_reg = Registry()
        register_memory(mem_reg, store, "s1")

        async def memory_checks() -> Results:
            out: Results = []
            s = await mem_reg.dispatch("memory_save",
                                       {"text": "用户是前端工程师，偏好 React Hooks",
                                        "tags": "用户,偏好"})
            add(out, "[记忆] 写入返回新 id", s["ok"] and s["meta"].get("id", 0) > 0,
                str(s.get("meta")))
            hit = await mem_reg.dispatch("memory_search", {"query": "React 偏好"})
            add(out, "[记忆] 相关查询命中刚写入的事实",
                hit["ok"] and "前端工程师" in hit["content"], hit.get("content", "")[:90])
            miss = await mem_reg.dispatch("memory_search", {"query": "猫咪粮食采购"})
            add(out, "[记忆] 无关查询不返回噪声",
                miss["ok"] and "没有相关记忆" in miss["content"],
                miss.get("content", "")[:60])
            empty = await mem_reg.dispatch("memory_save", {"text": "   "})
            add(out, "[记忆] 空内容被拒", not empty["ok"])
            return out

        r.extend(asyncio.run(memory_checks()))
        add(r, "[记忆] 计数与写入一致", store.count_facts() == 1, str(store.count_facts()))

        # 会话隔离
        a = store.create_session("A")
        b = store.create_session("B")
        store.append(a, {"role": "user", "content": "a-msg"})
        add(r, "[记忆] 会话之间消息互不串台",
            len(store.messages(a)) == 1 and len(store.messages(b)) == 0)

        # --- 10. 边界与辅助函数 -----------------------------------------------
        text, cut = clip("x" * 20000, limit=1000)
        add(r, "[边界] 超长观测被截断且标注", cut and "已截断" in text and len(text) < 1400,
            f"len={len(text)}")
        add(r, "[边界] 正常长度内容不截断", clip("短内容")[1] is False)
        add(r, "[边界] ok()/fail() 必然带 ok 字段",
            ok("x")["ok"] is True and fail("x")["ok"] is False)
        add(r, "[边界] 非法 tool_calls 结构被安全归一",
            _normalise_tool_calls([{"function": {"name": "t", "arguments": "{bad json"}}])
            == [{"name": "t", "arguments": {"__raw__": "{bad json"}}])
        add(r, "[边界] 非列表 tool_calls 归零", _normalise_tool_calls("nope") == [])

        # --- 11. SSE 帧格式（直接复用 server 的编码器）------------------------
        from .server import sse
        frame = sse({"type": "text", "delta": "你好\n世界"})
        add(r, "[SSE] 帧以 data: 开头并以空行结束",
            frame.startswith("data: ") and frame.endswith("\n\n"), repr(frame[:40]))
        add(r, "[SSE] 载荷可无损还原",
            json.loads(frame[6:].strip())["delta"] == "你好\n世界")
        add(r, "[SSE] 换行不会破坏帧结构（json 转义）",
            frame.count("\n\n") == 1, repr(frame))

        # --- 12. MCP 桥接（不真的起进程，测翻译层）----------------------------
        from types import SimpleNamespace

        from .mcp import MCPHub, MCPServerConn, _describe_exc, _field
        from .tools import Tool as _Tool  # noqa: F401

        # 用 SimpleNamespace 而不是自定义类：这里要的只是「有这些字段的对象」，
        # 类定义会多余，还会被 lint 规则当成可变类属性。
        camel = SimpleNamespace(
            inputSchema={"type": "object", "properties": {"a": {"type": "string"}}},
            structuredContent={"x": 1},
            isError=True,
        )
        snake = SimpleNamespace(
            input_schema={"type": "object", "properties": {"b": {"type": "integer"}}},
            structured_content={"y": 2},
            is_error=False,
        )

        add(r, "[MCP] 字段名兼容 1.x 驼峰写法",
            _field(camel, "input_schema", "inputSchema")["properties"]["a"]["type"] == "string"
            and _field(camel, "is_error", "isError") is True
            and _field(camel, "structured_content", "structuredContent") == {"x": 1})
        add(r, "[MCP] 字段名兼容 2.x 下划线写法",
            _field(snake, "input_schema", "inputSchema")["properties"]["b"]["type"] == "integer"
            and _field(snake, "is_error", "isError") is False)
        add(r, "[MCP] 字段缺失时返回 default 而非抛异常",
            _field(object(), "nope", default="D") == "D")

        hub = MCPHub(tmp / "no-such-servers.json")
        add(r, "[MCP] 无配置文件时安静地返回空", hub.load_config() == {})
        add(r, "[MCP] 无配置时 summary 结构完整",
            hub.summary() == {"servers": [], "alive_servers": 0, "tools": 0, "configured": 0})

        cfg = tmp / "servers.json"
        cfg.write_text(json.dumps({"mcpServers": {
            "a": {"command": "{python}", "args": ["{root}/mcp/system_server.py"]},
            "b": {"command": "x", "disabled": True},
            "c": {"args": ["no-command"]},
        }}), encoding="utf-8")
        loaded = MCPHub(cfg).load_config()
        add(r, "[MCP] 占位符 {python}/{root} 被展开成绝对可用路径",
            loaded["a"]["command"] == sys.executable
            and loaded["a"]["args"][0].endswith("system_server.py")
            and "{root}" not in loaded["a"]["args"][0],
            str(loaded.get("a")))
        add(r, "[MCP] 显式 disabled 的服务器被跳过", "b" not in loaded)
        add(r, "[MCP] 缺 command 的条目被跳过", "c" not in loaded)

        # 工具重名 → 自动加服务器前缀，绝不互相覆盖
        collide = Registry()
        collide.register(Tool("dup", "内置的", {"type": "object", "properties": {}},
                              lambda **_: None))  # type: ignore[arg-type]
        fake = MCPServerConn(name="srv", command="x")
        fake._tools = [{"name": "dup", "description": "来自 MCP",
                        "inputSchema": {"type": "object", "properties": {}}}]
        MCPHub(cfg)._register(collide, fake)
        add(r, "[MCP] 工具重名时自动加服务器前缀，不互相覆盖",
            collide.get("dup").source == "builtin"
            and collide.get("srv__dup") is not None
            and collide.get("srv__dup").source == "mcp:srv",
            str(collide.names()))

        class _Group(BaseExceptionGroup):  # type: ignore[misc]
            pass

        nested = _Group("outer", [_Group("inner", [ValueError("真正的原因")])])
        add(r, "[MCP] ExceptionGroup 被摊平成可读的根因",
            "真正的原因" in _describe_exc(nested)
            and "TaskGroup" not in _describe_exc(nested),
            _describe_exc(nested))

        # --- 13. 前端资源（单文件、零外链）------------------------------------
        page = config.STATIC_DIR / "index.html"
        html = page.read_text(encoding="utf-8") if page.exists() else ""
        add(r, "[界面] static/index.html 存在且不是占位文件", len(html) > 4000,
            f"{len(html)} 字节")
        needed = ['id="transcript"', 'id="input"', "/api/chat", "/api/health",
                  "/api/sessions", "/api/tools"]
        missing = [x for x in needed if x not in html]
        add(r, "[界面] 关键挂载点与接口调用都在", not missing, f"缺: {missing}")
        # 零外部依赖：不能有任何指向外站的 script/link
        import re as _re
        ext = _re.findall(r'(?:src|href)\s*=\s*["\']https?://[^"\']+', html)
        add(r, "[界面] 零外部依赖（无 CDN/外链资源）", not ext, str(ext[:3]))
        add(r, "[界面] 用户输入经 esc() 转义后再渲染",
            "function esc(" in html and "esc(" in html.split("function render")[1][:2000])

        # --- 14. system 提示词每轮必注入 -------------------------------------
        # 曾经的写法是 `if not history: history = [system]` —— 结果从第二轮起
        # 模型就丢了全部行为约束，退化成裸模型。这条断言专门钉死它。
        from .server import build_messages

        m0 = build_messages([])
        add(r, "[提示词] 空历史也注入 system",
            len(m0) == 1 and m0[0]["role"] == "system"
            and m0[0]["content"] == SYSTEM_PROMPT, str(m0)[:80])

        hist = [{"role": "user", "content": "第一问"},
                {"role": "assistant", "content": "第一答"},
                {"role": "user", "content": "第二问"}]
        m1 = build_messages(hist)
        add(r, "[提示词] 第二轮（历史非空）依然注入 system",
            m1[0]["role"] == "system" and m1[0]["content"] == SYSTEM_PROMPT,
            str(m1[0])[:80])
        add(r, "[提示词] 注入后历史顺序与内容不变",
            m1[1:] == hist and len(m1) == len(hist) + 1)
        already = [{"role": "system", "content": "自定义"}, {"role": "user", "content": "x"}]
        add(r, "[提示词] 历史里已有 system 时不重复注入",
            build_messages(already) == already)
        add(r, "[提示词] 原历史列表未被就地修改",
            hist == [{"role": "user", "content": "第一问"},
                     {"role": "assistant", "content": "第一答"},
                     {"role": "user", "content": "第二问"}])

        # 端到端确认：带历史的第二轮，模型端真的收到了 system
        probe = ScriptedBackend([{"text": "好的。"}])
        seeded = build_messages([{"role": "user", "content": "第一问"},
                                 {"role": "assistant", "content": "第一答"},
                                 {"role": "user", "content": "第二问"}])
        asyncio.run(drain(probe, reg, max_steps=2, messages=seeded))
        first_seen = probe.calls[0][0] if probe.calls else {}
        add(r, "[提示词] 送到模型的第一条消息就是 system",
            first_seen.get("role") == "system"
            and first_seen.get("content") == SYSTEM_PROMPT, str(first_seen)[:80])

        # --- 15. 线程数必须锁在 2–4 -------------------------------------------
        # 实测 16 线程比 4 线程慢 1.93×：小模型解码卡在内存带宽上，
        # 线程越多越互相抢总线。这条把它钉成回归测试，别让谁「顺手」改成
        # cpu_count-1。
        from .config import auto_threads

        picks = {n: auto_threads(n) for n in (1, 2, 4, 8, 16, 32, 64)}
        add(r, "[性能] 自动线程数恒在 [2, 4]",
            all(2 <= v <= 4 for v in picks.values()), str(picks))
        add(r, "[性能] 16 核机器上不跟随核数（锁 4）", auto_threads(16) == 4,
            f"得到 {auto_threads(16)}")
        add(r, "[性能] 核数很少时也不低于 2", auto_threads(2) == 2, f"得到 {auto_threads(2)}")

        # --- 16. 请求体构造（think 字段的取舍是实测结论，不是口味）---------------
        from .llm import build_payload

        p_auto = build_payload("m", [{"role": "user", "content": "x"}])
        add(r, "[请求体] think=None 时完全不发该字段（让后端按模板决定）",
            "think" not in p_auto, str(sorted(p_auto)))
        add(r, "[请求体] 显式 False / True 时才带 think",
            build_payload("m", [], think=False)["think"] is False
            and build_payload("m", [], think=True)["think"] is True)
        add(r, "[请求体] 线程数恒被显式指定",
            p_auto["options"]["num_thread"] == config.NUM_THREADS,
            str(p_auto["options"]))
        add(r, "[请求体] 无工具时不带 tools 字段",
            "tools" not in p_auto)
        add(r, "[请求体] 有工具时才带 tools 字段",
            build_payload("m", [], [{"type": "function"}]).get("tools")
            == [{"type": "function"}])
        add(r, "[请求体] 默认就是流式", p_auto["stream"] is True)
        add(r, "[请求体] 温度可覆盖且不污染全局",
            build_payload("m", [], temperature=0.1)["options"]["temperature"] == 0.1
            and build_payload("m", [])["options"]["temperature"] == config.TEMPERATURE)

    finally:
        config.WORKDIR = original_workdir

    return r


def main() -> int:
    results = run()
    failed = [x for x in results if not x[1]]
    width = max(len(n) for n, _, _ in results)
    for name, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        line = f"  [{mark}] {name.ljust(width)}"
        if detail and not passed:
            line += f"   <- {detail}"
        print(line)
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print(f"SOME FAILED ({len(failed)})")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("ALL GREEN")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

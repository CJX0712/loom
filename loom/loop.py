"""智能体循环 —— Loom 的心脏。

协议不变量（自检会逐条断言）
--------------------------
1. 每一条带 `tool_calls` 的 assistant 消息，后面**必须**紧跟**数量相等、id 一一对应**
   的 tool 消息。Ollama / OpenAI 都强制这个形状，缺一条整个会话就废了。
2. 循环**必然终止**：最多 `max_steps` 轮往返，且每一轮都至少消耗一次模型调用。
3. 工具失败**不会**中断循环 —— 失败被当作一次普通观测回灌给模型，让它自救。
4. 取消（cancel）发生在任意时刻，都会产出一条结构完整的 transcript。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from . import config
from .llm import ChatBackend, LLMError
from .tools import Registry

SYSTEM_PROMPT = """你是 Loom，一个完全运行在本地的智能体。

工作方式：
- 你手上有真实工具。**需要事实、文件内容或计算结果时就去调用工具，不要凭记忆编造。**
- 一次只做一件可验证的小事：先看清楚现状，再动手，再看结果。
- 工具返回 `{"ok": false}` 时不要放弃，读错误信息、换一个参数或换一个工具重试。
- 所有文件操作都限制在当前工作目录内，越界会被拒绝 —— 不要试图绕开。
- 回答用中文。

**输出纪律（很重要）：**
- **直接给结论，不要复述你的思考过程。** 不要写"用户让我…""首先我需要…""我尝试了…"
  "综上来看…"这类内心独白的转述。
- 不要向用户解释你要做什么，直接做、然后给结果。
- 不要贴大段原始数据；该归纳的归纳，该给文件路径的给路径。
- 一次回答控制在一小段内。用户要的是一句话就说清的东西，就只给一句话。
"""


@dataclass
class Step:
    """一轮「模型 → 工具 → 观测」。"""

    index: int
    text: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    ms: float = 0.0


def trim_context(messages: list[dict], limit: int | None = None) -> list[dict]:
    """滑动窗口裁剪上下文，**但绝不拆散 tool_call 与它的 tool 响应**。

    做法：从尾部往前收集，遇到 `role == "tool"` 就继续往前直到找到发起的
    assistant 消息，然后整块保留。system 消息永远保留。
    """
    limit = limit or config.MAX_CONTEXT_MESSAGES
    if len(messages) <= limit:
        return list(messages)
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    kept: list[dict] = []
    i = len(rest) - 1
    while i >= 0 and len(kept) + len(system) < limit:
        msg = rest[i]
        if msg.get("role") == "tool":
            # 往前找这条 tool 响应所属的 assistant 消息，整组一起收
            j = i
            while j >= 0 and rest[j].get("role") == "tool":
                j -= 1
            if j >= 0 and rest[j].get("role") == "assistant":
                block = rest[j : i + 1]
            else:
                block = rest[i : i + 1]
            if len(kept) + len(system) + len(block) > limit and kept:
                break
            kept = block + kept
            i = j - 1
        else:
            kept.insert(0, msg)
            i -= 1
    return system + kept


def _jsonify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


async def run_agent(
    backend: ChatBackend,
    registry: Registry,
    messages: list[dict],
    *,
    model: str | None = None,
    max_steps: int | None = None,
    think: bool | None = None,
    cancel: asyncio.Event | None = None,
) -> AsyncIterator[dict]:
    """跑一轮完整的智能体对话，流式产出事件，并把新消息写回 `messages`。

    调用方负责持有 `messages`（含 system 消息）并把它持久化。
    """
    max_steps = max_steps or config.MAX_STEPS
    think = config.THINK if think is None else think
    schemas = registry.schemas()
    total_prompt = total_completion = 0

    for step_idx in range(1, max_steps + 1):
        if cancel is not None and cancel.is_set():
            yield {"type": "final", "text": "", "steps": step_idx - 1,
                   "reason": "cancelled"}
            return

        yield {"type": "step", "n": step_idx, "max": max_steps}
        step = Step(index=step_idx)
        t0 = time.perf_counter()

        window = trim_context(messages)
        calls: list[dict] = []
        text_parts: list[str] = []
        try:
            async for ev in backend.chat_stream(window, schemas, model=model, think=think):
                if cancel is not None and cancel.is_set():
                    break
                if ev["type"] == "text":
                    text_parts.append(ev["text"])
                    yield {"type": "text", "delta": ev["text"]}
                elif ev["type"] == "thinking":
                    yield {"type": "thinking", "delta": ev["text"]}
                elif ev["type"] == "usage":
                    step.usage = ev
                    total_prompt += ev.get("prompt_tokens", 0)
                    total_completion += ev.get("completion_tokens", 0)
                elif ev["type"] == "tool_calls":
                    calls = ev["calls"]
        except LLMError as exc:
            yield {"type": "error", "message": str(exc)}
            yield {"type": "final", "text": "", "steps": step_idx - 1, "reason": "llm_error"}
            return

        step.text = "".join(text_parts)
        step.tool_calls = calls
        step.ms = round((time.perf_counter() - t0) * 1000, 1)

        # --- 记录 assistant 消息（带 tool_calls 的形状必须完整）-------------
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": step.text}
        call_ids: list[str] = []
        if calls:
            tool_calls_payload = []
            for i, c in enumerate(calls):
                cid = c.get("id") or f"call_{step_idx}_{i}"
                call_ids.append(cid)
                tool_calls_payload.append({
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": c.get("name", ""),
                        "arguments": c.get("arguments", {}),
                    },
                })
            assistant_msg["tool_calls"] = tool_calls_payload
            calls = [{**c, "id": cid} for c, cid in zip(calls, call_ids, strict=True)]
        messages.append(assistant_msg)
        yield {"type": "usage", **step.usage, "step": step_idx, "ms": step.ms}

        if not calls:
            yield {
                "type": "final",
                "text": step.text,
                "steps": step_idx,
                "reason": "complete",
                "usage": {"prompt_tokens": total_prompt,
                          "completion_tokens": total_completion},
            }
            return

        # --- 执行工具，并把「每一条都必有响应」落实 -------------------------
        for c in calls:
            name = c.get("name") or ""
            args = c.get("arguments") or {}
            yield {"type": "tool_call", "id": c["id"], "name": name,
                   "arguments": args, "step": step_idx}

            if not name:
                result = {"ok": False, "content": "模型没有给出工具名",
                          "meta": {"kind": "no_name"}}
            else:
                result = await registry.dispatch(name, args)

            step.results.append({"name": name, "id": c["id"], **result})
            payload = _jsonify({"ok": result.get("ok"),
                                "content": result.get("content", ""),
                                "meta": result.get("meta", {})})
            messages.append({
                "role": "tool",
                "tool_call_id": c["id"],
                "name": name or "unknown",
                "content": payload,
            })
            yield {"type": "tool_result", "id": c["id"], "name": name,
                   "ok": bool(result.get("ok")), "content": result.get("content", ""),
                   "meta": result.get("meta", {}), "step": step_idx}

    yield {"type": "final", "text": "", "steps": max_steps, "reason": "max_steps"}

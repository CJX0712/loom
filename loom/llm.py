"""Ollama 推理客户端 —— 流式输出 + 原生工具调用。

只依赖 Ollama 的 HTTP API，不绑定任何 SDK，因此可以被任何实现了同样的
`chat_stream()` 后端的对象替换（见文件末尾的 `ScriptedBackend`，自检用它做
确定性回放，不需要下载任何模型）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol

import httpx

from . import config


class LLMError(RuntimeError):
    """推理后端不可用或返回了非法响应。"""


class ChatBackend(Protocol):
    """所有推理后端必须实现的最小接口。"""

    name: str

    def chat_stream(
        self, messages: list[dict], tools: list[dict] | None = None, **opts: Any
    ) -> AsyncIterator[dict]:  # pragma: no cover - 协议声明
        ...


def _normalise_tool_calls(raw: Any) -> list[dict]:
    """把后端给的 tool_calls 归一成 `{"name": str, "arguments": dict}`。

    Ollama 在不同版本里 arguments 可能是 dict，也可能是 JSON 字符串；
    流式分片时还可能是半个 JSON。这里能解析就解析，解析不了就原样保留，
    由上层在流结束时做最后一次合并重试 —— 但绝不抛异常。
    """
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = fn.get("name") or item.get("name") or ""
        args = fn.get("arguments", item.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"__raw__": args}
        if not isinstance(args, dict):
            args = {"__raw__": args}
        if name:
            out.append({"name": name, "arguments": args})
    return out


def build_payload(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    temperature: float | None = None,
    think: bool | None = None,
) -> dict:
    """构造发给 Ollama `/api/chat` 的请求体。

    单独抽出来是为了能被自检直接断言 —— 尤其是 `think` 的处理规则：
    **只有显式给 True/False 时才带这个字段，None 表示完全不提**，让 Ollama
    按模型自己的模板决定。见 config.THINK 那段注释里为什么默认不提。
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            # 显式锁线程数。跟随核数会让小模型解码慢近一倍（见 config.auto_threads）。
            "num_thread": config.NUM_THREADS,
        },
    }
    if tools:
        payload["tools"] = tools
    if think is not None:
        payload["think"] = think
    return payload


class OllamaBackend:
    """Ollama 的流式聊天后端。"""

    name = "ollama"

    def __init__(self, host: str | None = None, timeout: float | None = None) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        # 读超时给足：CPU 上新模型第一次加载可能要几十秒
        self._timeout = timeout or 600.0

    # -- 服务发现 ---------------------------------------------------------
    async def available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=4.0) as c:
                r = await c.get(f"{self.host}/api/version")
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{self.host}/api/tags")
                r.raise_for_status()
                data = r.json()
        except Exception as exc:  # 服务没起 / 端口不对
            raise LLMError(f"无法连接 Ollama ({self.host}): {exc}") from exc
        return [m["name"] for m in data.get("models", []) if m.get("name")]

    async def pick_model(self) -> str:
        """按偏好表挑一个本地已有的模型；一个都没有时抛错并给出修复命令。"""
        installed = await self.list_models()
        if not installed:
            raise LLMError(
                "Ollama 里没有可用模型。先执行：ollama pull "
                f"{config.DEFAULT_MODEL}"
            )
        if config.DEFAULT_MODEL in installed:
            return config.DEFAULT_MODEL
        for pref in config.MODEL_PREFERENCES:
            if pref in installed:
                return pref
        # 退一步：任意一个，但明确告知调用方这不是首选
        return sorted(installed)[0]

    # -- 主循环 -----------------------------------------------------------
    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[dict]:
        """流式对话。产出的事件类型：

        - ``{"type": "text", "text": str}``        增量正文
        - ``{"type": "thinking", "text": str}``    增量思维链（模型支持时）
        - ``{"type": "usage", ...}``               结束时的一次性统计
        - ``{"type": "tool_calls", "calls": [...]}`` 本轮的工具调用（已合并完整）
        """
        payload = build_payload(
            model or config.DEFAULT_MODEL, messages, tools,
            temperature=temperature, think=think,
        )

        acc: dict[int, dict] = {}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c, c.stream(
                "POST", f"{self.host}/api/chat", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise LLMError(f"Ollama {resp.status_code}: {body[:400]}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if err := chunk.get("error"):
                        raise LLMError(str(err))
                    msg = chunk.get("message") or {}

                    if piece := msg.get("thinking"):
                        yield {"type": "thinking", "text": piece}
                    if piece := msg.get("content"):
                        yield {"type": "text", "text": piece}

                    for i, tc in enumerate(msg.get("tool_calls") or []):
                        slot = acc.setdefault(i, {"name": "", "arguments": {}})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        args = fn.get("arguments", {})
                        if isinstance(args, dict):
                            slot["arguments"].update(args)
                        elif isinstance(args, str) and args.strip():
                            merged = slot.get("_raw", "") + args
                            slot["_raw"] = merged
                            try:
                                slot["arguments"] = json.loads(merged)
                                slot.pop("_raw", None)
                            except json.JSONDecodeError:
                                pass

                    if chunk.get("done"):
                        yield {
                            "type": "usage",
                            "prompt_tokens": chunk.get("prompt_eval_count", 0),
                            "completion_tokens": chunk.get("eval_count", 0),
                            "total_ms": round(
                                (chunk.get("total_duration") or 0) / 1e6, 1
                            ),
                        }
        except httpx.HTTPError as exc:
            raise LLMError(f"与 Ollama 通信失败: {exc}") from exc

        calls = [acc[i] for i in sorted(acc)]
        for c in calls:
            c.pop("_raw", None)
        yield {"type": "tool_calls", "calls": calls}


class ScriptedBackend:
    """确定性的假后端：按脚本逐轮流式吐出预设内容。

    自检用。**不下载任何模型、不联网**，因此可以放进 CI 当硬门禁。
    """

    name = "scripted"

    def __init__(self, turns: Iterable[dict] | None = None) -> None:
        self.turns = list(turns or [])
        self.calls: list[list[dict]] = []  # 记录每次收到的 messages，供断言
        self._i = 0

    async def list_models(self) -> list[str]:
        return ["scripted"]

    async def pick_model(self) -> str:
        return "scripted"

    async def available(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.turns)

    async def chat_stream(
        self, messages: list[dict], tools: list[dict] | None = None, **opts: Any
    ) -> AsyncIterator[dict]:
        self.calls.append([dict(m) for m in messages])
        turn = self.turns[self._i] if self._i < len(self.turns) else {"text": ""}
        self._i += 1

        for chunk in turn.get("text", "").split("\u0000"):
            if chunk:
                yield {"type": "text", "text": chunk}
        yield {
            "type": "usage",
            "prompt_tokens": sum(len(str(m.get("content", ""))) for m in messages),
            "completion_tokens": len(turn.get("text", "")),
            "total_ms": 0.0,
        }
        yield {"type": "tool_calls", "calls": [dict(c) for c in turn.get("calls", [])]}

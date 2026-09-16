"""HTTP 服务：FastAPI + SSE 流式。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .llm import LLMError, OllamaBackend
from .loop import SYSTEM_PROMPT, run_agent
from .mcp import MCPHub
from .memory import Store, register_memory
from .tools import Registry, build_registry

STATE: dict[str, Any] = {}


class ChatIn(BaseModel):
    session_id: str | None = None
    text: str = Field(min_length=1, max_length=32000)
    model: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=40)


class SessionIn(BaseModel):
    title: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = build_registry()
    store = Store(config.DB_PATH)
    hub = MCPHub()
    backend = OllamaBackend()

    STATE.update(
        registry=registry,
        store=store,
        hub=hub,
        backend=backend,
        cancel={},
        model=None,
        started_at=time.time(),
        mcp_status=[],
    )

    # 长期记忆工具（跨会话）
    register_memory(registry, store)
    # MCP：连不上不影响启动，状态照实汇报
    try:
        STATE["mcp_status"] = await hub.start(registry)
    except Exception as exc:
        STATE["mcp_status"] = [{"name": "hub", "alive": False, "tools": 0, "error": str(exc)}]

    try:
        STATE["model"] = await backend.pick_model()
    except LLMError as exc:
        STATE["model_error"] = str(exc)
    yield
    await hub.stop()
    store.close()


app = FastAPI(title="Loom", version=config.__version__, lifespan=lifespan)


# ---------------------------------------------------------------------------
# 静态资源与健康检查
# ---------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    page = config.STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "static/index.html 缺失")
    return FileResponse(page, media_type="text/html; charset=utf-8")


@app.get("/api/health")
async def health() -> dict:
    backend: OllamaBackend = STATE["backend"]
    registry: Registry = STATE["registry"]
    hub: MCPHub = STATE["hub"]
    ollama_up = await backend.available()
    return {
        "ok": ollama_up and STATE.get("model") is not None,
        "config": config.describe(),
        "ollama": {"up": ollama_up, "host": config.OLLAMA_HOST},
        "model": STATE.get("model"),
        "model_error": STATE.get("model_error"),
        "tools": len(registry.names()),
        "tool_sources": sorted({t.source for t in registry.all()}),
        "mcp": hub.summary(),
        "memory_facts": STATE["store"].count_facts(),
        "sessions": len(STATE["store"].list_sessions(limit=1000)),
        "uptime_s": round(time.time() - STATE["started_at"], 1),
    }


@app.get("/api/tools")
async def tools() -> dict:
    registry: Registry = STATE["registry"]
    return {"count": len(registry.names()), "tools": registry.describe()}


@app.get("/api/models")
async def models() -> dict:
    backend: OllamaBackend = STATE["backend"]
    try:
        return {"models": await backend.list_models(), "current": STATE.get("model")}
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------
@app.post("/api/sessions")
async def create_session(body: SessionIn) -> dict:
    sid = STATE["store"].create_session(body.title)
    return {"id": sid, "messages": []}


@app.get("/api/sessions")
async def list_sessions() -> dict:
    return {"sessions": STATE["store"].list_sessions()}


@app.get("/api/sessions/{sid}")
async def get_session(sid: str) -> dict:
    store: Store = STATE["store"]
    if not store.exists(sid):
        raise HTTPException(404, "会话不存在")
    return {"id": sid, "messages": store.messages(sid)}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str) -> dict:
    return {"deleted": STATE["store"].delete_session(sid)}


@app.post("/api/sessions/{sid}/cancel")
async def cancel_session(sid: str) -> dict:
    ev: asyncio.Event | None = STATE["cancel"].get(sid)
    if ev is None:
        return {"cancelled": False, "reason": "该会话当前没有在跑的轮次"}
    ev.set()
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# 聊天（SSE）
# ---------------------------------------------------------------------------
def build_messages(history: list[dict]) -> list[dict]:
    """把库里的历史拼成这一轮要发给模型的完整消息列表。

    **system 提示词每一轮都要重新注入。** 它属于代码、不属于用户数据，所以不入库；
    但反过来说，绝不能只在会话第一轮注入 —— 否则从第二轮起模型就丢了全部行为约束，
    变成裸模型。这是个很容易写错的形状：

        if not history:
            history = [{"role": "system", ...}]   # ← 只有首轮有，之后全丢

    历史里若已经带了 system（理论上不该有），也不重复注入。
    """
    if history and history[0].get("role") == "system":
        return list(history)
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history]


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request) -> StreamingResponse:
    store: Store = STATE["store"]
    registry: Registry = STATE["registry"]
    backend: OllamaBackend = STATE["backend"]

    sid = body.session_id
    if sid and not store.exists(sid):
        raise HTTPException(404, "会话不存在")
    if not sid:
        sid = store.create_session(body.text[:60])

    history = store.messages(sid)
    messages: list[dict] = build_messages(history)

    user_msg = {"role": "user", "content": body.text}
    messages.append(user_msg)
    store.append(sid, user_msg)
    store.touch(sid, title=body.text[:60] if len(store.messages(sid)) <= 2 else None)

    cancel = asyncio.Event()
    STATE["cancel"][sid] = cancel
    model = body.model or STATE.get("model") or config.DEFAULT_MODEL

    async def gen() -> AsyncIterator[str]:
        yield sse({"type": "session", "session_id": sid, "model": model})
        persisted = len(messages)
        t0 = time.perf_counter()
        try:
            async for ev in run_agent(
                backend, registry, messages,
                model=model, max_steps=body.max_steps, cancel=cancel,
            ):
                # 把新产生的消息增量落盘，客户端中途断开也不丢 transcript
                while persisted < len(messages):
                    store.append(sid, messages[persisted])
                    persisted += 1
                yield sse(ev)
                if await request.is_disconnected():
                    cancel.set()
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            yield sse({"type": "final", "text": "", "steps": 0, "reason": "exception"})
        finally:
            while persisted < len(messages):
                store.append(sid, messages[persisted])
                persisted += 1
            STATE["cancel"].pop(sid, None)
            yield sse({"type": "done", "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=503)

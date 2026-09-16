"""持久化：会话、消息、长期记忆。

用标准库 sqlite3，零额外依赖。所有写入走同一个连接 + 一把锁，
既可以给异步服务用（通过 `asyncio.to_thread`），也可以给同步自检用。
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config
from .llm import ChatBackend
from .rag import EmbeddingBackend, cosine
from .tools import Registry, Tool, fail, ok

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    session_id  TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_text ON facts(text);
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id     INTEGER PRIMARY KEY,
    embedding   TEXT NOT NULL
);
"""

_WORD_RX = re.compile(r"[\w\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    """中英混排的粗粒度分词：英文按词、中文按 2-gram。够用且零依赖。"""
    out: set[str] = set()
    for w in _WORD_RX.findall((text or "").lower()):
        if w.isascii():
            if len(w) > 2:
                out.add(w)
        else:
            out.add(w)
            for i in range(len(w) - 1):
                out.add(w[i : i + 2])
    return out


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- 会话 -------------------------------------------------------------
    def create_session(self, title: str = "") -> str:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (sid, title[:120], now, now),
            )
            self._db.commit()
        return sid

    def touch(self, sid: str, title: str | None = None) -> None:
        with self._lock:
            if title:
                self._db.execute(
                    "UPDATE sessions SET updated_at=?, title=? WHERE id=?",
                    (time.time(), title[:120], sid),
                )
            else:
                self._db.execute(
                    "UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid)
                )
            self._db.commit()

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.id, s.title, s.created_at, s.updated_at,"
                " (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS n"
                " FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, sid: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM sessions WHERE id=?", (sid,))
            self._db.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            self._db.commit()
        return cur.rowcount > 0

    def exists(self, sid: str) -> bool:
        with self._lock:
            return self._db.execute(
                "SELECT 1 FROM sessions WHERE id=?", (sid,)
            ).fetchone() is not None

    # -- 消息 -------------------------------------------------------------
    def append(self, sid: str, message: dict) -> None:
        with self._lock:
            nxt = self._db.execute(
                "SELECT COALESCE(MAX(seq),0)+1 FROM messages WHERE session_id=?", (sid,)
            ).fetchone()[0]
            self._db.execute(
                "INSERT INTO messages(session_id,seq,role,payload,created_at)"
                " VALUES(?,?,?,?,?)",
                (sid, nxt, message.get("role", "?"),
                 json.dumps(message, ensure_ascii=False), time.time()),
            )
            self._db.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid)
            )
            self._db.commit()

    def messages(self, sid: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM messages WHERE session_id=? ORDER BY seq", (sid,)
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # -- 长期记忆 ---------------------------------------------------------
    def save_fact(self, text: str, tags: str = "", session_id: str | None = None,
                  vec: list[float] | None = None) -> int:
        text = text.strip()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO facts(text,tags,session_id,created_at) VALUES(?,?,?,?)",
                (text, tags.strip(), session_id, time.time()),
            )
            fid = int(cur.lastrowid)
            if vec is not None:
                self._db.execute(
                    "INSERT INTO fact_embeddings(fact_id,embedding) VALUES(?,?)",
                    (fid, json.dumps([float(x) for x in vec])),
                )
            self._db.commit()
        return fid

    def recall_facts(self, vec: list[float], k: int = 8) -> list[dict]:
        """语义召回：和所有带向量的事实算余弦，返回最相似的 k 条（0 向量返回空）。"""
        if not vec:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT f.id, f.text, f.tags, f.created_at, fe.embedding"
                " FROM facts f JOIN fact_embeddings fe ON fe.fact_id=f.id"
                " ORDER BY f.id DESC LIMIT 5000"
            ).fetchall()
        scored: list[tuple[float, dict]] = []
        for r in rows:
            s = cosine(json.loads(r["embedding"]), vec)
            if s > 0:
                scored.append((s, dict(r)))
        scored.sort(key=lambda x: (-x[0], -x[1]["id"]))
        return [dict(r, score=round(s, 4)) for s, r in scored[:k]]

    def search_facts(self, query: str, limit: int = 8) -> list[dict]:
        q = _tokens(query)
        with self._lock:
            rows = self._db.execute(
                "SELECT id,text,tags,created_at FROM facts ORDER BY id DESC LIMIT 2000"
            ).fetchall()
        scored: list[tuple[float, dict]] = []
        for r in rows:
            hay = _tokens(r["text"] + " " + (r["tags"] or ""))
            if not hay:
                continue
            inter = len(q & hay)
            if inter == 0:
                continue
            scored.append((inter / (len(q) ** 0.5 + 1e-9), dict(r)))
        scored.sort(key=lambda x: (-x[0], -x[1]["id"]))
        return [dict(r, score=round(s, 4)) for s, r in scored[:limit]]

    def count_facts(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM facts").fetchone()[0])


def memory_tools(store: Store, session_id: str | None = None,
                 backend: EmbeddingBackend | None = None) -> list[Tool]:
    """把长期记忆暴露成工具，注册进 Registry 后模型就能自己记、自己回想了。

    `backend` 提供嵌入能力时：保存事实会同步生成向量，`memory_recall` 做语义召回；
    不提供时降级为纯关键词记忆（recall 优雅报错，save 照常工作）。
    """

    async def memory_save(text: str, tags: str = "") -> dict:
        if not (text or "").strip():
            return fail("text 不能为空")
        vec = None
        if backend is not None:
            try:
                vec = (await backend.embed([text.strip()]))[0]
            except Exception:
                vec = None  # 嵌入失败也不耽误记忆，只是这条不能被语义召回
        fid = store.save_fact(text, tags, session_id, vec=vec)
        return ok(f"已记住 #{fid}：{text.strip()[:120]}", id=fid)

    async def memory_search(query: str, limit: int = 8) -> dict:
        hits = store.search_facts(query, int(limit))
        if not hits:
            return ok("没有相关记忆。", hits=0)
        body = "\n".join(
            f"- #{h['id']} (score {h['score']}) {h['text']}" for h in hits
        )
        return ok(body, hits=len(hits))

    async def memory_recall(query: str, limit: int = config.MEMORY_RECALL_K) -> dict:
        if backend is None:
            return fail("语义召回未启用（无嵌入后端）", kind="config")
        try:
            qv = (await backend.embed([query]))[0]
        except Exception as exc:
            return fail(
                f"召回失败（嵌入模型可能未就绪，先 ollama pull {config.EMBED_MODEL}）: "
                f"{type(exc).__name__}: {exc}", kind="embed")
        hits = store.recall_facts(qv, int(limit))
        if not hits:
            return ok("没有可语义召回的记忆。", hits=0)
        body = "\n".join(
            f"- #{h['id']} (score {h['score']}) {h['text']}" for h in hits
        )
        return ok(body, hits=len(hits))

    return [
        Tool(
            "memory_save",
            "把用户告知的长期事实、偏好或结论写入长期记忆，跨会话可检索。",
            {"type": "object", "properties": {
                "text": {"type": "string", "description": "要记住的事实，一句话说清"},
                "tags": {"type": "string", "description": "可选标签，逗号分隔"}},
             "required": ["text"]},
            memory_save,
            tags=["memory"],
        ),
        Tool(
            "memory_search",
            "在自己的长期记忆里按关键词检索（字面匹配）。回答涉及用户偏好或历史事实前先查一下。",
            {"type": "object", "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "返回条数，默认 8"}},
             "required": ["query"]},
            memory_search,
            tags=["memory"],
        ),
        Tool(
            "memory_recall",
            "在语义层面召回长期记忆：把问题嵌入后，找回含义最相近的事实（而非字面匹配）。"
            "回答涉及用户偏好、历史结论或跨会话上下文前，优先用这个。",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "自然语言问题"},
                "limit": {"type": "integer", "description": "返回条数，默认 8"}},
             "required": ["query"]},
            memory_recall,
            tags=["memory"],
        ),
    ]


def register_memory(reg: Registry, store: Store, session_id: str | None = None,
                    backend: EmbeddingBackend | None = None) -> None:
    for t in memory_tools(store, session_id, backend):
        if reg.get(t.name) is None:
            reg.register(t)


# ---------------------------------------------------------------------------
# 记忆自动沉淀（MemGPT 式 consolidation）
# ---------------------------------------------------------------------------
# 提示词刻意写得很"机械"：蒸馏是抽取任务，不需要模型自我辩论。
# 限制条数与字数是在省 CPU —— 思考型模型会为"哪条值得记"纠结很久，
# 而这段纠结的每一个 token 都是真金白银的解码时间。
CONSOLIDATE_SYSTEM = (
    "你是记忆整理器。阅读对话，抽出值得长期记住的事实：稳定偏好、明确结论、"
    "关键决策、项目约定。\n"
    "规则：\n"
    "1) 每条一句话，不超过 30 字，独立可读\n"
    "2) 最多 3 条；没有值得记的就输出 []\n"
    "3) 不要解释、不要复述对话，直接输出 JSON 数组\n"
    "示例：[\"用户是前端工程师，主用 React\", \"项目禁止在提交里写 any\"]"
)


async def _complete(backend: ChatBackend, messages: list[dict],
                    model: str | None = None,
                    num_predict: int | None = None,
                    temperature: float | None = None) -> str:
    """把一次（无工具）对话跑完，拼接出完整正文。

    `num_predict` 用来给生成长度封顶 —— 批处理任务（如沉淀）必须封，
    否则思考型模型会一路生成到上下文上限。
    """
    parts: list[str] = []
    async for ev in backend.chat_stream(messages, tools=None, model=model,
                                        num_predict=num_predict,
                                        temperature=temperature):
        if ev["type"] == "text":
            parts.append(ev["text"])
    return "".join(parts)


def _recent_text(messages: list[dict], n: int) -> str:
    """取最近 n 条 user/assistant/tool 消息的纯文本，tool 消息解包内层 content。"""
    picked: list[str] = []
    for m in reversed(messages):
        role = m.get("role")
        if role not in ("user", "assistant", "tool"):
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False)
        if role == "tool":
            # tool 的 content 形如 {"ok":..., "content":"..."} —— 取内层可读内容
            try:
                text = str(json.loads(content).get("content", "") or "")
            except Exception:
                text = str(content)
        if text.strip():
            picked.append(f"{role}: {text.strip()}")
        if len(picked) >= n:
            break
    return "\n\n".join(reversed(picked))


def _parse_facts(raw: str) -> list[str]:
    """从模型输出里稳健解析出事实字符串列表。

    容忍 ```json 围栏、前后废话、以及整段都不是 JSON 时按行兜底。
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    i, j = raw.find("["), raw.rfind("]")
    if i >= 0 and j > i:
        try:
            arr = json.loads(raw[i : j + 1])
        except json.JSONDecodeError:
            arr = None
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    # 兜底：按非空短行拆
    return [ln.strip(" -。，、").strip() for ln in raw.splitlines()
            if ln.strip() and len(ln.strip()) > 2]


async def consolidate_memory(
    backend: ChatBackend,
    store: Store,
    messages: list[dict],
    *,
    backend_embed: EmbeddingBackend | None = None,
    session_id: str | None = None,
    model: str | None = None,
    max_facts: int | None = None,
) -> dict:
    """对话结束后，用 LLM 把值得长期记住的事实蒸馏进 facts 表。

    - 只看最近 N 条消息（user/assistant/tool 的文本内容）
    - 让模型只输出 JSON 数组（每个元素是一句话事实）
    - 与已有事实做阈值去重，避免反复记同一条
    - 嵌入失败也不耽误落库（降级为纯关键词记忆）
    """
    max_facts = max_facts or config.MEMORY_CONSOLIDATE_MAX
    window = _recent_text(messages, config.MEMORY_CONSOLIDATE_WINDOW)
    # 太短的对话（比如一句"收到"）不值得花一次推理去蒸馏
    if len(window) < config.MEMORY_CONSOLIDATE_MIN_CHARS:
        return {"ok": True, "saved": 0, "skipped": 0, "reason": "too_short"}
    # 沉淀模型显式指定时优先 —— CPU 上建议换成不思考的小模型，见 config 注释
    model = config.CONSOLIDATE_MODEL or model

    try:
        # 双重保险：num_predict 封住生成长度，wait_for 封住墙钟时间。
        # 前者防"模型一直想"，后者防"慢慢吐 token"（HTTP 读超时救不了后者）。
        raw = await asyncio.wait_for(
            _complete(backend, [
                {"role": "system", "content": CONSOLIDATE_SYSTEM},
                {"role": "user", "content": window},
            ], model=model,
                num_predict=config.CONSOLIDATE_MAX_TOKENS,
                temperature=config.CONSOLIDATE_TEMPERATURE),
            timeout=config.CONSOLIDATE_TIMEOUT,
        )
    except Exception as exc:  # 含 TimeoutError：超时就放弃这一轮，不占着 CPU
        return {"ok": False, "saved": 0, "skipped": 0,
                "error": f"{type(exc).__name__}: {exc}"}

    facts = _parse_facts(raw)
    if not facts:
        return {"ok": True, "saved": 0, "skipped": 0, "reason": "none"}

    # 去重：拉候选事实的命中项，按 token 重叠比跳过近似项（含同对话二次沉淀的幂等）
    existing = store.search_facts(" ".join(facts), limit=200)
    existing_tokens = [_tokens(r["text"]) for r in existing]
    saved = skipped = 0
    for f in facts[:max_facts]:
        ft = _tokens(f)
        if any(len(ft & et) / max(1, min(len(ft), len(et))) >= 0.9
               for et in existing_tokens):
            skipped += 1
            continue
        vec = None
        if backend_embed is not None:
            try:
                vec = (await backend_embed.embed([f]))[0]
            except Exception:
                vec = None  # 嵌入失败也不耽误记忆，只是这条不能被语义召回
        store.save_fact(f, tags="auto", session_id=session_id, vec=vec)
        saved += 1
    return {"ok": True, "saved": saved, "skipped": skipped}


__all__ = ["Any", "Store", "memory_tools", "register_memory", "consolidate_memory"]

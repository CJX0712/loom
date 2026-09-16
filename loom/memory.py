"""持久化：会话、消息、长期记忆。

用标准库 sqlite3，零额外依赖。所有写入走同一个连接 + 一把锁，
既可以给异步服务用（通过 `asyncio.to_thread`），也可以给同步自检用。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config
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


__all__ = ["Any", "Store", "memory_tools", "register_memory"]

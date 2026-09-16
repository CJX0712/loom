"""RAG：本地嵌入 + SQLite 向量库。

不自己写向量引擎，也不引 numpy —— 纯标准库 + 直接打 Ollama 的 `/api/embed`
端点。本地规模（几千块）下，把向量全读出来在 Python 里算余弦完全够用，
换来的是零依赖和绝对可审计。

设计要点
--------
1. **嵌入后端可替换**：`OllamaEmbeddings`（真）和 `FakeEmbeddings`（确定性假，
   供自检）都实现同一个 `async embed(texts) -> list[list[float]]` 接口。
   自检不下载任何模型，因此 CI 永远绿灯。
2. **切块是边界优先的**：尽量在段落 / 句末截断，不会把一句话拦腰斩在块中间。
3. **向量存成 JSON 文本**：可读、可调试、零依赖。规模上十万块再换专用向量库。
4. **工具结果结构化**：和全项目一致，失败返回 `{"ok": false, ...}`。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

from . import config
from .tools import Registry, SandboxError, Tool, fail, ok, rel, safe_path

# 摄入时跳过的目录
RAG_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".loom", ".idea", ".vscode",
}

# 当作文档喂给嵌入模型的后缀（其余按二进制 / 无意义文本跳过）
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".yaml", ".yml", ".toml", ".rst", ".csv", ".html", ".htm", ".css", ".scss",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".sh", ".bash", ".bat",
    ".ps1", ".sql", ".log", ".ini", ".cfg", ".conf", ".tex", ".xml", ".svg",
}


# ---------------------------------------------------------------------------
# 嵌入后端
# ---------------------------------------------------------------------------
class EmbeddingBackend(Protocol):
    """所有嵌入后端必须实现的最小接口。"""

    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        ...


class OllamaEmbeddings:
    """打 Ollama 的 `/api/embed` 端点。本地、零 API Key。"""

    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None,
                 timeout: float = 90.0) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.EMBED_MODEL
        self.timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": list(texts)},
            )
            r.raise_for_status()
            data = r.json()
        embs = data.get("embeddings") or []
        if len(embs) != len(texts):
            raise RuntimeError(
                f"embed 返回数量不符：要 {len(texts)} 个，得 {len(embs)} 个"
            )
        return [[float(x) for x in e] for e in embs]


class FakeEmbeddings:
    """确定性假嵌入：把词哈希成固定维度的词袋向量。

    **仅供自检使用** —— 不下载模型、不联网，且相同文本必得相同向量，
    词重叠越高的两段余弦越接近 1。这就足以把「检索是否返回了相关块」
    这条不变量钉死。
    """

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        # 中英文混合分词：英文/数字按词，中文**逐字**（这样"猫"和"猫咪"能共享字符，
        # 余弦才有意义；否则整串中文被当成一个 token，语义检索彻底失效）。
        for tok in re.findall(r"[a-z0-9_]+|[一-鿿]", (text or "").lower()):
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm:
            return [x / norm for x in v]
        return v


# ---------------------------------------------------------------------------
# 切块 + 余弦
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int | None = None,
               overlap: int | None = None) -> list[str]:
    """边界优先的滑动窗口切块。

    尽量在段落空行 / 句末（。！？等）截断，不在块尾硬切一句话；相邻块重叠
    `overlap` 个字符，防止一句话被切在两半、上下文断裂。
    """
    size = size or config.RAG_CHUNK_SIZE
    overlap = overlap or config.RAG_CHUNK_OVERLAP
    overlap = max(0, min(overlap, size - 1))  # 重叠必须小于块大小，否则死循环

    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    # 边界字符：在它们之后切，读起来最自然
    boundary = set("\n。！？!?；;")
    while start < n:
        end = min(start + size, n)
        if end < n:
            # 在候选窗口的最后 `overlap` 个字符里，从后往前找最近的边界
            span = text[end - overlap:end]
            cut = -1
            for i in range(len(span) - 1, -1, -1):
                if span[i] in boundary:
                    cut = i
                    break
            if cut >= 0:
                end = end - overlap + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        nxt = end - overlap
        # 防止死循环：若重叠回退没推进，就硬进到 end
        start = nxt if nxt > start else end
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。任一为零向量时定义为 0（不抛异常）。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# 向量库（SQLite）
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source);
"""


class VectorStore:
    """SQLite 向量库：纯 stdlib，余弦在内存里算。

    规模几千块时足够快；更大规模再换专用向量索引（接口不变）。
    """

    def __init__(self, path: str | Path, backend: EmbeddingBackend | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend or OllamaEmbeddings()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- 写入 -------------------------------------------------------------
    async def ingest_text(self, source: str, text: str) -> int:
        """把一个来源的文本切块并嵌入入库。同源重复摄入幂等覆盖。"""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        vecs = await self.backend.embed(chunks)
        with self._lock:
            # 先清掉旧的同名来源，保证幂等（同一份文档重新摄入不会翻倍）
            self._db.execute("DELETE FROM rag_chunks WHERE source=?", (source,))
            for i, (ch, vec) in enumerate(zip(chunks, vecs, strict=True)):
                self._db.execute(
                    "INSERT INTO rag_chunks(source,chunk_index,text,embedding,created_at)"
                    " VALUES(?,?,?,?,?)",
                    (source, i, ch, json.dumps(vec), time.time()),
                )
            self._db.commit()
        return len(chunks)

    async def ingest_path(self, path: str | Path, recursive: bool = True) -> dict:
        """摄入一个文件或目录（目录递归，跳过敏感目录与非文本后缀）。"""
        p = Path(path)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            if recursive:
                walker = p.rglob("*")
            else:
                walker = p.glob("*")
            files = [
                f for f in walker
                if f.is_file()
                and f.suffix.lower() in TEXT_EXTS
                and not any(part in RAG_SKIP_DIRS for part in f.parts)
            ]
        else:
            return {"ok": False, "count": 0, "files": 0, "error": "路径不存在"}

        total = 0
        ingested_files = 0
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            n = await self.ingest_text(str(f), text)
            if n:
                total += n
                ingested_files += 1
        if ingested_files == 0:
            return {"ok": False, "count": 0, "files": 0,
                    "error": "没有可读的文本文件"}
        return {"ok": True, "count": total, "files": ingested_files}

    # -- 检索 -------------------------------------------------------------
    async def search(self, query: str, k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        qv = (await self.backend.embed([query]))[0]
        with self._lock:
            rows = self._db.execute(
                "SELECT id,source,chunk_index,text,embedding FROM rag_chunks"
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            vec = json.loads(r["embedding"])
            s = cosine(qv, vec)
            if s > 0:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        out: list[dict] = []
        for s, r in scored[:k]:
            out.append({
                "score": round(s, 4),
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
            })
        return out

    # -- 管理 -------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])

    def sources(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT source, COUNT(*) AS n FROM rag_chunks GROUP BY source"
            ).fetchall()
        return [{"source": r["source"], "chunks": int(r["n"])} for r in rows]

    def clear(self) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM rag_chunks")
            self._db.commit()
            return int(cur.rowcount)


# ---------------------------------------------------------------------------
# 工具（暴露给智能体；闭包包住 store，注册进 Registry）
# ---------------------------------------------------------------------------
def rag_tools(store: VectorStore) -> list[Tool]:
    """把 RAG 暴露成两个工具，注册进 Registry 后模型就能自己查资料了。"""

    async def rag_ingest(path: str, recursive: bool = True) -> dict:
        if not (path or "").strip():
            return fail("path 不能为空")
        try:
            p = safe_path(path)
        except SandboxError as exc:
            return fail(str(exc), kind="sandbox")
        if p.is_dir() and not recursive:
            return fail("这是一个目录，请设 recursive=true 递归摄入")
        res = await store.ingest_path(p)
        if not res.get("ok"):
            return fail(res.get("error", "摄入失败"), kind="ingest")
        return ok(
            f"已摄入 {res['files']} 个文件，生成 {res['count']} 个文本块（来源：{rel(p)}）",
            chunks=res["count"], files=res["files"],
        )

    async def rag_search(query: str, k: int = 5) -> dict:
        if not (query or "").strip():
            return fail("query 不能为空")
        k = max(1, min(int(k), 20))
        try:
            hits = await store.search(query, k)
        except Exception as exc:  # 嵌入模型没就绪时会抛连接错误
            return fail(
                f"检索失败（嵌入模型可能未就绪，先 ollama pull {config.EMBED_MODEL}）: "
                f"{type(exc).__name__}: {exc}", kind="embed"
            )
        if not hits:
            return ok("知识库为空或没有相关内容。先用 rag_ingest 摄入文档。", hits=0)
        body = "\n\n".join(
            f"[#{i + 1} 来源 {h['source']} 相似度 {h['score']}]\n{h['text']}"
            for i, h in enumerate(hits)
        )
        return ok(body, hits=len(hits))

    return [
        Tool(
            "rag_ingest",
            "把工作目录内的文件或目录摄入本地知识库（本地嵌入 + 向量检索，零 API Key）。"
            "摄入后即可用 rag_search 做语义检索。已摄入的源会幂等覆盖。",
            {"type": "object", "properties": {
                "path": {"type": "string", "description": "文件或目录的相对路径"},
                "recursive": {"type": "boolean", "description": "目录是否递归，默认 true"}},
             "required": ["path"]},
            rag_ingest,
            tags=["rag"],
        ),
        Tool(
            "rag_search",
            "在本地知识库里做语义检索，返回最相关的文本块。回答涉及已摄入文档前先查这里。",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "自然语言问题"},
                "k": {"type": "integer", "description": "返回几条，默认 5"}},
             "required": ["query"]},
            rag_search,
            tags=["rag"],
        ),
    ]


def register_rag(reg: Registry, store: VectorStore) -> None:
    for t in rag_tools(store):
        if reg.get(t.name) is None:
            reg.register(t)


__all__ = [
    "FakeEmbeddings", "OllamaEmbeddings", "VectorStore", "chunk_text", "cosine",
    "rag_tools", "register_rag",
]

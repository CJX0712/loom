"""Loom 配置 —— 全部通过环境变量覆盖，零配置文件、零 API Key。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

# --- 推理后端：Ollama（本地，无 API Key） -----------------------------------
OLLAMA_HOST = os.environ.get("LOOM_OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("LOOM_MODEL", "qwen3:4b")
MODEL_PREFERENCES = [
    m.strip()
    for m in os.environ.get(
        "LOOM_MODEL_PREFS",
        "qwen3:4b,qwen3:8b,qwen2.5:7b,qwen2.5:3b,llama3.2:3b,llama3.1:8b",
    ).split(",")
    if m.strip()
]

# --- 智能体循环 --------------------------------------------------------------
# 单轮用户输入最多允许多少次「模型 → 工具 → 观测」往返。硬上限，防跑飞。
MAX_STEPS = int(os.environ.get("LOOM_MAX_STEPS", "12"))
# 单次工具调用最长执行秒数
TOOL_TIMEOUT = float(os.environ.get("LOOM_TOOL_TIMEOUT", "60"))
# 单个工具结果回灌给模型时的最大字符数（防止上下文被一个大文件撑爆）
MAX_OBSERVATION_CHARS = int(os.environ.get("LOOM_MAX_OBSERVATION_CHARS", "6000"))
# 模型采样
TEMPERATURE = float(os.environ.get("LOOM_TEMPERATURE", "0.6"))
# 思维链开关：True / False / None（auto）
#
# **默认 auto —— 即不向 Ollama 发送 `think` 字段。这看起来违反直觉，但实测是这样：**
#
# Qwen3-4B 无论你怎么要求都会先想一遍。`think: false` 并不能让它少想，
# 只能让 Ollama **停止把推理分流到 `thinking` 字段**，于是那一大段"首先我需要确认…"
# 直接落进 `content`，变成对外的正式回答。同样的 500 token，速度也几乎一样
# （4.51 vs 4.59 tok/s）。
#
# 实测（qwen3:4b，同一提示词，num_predict=500）：
#
# | 传参          | thinking 字段 | content        |
# |---------------|---------------|----------------|
# | 不传 / true   | 671 字符      | **0（干净）**  |
# | false         | 0             | **671 字符**   |
#
# 也就是说 `think:false` 一个 token 都没省，只是把推理倒进了答案里。
# 所以默认 auto：让后端照模型模板正确处理，推理走 `thinking` 通道，
# Loom 把它渲染成可折叠的"思考"块，`content` 保持干净。
#
# 真的想省 CPU，换一个本来就不思考的模型（如 qwen2.5:3b-instruct），
# 而不是靠这个字段。
_think_raw = os.environ.get("LOOM_THINK", "auto").strip().lower()
if _think_raw in ("1", "true", "yes", "on"):
    THINK: bool | None = True
elif _think_raw in ("0", "false", "no", "off"):
    THINK = False
else:
    THINK = None  # auto
# 上下文窗口内保留的最大消息条数（滑动窗口，保护 16GB 内存机器）
MAX_CONTEXT_MESSAGES = int(os.environ.get("LOOM_MAX_CONTEXT_MESSAGES", "40"))


def auto_threads(cpu_count: int | None = None) -> int:
    r"""给单路小模型推理挑线程数。

    **小模型解码是内存带宽瓶颈，不是算力瓶颈。** 线程开得越多，核心之间
    抢内存总线越凶，反而更慢。实测（Ryzen 7 8C/16T · qwen3:4b Q4_K_M · 128 token）：

    | num_thread | 生成 tok/s |
    |---|---|
    | 2 | 4.83 |
    | **4** | **4.85** |
    | 6 | 4.66 |
    | 8 | 4.09 |
    | 12 | 2.87 |
    | 16 | 2.52 |

    16 线程比 4 线程**慢 1.93×**。所以默认锁在 2–4，不跟随 CPU 核数。

    注：预填充（prompt eval）反而是 8 线程最快。但预填充比解码快一个数量级，
    端到端耗时由解码主导，所以按解码调优。
    """
    n = cpu_count if cpu_count is not None else (os.cpu_count() or 4)
    return max(2, min(4, n // 4 or 2))


# 0 表示按上面的规则自动；显式给值则完全听用户的
_num_threads_env = os.environ.get("LOOM_NUM_THREADS", "0")
NUM_THREADS = int(_num_threads_env) if int(_num_threads_env) > 0 else auto_threads()

# --- 沙箱 --------------------------------------------------------------------
# 所有文件读写被限制在这个目录内。默认是启动时的当前目录。
WORKDIR = Path(os.environ.get("LOOM_WORKDIR", os.getcwd())).resolve()
# 是否放开 shell 工具。默认开启，但危险命令仍被拒绝（见 tools.py）。
SHELL_ENABLED = os.environ.get("LOOM_SHELL", "1") not in ("0", "false", "no")
SHELL_TIMEOUT = float(os.environ.get("LOOM_SHELL_TIMEOUT", "60"))

# --- 抓取：优先用已经在跑的 crawl4ai，缺失时回退到内置抓取 -------------------
CRAWL4AI_URL = os.environ.get("LOOM_CRAWL4AI_URL", "http://127.0.0.1:11235")
CRAWL4AI_ENABLED = os.environ.get("LOOM_CRAWL4AI", "1") not in ("0", "false", "no")
HTTP_TIMEOUT = float(os.environ.get("LOOM_HTTP_TIMEOUT", "30"))

# --- RAG（本地知识库）-------------------------------------------------------
# 用 Ollama 的嵌入模型做语义检索，零额外依赖（直接打 /api/embed 端点）。
EMBED_MODEL = os.environ.get("LOOM_EMBED_MODEL", "nomic-embed-text")
# 文本切块：每块约多少字符，相邻块重叠多少（重叠防止一句话被切在两半）
RAG_CHUNK_SIZE = int(os.environ.get("LOOM_RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP = int(os.environ.get("LOOM_RAG_CHUNK_OVERLAP", "150"))
# 语义记忆召回条数（长期记忆用 embedding 找回含义相近的事实）
MEMORY_RECALL_K = int(os.environ.get("LOOM_MEMORY_RECALL_K", "8"))

# --- MCP（Model Context Protocol）-------------------------------------------
MCP_CONFIG = Path(
    os.environ.get("LOOM_MCP_CONFIG", str(ROOT / "mcp" / "servers.json"))
).resolve()

# --- 服务 --------------------------------------------------------------------
HOST = os.environ.get("LOOM_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOOM_PORT", "8790"))
STATE_DIR = Path(os.environ.get("LOOM_STATE", str(ROOT / ".loom"))).resolve()
DB_PATH = STATE_DIR / "loom.sqlite3"
RAG_DB_PATH = STATE_DIR / "rag.sqlite3"

# --- 版本 --------------------------------------------------------------------
__version__ = "0.1.0"


def describe() -> dict:
    """给 /api/health 和 selftest 用的配置快照。"""
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "ollama_host": OLLAMA_HOST,
        "model": DEFAULT_MODEL,
        "workdir": str(WORKDIR),
        "shell_enabled": SHELL_ENABLED,
        "max_steps": MAX_STEPS,
        "think": {None: "auto", True: "on", False: "off"}[THINK],
        "num_threads": NUM_THREADS,
        "crawl4ai": CRAWL4AI_URL if CRAWL4AI_ENABLED else None,
        "mcp_config": str(MCP_CONFIG),
        "embed_model": EMBED_MODEL,
        "rag_chunk_size": RAG_CHUNK_SIZE,
        "rag_chunk_overlap": RAG_CHUNK_OVERLAP,
        "memory_recall_k": MEMORY_RECALL_K,
    }

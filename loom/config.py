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
# 子智能体（agent_delegate 派出的 worker）最多几轮工具往返。
# 比主循环更紧：worker 不该跑太久，且多个 worker 并行时别把 CPU 抢光。
DELEGATE_MAX_STEPS = int(os.environ.get("LOOM_DELEGATE_MAX_STEPS", "8"))
# agent_delegate 自身允许的墙钟超时（秒）。它要跑完一整轮子智能体循环
# （含模型加载 + 多轮工具往返），远长于普通单工具，故单独放宽。
DELEGATE_TIMEOUT = float(os.environ.get("LOOM_DELEGATE_TIMEOUT", "300"))

# --- 记忆自动沉淀（MemGPT 式 consolidation）----------------------------------
# 对话结束后用 LLM 蒸馏长期事实入库。默认开启：多花一次推理，换来智能体
# 真正「越用越懂用户」。沉淀发生在后台，失败静默、不影响主链路。
MEMORY_CONSOLIDATE = os.environ.get("LOOM_MEMORY_CONSOLIDATE", "1") not in (
    "0", "false", "no")
# 单次沉淀最多写几条事实（防一轮对话刷爆 facts 表）
MEMORY_CONSOLIDATE_MAX = int(os.environ.get("LOOM_MEMORY_CONSOLIDATE_MAX", "6"))
# 蒸馏时只看最近多少条消息（user/assistant/tool）
MEMORY_CONSOLIDATE_WINDOW = int(os.environ.get("LOOM_MEMORY_CONSOLIDATE_WINDOW", "24"))
# 对话太短就不值得花一次推理去蒸馏（字符数阈值，按窗口文本算）
MEMORY_CONSOLIDATE_MIN_CHARS = int(
    os.environ.get("LOOM_MEMORY_CONSOLIDATE_MIN_CHARS", "40"))
# 沉淀专用模型。留空 = 跟主对话用同一个。
#
# **CPU 上强烈建议换一个不思考的小模型**（如 qwen2.5:3b-instruct）：
# 蒸馏只要一句话结论，不需要思维链；而 Qwen3 无论怎么要求都会先想一大段，
# 白白多烧一倍 token。实测主对话一轮约 100s，沉淀再来一次同样量级的推理不划算。
CONSOLIDATE_MODEL = os.environ.get("LOOM_CONSOLIDATE_MODEL", "").strip() or None
# 沉淀输出 token 上限。**必须封顶** —— 不封顶的话思考型模型会一路生成到上下文
# 上限（qwen3:4b 是 4 万 token），一次"蒸馏只要一句话"的任务能跑几小时。
#
# 但也不能太小：**思考型模型的思维链和正文共用这个预算**。实测（qwen3:4b）：
# 设 400 或 1024 时，模型把预算全用在"想"上，正文（content）直接是空字符串，
# 一条事实都提不出来 —— 而这个模型一次简单任务的推理就有 2000+ token
# （见 README 性能一节）。所以默认给到 3072：够装思维链 + 几条事实。
CONSOLIDATE_MAX_TOKENS = int(os.environ.get("LOOM_CONSOLIDATE_MAX_TOKENS", "3072"))
# 沉淀用低温：蒸馏是机械抽取，不需要创造性；温度高了小模型容易反复改口、
# 越写越长（CPU 上这就是几分钟的差别）。
CONSOLIDATE_TEMPERATURE = float(os.environ.get("LOOM_CONSOLIDATE_TEMPERATURE", "0.1"))
# 沉淀的墙钟超时（秒）。超时就放弃这一轮，宁可少记一条也不让它挂着占 CPU。
# 注意：HTTP 读超时救不了"慢慢吐 token"的场景（每读都有数据就不触发），
# 所以这里必须自己加一层 asyncio 超时。
# 这对数字要一起看：qwen3:4b 在 CPU 上约 4.5 tok/s，3072 token ≈ 680 秒。
# 超时给 900 秒是留余量（真正先撞到的通常是 token 上限）。
#
# **慢是物理限制，不是 bug**：思考型模型在 CPU 上做一次沉淀就要 10 分钟量级。
# 它跑在后台、有并发闸门（同时只允许一个）、可整体关闭，所以代价可控。
# 想要秒级沉淀，装一个不思考的小模型并设 LOOM_CONSOLIDATE_MODEL。
CONSOLIDATE_TIMEOUT = float(os.environ.get("LOOM_CONSOLIDATE_TIMEOUT", "900"))

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
        "memory_consolidate": MEMORY_CONSOLIDATE,
        "memory_consolidate_max": MEMORY_CONSOLIDATE_MAX,
        "consolidate_model": CONSOLIDATE_MODEL,
        "delegate_max_steps": DELEGATE_MAX_STEPS,
        "delegate_timeout": DELEGATE_TIMEOUT,
    }

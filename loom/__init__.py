"""Loom —— 本地优先的 MCP 原生智能体运行时。

把现成最强的开源件组装成一个能真正干活的本地智能体：

- **Ollama** 负责推理（本地模型，零 API Key）
- **MCP** 负责工具（官方 SDK，接入整个生态）
- **crawl4ai** 负责抓取（本地部署，正文提取质量高）

Loom 自己只写两样东西：**协议翻译**和**产品体验**。
"""

from .config import __version__

__all__ = ["__version__"]

"""多智能体编排 —— orchestrator-worker 模式。

设计（业界已验证的最佳实践，见 Anthropic «Effective Multi-Agent Orchestration»）：
- 主智能体是 orchestrator，遇到能拆成独立子任务的工作时，用 `agent_delegate`
  派一个专职子智能体（worker）去做，再把结果综合起来。
- 子智能体**复用主机的同一套工具集**（fs/shell/web/python/memory/rag/mcp），
  但**自动剪除 `agent_delegate` 本身** —— 否则子代理会再派子代理，递归失控、
  上下文爆炸、CPU 被小模型们瓜分干净。
- 每个子智能体是**独立、无状态**的一轮 `run_agent`：自己的消息列表、自己的
  system 提示词（按角色裁剪），跑到底拿到结论就返回，不污染主会话的 transcript。
- 资源有硬上限：`DELEGATE_MAX_STEPS` 比主循环更紧（子代理不该跑太久）。

全部复用 `loop.run_agent` 与全局 `Registry`，不重写任何协议逻辑。
"""

from __future__ import annotations

from typing import Any

from . import config
from .loop import run_agent
from .tools import Registry, Tool, fail, ok

# 角色预设：把「专职子智能体」该做什么、不该做什么说清，避免子任务膨胀。
ROLE_TEMPLATES: dict[str, str] = {
    "research": "专职信息搜集：用 web_search/web_fetch/fs_read/rag_search 把事实查全，"
                "不要写文件、不要产出结论之外的任何东西，只回报查到的事实。",
    "executor": "专职动手执行：用 shell_run/python_run/fs_write 把一件具体的事做完，"
                "完成后回报结果、产物路径与退出状态。",
    "reviewer": "专职审查校验：对照给出的材料挑错、验证事实、指出风险与遗漏，"
                "只输出审查意见，不要重新做一遍。",
    "planner": "专职规划拆解：把一个大目标拆成有序、可执行、互不依赖的步骤清单，"
                "不亲自执行，只输出计划。",
}


def build_sub_prompt(role: str, task: str) -> str:
    """拼一个「只盯这一件事」的子智能体系统提示词。"""
    preset = ROLE_TEMPLATES.get(role.strip().lower(), "")
    role_line = role.strip() or "worker"
    discipline = (
        "你是 Loom 主智能体派出的专职子智能体。\n"
        f"你的专职角色：{role_line}\n"
        f"{preset}\n"
        "你的唯一任务（用户给出，做完即止）：\n"
        f"{task}\n\n"
        "工作纪律：\n"
        "- 只做上面这一件事，范围不要扩张；没有工具能解决就直接说清楚，不要硬编。\n"
        "- 你可以调用工具（文件 / 命令 / 网络 / 计算 / 记忆 / 知识库），需要事实就去查。\n"
        "- 你**不能**再派生子智能体——把复杂工作自己用工具一步步做完。\n"
        "- 直接给结论，不要复述思考过程，控制在几句话内。回答用中文。\n"
    )
    return discipline


def scoped_registry(live: Registry) -> Registry:
    """返回主注册表的副本，**剔除 `agent_delegate`** 以阻断递归委派。

    这是多智能体系统不跑飞的关键开关：worker 手里有全集工具，唯独没有
    「再派一个 worker」的权限。
    """
    sub = Registry()
    for t in live.all():
        if t.name == "agent_delegate":
            continue
        sub.register(t)
    return sub


async def run_delegate(
    backend: Any,
    live: Registry,
    task: str,
    role: str = "",
    *,
    max_steps: int | None = None,
    model: str | None = None,
    think: bool | None = None,
) -> dict:
    """跑一个专职子智能体，直到它给出结论，返回结构化结果。

    - `backend`    : 复用主机的推理后端（OllamaBackend 或自检用的 ScriptedBackend）
    - `live`      : 主机的工具注册表（会被剪枝后交给子代理）
    - 返回 `{"answer": str, "steps": int, "reason": str}`
    """
    task = (task or "").strip()
    if not task:
        raise ValueError("delegate task 不能为空")
    max_steps = max(1, min(int(max_steps or config.DELEGATE_MAX_STEPS), config.MAX_STEPS))
    sub_sys = build_sub_prompt(role, task)
    sub_reg = scoped_registry(live)
    sub_messages = [
        {"role": "system", "content": sub_sys},
        {"role": "user", "content": task},
    ]

    collected: list[str] = []
    last_text = ""
    steps = 0
    reason = "unknown"
    async for ev in run_agent(
        backend, sub_reg, sub_messages,
        model=model, max_steps=max_steps, think=think,
    ):
        if ev["type"] == "text":
            collected.append(ev["delta"])
        elif ev["type"] == "final":
            last_text = ev.get("text", "")
            steps = ev.get("steps", 0)
            reason = ev.get("reason", "unknown")

    answer = "".join(collected).strip() or last_text.strip()
    if not answer:
        answer = "(子智能体未产出文本结论)"
    return {"answer": answer, "steps": steps, "reason": reason}


def delegate_tools(backend: Any, live: Registry) -> list[Tool]:
    """生成 `agent_delegate` 工具。闭包捕获 backend 与真实工具表。"""

    async def agent_delegate(
        task: str,
        role: str = "worker",
        max_steps: int | None = None,
    ) -> dict:
        """派一个专职子智能体去独立完成一件任务，再把它的最终结论拿回来。

        适合：能独立成块、可以用工具查证的工作（查资料 / 跑代码 / 读文件 /
        核对事实）。子智能体有自己的工具权限，但**不能再派子智能体**，
        所以放心派，不会递归爆炸。把拿回来的结论综合进你的回答即可。
        """
        try:
            res = await run_delegate(backend, live, task, role, max_steps=max_steps)
        except Exception as exc:  # 子代理任何异常都变成结构化失败，不连累主循环
            return fail(
                f"子智能体执行失败: {type(exc).__name__}: {exc}",
                kind="delegate_error",
            )
        return ok(
            res["answer"],
            kind="delegate",
            role=role.strip() or "worker",
            steps=res["steps"],
            reason=res["reason"],
        )

    return [Tool(
        "agent_delegate",
        "派一个专职子智能体独立完成任务并拿回结论：查资料 / 跑代码 / 读文件 / 核对。"
        "子智能体有完整工具权限但不可再委派，故不会递归爆炸。",
        {"type": "object", "properties": {
            "task": {"type": "string",
                     "description": "给子智能体的明确任务，一句话说清要什么结果"},
            "role": {"type": "string",
                     "description": "角色预设：worker|research|executor|reviewer|planner，"
                                    "默认 worker"},
            "max_steps": {"type": "integer",
                          "description": "子智能体最多几轮工具往返，默认取全局配置"}},
         "required": ["task"]},
        agent_delegate,
        source="builtin",
        tags=["orchestration"],
        timeout=config.DELEGATE_TIMEOUT,
    )]


def register_delegate(registry: Registry, backend: Any) -> None:
    """把 `agent_delegate` 注册进主注册表（捕获 backend + 自身作为 live 表）。"""
    for t in delegate_tools(backend, registry):
        if registry.get(t.name) is None:
            registry.register(t)

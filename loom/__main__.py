"""命令行入口：python -m loom <command>"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from . import config

BANNER = r"""
   __
  / /  ___  ___  __ _ _ __
 / /  / _ \/ _ \/ _` | '_ \
/ /__| (_) | (_) | (_| | | | |
\____/\___/ \___/ \__,_|_| |_|
 本地优先 · MCP 原生 · 零 API Key
"""


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(BANNER)
    print(f"  模型   {config.DEFAULT_MODEL}  (ollama @ {config.OLLAMA_HOST})")
    print(f"  工作区 {config.WORKDIR}")
    print(f"  界面   http://{args.host}:{args.port}")
    print()
    uvicorn.run("loom.server:app", host=args.host, port=args.port,
                log_level=args.log_level, access_log=False)
    return 0


def cmd_selftest(_: argparse.Namespace) -> int:
    from .selftest import main as selftest_main

    return selftest_main()


def cmd_tools(_: argparse.Namespace) -> int:
    from .tools import build_registry

    reg = build_registry()
    print(f"{len(reg.names())} 个内置工具：")
    for t in reg.all():
        flag = " [危险]" if t.dangerous else ""
        print(f"  {t.name:<14}{flag}  {t.description}")
    return 0


def cmd_mcp(_: argparse.Namespace) -> int:
    from .mcp import MCPHub
    from .tools import build_registry

    async def go() -> int:
        hub = MCPHub()
        reg = build_registry()
        status = await hub.start(reg)
        if not status:
            print(f"没有配置任何 MCP 服务器（{config.MCP_CONFIG}）")
            print("复制 mcp/servers.example.json 为 mcp/servers.json 后重试。")
            await hub.stop()
            return 0
        for s in status:
            state = "OK " if s["alive"] else "ERR"
            print(f"  [{state}] {s['name']:<16} tools={s['tools']:<3} {s.get('error') or ''}")
        mcp_tools = [t for t in reg.all() if t.source.startswith("mcp:")]
        print(f"\n合计并入 {len(mcp_tools)} 个 MCP 工具：")
        for t in mcp_tools:
            print(f"  {t.name:<28} ({t.source})")
        await hub.stop()
        return 0

    return asyncio.run(go())


def cmd_chat(args: argparse.Namespace) -> int:
    from .llm import LLMError, OllamaBackend
    from .loop import SYSTEM_PROMPT, run_agent
    from .mcp import MCPHub
    from .memory import Store, register_memory
    from .tools import build_registry

    async def go() -> int:
        reg = build_registry()
        store = Store(config.DB_PATH)
        register_memory(reg, store)
        hub = MCPHub()
        await hub.start(reg)
        backend = OllamaBackend()
        try:
            model = args.model or await backend.pick_model()
        except LLMError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": args.prompt}]
        try:
            async for ev in run_agent(backend, reg, messages, model=model,
                                      max_steps=args.max_steps):
                t = ev["type"]
                if t == "text":
                    sys.stdout.write(ev["delta"])
                    sys.stdout.flush()
                elif t == "step":
                    print(f"\n\033[2m── 第 {ev['n']}/{ev['max']} 步 ──\033[0m")
                elif t == "tool_call":
                    args_preview = json.dumps(ev["arguments"], ensure_ascii=False)[:160]
                    print(f"\n\033[36m▶ {ev['name']} {args_preview}\033[0m")
                elif t == "tool_result":
                    mark = "\033[32m✓\033[0m" if ev["ok"] else "\033[31m✗\033[0m"
                    first = (ev["content"] or "").strip().splitlines()
                    preview = first[0][:160] if first else ""
                    print(f"{mark} {ev['name']}  {ev['meta'].get('ms', '?')}ms  {preview}")
                elif t == "final":
                    print(f"\n\n\033[2m[{ev['reason']} · {ev['steps']} 步]\033[0m")
        finally:
            await hub.stop()
        return 0

    return asyncio.run(go())


def cmd_rag(args: argparse.Namespace) -> int:
    """本地知识库：摄入文件/目录，或在已摄入内容上做语义检索。"""
    from .rag import OllamaEmbeddings, VectorStore
    from .tools import SandboxError, safe_path

    async def go() -> int:
        store = VectorStore(config.RAG_DB_PATH, OllamaEmbeddings())
        if args.rag_cmd == "ingest":
            try:
                p = safe_path(args.path)
            except SandboxError as exc:
                print(f"FAIL {exc}", file=sys.stderr)
                return 2
            try:
                res = await store.ingest_path(p, recursive=getattr(args, "recursive", True))
            except Exception as exc:
                print(f"FAIL 摄入失败: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            if not res.get("ok"):
                print(f"FAIL {res.get('error', '摄入失败')}", file=sys.stderr)
                return 1
            print(f"  ok  摄入 {res['files']} 个文件，生成 {res['count']} 个文本块")
            print("RAG INGEST OK")
            return 0

        # search
        try:
            hits = await store.search(args.query, int(args.k))
        except Exception as exc:
            print(f"FAIL 检索失败（嵌入模型可能未就绪，先 ollama pull {config.EMBED_MODEL}）: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if not hits:
            print("  （知识库为空或没有相关内容，先用 rag ingest 摄入文档）")
            return 0
        for i, h in enumerate(hits, 1):
            print(f"  [#{i}] {h['source']}  (相似度 {h['score']})")
            print("  " + h["text"][:300].replace("\n", "\n  "))
        print(f"\n  RAG SEARCH OK  ({len(hits)} hits)")
        return 0

    return asyncio.run(go())


def cmd_smoke(_: argparse.Namespace) -> int:
    """端到端：真模型 + 真工具，验证整条链路。"""
    from .llm import LLMError, OllamaBackend

    async def go() -> int:
        backend = OllamaBackend()
        if not await backend.available():
            print(f"FAIL Ollama 未运行（{config.OLLAMA_HOST}）")
            return 1
        try:
            models = await backend.list_models()
        except LLMError as exc:
            print(f"FAIL {exc}")
            return 1
        print(f"  ok  Ollama 在线，本地模型 {len(models)} 个: {models}")
        if not models:
            print(f"FAIL 没有本地模型，先执行: ollama pull {config.DEFAULT_MODEL}")
            return 1

        model = await backend.pick_model()
        print(f"  ok  选用模型 {model}")

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "loom", "chat",
            "用 python_run 算出 1234 * 5678 等于多少，然后只回答这个数字。",
            "--model", model, "--max-steps", "3",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        text = out.decode("utf-8", "replace")
        print("  ── 子进程输出 ──")
        print("  " + text.strip().replace("\n", "\n  ")[-1200:])
        good = proc.returncode == 0 and "7006652" in text
        print(f"\n  {'ok ' if good else 'FAIL'} 端到端：模型调用工具并算出正确结果")
        print("SMOKE OK" if good else "SMOKE FAILED")
        return 0 if good else 1

    return asyncio.run(go())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loom", description="本地优先的 MCP 原生智能体运行时")
    p.add_argument("--version", action="version", version=f"loom {config.__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="启动 Web 服务与界面")
    s.add_argument("--host", default=config.HOST)
    s.add_argument("--port", type=int, default=config.PORT)
    s.add_argument("--log-level", default="warning")
    s.set_defaults(func=cmd_serve)

    sub.add_parser("selftest", help="跑不变量自检（离线、不需要模型）") \
        .set_defaults(func=cmd_selftest)
    sub.add_parser("tools", help="列出内置工具") \
        .set_defaults(func=cmd_tools)
    sub.add_parser("mcp", help="列出 MCP 服务器与并入的工具") \
        .set_defaults(func=cmd_mcp)
    sub.add_parser("smoke", help="端到端冒烟（需要 Ollama + 一个本地模型）") \
        .set_defaults(func=cmd_smoke)

    rag = sub.add_parser("rag", help="本地知识库：摄入 / 检索")
    rag_sub = rag.add_subparsers(dest="rag_cmd", required=True)
    ri = rag_sub.add_parser("ingest", help="摄入文件或目录到知识库")
    ri.add_argument("path", help="文件或目录的相对路径")
    ri.add_argument("--no-recursive", dest="recursive", action="store_false",
                    help="目录不递归")
    rs = rag_sub.add_parser("search", help="在知识库上做语义检索")
    rs.add_argument("query", help="自然语言问题")
    rs.add_argument("--k", type=int, default=5, help="返回条数，默认 5")
    rag.set_defaults(func=cmd_rag)

    c = sub.add_parser("chat", help="命令行一轮对话")
    c.add_argument("prompt")
    c.add_argument("--model", default=None)
    c.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    c.set_defaults(func=cmd_chat)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

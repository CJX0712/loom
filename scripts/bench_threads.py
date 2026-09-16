"""Ollama 线程数扫描：同一提示词、同一模型，只改 num_thread，看吞吐怎么变。

结论预期：小模型 + 单路推理是**内存带宽瓶颈**，不是算力瓶颈。
线程超订（比如 16）不会更快，只会让核心互相抢总线。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

HOST = "http://127.0.0.1:11434"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b"
THREADS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                            else ["2", "4", "6", "8", "12", "16"])]
PROMPT = ("用中文写一段 200 字左右的说明，介绍本地大模型推理为什么是内存带宽瓶颈"
          "而不是算力瓶颈。直接开始写，不要客套。")


def run(num_thread: int, num_predict: int = 128) -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "think": False,
        "options": {"num_thread": num_thread, "num_predict": num_predict,
                    "temperature": 0.3, "seed": 42},
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    wall = time.perf_counter() - t0
    return {
        "num_thread": num_thread,
        "eval_count": d.get("eval_count", 0),
        "eval_s": (d.get("eval_duration") or 1) / 1e9,
        "prompt_count": d.get("prompt_eval_count", 0),
        "prompt_s": (d.get("prompt_eval_duration") or 1) / 1e9,
        "load_s": (d.get("load_duration") or 0) / 1e9,
        "wall_s": wall,
    }


def main() -> int:
    print(f"模型 {MODEL}   每档 num_predict=128   提示词固定\n")
    print(f"{'线程':>5} {'生成 tok/s':>11} {'预填 tok/s':>11} {'加载 s':>7} {'整轮 s':>7}")
    print("-" * 46)
    rows = []
    for n in THREADS:
        try:
            r = run(n)
        except Exception as exc:
            print(f"{n:>5}   失败: {exc}")
            continue
        gen = r["eval_count"] / r["eval_s"] if r["eval_s"] else 0
        pre = r["prompt_count"] / r["prompt_s"] if r["prompt_s"] else 0
        rows.append((n, gen))
        print(f"{n:>5} {gen:>11.2f} {pre:>11.2f} {r['load_s']:>7.1f} {r['wall_s']:>7.1f}")

    if not rows:
        return 1
    best = max(rows, key=lambda x: x[1])
    worst = min(rows, key=lambda x: x[1])
    print(f"\n最快 {best[0]} 线程 ({best[1]:.2f} tok/s)，"
          f"最慢 {worst[0]} 线程 ({worst[1]:.2f} tok/s)，"
          f"相差 {best[1] / max(worst[1], 1e-9):.2f}×")
    print(f"\n推荐：num_thread={best[0]}"
          f"  →  export OLLAMA_NUM_THREADS={best[0]}  （重启 ollama serve 生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

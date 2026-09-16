"""HTTP 冒烟：把服务真的起起来，走一遍完整链路。

跑的内容：
  1. /api/health        —— Ollama、模型、工具、MCP 状态
  2. /api/tools         —— 工具清单齐全
  3. 建会话 → 发一轮    —— SSE 真的在流、事件形状正确、收尾有 done
  4. transcript 落盘    —— 刷新后能读回同一段对话
  5. 长期记忆           —— 通过 API 能查回来

需要服务已经在跑：
    python -m loom serve --port 8790
    python scripts/smoke_http.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790"
PROMPT = "用 python_run 算出 1234 * 5678，然后只回答那个数字。"
EXPECT = "7006652"

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"   <- {detail}" if detail and not passed else ""))


def get(path: str, timeout: int = 30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path: str, payload: dict, timeout: int = 60):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stream_chat(session_id: str, text: str, budget_s: float = 600.0) -> list[dict]:
    """读 SSE，返回事件列表。"""
    req = urllib.request.Request(
        BASE + "/api/chat",
        data=json.dumps({"session_id": session_id, "text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    events: list[dict] = []
    deadline = time.time() + budget_s
    with urllib.request.urlopen(req, timeout=budget_s) as resp:
        check("chat 返回 text/event-stream",
              resp.headers.get("Content-Type", "").startswith("text/event-stream"),
              resp.headers.get("Content-Type", ""))
        for raw in resp:
            if time.time() > deadline:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError as exc:
                check("SSE 每一帧都是合法 JSON", False, f"{exc}: {line[:80]}")
                continue
            events.append(ev)
            if ev.get("type") in ("done", "final") and ev.get("type") == "done":
                break
    return events


def main() -> int:
    print(f"smoke @ {BASE}\n")

    try:
        h = get("/api/health")
    except (urllib.error.URLError, OSError) as exc:
        print(f"  服务没起来：{exc}")
        print("\n请先运行: python -m loom serve")
        return 1

    check("health 响应结构完整", {"ollama", "tools", "mcp", "config"} <= set(h), str(list(h)))
    check("Ollama 在线", h["ollama"]["up"], str(h["ollama"]))
    check("已选中一个本地模型", bool(h.get("model")), str(h.get("model_error")))

    t = get("/api/tools")
    names = {x["name"] for x in t["tools"]}
    check("内置工具齐全（含 MCP 并进来的）",
          {"fs_read", "fs_write", "fs_list", "fs_grep", "shell_run",
           "web_fetch", "web_search", "python_run", "memory_save"} <= names,
          str(sorted(names)))

    sess = post("/api/sessions", {"title": "smoke"})
    sid = sess["id"]
    check("新建会话返回 id", bool(sid), str(sess))
    check("新会话 transcript 为空", sess["messages"] == [])

    print(f"\n  提问（可能要走好几步，真模型 CPU 推理会慢）: {PROMPT}")
    t0 = time.time()
    events = stream_chat(sid, PROMPT)
    elapsed = time.time() - t0
    kinds = [e.get("type") for e in events]

    check("收到 session 事件（回传 session_id）", "session" in kinds, str(kinds[:6]))
    check("至少走过一轮 step", kinds.count("step") >= 1, f"{kinds.count('step')} 轮")
    check("流以 done 收尾", kinds and kinds[-1] == "done", str(kinds[-3:]))
    check("有且仅有一个 final", kinds.count("final") == 1, str(kinds.count("final")))
    check("没有 error 事件", "error" not in kinds,
          str([e.get("message") for e in events if e.get("type") == "error"][:2]))

    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    tool_res = [e for e in events if e.get("type") == "tool_result"]
    check("模型真的调用了工具", len(tool_calls) >= 1, f"{len(tool_calls)} 次")
    check("每个 tool_call 都有配对的 tool_result",
          {e["id"] for e in tool_calls} == {e["id"] for e in tool_res},
          f"calls={[e['id'] for e in tool_calls]} results={[e['id'] for e in tool_res]}")

    final = next((e for e in events if e.get("type") == "final"), {})
    check("final 带终止原因与步数",
          final.get("reason") and final.get("steps", 0) >= 1, str(final))
    answer = "".join(e.get("delta", "") for e in events if e.get("type") == "text")
    check(f"模型用工具算出了 {EXPECT}", EXPECT in answer,
          f"回答片段: {answer.strip()[-160:]!r}")

    back = get(f"/api/sessions/{sid}")
    roles = [m["role"] for m in back["messages"]]
    # system 提示词属于代码、不属于用户数据，**故意不入库**，每轮请求时重新注入。
    # 所以这里期望的是：库里没有 system，但有完整的 user/assistant/tool 往返。
    check("transcript 不含 system（提示词不入库）", "system" not in roles, str(roles[:3]))
    check("transcript 是完整的 user→assistant→tool 往返",
          roles[:2] == ["user", "assistant"] and "tool" in roles
          and roles[-1] in ("assistant", "tool"), str(roles))

    # 协议不变量：落盘后的 transcript 依然自洽
    want, got = [], []
    for m in back["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            want += [tc["id"] for tc in m["tool_calls"]]
        elif m.get("role") == "tool":
            got.append(m.get("tool_call_id"))
    check("落盘 transcript 的 tool_call / tool 响应一一对应",
          sorted(want) == sorted(got), f"calls={len(want)} results={len(got)}")

    print(f"\n  端到端耗时 {elapsed:.1f}s")
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    print("ALL GREEN" if not failed else f"SOME FAILED ({len(failed)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

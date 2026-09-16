#!/usr/bin/env bash
# Loom 一键启动（macOS / Linux）
#   首次运行会自动建 venv、装依赖；之后直接起服务。
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "[loom] 创建虚拟环境 $VENV ..."
  python3 -m venv "$VENV"
  echo "[loom] 安装依赖 ..."
  "$PY" -m pip install --upgrade pip -q
  "$PY" -m pip install -r requirements.txt -q
fi

case "${1:-serve}" in
  selftest|tools|mcp|smoke)
    exec "$PY" -X utf8 -m loom "$@"
    ;;
  serve)
    shift || true
    if ! command -v ollama >/dev/null 2>&1; then
      echo "[loom] 未检测到 ollama。请先安装并执行: ollama pull qwen3:4b"
    fi
    echo "[loom] 启动服务 http://127.0.0.1:${LOOM_PORT:-8790}"
    exec "$PY" -X utf8 -m loom serve "$@"
    ;;
  *)
    exec "$PY" -X utf8 -m loom "$@"
    ;;
esac

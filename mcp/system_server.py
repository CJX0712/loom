#!/usr/bin/env python3
"""一个真实可用的 MCP 服务器示例 —— 把本机运行状况暴露给智能体。

**协议部分完全交给官方 SDK**（`mcp.server.FastMCP`），这里只写工具本身。
把它接进 Loom 之后，模型就能自己回答「这台机器现在什么情况」。

用标准库实现，不装 psutil 也能跑；装了 psutil 会自动用上更准的数据。
"""

from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# MCP SDK 2.x 把 FastMCP 改名成了 MCPServer。两种都认，平滑跨过这次迁移。
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]

mcp = MCPServer("loom-system", version="0.1.0",
                instructions="提供本机运行状况，让智能体知道自己跑在什么环境里。")


def _mem() -> tuple[int, int]:
    """返回 (总内存字节, 已用字节)。跨平台，零依赖。"""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.total - vm.available)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullTotalPhys), int(st.ullTotalPhys - st.ullAvailPhys)
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k.strip()] = int(v.split()[0]) * 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return total, total - avail
    except Exception:
        return 0, 0


def _gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"


@mcp.tool()
def sys_overview() -> str:
    """这台机器的整体状况：系统、CPU、内存、磁盘占用、Python 版本。"""
    total, used = _mem()
    disk = shutil.disk_usage(os.path.abspath(os.sep if sys.platform != "win32" else "C:\\"))
    pct = (used / total * 100) if total else 0
    lines = [
        f"系统      {platform.system()} {platform.release()} ({platform.machine()})",
        f"主机名    {platform.node()}",
        f"CPU       {os.cpu_count()} 逻辑核",
        f"内存      {_gb(used)} / {_gb(total)}  已用 {pct:.1f}%",
        f"系统盘    剩 {_gb(disk.free)} / 共 {_gb(disk.total)}",
        f"Python    {platform.python_version()} @ {sys.executable}",
        f"负载均值  {', '.join(f'{x:.2f}' for x in os.getloadavg())}"
        if hasattr(os, "getloadavg") else "负载均值  (此平台不支持)",
    ]
    return "\n".join(lines)


@mcp.tool()
def top_processes(limit: int = 10) -> str:
    """按内存占用降序列出占用最高的进程。limit 为返回条数（1-40）。"""
    limit = max(1, min(int(limit), 40))
    rows: list[tuple[str, int, str]] = []
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=30
        ).stdout
        for r in csv.reader(io.StringIO(out)):
            if len(r) >= 5:
                try:
                    kb = int(r[4].replace(",", "").replace(" K", "").strip())
                except ValueError:
                    continue
                rows.append((r[0], kb * 1024, r[1]))
    else:
        out = subprocess.run(
            ["ps", "-eo", "comm,rss,pid", "--sort=-rss"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in out.splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) == 3:
                try:
                    rows.append((parts[0], int(parts[1]) * 1024, f"pid {parts[2]}"))
                except ValueError:
                    continue
    rows.sort(key=lambda x: -x[1])
    if not rows:
        return "拿不到进程列表"
    return "\n".join(
        f"{i + 1:>3}. {name[:34]:<34} {_gb(mem):>10}  {extra}"
        for i, (name, mem, extra) in enumerate(rows[:limit])
    )


@mcp.tool()
def listening_ports(limit: int = 25) -> str:
    """列出正在监听的 TCP 端口。用于确认某个服务到底起没起。"""
    limit = max(1, min(int(limit), 100))
    hits: list[str] = []
    try:
        if sys.platform == "win32":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True, timeout=30).stdout
            for line in out.splitlines():
                f = line.split()
                if len(f) >= 4 and f[3] == "LISTENING" and f[1].endswith(":0"):
                    hits.append(f"{f[1]:<24} pid={f[4]}")
        else:
            out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True,
                                 timeout=30).stdout
            for line in out.splitlines()[1:]:
                f = line.split()
                if len(f) >= 4:
                    hits.append(f"{f[3]:<24} {f[-1] if len(f) > 5 else ''}")
    except Exception as exc:
        return f"查询失败: {exc}"
    if not hits:
        return "没有监听中的 TCP 端口"
    return "\n".join(hits[:limit])


@mcp.tool()
def now(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """当前时间。fmt 为 strftime 格式串。模型没有时钟，需要时间就问这里。"""
    t = time.localtime()
    try:
        return time.strftime(fmt, t)
    except ValueError as exc:
        return f"格式非法: {exc}"


if __name__ == "__main__":
    mcp.run()

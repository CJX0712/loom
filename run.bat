@echo off
REM Loom 一键启动（Windows）
REM   首次运行会自动建 venv、装依赖；之后直接起服务。
setlocal enabledelayedexpansion
cd /d "%~dp0"

set VENV=.venv
set PY=%VENV%\Scripts\python.exe

if not exist "%PY%" (
  echo [loom] 创建虚拟环境 %VENV% ...
  python -m venv %VENV% || (echo [loom] 建 venv 失败，请确认已安装 Python 3.11+ && pause && exit /b 1)
  echo [loom] 安装依赖 ...
  "%PY%" -m pip install --upgrade pip -q
  "%PY%" -m pip install -r requirements.txt -q || (echo [loom] 装依赖失败 && pause && exit /b 1)
)

if /i "%~1"=="selftest" goto selftest
if /i "%~1"=="tools"    goto passthru
if /i "%~1"=="mcp"      goto passthru
if /i "%~1"=="smoke"    goto passthru
if /i "%~1"=="chat"     goto passthru

echo [loom] 检查 Ollama ...
where ollama >nul 2>nul && (ollama list >nul 2>nul || echo [loom] 提示：先执行 ollama pull qwen3:4b)

echo [loom] 启动服务 http://127.0.0.1:8790
"%PY%" -X utf8 -m loom serve %*
exit /b %ERRORLEVEL%

:selftest
"%PY%" -X utf8 -m loom selftest
pause
exit /b %ERRORLEVEL%

:passthru
"%PY%" -X utf8 -m loom %*
exit /b %ERRORLEVEL%

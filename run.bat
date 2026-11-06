@echo off
chcp 65001 >nul
cd /d %~dp0
title CourseMate 选课咨询机器人

REM ============ 找到 Python 解释器 ============
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [1/3] 首次运行: 创建虚拟环境并安装依赖(约 3-10 分钟, 取决于网速)...
    where uv >nul 2>nul
    if %errorlevel%==0 (
        uv venv .venv
        uv pip install -r requirements.txt
    ) else (
        where python >nul 2>nul
        if %errorlevel%==0 (
            python -m venv .venv
            call .venv\Scripts\activate.bat
            pip install -r requirements.txt
        ) else (
            echo.
            echo [错误] 未找到 Python。请先安装 Python 3.11+ (https://www.python.org/downloads/),
            echo        安装时勾选 "Add python.exe to PATH", 然后重新双击本文件。
            echo        或安装 uv (https://docs.astral.sh/uv/getting-started/installation/) 后重试。
            pause
            exit /b 1
        )
    )
)

REM ============ 构建知识库索引(缺少时) ============
if not exist vector_index\vectors.npy (
    echo [2/3] 构建课程知识库向量索引(首次约 1-2 分钟)...
    "%PY%" build_kb.py
    if errorlevel 1 ( echo 索引构建失败 & pause & exit /b 1 )
)

REM ============ 启动 ============
echo [3/3] 启动 CourseMate... 浏览器将自动打开 http://127.0.0.1:7860
echo(   提示: 关闭本窗口即停止服务。没有 Kimi API Key 时, 请在界面右下角
echo          切换为 "本地微调模型" 引擎(无需联网申请, 首次加载模型约需下载 1GB)。
"%PY%" src\app.py
pause

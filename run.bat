@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "HF_HOME=%~dp0.hf_cache"
title CourseMate

set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto check_dependencies

echo [1/3] Creating the project virtual environment...
where uv >nul 2>nul
if not errorlevel 1 goto create_with_uv
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Install Python 3.11+ with "Add python.exe to PATH", or install uv.
    goto failed
)
python -m venv .venv
if errorlevel 1 goto failed
goto check_dependencies

:create_with_uv
uv venv --seed .venv
if errorlevel 1 goto failed

:check_dependencies
REM Save requirements only after a successful installation so failures are retried.
fc /b requirements.txt .venv\installed-requirements.txt >nul 2>nul
if errorlevel 1 goto install_dependencies
"%PY%" -c "from importlib.metadata import distribution; [distribution(p) for p in ('gradio', 'openai', 'sentence-transformers', 'faster-whisper', 'av', 'edge-tts', 'python-dotenv', 'numpy', 'torch', 'transformers', 'peft', 'ddgs')]" >nul 2>nul
if errorlevel 1 goto install_dependencies
"%PY%" -m pip check
if errorlevel 1 goto install_dependencies
goto check_index

:install_dependencies
echo [1/3] Installing dependencies. Keep this window open until installation finishes.
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check the error above and run this file again.
    goto failed
)
"%PY%" -m pip check
if errorlevel 1 goto failed
copy /y requirements.txt .venv\installed-requirements.txt >nul
if errorlevel 1 goto failed

:check_index
echo [CHECK] Loading native dependencies to detect blocked DLLs...
"%PY%" -B check_runtime.py
if errorlevel 3 goto failed
if errorlevel 2 goto torch_cpu_fallback
if errorlevel 1 goto failed
goto validate_index

:torch_cpu_fallback
echo [WARNING] Windows application control blocked Torch. Trying torch 2.6.0+cpu once.
echo [WARNING] This replaces Torch in this project environment and disables CUDA acceleration.
"%PY%" -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu"
if errorlevel 1 goto failed
"%PY%" -m pip check
if errorlevel 1 goto failed
echo [CHECK] Checking runtime dependencies again after the CPU fallback...
"%PY%" -B check_runtime.py
if errorlevel 1 goto failed

:validate_index
if not exist vector_index\vectors.npy goto build_index
if not exist vector_index\meta.json goto build_index
goto launch

:build_index
echo [2/3] Building the course knowledge index...
"%PY%" build_kb.py
if errorlevel 1 goto failed

:launch
echo [3/3] Starting CourseMate. The browser will open when the server is ready.
echo Keep this window open while using the app.
"%PY%" src\app.py
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo [ERROR] CourseMate could not start. Please check the messages above.
pause
exit /b 1

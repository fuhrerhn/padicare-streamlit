@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Follow README setup first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause

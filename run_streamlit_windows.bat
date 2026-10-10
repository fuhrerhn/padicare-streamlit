@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Follow README setup first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run frontend/app.py
pause

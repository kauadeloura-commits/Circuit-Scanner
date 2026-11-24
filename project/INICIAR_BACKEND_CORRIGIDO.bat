@echo off
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" iniciar_scanner.py --no-browser
) else (
    python iniciar_scanner.py --no-browser
)

if errorlevel 1 pause

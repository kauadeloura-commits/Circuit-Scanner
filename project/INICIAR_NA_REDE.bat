@echo off
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" iniciar_scanner.py
) else (
  py iniciar_scanner.py
)
if errorlevel 1 pause

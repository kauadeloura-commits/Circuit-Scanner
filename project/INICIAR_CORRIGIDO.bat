@echo off
cd /d "%~dp0"

start "Circuit Scanner - Backend corrigido" "%~dp0INICIAR_BACKEND_CORRIGIDO.bat"
timeout /t 3 /nobreak >nul
start "Circuit Scanner - Frontend corrigido" "%~dp0INICIAR_FRONTEND_CORRIGIDO.bat"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5510"


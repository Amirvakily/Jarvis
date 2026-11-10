@echo off
rem Double-click to start Jarvis (window + microphone).
cd /d "%~dp0"
if not exist .venv\Scripts\jarvis.exe (
  echo Jarvis is not installed yet. Run scripts\install.ps1 first.
  pause
  exit /b 1
)
.venv\Scripts\jarvis.exe %*
if errorlevel 1 pause

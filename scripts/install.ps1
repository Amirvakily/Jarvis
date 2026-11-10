# Jarvis installer for Windows (PowerShell).
#   Right-click > Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python 3.11+ is required. Install it from https://www.python.org/downloads/ (tick 'Add to PATH')."
    exit 1
}
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[all]"
if (-not (Test-Path jarvis.toml)) { & .\.venv\Scripts\jarvis.exe init }
& .\.venv\Scripts\jarvis.exe setup-voice
& .\.venv\Scripts\jarvis.exe doctor

Write-Host ""
Write-Host "Done. Start Jarvis by double-clicking start-jarvis.bat"
Write-Host "For a local model, install Ollama (https://ollama.com) and run:  ollama pull llama3.1:8b"

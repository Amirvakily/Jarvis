#!/usr/bin/env sh
# Jarvis installer for macOS and Linux:  sh scripts/install.sh
set -eu
cd "$(dirname "$0")/.."

command -v python3 >/dev/null || { echo "Python 3.11+ is required."; exit 1; }
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[all]"
[ -f jarvis.toml ] || .venv/bin/jarvis init
.venv/bin/jarvis setup-voice || echo "Voice download skipped; run '.venv/bin/jarvis setup-voice' later."
.venv/bin/jarvis doctor || true

echo
echo "Done. Start Jarvis with:  .venv/bin/jarvis"
echo "For a local model, install Ollama (https://ollama.com) and run:  ollama pull llama3.1:8b"
echo "Linux: microphone access needs PortAudio (sudo apt install libportaudio2)."

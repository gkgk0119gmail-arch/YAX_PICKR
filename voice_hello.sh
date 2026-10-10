#!/usr/bin/env bash
# Voice "인사" -> hello motion: ./voice_hello.sh [--test] [--vel 150]
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PICKR_PY:-$HERE/perception-main/ai_bubble/.venv/bin/python}"
[ -x "$PY" ] || PY="$HERE/../perception-main/ai_bubble/.venv/bin/python"     # working-folder venv
export PYTHONNOUSERSITE=1
exec "$PY" "$HERE/robot_piper/voice_hello.py" "$@"

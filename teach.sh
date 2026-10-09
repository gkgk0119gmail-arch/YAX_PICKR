#!/usr/bin/env bash
# Teach & replay: ./teach.sh record NAME | ./teach.sh play NAME [--vel 20] [--dry-run] | ./teach.sh list
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PICKR_PY:-$HERE/perception-main/ai_bubble/.venv/bin/python}"
[ -x "$PY" ] || PY="$HERE/../perception-main/ai_bubble/.venv/bin/python"     # working-folder venv
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
exec "$PY" "$HERE/robot_piper/teach_replay.py" "$@"

#!/usr/bin/env bash
# Demo mode for the judges: partition -> blocks -> drag the basket, START / STOP in the live window.
#   ./demo.sh            ./demo.sh --no-qwen     (skips the ~30 s Qwen check of the partition scan)
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PICKR_PY:-$HERE/perception-main/ai_bubble/.venv/bin/python}"
[ -x "$PY" ] || PY="$HERE/../perception-main/ai_bubble/.venv/bin/python"     # working-folder venv
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
exec "$PY" "$HERE/robot_piper/demo.py" "$@"

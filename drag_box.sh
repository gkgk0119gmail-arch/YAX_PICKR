#!/usr/bin/env bash
# Drag the box: grip the middle of a side wall and pull it toward that side.
#   ./drag_box.sh --dry-run     ./drag_box.sh [--side right] [--distance 0.30]
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PICKR_PY:-$HERE/perception-main/ai_bubble/.venv/bin/python}"
[ -x "$PY" ] || PY="$HERE/../perception-main/ai_bubble/.venv/bin/python"     # working-folder venv
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
exec "$PY" "$HERE/robot_piper/drag_box.py" "$@"

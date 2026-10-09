#!/usr/bin/env bash
# Partition mission launcher: runs robot_piper/move_partition.py in the pipeline venv (offline models).
#   ./partition.sh --dry-run     scan + plan only        ./partition.sh     move (asks once)
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PICKR_PY:-$HERE/perception-main/ai_bubble/.venv/bin/python}"
[ -x "$PY" ] || PY="$HERE/../perception-main/ai_bubble/.venv/bin/python"     # working-folder venv
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
exec "$PY" "$HERE/robot_piper/move_partition.py" "$@"

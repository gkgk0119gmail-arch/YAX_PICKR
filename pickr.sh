#!/usr/bin/env bash
# PICKR launcher: runs pickr.py in the pipeline venv with offline model loading.
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PICKR_PY:-$HERE/perception-main/ai_bubble/.venv/bin/python}"
[ -x "$PY" ] || PY="$HERE/../perception-main/ai_bubble/.venv/bin/python"     # working-folder venv
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
exec "$PY" "$HERE/pickr.py" "$@"

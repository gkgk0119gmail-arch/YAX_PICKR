#!/usr/bin/env bash
# One-time setup on a new machine (Ubuntu, NVIDIA GPU). Creates the venv the pipeline expects:
#   perception-main/ai_bubble/.venv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/perception-main/ai_bubble/.venv"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv "$VENV" --python 3.11
uv pip install --python "$VENV/bin/python" -r "$ROOT/setup/requirements.txt"

# VGGT (Meta) from source - code only; the commercial checkpoint is downloaded by get_models.py
if [ ! -d "$ROOT/perception-main/vggt" ]; then
  git clone --depth 1 https://github.com/facebookresearch/vggt.git "$ROOT/perception-main/vggt"
fi
uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT/perception-main/vggt"

# system tools for the PiPER CAN bus (needs sudo)
echo "next: sudo apt install -y can-utils ethtool v4l-utils"
echo "then: $VENV/bin/python $ROOT/setup/get_models.py"

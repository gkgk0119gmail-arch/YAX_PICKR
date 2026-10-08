#!/usr/bin/env bash
# Run a script from sim_tools/ in the piper_sim conda env (ROS PYTHONPATH unset).
#   ./sim_tools/run.sh pick_place_demo.py [args]
source ~/miniconda3/etc/profile.d/conda.sh
conda activate piper_sim
script="$1"; shift
exec env -u PYTHONPATH python "$(dirname "$0")/$script" "$@"

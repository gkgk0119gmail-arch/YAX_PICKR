#!/usr/bin/env bash
# PiPER MuJoCo viewer — conda env `piper_sim`; ROS PYTHONPATH unset so humble's packages don't leak in.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate piper_sim
exec env -u PYTHONPATH python "$(dirname "$0")/view_piper.py" "$@"

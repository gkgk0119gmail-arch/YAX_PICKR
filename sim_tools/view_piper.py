"""Open the WeGo PiPER MuJoCo scene (arm + gripper + cube + bin) in the interactive viewer.

Run:  ./sim_tools/run_viewer.sh
Viewer tips: double-click a body to select, Ctrl+right-drag to push it,
right panel "Control" tab has sliders for joint1..6 / gripper actuators.
"""
import mujoco
import mujoco.viewer

from piper_sim.scene import load_model  # bakes default_scene.json (cube, bin) into piper_scene.xml

model = load_model()
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)

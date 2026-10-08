"""PiPER pick & place in MuJoCo — standalone (no web backend / redis / shm).

Uses the same IK model, grasp pose and phase heights as
wego-piper-web/backend/app/services/sim_demo.py, but drives the MuJoCo
actuators directly instead of publishing to a leader segment.

Run:
  ./sim_tools/run.sh pick_place_demo.py                 # viewer
  ./sim_tools/run.sh pick_place_demo.py --headless      # no window, prints result
  ./sim_tools/run.sh pick_place_demo.py --cube 0.30 0.08 --episodes 3
"""
import argparse
import math
import time

import mujoco
import numpy as np

from piper_robot.armmodel import ArmModel
from piper_robot.joints import JOINT_CALIBRATION
from piper_sim.scene import ARM_JOINTS, FINGER_STROKE_M, JointMap, load_model

# ── values taken from sim_demo.py (measured there) ─────────────────────────────
GRASP_SEED_DEG = (-5.9, 121.6, -76.9, -0.8, 43.0, 81.5)  # a real top-down IK solution
GRASP_Z, APPROACH_Z, LIFT_Z, DROP_Z = 0.095, 0.195, 0.20, 0.16  # link6 heights (m)
OPEN, CLOSED = 1.0, 0.0  # gripper fraction
PHASES = [  # (name, target, link6 z, gripper, seconds)
    ("approach", "cube", APPROACH_Z, OPEN, 1.5),
    ("descend", "cube", GRASP_Z, OPEN, 1.2),
    ("grasp", "cube", GRASP_Z, CLOSED, 2.0),
    ("lift", "cube", LIFT_Z, CLOSED, 1.2),
    ("carry", "bin", LIFT_Z, CLOSED, 1.8),
    ("lower", "bin", DROP_Z, CLOSED, 0.9),
    ("release", "bin", DROP_Z, OPEN, 0.9),
    ("retreat", "bin", LIFT_Z, OPEN, 0.9),
]


def arm_model() -> ArmModel:
    """IK model with limits narrowed to the real robot's calibration range."""
    m = ArmModel.load("piper")
    cal = np.array([JOINT_CALIBRATION[j] for j in ARM_JOINTS]) * math.pi / 180000.0
    m.limits = np.stack([np.maximum(m.limits[:, 0], cal[:, 0]),
                         np.minimum(m.limits[:, 1], cal[:, 1])], axis=1)
    return m


class Sim:
    def __init__(self, headless: bool, model=None):
        self.model = model if model is not None else load_model()
        self.data = mujoco.MjData(self.model)
        self.jm = JointMap(self.model)
        self.viewer = None if headless else mujoco.viewer.launch_passive(self.model, self.data)
        self._last_sync = -1.0
        self.on_step = None  # optional callback(sim) after every physics step
        mujoco.mj_forward(self.model, self.data)
        self.jm.hold(self.data)

    def body_xy(self, name):
        return self.data.body(name).xpos[:2].copy()

    def place_body(self, name, x, y, z):
        adr = self.model.jnt_qposadr[self.model.joint(f"{name}_free").id]
        self.data.qpos[adr:adr + 7] = [x, y, z, 1, 0, 0, 0]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_ctrl(self, q_rad, grip):
        for j, q in zip(ARM_JOINTS, q_rad):
            self.data.ctrl[self.jm.act[j]] = q
        for f in ("gripper_l", "gripper_r"):
            self.data.ctrl[self.jm.act[f]] = grip * FINGER_STROKE_M

    def run(self, secs, realtime):
        for _ in range(int(secs / self.model.opt.timestep)):
            t0 = time.perf_counter()
            mujoco.mj_step(self.model, self.data)
            if self.on_step:
                self.on_step(self)
            if self.viewer and self.data.time - self._last_sync >= 1 / 60:  # ~60 fps redraw
                if not self.viewer.is_running():
                    raise SystemExit
                self.viewer.sync()
                self._last_sync = self.data.time
            if realtime:
                time.sleep(max(0.0, self.model.opt.timestep - (time.perf_counter() - t0)))

    def ramp(self, q_from, q_to, g_from, g_to, secs, realtime, hz=50):
        n = max(2, int(secs * hz))
        for i in range(1, n + 1):
            a = i / n
            self.set_ctrl(q_from + (q_to - q_from) * a, g_from + (g_to - g_from) * a)
            self.run(1.0 / hz, realtime)


def episode(sim: Sim, ik: ArmModel, realtime: bool) -> bool:
    R = ik.fk(np.radians(GRASP_SEED_DEG))[:3, :3]
    where = {"cube": sim.body_xy("cube"), "bin": sim.body_xy("bin")}
    q = sim.jm.q_rad(sim.data)
    grip = CLOSED
    seed = np.radians(GRASP_SEED_DEG)
    for name, what, z, want_grip, secs in PHASES:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [*where[what], z]
        sol = ik.ik(T, seed)
        if not sol.ok:
            print(f"  [{name}] IK failed ({sol.pos_mm:.0f} mm short) — target out of reach")
            return False
        print(f"  [{name:8s}] link6 -> ({T[0, 3]:.3f}, {T[1, 3]:.3f}, {z:.3f})")
        sim.ramp(q, sol.q, grip, want_grip, secs, realtime)
        q, seed, grip = sol.q, sol.q, want_grip
    sim.run(1.0, realtime)  # let the cube settle
    cube = sim.data.body("cube").xpos
    ok = np.linalg.norm(cube[:2] - where["bin"]) < 0.08 and cube[2] < 0.06
    print(f"  cube final = ({cube[0]:.3f}, {cube[1]:.3f}, {cube[2]:.3f}) -> {'IN BIN ✅' if ok else 'MISS ❌'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no window, run as fast as possible")
    ap.add_argument("--cube", nargs=2, type=float, metavar=("X", "Y"), help="cube start xy (m)")
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()
    if not args.headless:
        import mujoco.viewer  # noqa: F401

    sim, ik = Sim(args.headless), arm_model()
    rng = np.random.default_rng()
    wins = 0
    for ep in range(args.episodes):
        xy = args.cube if args.cube else (
            (0.35, 0.0) if ep == 0 else (rng.uniform(0.28, 0.42), rng.uniform(-0.12, 0.12)))
        sim.place_body("cube", *xy, 0.02)
        print(f"episode {ep + 1}: cube at ({xy[0]:.3f}, {xy[1]:.3f})")
        wins += episode(sim, ik, realtime=not args.headless)
    print(f"success {wins}/{args.episodes}")
    if sim.viewer:
        print("close the viewer window to exit")
        while sim.viewer.is_running():
            sim.run(0.02, True)


if __name__ == "__main__":
    import mujoco.viewer  # noqa: F401  (launch_passive lives here)
    main()

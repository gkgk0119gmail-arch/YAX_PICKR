"""Digital twin: the real pac2026 pipeline (real cameras + perception) drives a MuJoCo PiPER.

The objects look different in sim, so perception does NOT run on sim images. Instead the
pipeline sends what it perceived (box pose / size / grid, block poses / sizes, robot frame)
and this server builds the same scene, then executes the pipeline's robot commands with real
physics and answers the questions the real arm would (gripper width, is the block in the cell).

  ./sim_tools/run.sh twin_server.py                 # viewer, port 8765
  ./sim_tools/run.sh twin_server.py --headless      # no window, as fast as possible
  cd pac2026 && python run.py --robot sim [--image ...]   # in the perception venv

Protocol: one JSON object per line over TCP, one reply line per request.
  {"op": "scene", "box": {center, yaw_deg, rows, cols, cell, height}, "blocks": [{xy, yaw_deg, dims}]}
  {"op": "move", "x", "y", "z", "yaw"}      gripper TIP, robot base frame (m, rad); yaw = finger-opening axis
  {"op": "grip", "width_mm"}                 distance between the finger pads (what the real arm reports)
  {"op": "width"} -> {"width_mm"}
  {"op": "cell_state", "x", "y"} -> {"state": empty | seated | straddle}
  {"op": "report"} -> {"hits": arm<->partition contacts, "blocks": [...]}
"""
import argparse
import json
import socket

import numpy as np

from pac_demo import FINGER_REACH, PartitionWatch
from pac_scene import FINGER_GAP0
from pac_scene import Blocks, PartitionBox, build
from pick_place_demo import GRASP_SEED_DEG, Sim, arm_model
from piper_sim.scene import FINGER_STROKE_M

TILTS_DEG = tuple(range(0, 31, 3))  # tool may lean (top toward the robot): with the real 0.135 m gripper and
#                                     joint5 ≤ 65°, straight down only reaches tip z ≲ 0.08 m
SEG_M = 0.01                   # straight tip lines are solved by IK every SEG_M
TIP_SPEED = 0.12               # m/s
JOINT_SPEED = 1.2              # rad/s


def tool_rotation(x, y, yaw, tilt_deg):
    """Top-down tool frame (tool y = finger-opening axis at `yaw`), leaned `tilt_deg` so its top
    points back toward the robot base (fingertips forward) — what pac_demo does along +x."""
    yv = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    R = np.stack([np.cross(yv, z), yv, z], 1)
    r = np.array([x, y, 0.0])
    if tilt_deg == 0 or np.linalg.norm(r) < 1e-6:
        return R
    a = np.cross([0, 0, 1.0], r / np.linalg.norm(r))           # axis ⟂ to the radial direction
    t = -np.radians(tilt_deg)                                # tool z (down) swings outward
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K) @ R


class Twin:
    def __init__(self, headless, realtime):
        self.headless, self.realtime = headless, realtime
        self.ik = arm_model()
        self.sim = None
        self.load(PartitionBox(), Blocks(count=0))

    # ------------------------------------------------------------------ scene
    def load(self, box, blocks):
        if self.sim and self.sim.viewer:
            self.sim.viewer.close()
        self.box, self.blocks = box, blocks
        self.sim = Sim(self.headless, build(box, blocks))
        self.watch = PartitionWatch(self.sim.model)
        self.sim.on_step = self.watch
        self.sim.run(0.3, False)
        self.q = self.sim.jm.q_rad(self.sim.data)
        self.grip = 0.0
        self.tip = None                     # unknown until the first move (start pose is folded)

    def scene(self, msg):
        b = msg["box"]
        box = PartitionBox(rows=int(b["rows"]), cols=int(b["cols"]), cell=float(b["cell"]),
                           height=float(b["height"]), center=tuple(b["center"]), yaw_deg=float(b["yaw_deg"]))
        bl = msg.get("blocks", [])
        blocks = Blocks(count=len(bl), spots=[tuple(k["xy"]) for k in bl], yaws=[k["yaw_deg"] for k in bl],
                        dims_each=[tuple(k["dims"]) for k in bl])
        self.load(box, blocks)
        ix, iy = box.inner
        print(f"[twin] scene: {box.rows}x{box.cols} cell {box.cell * 100:.1f} cm, inner {ix * 100:.1f}x{iy * 100:.1f} cm, "
              f"h {box.height * 100:.1f} cm at ({box.center[0]:.3f}, {box.center[1]:.3f}) yaw {box.yaw_deg:.0f}°, "
              f"{len(bl)} block(s)", flush=True)
        return {"ok": True}

    # ------------------------------------------------------------------ motion
    def _line(self, R, start, goal):
        n = 1 if start is None else max(1, int(np.ceil(np.linalg.norm(goal - start) / SEG_M)))
        path, s = [], self.q if start is not None else np.radians(GRASP_SEED_DEG)
        for k in range(1, n + 1):
            p = goal if start is None else start + (goal - start) * k / n
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, p - R[:, 2] * FINGER_REACH         # fingertip -> link6
            sol = self.ik.ik(T, s)
            if not sol.ok:
                return None, sol.pos_mm
            path.append(sol.q)
            s = sol.q
        return path, 0.0

    def move(self, msg):
        goal = np.array([msg["x"], msg["y"], msg["z"]], float)
        worst = 0.0
        for tilt in TILTS_DEG:
            for yaw in (msg["yaw"], msg["yaw"] + np.pi):              # parallel gripper: ±180° is the same grasp
                R = tool_rotation(goal[0], goal[1], yaw, tilt)
                path, short = self._line(R, self.tip, goal)
                if path is None and self.tip is not None:            # no straight line: joint-space move
                    path, short = self._line(R, None, goal)
                if path is None:
                    worst = max(worst, short)
                    continue
                self._follow(path, goal)
                return {"ok": True, "tilt_deg": tilt}
        print(f"[twin] move ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) UNREACHABLE ({worst:.0f} mm short)",
              flush=True)
        return {"ok": False, "error": f"unreachable, {worst:.0f} mm short"}

    def _follow(self, path, goal):
        prev_tip = self.tip if self.tip is not None else goal
        for k, q_to in enumerate(path, 1):
            dq = float(np.max(np.abs(q_to - self.q)))
            secs = max(np.linalg.norm(goal - prev_tip) / len(path) / TIP_SPEED, dq / JOINT_SPEED, 0.02)
            self.sim.ramp(self.q, q_to, self.grip, self.grip, secs, self.realtime)
            self.q = q_to
        self.sim.run(0.15, self.realtime)                            # settle
        self.tip = goal

    def grip_to(self, msg):
        g = float(np.clip(msg["width_mm"] / 1000.0 / 2.0 / FINGER_STROKE_M, 0.0, 1.0))
        self.sim.ramp(self.q, self.q, self.grip, g, 0.8, self.realtime)
        self.grip = g
        self.sim.run(0.3, self.realtime)
        return self.width()

    def width(self, _msg=None):
        d, jm = self.sim.data, self.sim.jm
        gap = 2 * FINGER_GAP0 + float(d.qpos[jm.qadr["gripper_l"]] + d.qpos[jm.qadr["gripper_r"]])
        return {"ok": True, "width_mm": gap * 1000.0}           # inner-face distance, like the real arm reports

    # ------------------------------------------------------------------ checks
    def cell_state(self, msg):
        self.sim.run(0.5, self.realtime)                             # let a released block settle
        rc = self.box.cell_of(msg["x"], msg["y"])
        if rc is None:
            return {"ok": True, "state": "empty", "note": "target is not inside a sim cell"}
        cx, cy = self.box.cell_xy(*rc)
        reach = self.box.cell / 2 + self.box.div_t + 0.02
        state = "empty"
        for bid in self.blocks.ids():
            x, y, z = self.sim.data.body(bid).xpos
            if self.box.cell_of(x, y) == rc and z < self.box.height / 2:
                return {"ok": True, "state": "seated", "cell": rc, "block": bid}
            if np.hypot(x - cx, y - cy) < reach and z >= self.box.height / 2 and z < self.box.height + 0.05:
                state = "straddle"                                   # resting on the dividers / rim
        return {"ok": True, "state": state, "cell": rc}

    def report(self, _msg=None):
        blocks = []
        for bid in self.blocks.ids():
            x, y, z = self.sim.data.body(bid).xpos
            blocks.append({"id": bid, "pos": [round(float(v), 4) for v in (x, y, z)], "cell": self.box.cell_of(x, y)})
        return {"ok": True, "hits": self.watch.hits, "blocks": blocks}

    def handle(self, msg):
        op = msg.get("op")
        fn = {"scene": self.scene, "move": self.move, "grip": self.grip_to, "width": self.width,
              "cell_state": self.cell_state, "report": self.report}.get(op)
        if fn is None:
            return {"ok": False, "error": f"unknown op {op!r}"}
        return fn(msg)

    def idle(self):
        if self.sim.viewer:
            self.sim.run(0.02, True)


def serve(twin, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(0.02)
    print(f"[twin] listening on {host}:{port}", flush=True)
    while True:
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            twin.idle()
            continue
        print(f"[twin] pipeline connected from {addr[0]}", flush=True)
        conn.settimeout(0.02)
        buf = b""
        with conn:
            while True:
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    twin.idle()
                    continue
                if not chunk:
                    print("[twin] pipeline disconnected", flush=True)
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        reply = twin.handle(json.loads(line))
                    except Exception as e:  # noqa: BLE001 - report to the pipeline, keep serving
                        reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    conn.sendall((json.dumps(reply) + "\n").encode())


def main():
    ap = argparse.ArgumentParser(description="MuJoCo twin for the pac2026 pipeline (--robot sim)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--headless", action="store_true", help="no window, not paced to real time")
    a = ap.parse_args()
    if not a.headless:
        import mujoco.viewer  # noqa: F401  (launch_passive lives here)
    serve(Twin(a.headless, realtime=not a.headless), a.host, a.port)


if __name__ == "__main__":
    main()

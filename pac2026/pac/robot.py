"""Robot interface: positions are the gripper TIP (TCP) in the robot base frame, metres; yaw in rad
about the vertical axis for a top-down grasp.

MockRobot  - no hardware; logs every command (optionally fails grasps to exercise the retries).
PiperRobot - AgileX PiPER via piper_sdk (C_PiperInterface_V2). Written from the SDK docs / demos,
             NOT yet run on a real arm: check units, down_rpy_deg and tcp_offset_m on site.
SimRobot   - MuJoCo PiPER twin (sim_tools/twin_server.py) fed with the perceived scene."""
import json
import socket
import time

import numpy as np


class RobotBase:
    def connect(self): ...
    def move_to(self, x, y, z, yaw): ...
    def open(self, width_mm=None): ...
    def close(self): ...
    def gripper_width_mm(self) -> float: ...
    def home(self): ...
    def stop(self): ...


class MockRobot(RobotBase):
    def __init__(self, cfg, log, seed=0):
        self.cfg, self.log = cfg["robot"], log
        self.rng = np.random.default_rng(seed)
        self.width = self.cfg["gripper_open_mm"]
        self.pose = (0.3, 0.0, 0.3, 0.0)
        self.over_block = None                              # opening (mm) of the block under the fingers

    def connect(self):
        self.log.info("[mock robot] connected")

    def move_to(self, x, y, z, yaw):
        self.pose = (x, y, z, yaw)
        self.log.info(f"[mock robot] move_to x={x:.3f} y={y:.3f} z={z:.3f} m yaw={np.degrees(yaw):.1f} deg")

    def open(self, width_mm=None):
        self.width = width_mm or self.cfg["gripper_open_mm"]
        self.log.info(f"[mock robot] gripper open {self.width:.0f} mm")

    def close(self):
        miss = self.rng.random() < self.cfg.get("mock_fail_rate", 0.0)
        self.width = 0.0 if (miss or self.over_block is None) else self.over_block
        self.log.info(f"[mock robot] gripper close -> {self.width:.0f} mm{' (simulated miss)' if miss else ''}")

    def gripper_width_mm(self):
        return self.width

    def home(self):
        self.move_to(0.3, 0.0, 0.3, 0.0)

    def stop(self):
        self.log.info("[mock robot] stop")


class PiperRobot(RobotBase):
    def __init__(self, cfg, log):
        self.cfg, self.log = cfg["robot"], log
        self.p = None

    def connect(self):
        from piper_sdk import C_PiperInterface_V2
        self.p = C_PiperInterface_V2(self.cfg["can_port"])
        self.p.ConnectPort()
        t0 = time.time()
        while not self.p.EnablePiper():
            if time.time() - t0 > 10:
                raise RuntimeError("PiPER enable timed out (CAN up? arm powered?)")
            time.sleep(0.01)
        self.p.GripperCtrl(0, self.cfg["gripper_effort"], 0x01, 0)
        self.log.info("[piper] connected and enabled")

    def _flange_pose(self, x, y, z, yaw):
        """TCP target -> flange pose (mm, deg). R = Rz(yaw) * R_down; flange = tcp - R @ [0, 0, tcp_offset]."""
        from scipy.spatial.transform import Rotation as R
        r_down = R.from_euler("xyz", self.cfg["down_rpy_deg"], degrees=True)
        r = R.from_euler("z", yaw) * r_down
        f = np.array([x, y, z]) - r.apply([0, 0, self.cfg["tcp_offset_m"]])
        rx, ry, rz = r.as_euler("xyz", degrees=True)          # extrinsic xyz == Rz*Ry*Rx, as PiPER reports
        return f * 1000.0, (rx, ry, rz)

    def move_to(self, x, y, z, yaw, tol_mm=2.0, timeout=10.0):
        (fx, fy, fz), (rx, ry, rz) = self._flange_pose(x, y, z, yaw)
        k = 1000
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.p.MotionCtrl_2(0x01, 0x00, self.cfg["speed_percent"], 0x00)       # CAN control, MOVE P
            self.p.EndPoseCtrl(round(fx * k), round(fy * k), round(fz * k), round(rx * k), round(ry * k), round(rz * k))
            e = self.p.GetArmEndPoseMsgs().end_pose
            err = np.linalg.norm([e.X_axis / k - fx, e.Y_axis / k - fy, e.Z_axis / k - fz])
            if err < tol_mm:
                return True
            time.sleep(0.01)
        self.log.info(f"[piper] move_to timeout, remaining error {err:.1f} mm")
        return False

    def open(self, width_mm=None):
        w = width_mm or self.cfg["gripper_open_mm"]
        for _ in range(50):
            self.p.GripperCtrl(round(w * 1000), self.cfg["gripper_effort"], 0x01, 0)
            time.sleep(0.01)

    def close(self):
        for _ in range(80):
            self.p.GripperCtrl(0, self.cfg["gripper_effort"], 0x01, 0)
            time.sleep(0.01)

    def gripper_width_mm(self):
        return self.p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0

    def home(self):
        self.move_to(0.3, 0.0, 0.3, 0.0)

    def stop(self):
        # damped stop; do NOT DisablePiper()/ResetPiper() here - the arm would drop
        self.p.MotionCtrl_1(0x01, 0, 0)


class SimRobot(RobotBase):
    """MuJoCo twin (sim_tools/twin_server.py) over TCP. The pipeline's perception still runs on the
    real cameras; the twin rebuilds the perceived scene and answers gripper width / cell state."""
    twin = True

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg["robot"], log
        self.sock, self.buf = None, b""

    def _call(self, **msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("twin server closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        r = json.loads(line)
        if not r.get("ok"):
            self.log.warn(f"[sim] {msg['op']} failed: {r.get('error')}")
        return r

    def connect(self):
        host, port = self.cfg.get("sim_host", "localhost"), self.cfg.get("sim_port", 8765)
        try:
            self.sock = socket.create_connection((host, port), timeout=120)
        except OSError as e:
            raise RuntimeError(f"twin server not reachable at {host}:{port} "
                               f"(start ./sim_tools/run.sh twin_server.py)") from e
        self.log.info(f"[sim] connected to MuJoCo twin {host}:{port}")

    def sync_scene(self, box, part, grasps, calib, block_h):
        from .twin import scene_msg
        msg = scene_msg(box, part, grasps, calib, block_h)
        b = msg["box"]
        self.log.info(f"[sim] scene -> twin: {b['rows']}x{b['cols']} cell {b['cell']*100:.1f} cm, "
                      f"h {b['height']*100:.1f} cm, centre ({b['center'][0]:.3f}, {b['center'][1]:.3f}) m, "
                      f"yaw {b['yaw_deg']:.0f} deg, {len(msg['blocks'])} block(s)")
        self._call(**msg)

    def move_to(self, x, y, z, yaw):
        r = self._call(op="move", x=x, y=y, z=z, yaw=yaw)
        self.log.info(f"[sim] move_to x={x:.3f} y={y:.3f} z={z:.3f} m yaw={np.degrees(yaw):.1f} deg"
                      + (f" (tilt {r['tilt_deg']} deg)" if r.get("tilt_deg") else ""))
        return r.get("ok", False)

    def open(self, width_mm=None):
        self._call(op="grip", width_mm=width_mm or self.cfg["gripper_open_mm"])

    def close(self):
        self._call(op="grip", width_mm=0.0)

    def gripper_width_mm(self):
        return self._call(op="width")["width_mm"]

    def cell_state(self, x, y):
        return self._call(op="cell_state", x=x, y=y)["state"]

    def home(self):
        self.move_to(0.3, 0.0, 0.3, 0.0)

    def report(self):
        r = self._call(op="report")
        if r.get("hits"):
            top = sorted(r["hits"].items(), key=lambda kv: -kv[1])[:5]
            self.log.warn("[sim] arm touched the partition: " + ", ".join(f"{k}x{v}" for k, v in top))
        else:
            self.log.info("[sim] arm never touched the partition")
        return r

    def stop(self):
        self.log.info("[sim] stop")


def make_robot(cfg, log):
    kind = cfg["robot"]["type"]
    return {"piper": PiperRobot, "sim": SimRobot}.get(kind, MockRobot)(cfg, log)

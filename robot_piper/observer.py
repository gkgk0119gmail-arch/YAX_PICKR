"""Fast top-camera checks while the arm works: one D435i frame -> Grounding DINO + SAM2 + RealSense top face
(no VGGT, ~1-2 s) in a background thread. The part of the image covered by the arm (and a held block) is
computed from the joint angles and ignored, so an occluded block is not mistaken for a moved one.

  obs = Observer(finder)            # opens the top camera (keep it open for the whole run)
  obs.start(q, tcp, holding=True)   # returns at once
  ...move the arm...
  res = obs.wait()                  # dict(blocks=[...], containers=[...], mask=..., color=...)
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "pac2026"), str(ROOT / "perception-main")]
import arm_ik  # noqa: E402
import collide  # noqa: E402
from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics  # noqa: E402

T_BASE_CAM = np.array(json.loads((ROOT / "pac2026" / "calib" / "top_cam_to_base.json").read_text())["T_base_cam"])
T_CAM_BASE = np.linalg.inv(T_BASE_CAM)


def project(P, K):
    pc = P @ T_CAM_BASE[:3, :3].T + T_CAM_BASE[:3, 3]
    return np.stack([pc[:, 0] / pc[:, 2] * K["fx"] + K["cx"], pc[:, 1] / pc[:, 2] * K["fy"] + K["cy"]], 1)


def pixel_to_plane(u, v, z_plane, K):
    """Base-frame xy where the camera ray through pixel (u, v) meets the horizontal plane z = z_plane."""
    d_cam = np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
    d = T_BASE_CAM[:3, :3] @ d_cam
    o = T_BASE_CAM[:3, 3]
    t = (z_plane - o[2]) / d[2]
    return (o + t * d)[:2]


def arm_mask(q, tcp, K, shape, holding=False, grow_px=25):
    """Image mask of the arm links, gripper (+ D405) and a held 90x90x120 block for joints q (deg)."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    links = C_PiperForwardKinematics(0x01).CalFK([np.radians(v) for v in q])
    pts = np.array([[0, 0, 0.0], [0, 0, 0.123]] + [np.array(l[:3]) / 1000 for l in links])
    uv = project(pts, K).astype(np.int32)
    for a, b in zip(uv[:-1], uv[1:]):
        cv2.line(m, tuple(a), tuple(b), 1, 70)
    P, T = collide.body_world(q, tcp, 0.10)
    if holding:                                             # a block hanging under the fingertips
        tip = arm_ik.tip_of(q, tcp)[0]
        blk = collide.block_points(tip[:2], 0, tip[2] - 0.065, 0.12)
        P = np.vstack([P, blk])
    hull = cv2.convexHull(project(P, K).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    return cv2.dilate(m, np.ones((grow_px, grow_px), np.uint8)).astype(bool)


class Observer:
    def __init__(self, finder):
        from pac.cameras import RealSenseCam
        cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["camera"]
        prof_path = ROOT / "pac2026" / cfg.get("profile", "camera_profile.yaml")
        prof = (yaml.safe_load(prof_path.read_text()) or {}).get("top") if prof_path.exists() else None
        c = cfg["top"]
        self.cam = RealSenseCam(c["serial"], c["color_size"], c["depth_size"], c["fps"], 3, prof)
        self.finder = finder
        self._thread, self._res = None, None

    def start(self, q, tcp, holding=False):
        q = np.array(q, float)
        self._res = None
        self._thread = threading.Thread(target=self._run, args=(q, tcp, holding), daemon=True)
        self._thread.start()

    def pending(self):
        return self._thread is not None

    def wait(self, timeout=15.0):
        if self._thread is None:
            return None
        self._thread.join(timeout)
        res, self._thread = self._res, None
        return res

    def _run(self, q, tcp, holding):
        t0 = time.time()
        try:
            time.sleep(0.2)                                 # a frame taken after the last move started
            f = self.cam.grab(with_depth=True)
            K = f.intrinsics
            mask = arm_mask(q, tcp, K, f.color.shape[:2], holding)
            vg = self.finder.use_vggt
            self.finder.use_vggt = False                    # fast: RealSense top face only
            try:
                objs = self.finder(f.color, f.depth_m, K)
            finally:
                self.finder.use_vggt = vg
            keep = []
            for o in objs:
                x1, y1, x2, y2 = (int(round(v)) for v in o["box"])
                box = np.zeros(mask.shape, bool)
                box[max(y1, 0):y2, max(x1, 0):x2] = True
                covered = (mask & box).sum() / max(box.sum(), 1)
                o["covered_by_arm"] = float(covered)
                if covered < 0.3:
                    keep.append(o)
            self._res = dict(blocks=[o for o in keep if o["label"] == "block" and o.get("robot")],
                             containers=[o for o in keep if o["label"] == "container"],
                             color=f.color, depth=f.depth_m, mask=mask, K=K, q=q, secs=time.time() - t0)
        except Exception as e:                              # a failed check must not stop the arm
            self._res = dict(error=str(e), blocks=[], containers=[], secs=time.time() - t0)

    def close(self):
        if self._thread is not None:
            self._thread.join(10)
        self.cam.close()

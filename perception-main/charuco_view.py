"""Live ChArUco check on both cameras (top D435i + wrist D405) side by side: board detection, board pose
in each camera frame, a print-scale check (corner spacing from RealSense depth vs the nominal 40 mm), and,
when both see the board at the same moment, the wrist camera's pose in the top camera's frame.

  conda activate piper_sim && env -u PYTHONPATH python perception-main/charuco_view.py [--cams top wrist]
  q / Esc: quit    s: save frames + poses (data/charuco/)

Board = the A4 print ~/Desktop/charuco_A4_7x5_40mm.pdf (same as realsense_calib*.py):
DICT_4X4_100, 7x5 squares, 40 mm squares, 30 mm markers.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE.parent / "pac2026" / "config.yaml").read_text())["camera"]
PROFILE = HERE.parent / "pac2026" / "camera_profile.yaml"
SQUARES, SQUARE, MARKER = (7, 5), 0.040, 0.030
BOARD = cv2.aruco.CharucoBoard(SQUARES, SQUARE, MARKER, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
DETECTOR = cv2.aruco.CharucoDetector(BOARD)
NAMES = {"top": "D435i top", "wrist": "D405 wrist"}
TILE = (960, 540)
INFO_H = 125
FRESH_S = 0.2           # both poses within this time -> camera-to-camera transform


class Cam:
    def __init__(self, role):
        self.role = role
        self.open()

    def open(self):
        c = CFG[self.role]
        cfg = rs.config()
        cfg.enable_device(str(c["serial"]))
        cfg.enable_stream(rs.stream.color, *c["color_size"], rs.format.bgr8, c["fps"])
        cfg.enable_stream(rs.stream.depth, *c["depth_size"], rs.format.z16, c["fps"])
        self.pipe = rs.pipeline()
        prof = self.pipe.start(cfg)
        p = (yaml.safe_load(PROFILE.read_text()) or {}).get(self.role) if PROFILE.exists() else None
        if p and str(p.get("serial")) == str(c["serial"]):         # grey-card exposure / white balance
            sensor = next(s for s in prof.get_device().query_sensors()
                          if any(sp.stream_type() == rs.stream.color for sp in s.get_stream_profiles()))
            for name in ("enable_auto_exposure", "enable_auto_white_balance", "exposure", "gain", "white_balance"):
                if name in p:
                    sensor.set_option(getattr(rs.option, name), float(p[name]))
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
        self.D = np.array(i.coeffs[:5], float)
        self.align = rs.align(rs.stream.color)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()   # D435i 1 mm, D405 0.1 mm
        self.last_frame_t = time.time()
        self.tile = np.zeros((TILE[1] + INFO_H, TILE[0], 3), np.uint8)
        self.img = None
        self.pose = None            # dict, see process()
        self.T = None               # 4x4 board -> camera
        self.T_t = 0.0

    def poll(self):
        fs = self.pipe.poll_for_frames()
        if not fs:
            if time.time() - self.last_frame_t > 6:                 # stalled stream: reopen
                print(f"{self.role}: no frames, reopening", flush=True)
                self.pipe.stop()
                time.sleep(1.0)
                self.open()
            return False
        self.last_frame_t = time.time()
        fs = self.align.process(fs)
        cf, df = fs.get_color_frame(), fs.get_depth_frame()
        if not cf or not df:
            return False
        img = np.asanyarray(cf.get_data()).copy()
        depth = np.asanyarray(df.get_data()).astype(np.float32) * self.scale
        self.process(img, depth)
        return True

    def process(self, img, depth):
        self.img = img.copy()
        cc, ci, mc, mi = DETECTOR.detectBoard(img)
        lines, self.pose = [], None
        if mi is not None:
            cv2.aruco.drawDetectedMarkers(img, mc, mi)
        if ci is not None and len(ci) >= 6:
            for (u, v), k in zip(np.asarray(cc).reshape(-1, 2), np.asarray(ci).reshape(-1)):   # (OpenCV 5's
                cv2.circle(img, (int(u), int(v)), 5, (0, 255, 0), 2)                         # drawDetectedCornersCharuco
                cv2.putText(img, str(int(k)), (int(u) + 6, int(v) - 6), 0, 0.45, (0, 255, 0), 1)  # asserts on shapes)
            obj, imgp = BOARD.matchImagePoints(cc, ci)
            ok, rvec, tvec = cv2.solvePnP(obj, imgp, self.K, self.D)
            if ok:
                proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)
                err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, -1))))
                cv2.drawFrameAxes(img, self.K, self.D, rvec, tvec, 2 * SQUARE, 3)
                R, _ = cv2.Rodrigues(rvec)
                self.T = np.eye(4)
                self.T[:3, :3], self.T[:3, 3] = R, tvec.ravel()
                self.T_t = time.time()
                tilt = float(np.degrees(np.arccos(abs(R[2, 2]))))     # board normal vs optical axis
                sq, n = square_from_depth(cc, ci, depth, self.K)
                self.pose = dict(rvec=rvec.ravel().tolist(), tvec_m=tvec.ravel().tolist(), reproj_px=err,
                                 corners=int(len(ci)), tilt_deg=tilt, square_from_depth_mm=sq * 1000)
                t = tvec.ravel() * 100
                lines = [f"corners {len(ci)}/24   reproj {err:.2f} px",
                         f"board x {t[0]:+.1f} y {t[1]:+.1f} z {t[2]:.1f} cm",
                         f"tilt {tilt:.1f} deg   square(depth) {sq * 1000:.1f} mm"]
        if not lines:
            lines = [f"board not found (markers: {0 if mi is None else len(mi)})"]
        h, w = img.shape[:2]
        lines = [f"{NAMES[self.role]}  {w}x{h} (full frame)"] + lines
        text = np.zeros((INFO_H, TILE[0], 3), np.uint8)          # info under the image, not over it
        for k, s in enumerate(lines):
            cv2.putText(text, s, (10, 28 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255) if k == 0 else (255, 255, 255), 2)
        self.tile = np.vstack([cv2.resize(img, TILE), text])


def square_from_depth(corners, ids, depth_m, K):
    """Median 3D distance between horizontally/vertically adjacent ChArUco corners (m), from depth only."""
    nx = SQUARES[0] - 1
    pts = {}
    for (u, v), k in zip(np.asarray(corners).reshape(-1, 2), np.asarray(ids).reshape(-1)):
        x, y = int(round(u)), int(round(v))
        win = depth_m[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
        win = win[win > 0]
        if win.size:
            z = float(np.median(win))
            pts[int(k)] = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    d = [np.linalg.norm(pts[k] - pts[k + 1]) for k in pts if k + 1 in pts and (k + 1) % nx]
    d += [np.linalg.norm(pts[k] - pts[k + nx]) for k in pts if k + nx in pts]
    return (float(np.median(d)), len(d)) if d else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="+", default=["top", "wrist"], choices=["top", "wrist"])
    args = ap.parse_args()
    connected = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
    cams = []
    for role in args.cams:
        if str(CFG[role]["serial"]) in connected:
            cams.append(Cam(role))
        else:
            print(f"{NAMES[role]} ({CFG[role]['serial']}) not connected — skipped", flush=True)
    if not cams:
        raise SystemExit("no camera")
    out = HERE / "data" / "charuco"
    win = "ChArUco " + " + ".join(NAMES[c.role] for c in cams)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TILE[0] * len(cams), TILE[1] + INFO_H + 60)
    rel = None
    while True:
        for c in cams:
            c.poll()
        bar = np.zeros((60, TILE[0] * len(cams), 3), np.uint8)
        by = {c.role: c for c in cams}
        if "top" in by and "wrist" in by and by["top"].T is not None and by["wrist"].T is not None \
                and abs(by["top"].T_t - by["wrist"].T_t) < FRESH_S and time.time() - by["top"].T_t < FRESH_S:
            rel = by["top"].T @ np.linalg.inv(by["wrist"].T)       # wrist camera -> top camera
            t = rel[:3, 3] * 100
            msg = (f"both see the board: D405 in D435i frame  x {t[0]:+.1f} y {t[1]:+.1f} z {t[2]:+.1f} cm"
                   f"  (dist {np.linalg.norm(t):.1f} cm)")
            col = (0, 255, 0)
        else:
            msg, col = "camera-to-camera: needs the board in BOTH views at the same time", (0, 165, 255)
        cv2.putText(bar, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.imshow(win, np.vstack([np.hstack([c.tile for c in cams]), bar]))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            out.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            rec = {}
            for c in cams:
                if c.img is not None:
                    cv2.imwrite(str(out / f"{stamp}_{c.role}.png"), c.img)
                rec[c.role] = dict(pose=c.pose, K=c.K.tolist(), dist=c.D.tolist())
            rec["T_top_from_wrist"] = None if rel is None else rel.tolist()
            (out / f"{stamp}.json").write_text(json.dumps(rec, indent=1))
            print("saved", out / stamp, flush=True)
    for c in cams:
        c.pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

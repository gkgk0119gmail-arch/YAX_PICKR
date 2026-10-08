"""
Eye-in-hand calibration for a RealSense camera mounted on the robot flange.

Unknown to solve:    T_flange_camera   (constant rigid transform)

Workflow:
  1. Fix a ChArUco board somewhere in the workspace (does NOT need to be
     in a known pose). Keep it absolutely still for the whole session.
  2. This script opens a live preview window. Move the arm by hand or via
     teach pendant to a pose where the camera sees the board.
  3. Press ENTER to record a sample. The script:
        - averages the camera->board pose over ~30 frames
        - reads the base->flange pose from the robot
     Status text shows how many corners are detected.
  4. Repeat for 10-20 poses. IMPORTANT: vary the orientation of the
     camera around at least 3 distinct rotation axes between poses.
     Pure translations make the rotation under-determined.
  5. Press 'c' to compute the calibration (or 'q' to quit without solving).
  6. The script runs cv2.calibrateHandEye with several methods, prints
     the result, validates it (T_base_board should be the same across
     all samples), and saves T_flange_camera to flange_camera_transform.npz.

Keys:
    ENTER / SPACE   capture a sample
    c               compute calibration and exit
    r               reset (discard captured samples)
    u               undo last sample
    q               quit without solving

Keyboard jog (teleoperation)
----------------------------
    Instead of a teach pendant, jog the flange in base-frame Cartesian space
    by a discrete step each press (blocking IK + MoveJ). Works in the live
    preview and over the --web viewer.

    w / s          +X / -X (mm)        t / y          +RX / -RX (deg)
    a / d          +Y / -Y (mm)        h / n          +RY / -RY (deg)
    e / f          +Z / -Z (mm)        j / m          +RZ / -RZ (deg)
    - / =          halve / double the XYZ step
    , / .          halve / double the RPY step

    Defaults: --xyz-step 10 (mm), --rpy-step 5 (deg), --jog-vel 20 (%).

Remote / SSH use
----------------
    Pass --web to serve the window in a browser instead of opening a native
    OpenCV window (no X11 forwarding needed). Forward the port over SSH
    (ssh -L 7000:localhost:7000 <host>) and open http://localhost:7000/.
    The same keys work in the browser (click the page first). Options:
    --web-host, --web-port (default 7000), --web-quality (JPEG, default 80).
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

# Make Fairino SDK importable.
_FAIRINO_SDK_PATH = "/home/pickr0001/fairino-python-sdk/linux"
if _FAIRINO_SDK_PATH not in sys.path:
    sys.path.insert(0, _FAIRINO_SDK_PATH)
from fairino import Robot  # noqa: E402


#### PARAMETERS

# Robot
ROBOT_IP = "192.168.58.2"

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ChArUco board settings (must match the printed board)
ARUCO_DICT = cv2.aruco.DICT_4X4_100
BOARD_SQUARES_X = 7
BOARD_SQUARES_Y = 5
BOARD_SQUARE_LENGTH = 0.04  # meters
BOARD_MARKER_LENGTH = 0.03  # meters

# Per-sample averaging
FRAMES_PER_SAMPLE = 30
# A 7x5 ChArUco board has up to 24 inner corners. Requiring only a handful
# yields unreliable PnP poses (noisy/flipped rotation) that poison the
# hand-eye solve, so demand a healthy fraction of the board be visible.
MIN_CORNERS_FOR_POSE = 12

# Reject captures where the board is too far / too close. Planar PnP gets
# noisy and flip-prone far away, and the board leaves the frame up close;
# 0.3-0.5 m is the sweet spot. Set BOARD_DIST_MAX_M = 0 to disable the check.
BOARD_DIST_MIN_M = 0.30
BOARD_DIST_MAX_M = 0.50

# Reject a captured sample whose board pose disagrees with the consensus by
# more than this, then re-solve. Set to 0 to disable outlier rejection.
OUTLIER_T_MM = 8.0      # translation deviation of T_base_board from the mean
OUTLIER_R_DEG = 1.0     # rotation deviation of T_base_board from the mean

# Fairino pose [x,y,z,rx,ry,rz] uses degrees for rotation and millimetres
# for translation. The rotation convention is XYZ fixed-axis (extrinsic),
# i.e. R = Rz(rz) @ Ry(ry) @ Rx(rx). Scipy spells that 'XYZ' (uppercase).
# If your reprojection check below gives huge errors, try 'xyz' (lowercase
# = intrinsic) instead -- some controller firmwares disagree on this.
EULER_CONVENTION = "xyz"

OUTPUT_FILE = "flange_camera_transform.npz"

WINDOW_CALIB = "Eye-in-Hand Calibration"


# ---------------------------------------------------------------------------
# Keyboard jog (teleoperation)
#
# Instead of reaching for a teach pendant, these keys jog the flange in its
# own (base-frame) Cartesian frame by a discrete step each press. The motion
# is a blocking IK + MoveJ (mirrors refine_eye_in_hand.py / keyboard_ik_control
# .py). Jog works in the live preview and through the --web viewer (the
# browser forwards each letter as its char code).
#
#   Translation (mm):   w/s = +X/-X   a/d = +Y/-Y   e/f = +Z/-Z
#   Rotation (deg):     t/y = +RX/-RX h/n = +RY/-RY j/m = +RZ/-RZ
#   Step adjust:        -/= halve/double XYZ step    ,/. halve/double RPY step
#
# Map: key -> (unit_x, unit_y, unit_z) signed direction, scaled by the step.
JOG_TRANS_KEYS = {
    ord('w'): (1.0, 0.0, 0.0),
    ord('s'): (-1.0, 0.0, 0.0),
    ord('a'): (0.0, 1.0, 0.0),
    ord('d'): (0.0, -1.0, 0.0),
    ord('e'): (0.0, 0.0, 1.0),
    ord('f'): (0.0, 0.0, -1.0),
}
JOG_ROT_KEYS = {
    ord('t'): (1.0, 0.0, 0.0),
    ord('y'): (-1.0, 0.0, 0.0),
    ord('h'): (0.0, 1.0, 0.0),
    ord('n'): (0.0, -1.0, 0.0),
    ord('j'): (0.0, 0.0, 1.0),
    ord('m'): (0.0, 0.0, -1.0),
}
JOG_XYZ_STEP_MM = 10.0      # default per-press translation
JOG_RPY_STEP_DEG = 5.0      # default per-press rotation
JOG_VEL_PCT = 20.0          # MoveJ velocity (% of max); keep modest for jogging


#### DISPLAY BACKENDS
#
# The tool is interactive: it shows an OpenCV window and reads single key
# presses via cv2.waitKey. That requires a local display (or X11 forwarding),
# which is painful over SSH. To support headless / remote use we abstract
# "show an image in a named window" and "wait for a key" behind a tiny
# interface with two implementations:
#
#   CvDisplay  - the original cv2.imshow / cv2.waitKey behaviour.
#   WebDisplay - serves every window as an MJPEG stream in the browser and
#                forwards key presses from the page back to the program,
#                so it works over plain SSH (just forward the port).
#
# Both wait_key() return a raw int matching cv2's convention (-1 when no key
# was pressed), so callers can keep using `key = display.wait_key(t) & 0xFF`.

class CvDisplay:
    """Thin wrapper around the native OpenCV highgui windows."""

    def named_window(self, name, width=None, height=None):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if width and height:
            cv2.resizeWindow(name, int(width), int(height))

    def imshow(self, name, img):
        cv2.imshow(name, img)

    def wait_key(self, timeout_ms):
        return cv2.waitKey(int(timeout_ms))

    def destroy_window(self, name):
        cv2.destroyWindow(name)

    def destroy_all(self):
        cv2.destroyAllWindows()


class WebDisplay:
    """Serve named windows as MJPEG streams and collect key presses.

    Open http://<host>:<port>/ in a browser. Every window shown via
    imshow() appears as a live image; key presses on the page are queued
    and returned by wait_key(), mirroring cv2.waitKey semantics.
    """

    def __init__(self, host="0.0.0.0", port=7000, quality=80):
        self._cond = threading.Condition()
        self._frames = {}      # name -> encoded jpeg bytes
        self._versions = {}    # name -> monotonically increasing int
        self._order = []       # window creation order (for stable layout)
        self._keys = queue.Queue()
        self._quality = int(quality)
        self.host = host
        self.port = port
        self._server = None
        self._thread = None
        self._start()

    # --- public display API ------------------------------------------------
    def _placeholder(self, name, width=None, height=None):
        """A 'waiting for frames' JPEG so the browser <img> renders something
        immediately, even before the first real imshow()."""
        w = int(width) if width else 640
        h = int(height) if height else 360
        img = np.full((h, w, 3), 24, dtype=np.uint8)
        cv2.putText(img, "waiting for frames...", (16, max(28, h // 2 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(img, name, (16, max(52, h // 2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        return buf.tobytes() if ok else None

    def named_window(self, name, width=None, height=None):
        with self._cond:
            if name not in self._versions:
                self._order.append(name)
                # Seed a placeholder frame (version 1) so the stream has data
                # to send right away; the first real imshow() bumps it to 2.
                self._frames[name] = self._placeholder(name, width, height)
                self._versions[name] = 1
                self._cond.notify_all()

    def imshow(self, name, img):
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            return
        data = buf.tobytes()
        with self._cond:
            if name not in self._versions:
                self._order.append(name)
            self._frames[name] = data
            self._versions[name] = self._versions.get(name, 0) + 1
            self._cond.notify_all()

    def wait_key(self, timeout_ms):
        try:
            k = self._keys.get(timeout=max(0.0, float(timeout_ms) / 1000.0))
            return int(k)
        except queue.Empty:
            return -1

    def destroy_window(self, name):
        with self._cond:
            self._frames.pop(name, None)
            self._versions.pop(name, None)
            if name in self._order:
                self._order.remove(name)
            self._cond.notify_all()

    def destroy_all(self):
        with self._cond:
            self._frames.clear()
            self._versions.clear()
            self._order.clear()
            self._cond.notify_all()
        try:
            if self._server is not None:
                self._server.shutdown()
        except Exception:
            pass

    # --- internals ---------------------------------------------------------
    def push_key(self, k):
        self._keys.put(int(k))

    def _start(self):
        disp = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                disp._handle_get(self)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _handle_get(self, h):
        parsed = urlparse(h.path)
        path = parsed.path

        if path == "/":
            body = _WEB_PAGE.encode("utf-8")
            h.send_response(200)
            h.send_header("Content-Type", "text/html; charset=utf-8")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return

        if path == "/layout":
            with self._cond:
                windows = list(self._order)
            body = json.dumps({"windows": windows}).encode("utf-8")
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Cache-Control", "no-cache")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return

        if path.startswith("/key"):
            qs = parse_qs(parsed.query)
            k = qs.get("k", [None])[0]
            if k is not None:
                try:
                    self.push_key(int(k))
                except (TypeError, ValueError):
                    pass
            h.send_response(204)
            h.end_headers()
            return

        if path.startswith("/stream/"):
            name = unquote(path[len("/stream/"):])
            h.send_response(200)
            h.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame")
            h.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            h.send_header("Pragma", "no-cache")
            h.send_header("Connection", "close")
            h.end_headers()
            last_ver = -1
            try:
                while True:
                    with self._cond:
                        ver = self._versions.get(name)
                        if ver is None:
                            break
                        if ver == last_ver:
                            self._cond.wait(timeout=1.0)
                            ver = self._versions.get(name)
                            if ver is None:
                                break
                            if ver == last_ver:
                                continue
                        frame = self._frames.get(name)
                        last_ver = ver
                    if frame is None:
                        continue
                    h.wfile.write(b"--frame\r\n")
                    h.wfile.write(b"Content-Type: image/jpeg\r\n")
                    h.wfile.write(
                        ("Content-Length: %d\r\n\r\n" % len(frame)).encode())
                    h.wfile.write(frame)
                    h.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        h.send_response(404)
        h.end_headers()


_WEB_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eye-in-hand calibration (web)</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #14161a; color: #e6e6e6;
         font-family: ui-sans-serif, system-ui, sans-serif; }
  header { padding: 10px 16px; background: #1d2026;
           border-bottom: 1px solid #2c2f36; position: sticky; top: 0;
           display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .pill { font-size: 12px; padding: 3px 9px; border-radius: 999px;
          background: #2c2f36; }
  .pill b { color: #7ee787; }
  #key { font-family: ui-monospace, monospace; }
  .help { font-size: 12px; color: #9aa0a6; }
  main { display: flex; flex-wrap: wrap; gap: 16px; padding: 16px;
         align-items: flex-start; }
  figure { margin: 0; background: #1d2026; border: 1px solid #2c2f36;
           border-radius: 8px; overflow: hidden; }
  figcaption { font-size: 12px; padding: 6px 10px; background: #23262d;
               border-bottom: 1px solid #2c2f36; }
  img { display: block; max-width: 100%; }
  kbd { background: #2c2f36; border: 1px solid #3a3e46; border-radius: 4px;
        padding: 1px 6px; font-family: ui-monospace, monospace; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Eye-in-hand calibration &mdash; remote view</h1>
  <span class="pill">last key: <b id="key">&mdash;</b></span>
  <span class="help">Click the page first, then use the keys below.
  <kbd>Enter</kbd>/<kbd>Space</kbd> capture &middot;
  <kbd>c</kbd> compute &middot; <kbd>u</kbd> undo &middot;
  <kbd>r</kbd> reset &middot; <kbd>q</kbd> quit
  <br>jog: <kbd>w</kbd>/<kbd>s</kbd> <kbd>a</kbd>/<kbd>d</kbd> <kbd>e</kbd>/<kbd>f</kbd> = XYZ mm &middot;
  <kbd>t</kbd>/<kbd>y</kbd> <kbd>h</kbd>/<kbd>n</kbd> <kbd>j</kbd>/<kbd>m</kbd> = RPY deg &middot;
  <kbd>-</kbd>/<kbd>=</kbd> XYZ step &middot; <kbd>,</kbd>/<kbd>.</kbd> RPY step</span>
</header>
<main id="windows"></main>
<script>
const windowsEl = document.getElementById('windows');
const keyEl = document.getElementById('key');
const shown = new Map();

function syncLayout() {
  fetch('/layout').then(r => r.json()).then(data => {
    const names = data.windows || [];
    for (const name of names) {
      if (shown.has(name)) continue;
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = name;
      const img = document.createElement('img');
      const streamUrl = () =>
        '/stream/' + encodeURIComponent(name) + '?t=' + Date.now();
      img.src = streamUrl();
      img.alt = name;
      // MJPEG connections can drop (server restart, network blip); browsers
      // don't auto-retry, so reconnect after a short delay.
      img.onerror = () => { setTimeout(() => { img.src = streamUrl(); }, 1000); };
      fig.appendChild(cap);
      fig.appendChild(img);
      windowsEl.appendChild(fig);
      shown.set(name, fig);
    }
    for (const [name, fig] of shown) {
      if (!names.includes(name)) {
        fig.remove();
        shown.delete(name);
      }
    }
  }).catch(() => {});
}
setInterval(syncLayout, 1000);
syncLayout();

function sendKey(code, label) {
  keyEl.textContent = label;
  fetch('/key?k=' + code).catch(() => {});
}

document.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  let code = null, label = e.key;
  if (e.key === ' ') { code = 32; label = 'Space'; }
  else if (e.key === 'Enter') { code = 13; label = 'Enter'; }
  else if (e.key.length === 1) { code = e.key.toLowerCase().charCodeAt(0); }
  if (code !== null) {
    e.preventDefault();
    sendKey(code, label);
  }
});
</script>
</body>
</html>
"""


#### CHARUCO + REALSENSE HELPERS

def get_camera_intrinsics(profile):
    color_stream = profile.get_stream(rs.stream.color)
    video_profile = color_stream.as_video_stream_profile()
    intr = video_profile.get_intrinsics()
    K = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    D = np.array(intr.coeffs[:5], dtype=np.float32).reshape(1, 5)
    return K, D


def create_charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        BOARD_SQUARE_LENGTH,
        BOARD_MARKER_LENGTH,
        dictionary,
    )
    return board


def create_charuco_detector(board, K, D):
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = K
    charuco_params.distCoeffs = D

    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


def estimate_board_pose(charuco_corners, charuco_ids, board, K, D):
    charuco_corners = np.ascontiguousarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    charuco_ids = np.ascontiguousarray(charuco_ids, dtype=np.int32).reshape(-1, 1)

    obj_points = board.getChessboardCorners()
    obj_points = obj_points[charuco_ids.reshape(-1)]

    # ChArUco corners are coplanar (z=0), so use the IPPE planar solver. It is
    # far less prone to the rotation flips / noise that the default iterative
    # solver produces on far or oblique views (the main cause of bad samples).
    # IPPE needs >= 4 points; fall back to the iterative solver otherwise.
    if len(obj_points) >= 4:
        flags = cv2.SOLVEPNP_IPPE
    else:
        flags = cv2.SOLVEPNP_ITERATIVE
    success, rvec, tvec = cv2.solvePnP(
        obj_points, charuco_corners, K, D, flags=flags)
    return success, rvec, tvec


def rvec_tvec_to_matrix(rvec, tvec):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.reshape(-1)
    return T


def average_rotations_svd(rotation_matrices):
    R_sum = np.sum(rotation_matrices, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


#### ROBOT POSE HELPERS

def fairino_pose_to_matrix(pose):
    """
    Fairino pose: [x, y, z, rx, ry, rz]  (mm, deg).
    Returns 4x4 transform in metres.
    """
    pose = np.asarray(pose, dtype=np.float64)
    T = np.eye(4)
    T[:3, 3] = pose[:3] / 1000.0
    T[:3, :3] = R.from_euler(EULER_CONVENTION, pose[3:], degrees=True).as_matrix()
    return T


def matrix_to_fairino_pose(T):
    """Inverse of fairino_pose_to_matrix, mainly for sanity printing."""
    t = T[:3, 3] * 1000.0
    rpy = R.from_matrix(T[:3, :3]).as_euler(EULER_CONVENTION, degrees=True)
    return np.concatenate([t, rpy])


def get_flange_pose(robot):
    """Returns 4x4 T_base_flange in metres, or None on error."""
    err, pose = robot.GetActualToolFlangePose(flag=1)
    if err != 0 or pose is None:
        print(f"  [robot] GetActualToolFlangePose error={err}")
        return None
    return fairino_pose_to_matrix(pose), pose


#### KEYBOARD JOG (TELEOPERATION)

def setup_robot_for_motion(robot):
    """Put the controller in a state where IK + MoveJ jogging works.

    Clears errors, switches to manual mode, enables power, and refuses to run
    if a safety stop is latched. Called once in main() before the loop.
    """
    robot.ResetAllError()
    robot.Mode(1)            # 1 = Manual
    robot.RobotEnable(1)
    code = robot.GetSafetyCode()
    if code != 0:
        raise RuntimeError(
            f"Safety stop active (code {code}). Release it and retry.")


def jog_flange(robot, dx_mm, dy_mm, dz_mm, drx, dry, drz, vel):
    """Discrete Cartesian jog of the flange via IK + blocking MoveJ.

    Reads the current flange pose [x,y,z (mm), rx,ry,rz (deg)], adds the
    requested deltas, solves IK (config=-1 -> nearest solution) and runs a
    blocking MoveJ. Raises RuntimeError on any controller error so callers
    can surface it in the status line. Returns the commanded target pose.
    """
    err, pose = robot.GetActualToolFlangePose(flag=1)
    if err != 0 or pose is None:
        raise RuntimeError(f"GetActualToolFlangePose failed: {err}")
    target = [float(v) for v in pose]
    target[0] += dx_mm
    target[1] += dy_mm
    target[2] += dz_mm
    target[3] += drx
    target[4] += dry
    target[5] += drz

    err, joints = robot.GetInverseKin(0, desc_pos=target, config=-1)
    if err != 0:
        raise RuntimeError(f"GetInverseKin failed: {err}")
    err = robot.MoveJ(joints, tool=0, user=0, vel=vel, blendT=-1.0)
    if err != 0:
        e2, codes = robot.GetRobotErrorCode()
        raise RuntimeError(f"MoveJ failed: {err}, controller codes: {codes}")
    return target


def handle_jog_key(robot, key, jog):
    """If `key` is a jog / step-adjust key, act on it and return a status str.

    `jog` is a mutable dict: {"xyz": mm, "rpy": deg, "vel": pct}. Translation
    and rotation keys command the robot (and may raise on controller error);
    the '-/=' and ',/.' keys only mutate the step sizes. Returns None when the
    key isn't a jog key so the caller can fall through to its own handling.
    """
    if key in JOG_TRANS_KEYS:
        ux, uy, uz = JOG_TRANS_KEYS[key]
        d = jog["xyz"]
        jog_flange(robot, ux * d, uy * d, uz * d, 0.0, 0.0, 0.0, jog["vel"])
        return f"jog dT=({ux * d:+.1f},{uy * d:+.1f},{uz * d:+.1f}) mm"
    if key in JOG_ROT_KEYS:
        ux, uy, uz = JOG_ROT_KEYS[key]
        d = jog["rpy"]
        jog_flange(robot, 0.0, 0.0, 0.0, ux * d, uy * d, uz * d, jog["vel"])
        return f"jog dR=({ux * d:+.1f},{uy * d:+.1f},{uz * d:+.1f}) deg"
    if key == ord('-'):
        jog["xyz"] = max(0.1, jog["xyz"] / 2.0)
        return f"XYZ step = {jog['xyz']:.2f} mm"
    if key in (ord('='), ord('+')):
        jog["xyz"] = min(200.0, jog["xyz"] * 2.0)
        return f"XYZ step = {jog['xyz']:.2f} mm"
    if key == ord(','):
        jog["rpy"] = max(0.1, jog["rpy"] / 2.0)
        return f"RPY step = {jog['rpy']:.2f} deg"
    if key == ord('.'):
        jog["rpy"] = min(90.0, jog["rpy"] * 2.0)
        return f"RPY step = {jog['rpy']:.2f} deg"
    return None


#### SAMPLE CAPTURE

def capture_sample(pipeline, detector, board, K, D, robot):
    """
    Average ChArUco board pose over FRAMES_PER_SAMPLE frames, read robot
    flange pose, return (T_base_flange, T_camera_board, raw_pose) or None.
    """
    rvecs, tvecs = [], []
    n_detect = 0

    for _ in range(FRAMES_PER_SAMPLE):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image_bgr = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < MIN_CORNERS_FOR_POSE:
            continue
        ok, rvec, tvec = estimate_board_pose(ch_corners, ch_ids, board, K, D)
        if not ok:
            continue
        rvecs.append(rvec.reshape(-1))
        tvecs.append(tvec.reshape(-1))
        n_detect += 1

    if n_detect < FRAMES_PER_SAMPLE // 2:
        print(f"  [capture] only {n_detect}/{FRAMES_PER_SAMPLE} frames detected the board - rejected")
        return None

    # Average rvecs through their rotation matrices, average tvecs directly.
    Rs = [cv2.Rodrigues(rv)[0] for rv in rvecs]
    R_avg = average_rotations_svd(Rs)
    rvec_avg, _ = cv2.Rodrigues(R_avg)
    tvec_avg = np.mean(tvecs, axis=0)

    # Distance gate: reject board poses outside the reliable depth range.
    dist_m = float(np.linalg.norm(tvec_avg))
    if BOARD_DIST_MAX_M > 0 and not (BOARD_DIST_MIN_M <= dist_m <= BOARD_DIST_MAX_M):
        print(f"  [capture] board at {dist_m:.3f} m is outside "
              f"[{BOARD_DIST_MIN_M:.2f}, {BOARD_DIST_MAX_M:.2f}] m - rejected")
        return None

    T_camera_board = rvec_tvec_to_matrix(rvec_avg, tvec_avg)

    # Robot pose
    robot_result = get_flange_pose(robot)
    if robot_result is None:
        return None
    T_base_flange, raw_pose = robot_result

    # Print quick diagnostics
    t_b = T_base_flange[:3, 3]
    t_c = T_camera_board[:3, 3]
    print(f"  [capture] OK  flange(m)=({t_b[0]:+.3f},{t_b[1]:+.3f},{t_b[2]:+.3f})"
          f"  board_in_cam(m)=({t_c[0]:+.3f},{t_c[1]:+.3f},{t_c[2]:+.3f})"
          f"  ({n_detect} frames)")

    return T_base_flange, T_camera_board, raw_pose


#### HAND-EYE SOLVER + VALIDATION

HANDEYE_METHODS = [
    (cv2.CALIB_HAND_EYE_TSAI,       "Tsai"),
    (cv2.CALIB_HAND_EYE_PARK,       "Park"),
    (cv2.CALIB_HAND_EYE_HORAUD,     "Horaud"),
    (cv2.CALIB_HAND_EYE_ANDREFF,    "Andreff"),
    (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
]


def solve_hand_eye(samples):
    """
    samples: list of (T_base_flange, T_camera_board, raw_pose)
    Returns dict {method_name: (T_flange_camera, stats)}.
    """
    R_g2b = [s[0][:3, :3] for s in samples]
    t_g2b = [s[0][:3, 3]  for s in samples]
    R_t2c = [s[1][:3, :3] for s in samples]
    t_t2c = [s[1][:3, 3]  for s in samples]

    results = {}
    for method, name in HANDEYE_METHODS:
        try:
            Rcg, tcg = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
        except cv2.error as e:
            print(f"[{name}] solver failed: {e}")
            continue

        T_fc = np.eye(4)
        T_fc[:3, :3] = Rcg
        T_fc[:3, 3]  = tcg.reshape(-1)

        stats = validate_calibration(T_fc, samples)
        results[name] = (T_fc, stats)

        print(f"[{name}]  T_base_board std:  "
              f"trans={stats['t_std_mm']:.2f}mm   rot={stats['r_std_deg']:.3f}deg")
    return results


def validate_calibration(T_flange_camera, samples):
    """
    For each sample, compute T_base_board = T_base_flange @ T_flange_camera @ T_camera_board.
    A correct calibration makes these identical across all samples (board didn't move).
    """
    T_base_boards = []
    for T_bf, T_cb, _ in samples:
        T_bb = T_bf @ T_flange_camera @ T_cb
        T_base_boards.append(T_bb)
    T_base_boards = np.stack(T_base_boards, axis=0)

    # Translation spread
    centroids = T_base_boards[:, :3, 3]
    t_mean = centroids.mean(axis=0)
    t_std_mm = float(np.linalg.norm(centroids - t_mean, axis=1).std() * 1000.0)
    t_max_mm = float(np.linalg.norm(centroids - t_mean, axis=1).max() * 1000.0)

    # Rotation spread (angle from mean)
    Rs = T_base_boards[:, :3, :3]
    R_mean = average_rotations_svd([r for r in Rs])
    angles = []
    for Ri in Rs:
        Rd = R_mean.T @ Ri
        cos_a = np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    angles = np.asarray(angles)

    return {
        "t_std_mm":  t_std_mm,
        "t_max_mm":  t_max_mm,
        "r_std_deg": float(angles.std()),
        "r_max_deg": float(angles.max()),
        "t_base_board_mean":  t_mean,
        "R_base_board_mean":  R_mean,
    }


def per_sample_deviation(T_flange_camera, samples):
    """Per-sample deviation of T_base_board from the consensus across samples.

    Returns (t_dev_mm, r_dev_deg) arrays, one entry per sample: how far that
    sample's recovered board pose sits from the mean board pose. A good sample
    is near zero; an outlier (bad PnP / moved board) sticks out.
    """
    T_base_boards = np.stack(
        [T_bf @ T_flange_camera @ T_cb for T_bf, T_cb, _ in samples], axis=0)
    centroids = T_base_boards[:, :3, 3]
    t_mean = centroids.mean(axis=0)
    t_dev_mm = np.linalg.norm(centroids - t_mean, axis=1) * 1000.0

    Rs = T_base_boards[:, :3, :3]
    R_mean = average_rotations_svd([r for r in Rs])
    r_dev_deg = np.array([
        np.degrees(np.arccos(np.clip((np.trace(R_mean.T @ Ri) - 1.0) / 2.0,
                                     -1.0, 1.0)))
        for Ri in Rs])
    return t_dev_mm, r_dev_deg


def solve_hand_eye_robust(samples, t_thresh_mm=OUTLIER_T_MM,
                          r_thresh_deg=OUTLIER_R_DEG, min_samples=8):
    """Solve hand-eye, then iteratively drop the single worst outlier sample
    and re-solve until every remaining sample agrees with the consensus
    (within the thresholds) or we hit `min_samples`.

    A handful of bad camera->board PnP poses (board too far / oblique / few
    corners) otherwise dominate the least-squares solve. Returns:
        results        - {method: (T_fc, stats)} for the final inlier set
        best_name      - method with the smallest combined spread
        inliers        - the kept subset of `samples`
        dropped        - list of (original_index, t_dev_mm, r_dev_deg)
    """
    def score(stats):
        return stats["t_std_mm"] + 10.0 * stats["r_std_deg"]

    idx = list(range(len(samples)))
    dropped = []

    while True:
        cur = [samples[i] for i in idx]
        results = solve_hand_eye(cur)
        if not results:
            return results, None, cur, dropped
        best_name = min(results, key=lambda n: score(results[n][1]))
        T_fc_best = results[best_name][0]

        if t_thresh_mm <= 0 and r_thresh_deg <= 0:
            return results, best_name, cur, dropped
        if len(idx) <= min_samples:
            return results, best_name, cur, dropped

        t_dev, r_dev = per_sample_deviation(T_fc_best, cur)
        # Worst sample by whichever threshold it violates most.
        viol = np.maximum(t_dev / max(t_thresh_mm, 1e-9),
                          r_dev / max(r_thresh_deg, 1e-9))
        k = int(np.argmax(viol))
        if viol[k] <= 1.0:
            return results, best_name, cur, dropped  # all inliers

        orig = idx.pop(k)
        dropped.append((orig, float(t_dev[k]), float(r_dev[k])))
        print(f"  [outlier] dropped sample #{orig}  "
              f"t_dev={t_dev[k]:.1f}mm  r_dev={r_dev[k]:.2f}deg  "
              f"({len(idx)} left)")


#### MAIN LOOP

def draw_overlay(image, status_msg, status_color, n_samples, jog_status=""):
    H = image.shape[0]
    cv2.putText(image, status_msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(image, f"Captured: {n_samples}  (recommend >= 10)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if jog_status:
        cv2.putText(image, jog_status[:60], (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    cv2.putText(image,
                "jog: w/s a/d e/f = XYZmm   t/y h/n j/m = RPYdeg   -/= ,/. step",
                (10, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.putText(image, "ENTER=capture  c=compute  u=undo  r=reset  q=quit",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default=ROBOT_IP)
    ap.add_argument("--xyz-step", type=float, default=JOG_XYZ_STEP_MM,
                    help="Keyboard jog translation step in mm (w/s/a/d/e/f).")
    ap.add_argument("--rpy-step", type=float, default=JOG_RPY_STEP_DEG,
                    help="Keyboard jog rotation step in deg (t/y/h/n/j/m).")
    ap.add_argument("--jog-vel", type=float, default=JOG_VEL_PCT,
                    help="MoveJ velocity (%% of max) used for keyboard jogs.")
    ap.add_argument("--web", action="store_true",
                    help="Serve the window in a browser instead of opening a "
                         "native OpenCV window. Useful over SSH: forward the "
                         "port and open http://localhost:<port>/.")
    ap.add_argument("--web-host", default="0.0.0.0",
                    help="Interface the web viewer binds to (default 0.0.0.0).")
    ap.add_argument("--web-port", type=int, default=7000,
                    help="Port for the web viewer (default 7000).")
    ap.add_argument("--web-quality", type=int, default=80,
                    help="JPEG quality 1-100 for the web stream (default 80).")
    args = ap.parse_args()

    print("=" * 70)
    print("Eye-in-hand calibration  (camera mounted on flange)")
    print("=" * 70)

    # Connect to robot
    print(f"\nConnecting to robot at {args.robot_ip} ...")
    robot = Robot.RPC(args.robot_ip)
    time.sleep(1.0)
    err, pose = robot.GetActualToolFlangePose(flag=1)
    if err != 0:
        print(f"ERROR: cannot read flange pose (err={err}). Aborting.")
        return
    print(f"Robot OK. Initial flange pose: {[f'{v:.2f}' for v in pose]}")

    # Enable manual mode + power so the keyboard jog (IK + MoveJ) can move the
    # arm. If a safety stop is latched this raises and we abort early.
    try:
        setup_robot_for_motion(robot)
        print(f"[jog] enabled. xyz-step={args.xyz_step} mm, "
              f"rpy-step={args.rpy_step} deg, vel={args.jog_vel}%")
    except Exception as e:
        print(f"ERROR: could not enable robot for jogging: {e}")
        return

    # Mutable jog state; the step-adjust keys mutate it in place.
    jog = {"xyz": float(args.xyz_step),
           "rpy": float(args.rpy_step),
           "vel": float(args.jog_vel)}

    # RealSense
    print("\nStarting RealSense pipeline ...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT,
                         rs.format.bgr8, CAMERA_FPS)
    profile = pipeline.start(config)

    # Warm-up
    for _ in range(int(CAMERA_FPS * 1.5)):
        pipeline.wait_for_frames()

    K, D = get_camera_intrinsics(profile)
    print(f"K =\n{K}")
    print(f"D = {D.reshape(-1)}")

    board = create_charuco_board()
    detector = create_charuco_detector(board, K, D)

    if args.web:
        display = WebDisplay(host=args.web_host, port=args.web_port,
                             quality=args.web_quality)
        shown_host = "localhost" if args.web_host in ("0.0.0.0", "") else args.web_host
        print(f"\n[web] Viewer serving at http://{shown_host}:{args.web_port}/")
        print("[web] Over SSH, forward it, e.g.: "
              f"ssh -L {args.web_port}:localhost:{args.web_port} <host>")
        print("[web] Then open the URL in your local browser. Click the page "
              "before pressing keys.")
    else:
        display = CvDisplay()
    display.named_window(WINDOW_CALIB, CAMERA_WIDTH, CAMERA_HEIGHT)

    samples = []  # list of (T_base_flange, T_camera_board, raw_pose)
    do_compute = False
    jog_status = ""

    print("\nReady. Jog the arm with the keyboard (w/s a/d e/f, t/y h/n j/m) so")
    print("the camera sees the board, then press ENTER to capture.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data()).copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)

            if ch_ids is not None and len(ch_ids) >= MIN_CORNERS_FOR_POSE:
                cv2.aruco.drawDetectedCornersCharuco(image, ch_corners, ch_ids,
                                                    cornerColor=(0, 255, 0))
                ok, rvec, tvec = estimate_board_pose(ch_corners, ch_ids, board, K, D)
                if ok:
                    cv2.drawFrameAxes(image, K, D, rvec, tvec, 0.05)
                    dist_m = float(np.linalg.norm(tvec))
                    in_range = (BOARD_DIST_MAX_M <= 0 or
                                BOARD_DIST_MIN_M <= dist_m <= BOARD_DIST_MAX_M)
                    if in_range:
                        status = (f"Board: {len(ch_ids)} corners @ {dist_m:.2f} m. "
                                  "ENTER to capture.")
                        color = (0, 255, 0)
                    else:
                        where = "too close" if dist_m < BOARD_DIST_MIN_M else "too far"
                        status = (f"Board @ {dist_m:.2f} m {where} "
                                  f"(want {BOARD_DIST_MIN_M:.2f}-{BOARD_DIST_MAX_M:.2f} m).")
                        color = (0, 165, 255)
                else:
                    status = "Board detected but PnP failed."
                    color = (0, 165, 255)
            else:
                n = 0 if ch_ids is None else len(ch_ids)
                status = f"Board NOT detected ({n} corners). Reposition the arm."
                color = (0, 0, 255)

            draw_overlay(image, status, color, len(samples), jog_status)
            display.imshow(WINDOW_CALIB, image)

            key = display.wait_key(1) & 0xFF
            if key == 255:
                continue

            # Keyboard jog: frame the workspace without a teach pendant. Handled
            # before the capture/compute keys; on a jog (or step change) we loop
            # so the next frame re-detects the board from the new pose.
            try:
                jstat = handle_jog_key(robot, key, jog)
            except Exception as e:
                jog_status = f"[jog] {e}"
                print(jog_status)
                continue
            if jstat is not None:
                jog_status = jstat
                continue

            if key in (13, 10):  # Enter
                print(f"\n[sample {len(samples)+1}] capturing ...")
                sample = capture_sample(pipeline, detector, board, K, D, robot)
                if sample is not None:
                    samples.append(sample)
                    print(f"  total captured: {len(samples)}")
            elif key == ord(' '):
                print(f"\n[sample {len(samples)+1}] capturing ...")
                sample = capture_sample(pipeline, detector, board, K, D, robot)
                if sample is not None:
                    samples.append(sample)
                    print(f"  total captured: {len(samples)}")
            elif key == ord('u'):
                if samples:
                    samples.pop()
                    print(f"  undid last sample. total = {len(samples)}")
            elif key == ord('r'):
                samples.clear()
                print("  reset: all samples cleared.")
            elif key == ord('c'):
                do_compute = True
                break
            elif key == ord('q'):
                print("Quitting without solving.")
                break
    finally:
        display.destroy_all()
        pipeline.stop()
        try:
            robot.CloseRPC()
        except Exception:
            pass

    if not do_compute:
        return
    if len(samples) < 3:
        print(f"\nERROR: need at least 3 samples to solve, have {len(samples)}.")
        return

    print("\n" + "=" * 70)
    print(f"Solving hand-eye calibration with {len(samples)} samples ...")
    print("=" * 70)

    results, best_name, samples, dropped = solve_hand_eye_robust(samples)
    if not results or best_name is None:
        print("All solvers failed.")
        return
    if dropped:
        print(f"\n[outlier] removed {len(dropped)} sample(s) before the final "
              f"solve; {len(samples)} inliers remain.")

    T_fc_best, best_stats = results[best_name]

    print("\n" + "=" * 70)
    print(f"BEST method: {best_name}")
    print("=" * 70)
    print(f"\nT_flange_camera (m):\n{T_fc_best}")

    # Pretty representation
    fc_xyz_mm   = T_fc_best[:3, 3] * 1000.0
    fc_eul_deg  = R.from_matrix(T_fc_best[:3, :3]).as_euler(EULER_CONVENTION,
                                                            degrees=True)
    print(f"\nTranslation (mm):  x={fc_xyz_mm[0]:+8.2f}   "
          f"y={fc_xyz_mm[1]:+8.2f}   z={fc_xyz_mm[2]:+8.2f}")
    print(f"Euler '{EULER_CONVENTION}' (deg): "
          f"rx={fc_eul_deg[0]:+8.3f}   ry={fc_eul_deg[1]:+8.3f}   rz={fc_eul_deg[2]:+8.3f}")

    print(f"\nValidation (T_base_board consistency across {len(samples)} samples):")
    print(f"  translation std/max: {best_stats['t_std_mm']:.2f} / "
          f"{best_stats['t_max_mm']:.2f} mm")
    print(f"  rotation    std/max: {best_stats['r_std_deg']:.3f} / "
          f"{best_stats['r_max_deg']:.3f} deg")
    if best_stats["t_max_mm"] > 5.0 or best_stats["r_max_deg"] > 1.0:
        print("  WARNING: residuals are large. Likely causes:")
        print("    - Not enough rotation variety between poses (mostly translations)")
        print("    - Board moved during the session")
        print("    - Wrong EULER_CONVENTION (try 'XYZ' instead of 'xyz')")

    # Save
    np.savez(
        OUTPUT_FILE,
        T_flange_camera=T_fc_best,
        method=best_name,
        K=K, D=D,
        T_base_flange=np.stack([s[0] for s in samples]),
        T_camera_board=np.stack([s[1] for s in samples]),
        raw_flange_poses=np.array([s[2] for s in samples], dtype=object),
        euler_convention=EULER_CONVENTION,
    )
    print(f"\nSaved -> {OUTPUT_FILE}")
    print("\nAt runtime, compute the live camera pose as:")
    print("    T_base_camera = T_base_flange @ T_flange_camera")


if __name__ == "__main__":
    main()


# T_flange_camera (m):
# [[ 0.61618131 -0.78561938 -0.05588192  0.08737608]
#  [ 0.78477265  0.61842972 -0.04094585 -0.03452748]
#  [ 0.06672689 -0.01862453  0.99759744  0.1366223 ]
#  [ 0.          0.          0.          1.        ]]

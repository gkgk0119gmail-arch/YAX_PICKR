"""
Refine an eye-in-hand calibration (T_flange_camera) using RF-DETR + SAM2
detected centroids and gripper-tip ground-truth touches.

Workflow
--------
  1. Pose the robot so the camera sees one or more objects on the workspace.
     Press SPACE to freeze a frame: RF-DETR + SAM2 run on the still image,
     each detection gets a per-object 3D point cloud (deprojected from the
     SAM mask), and a top-face plane fit yields a centroid for each
     successful detection.

  2. The frozen view shows every centroid numbered 0, 1, 2, .... For each
     centroid you actually want to use:
        - jog the robot so the gripper TIP sits on the physical centroid
          (the tip frame is T_FLANGE_TIP below: 23 cm along flange +Z),
        - press 'g' to record (T_base_flange @ T_FLANGE_TIP)[:3, 3] as the
          ground-truth base-frame position of that centroid.
     Press 'k' to skip a centroid, 'b' to abandon the freeze (no touches
     saved from it), or 'x' when you're done with the current freeze and
     want to return to the live preview.

  3. Repeat steps 1-2 from a different camera pose and/or with new objects
     to gather more samples. Each touched centroid contributes one
     residual:

         residual_i = T_base_flange_capture_i @ X @ c_cam_i  -  q_base_i

     where X = X_0 @ exp(xi) is the unknown T_flange_camera and X_0 is the
     initial guess loaded from flange_camera_transform.npz.

  4. Press 'c' to run a Levenberg-Marquardt refinement over all samples
     and save the result to flange_camera_transform_refined.npz. The
     refined file can be used in place of the original by passing
     --flange-file flange_camera_transform_refined.npz to validate_eye_in_hand.py
     / live_detect_3d.py / cluster.py.

Keyboard jog (teleoperation, both phases)
-----------------------------------------
    Instead of a teach pendant, jog the flange in base-frame Cartesian space
    by a discrete step each press (blocking IK + MoveJ). Works in the live
    preview AND the frozen/touch phase, and over the --web viewer.

    w / s          +X / -X (mm)        t / y          +RX / -RX (deg)
    a / d          +Y / -Y (mm)        h / n          +RY / -RY (deg)
    e / f          +Z / -Z (mm)        j / m          +RZ / -RZ (deg)
    - / =          decrease / increase the XYZ step by 30%
    , / .          decrease / increase the RPY step by 30%
    p              face straight down (rx=-180, ry=0; keeps XYZ + RZ)

    Defaults: --xyz-step 10 (mm), --rpy-step 5 (deg), --jog-vel 20 (%).

Controls (live phase)
---------------------
    SPACE / ENTER  freeze frame and run RF-DETR + SAM2
    u              undo the LAST committed touch
    r              reset all samples
    c              compute refinement and save
    q              quit without saving

Controls (frozen / touch phase)
-------------------------------
    g              record gripper-tip GT for the SELECTED centroid
    k              skip the SELECTED centroid
    [ / ]          previous / next centroid
    x              done with this freeze, back to live
    b              abandon this freeze (no touches saved from it)

Remote / SSH use
----------------
    Pass --web to serve the windows in a browser instead of opening native
    OpenCV windows (no X11 forwarding needed).     Forward the port over SSH
    (ssh -L 7000:localhost:7000 <host>) and open http://localhost:7000/.
    The same keys work in the browser (click the page first). Options:
    --web-host, --web-port (default 7000), --web-quality (JPEG, default 80).
"""

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# Reuse the RGBD pipeline + DETR/SAM loading + 3D extraction from
# live_detect_3d so the centroids we touch are bit-identical to what
# the live viewer shows.
from live_detect_3d import (
    init_realsense_rgbd,
    detect_2d,
    extract_objects_3d,
    DEFAULT_ROBOT_IP,
)
from live_detect import (
    load_sam2_predictor,
    _pick_autocast_dtype,
    draw_detections,
    PRETRAINED_WEIGHTS,
    FINETUNED_WEIGHTS,
    SAM2_CKPT,
    SAM2_CFG,
)
from cluster import CAMERA_WIDTH, CAMERA_HEIGHT
from rfdetr import RFDETRMedium

# --- Fairino SDK -----------------------------------------------------------
# _FAIRINO_SDK_PATH = "/home/jeong/fr5sdk/linux"
_FAIRINO_SDK_PATH = '/home/pickr0001/fairino-python-sdk/linux'
if _FAIRINO_SDK_PATH not in sys.path:
    sys.path.insert(0, _FAIRINO_SDK_PATH)
from fairino import Robot  # noqa: E402


# ===========================================================================
# Constants
# ===========================================================================
ROBOT_IP = DEFAULT_ROBOT_IP
INPUT_FILE = "flange_camera_transform.npz"
OUTPUT_FILE = "flange_camera_transform.npz"

# Gripper-tip offset along flange +Z axis. The user touches the physical
# centroid with the gripper tip, so q_base = (T_base_flange @ T_FLANGE_TIP)[:3, 3].
# Override at runtime with --tip-offset.
T_FLANGE_TIP = np.eye(4)
T_FLANGE_TIP[:3, 3] = [0.0, 0.0, 0.23]  # 23 cm along flange Z

# Visualization
TOPDOWN_WINDOW_PX = 480
TOPDOWN_RANGE_M = 0.6  # half-extent shown around the plot center
OBJECT_COLORS_BGR = [
    (0, 255, 0),      # green
    (0, 200, 255),    # orange
    (255, 100, 100),  # blue-ish
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (100, 255, 255),  # yellow
    (180, 105, 255),  # pink
    (180, 255, 180),  # mint
]

WINDOW_LIVE = "Refine - Live (DETR+SAM)"
WINDOW_TOPDOWN = "Refine - Top-down"
WINDOW_FROZEN = "Refine - Frozen (touch each centroid)"

# ---------------------------------------------------------------------------
# Keyboard jog (teleoperation)
#
# The user has to position the gripper tip on each detected centroid. Rather
# than reaching for a teach pendant, these keys jog the flange in its own
# (base-frame) Cartesian frame by a discrete step each press. The motion is a
# blocking IK + MoveJ, mirroring keyboard_ik_control.py. Jog works in BOTH the
# live preview and the frozen/touch phase, and through the --web viewer (the
# browser forwards each letter as its char code).
#
# Default steps live in JogState; tune with --xyz-step / --rpy-step / --jog-vel
# at launch, or live with the step-adjust keys below.
#
#   Translation (mm):   w/s = +X/-X   a/d = +Y/-Y   e/f = +Z/-Z
#   Rotation (deg):     t/y = +RX/-RX h/n = +RY/-RY j/m = +RZ/-RZ
#   Step adjust:        -/= -30%/+30% XYZ step       ,/. -30%/+30% RPY step
#   Face down:          p = orient flange straight down (rx=-180, ry=0)
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


# ===========================================================================
# Display backends
#
# The tool is interactive: it shows OpenCV windows and reads single key
# presses via cv2.waitKey. That requires a local display (or X11
# forwarding), which is painful over SSH. To support headless / remote use
# we abstract "show an image in a named window" and "wait for a key" behind
# a tiny interface with two implementations:
#
#   CvDisplay  - the original cv2.imshow / cv2.waitKey behaviour.
#   WebDisplay - serves every window as an MJPEG stream in the browser and
#                forwards key presses from the page back to the program,
#                so it works over plain SSH (just forward the port).
#
# Both wait_key() return a raw int matching cv2's convention (-1 when no key
# was pressed), so callers can keep using `key = display.wait_key(t) & 0xFF`.
# ===========================================================================
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
<title>Refine eye-in-hand (web)</title>
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
  <h1>Refine eye-in-hand &mdash; remote view</h1>
  <span class="pill">last key: <b id="key">&mdash;</b></span>
  <span class="help">Click the page first, then use the keys below.
  <kbd>Space</kbd> freeze &middot; <kbd>g</kbd> touch &middot;
  <kbd>k</kbd> skip &middot; <kbd>[</kbd>/<kbd>]</kbd> cycle &middot;
  <kbd>x</kbd> done &middot; <kbd>b</kbd> abandon &middot;
  <kbd>u</kbd> undo &middot; <kbd>r</kbd> reset &middot;
  <kbd>c</kbd> compute &middot; <kbd>q</kbd> quit
  <br>jog: <kbd>w</kbd>/<kbd>s</kbd> <kbd>a</kbd>/<kbd>d</kbd> <kbd>e</kbd>/<kbd>f</kbd> = XYZ mm &middot;
  <kbd>t</kbd>/<kbd>y</kbd> <kbd>h</kbd>/<kbd>n</kbd> <kbd>j</kbd>/<kbd>m</kbd> = RPY deg &middot;
  <kbd>-</kbd>/<kbd>=</kbd> XYZ step &middot; <kbd>,</kbd>/<kbd>.</kbd> RPY step &middot;
  <kbd>p</kbd> face down</span>
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
  else if (e.key === 'ArrowLeft') { code = 91; label = '['; }   // map to [
  else if (e.key === 'ArrowRight') { code = 93; label = ']'; }  // map to ]
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


# ===========================================================================
# SE(3) helpers
# ===========================================================================
def _hat(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def se3_exp(xi):
    """xi = [omega(3); v(3)] -> 4x4 SE(3) transform.

    R = exp([omega]_x)
    t = V * v   with V = I + (1-cos)/theta^2 * [w]_x + (theta-sin)/theta^3 * [w]_x^2
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    omega = xi[:3]
    v = xi[3:]
    theta = float(np.linalg.norm(omega))
    W = _hat(omega)
    if theta < 1e-9:
        Rm = np.eye(3) + W + 0.5 * (W @ W)
        V = np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    else:
        s, c = np.sin(theta), np.cos(theta)
        Rm = np.eye(3) + (s / theta) * W + ((1.0 - c) / (theta * theta)) * (W @ W)
        V = (
            np.eye(3)
            + ((1.0 - c) / (theta * theta)) * W
            + ((theta - s) / (theta ** 3)) * (W @ W)
        )
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = V @ v
    return T


# ===========================================================================
# Geometry helpers
# ===========================================================================
def project_cam_to_pixel(c_cam, K):
    """Project a 3D camera-frame point to 2D color pixel. Returns (u, v) ints or None."""
    if c_cam[2] <= 1e-6:
        return None
    u = K[0, 0] * c_cam[0] / c_cam[2] + K[0, 2]
    v = K[1, 1] * c_cam[1] / c_cam[2] + K[1, 2]
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (int(round(u)), int(round(v)))


def base_to_cam(p_base, T_base_camera):
    Tinv = np.linalg.inv(T_base_camera)
    return (Tinv @ np.append(np.asarray(p_base, dtype=np.float64), 1.0))[:3]


def order_by_nearest_neighbour(points, start=None):
    """Greedy nearest-neighbour ordering of base-frame points.

    Returns the visiting order (a list of indices into `points`) such that each
    next point is the closest remaining one to the previous point. This keeps
    consecutive centroids physically adjacent so the user only has to nudge the
    gripper a short distance between touches instead of crossing the workspace.

    `start`, if given, is a 3D base-frame anchor (e.g. the current gripper tip):
    the first centroid chosen is the one nearest to it, so the very first move
    is short too. With no anchor we start from points[0].
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 1:
        return list(range(n))

    remaining = list(range(n))
    if start is not None:
        anchor = np.asarray(start, dtype=np.float64)
        first = min(remaining, key=lambda i: np.linalg.norm(pts[i] - anchor))
    else:
        first = remaining[0]

    order = [first]
    remaining.remove(first)
    while remaining:
        last = pts[order[-1]]
        nxt = min(remaining, key=lambda i: np.linalg.norm(pts[i] - last))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def y_for_sample(s, X):
    return (s["T_bf"] @ X @ np.append(s["c_cam"], 1.0))[:3]


# ===========================================================================
# RealSense capture (RGBD frame fetch + filter chain)
# ===========================================================================
def _apply_depth_filters(depth_frame, filters):
    """Apply the same filter chain order as live_detect_3d / cluster.py."""
    depth = depth_frame
    if filters.get("decimation"):
        depth = filters["decimation"].process(depth)
    if filters.get("depth_to_disparity"):
        depth = filters["depth_to_disparity"].process(depth)
    if filters.get("spatial"):
        depth = filters["spatial"].process(depth)
    if filters.get("temporal"):
        depth = filters["temporal"].process(depth)
    if filters.get("disparity_to_depth"):
        depth = filters["disparity_to_depth"].process(depth)
    if filters.get("hole_filling"):
        depth = filters["hole_filling"].process(depth)
    return depth


def grab_rgbd(pipe, align, filters, depth_scale):
    """Wait one frameset, return (color_bgr, depth_metres) or (None, None).

    color_bgr is HxWx3 uint8 aligned to color, depth_metres is HxW float32
    in meters (same shape as color).
    """
    try:
        frames = pipe.wait_for_frames()
    except RuntimeError:
        return None, None
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    depth_frame = _apply_depth_filters(depth_frame, filters)
    color = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_metres = depth_raw * float(depth_scale)
    return color, depth_metres


# ===========================================================================
# Robot helpers
# ===========================================================================
def get_flange_pose_matrix(robot, euler_conv):
    err, pose = robot.GetActualToolFlangePose(flag=1)
    if err != 0 or pose is None:
        raise RuntimeError(f"GetActualToolFlangePose failed: err={err}")
    T = np.eye(4)
    T[:3, 3] = np.asarray(pose[:3], dtype=np.float64) / 1000.0
    T[:3, :3] = R.from_euler(euler_conv, pose[3:], degrees=True).as_matrix()
    return T, np.asarray(pose, dtype=np.float64)


def setup_robot_for_motion(robot):
    """Put the controller in a state where IK + MoveJ jogging works.

    Mirrors pick_bt/robot.connect_robot / keyboard_ik_control.connect_robot:
    clear errors, manual mode, enable, and refuse to run if a safety stop is
    latched. Called once in main() before the interactive loop.
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


def face_down_flange(robot, vel):
    """Orient the flange to point straight down (rx=-180, ry=0 deg).

    Keeps the current XYZ position and current RZ (yaw), only overriding the
    roll/pitch so the end-effector +Z faces the workspace. Solves IK (config=-1
    -> nearest solution) and runs a blocking MoveJ. Returns the commanded pose.
    """
    err, pose = robot.GetActualToolFlangePose(flag=1)
    if err != 0 or pose is None:
        raise RuntimeError(f"GetActualToolFlangePose failed: {err}")
    target = [float(v) for v in pose]
    target[3] = -180.0  # rx
    target[4] = 0.0     # ry

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
    if key == ord('p'):
        face_down_flange(robot, jog["vel"])
        return "face down (rx=-180, ry=0)"
    if key == ord('-'):
        jog["xyz"] = max(0.1, jog["xyz"] * 0.7)
        return f"XYZ step = {jog['xyz']:.2f} mm"
    if key in (ord('='), ord('+')):
        jog["xyz"] = min(200.0, jog["xyz"] / 0.7)
        return f"XYZ step = {jog['xyz']:.2f} mm"
    if key == ord(','):
        jog["rpy"] = max(0.1, jog["rpy"] * 0.7)
        return f"RPY step = {jog['rpy']:.2f} deg"
    if key == ord('.'):
        jog["rpy"] = min(90.0, jog["rpy"] / 0.7)
        return f"RPY step = {jog['rpy']:.2f} deg"
    return None


# ===========================================================================
# Visualization
# ===========================================================================
def color_for_obj(obj_id):
    return OBJECT_COLORS_BGR[obj_id % len(OBJECT_COLORS_BGR)]


def _put_lines(img, lines, x, y, color=(255, 255, 255), scale=0.55, thickness=1):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * int(22 * scale * 1.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_live_overlay(img, samples, last_status):
    """Header on the live feed: total touched samples + last status line."""
    out = img.copy()
    H, W = out.shape[:2]

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, dst=out)

    cv2.putText(out, f"touched samples: {len(samples)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 2, cv2.LINE_AA)
    if last_status:
        cv2.putText(out, last_status[:100],
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(out,
                "jog: w/s a/d e/f = XYZmm   t/y h/n j/m = RPYdeg   -/= ,/. step   p=face down",
                (10, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(out,
                "SPACE=freeze+detect   u=undo   r=reset   c=compute   q=quit",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_frozen_overlay(frozen, K, selected_idx, last_status, tip_base=None,
                        class_names=None):
    """Render the frozen RGB with numbered centroids + per-status colours.

    Centroids carry one of three statuses:
        pending  - cyan, untouched
        touched  - green, GT recorded
        skipped  - grey, won't be saved as a sample
    The currently *selected* centroid is highlighted with a yellow ring.
    """
    img = frozen["base_img"].copy()
    H, W = img.shape[:2]

    masks = frozen.get("masks") or []
    dets = frozen.get("dets") or []
    if dets and masks and class_names is not None:
        try:
            img = draw_detections(img, dets, class_names, masks=masks)
        except Exception:
            pass

    centroids = frozen["centroids"]
    selected = centroids[selected_idx] if 0 <= selected_idx < len(centroids) else None

    for ci in centroids:
        idx = ci["label_idx"]
        status = frozen["status"][idx]
        c_cam = ci["c_cam"]
        uv = project_cam_to_pixel(c_cam, K)
        if uv is None:
            continue

        if status == "touched":
            color = (0, 255, 0)
            tag = "v"
        elif status == "skipped":
            color = (140, 140, 140)
            tag = "x"
        else:
            color = (0, 200, 255)
            tag = "?"

        if selected is not None and idx == selected["label_idx"]:
            cv2.circle(img, uv, 22, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.circle(img, uv, 12, color, -1, cv2.LINE_AA)
        cv2.drawMarker(img, uv, (0, 0, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
        label = f"#{idx}{tag}"
        cv2.putText(img, label, (uv[0] + 16, uv[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (uv[0] + 16, uv[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Live gripper-tip projected through the *capture* camera pose.
    # As the user jogs the arm the marker glides on the frozen image,
    # which gives a quick visual "is the tip already over centroid #2?".
    # The arm physically moves while the image is frozen, so this is an
    # approximation -- it's still useful as long as you mostly translate.
    if tip_base is not None:
        tip_cam = base_to_cam(tip_base, frozen["T_bc_capture"])
        uv = project_cam_to_pixel(tip_cam, K)
        if uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < H:
            cv2.drawMarker(img, uv, (255, 255, 255),
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=22, thickness=2)
            cv2.putText(img, "tip", (uv[0] + 8, uv[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)

    if selected is not None:
        c_base = selected["c_base_at_capture"]
        line = (f"selected #{selected['label_idx']}  "
                f"c_base=({c_base[0]:+.3f},{c_base[1]:+.3f},{c_base[2]:+.3f}) m")
        cv2.putText(img, line, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    if last_status:
        cv2.putText(img, last_status[:100], (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
                    1, cv2.LINE_AA)

    n_total = len(centroids)
    n_done = sum(1 for s in frozen["status"] if s != "pending")
    sel_idx = selected["label_idx"] if selected is not None else -1
    banner = np.zeros((78, W, 3), dtype=np.uint8)
    cv2.putText(
        banner,
        f"FROZEN  selected #{sel_idx}   {n_done}/{n_total} resolved   "
        "g=touch  k=skip  [ ]=cycle  x=done  b=abandon",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        banner,
        "jog: w/s a/d e/f = XYZmm   t/y h/n j/m = RPYdeg   -/= ,/. = step   p=face down",
        (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


def draw_topdown(samples, ground_truths, X):
    """Top-down (base-frame XY) plot of every captured sample + GT touch."""
    size = TOPDOWN_WINDOW_PX
    img = np.full((size, size, 3), 30, dtype=np.uint8)

    points = []
    for q in ground_truths.values():
        points.append(np.asarray(q)[:2])
    for s in samples:
        points.append(y_for_sample(s, X)[:2])
    if points:
        center = np.mean(np.asarray(points), axis=0)
    else:
        center = np.array([0.5, 0.0])

    half = TOPDOWN_RANGE_M / 2.0

    def to_px(xy):
        x, y = xy
        u = (x - (center[0] - half)) / (2 * half) * size
        v = size - (y - (center[1] - half)) / (2 * half) * size
        return int(round(u)), int(round(v))

    for g in np.linspace(-half, half, 7):
        u0, _ = to_px((center[0] + g, center[1] - half))
        u1, _ = to_px((center[0] + g, center[1] + half))
        cv2.line(img, (u0, 0), (u1, size), (60, 60, 60), 1)
        _, v0 = to_px((center[0] - half, center[1] + g))
        _, v1 = to_px((center[0] + half, center[1] + g))
        cv2.line(img, (0, v0), (size, v1), (60, 60, 60), 1)

    for s in samples:
        y = y_for_sample(s, X)
        uv = to_px(y[:2])
        c = color_for_obj(s["obj_id"])
        cv2.circle(img, uv, 4, c, -1, cv2.LINE_AA)

    for o, q in ground_truths.items():
        uv = to_px(np.asarray(q)[:2])
        c = color_for_obj(o)
        cv2.drawMarker(img, uv, c, markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=14, thickness=2)

    cv2.putText(img, "top-down (base XY)  m", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"center=({center[0]:+.2f},{center[1]:+.2f})  +/-{half:.2f}m",
                (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    return img


# ===========================================================================
# Refinement
# ===========================================================================
def make_residuals(samples, ground_truths, X0):
    obj_ids = sorted({s["obj_id"] for s in samples})
    free_obj = [o for o in obj_ids if o not in ground_truths]
    n_free = len(free_obj)

    def unpack(p):
        dxi = p[:6]
        X = X0 @ se3_exp(dxi)
        mu_map = {o: np.asarray(q, dtype=np.float64) for o, q in ground_truths.items()}
        for k, o in enumerate(free_obj):
            mu_map[o] = p[6 + 3 * k: 9 + 3 * k]
        return X, mu_map

    def residuals(p):
        X, mu_map = unpack(p)
        out = []
        for s in samples:
            mu_o = mu_map.get(s["obj_id"])
            if mu_o is None:
                continue
            y = (s["T_bf"] @ X @ np.append(s["c_cam"], 1.0))[:3]
            out.append(y - mu_o)
        if not out:
            return np.zeros(0)
        return np.concatenate(out)

    p0 = np.zeros(6 + 3 * n_free)
    for k, o in enumerate(free_obj):
        ys = [y_for_sample(s, X0) for s in samples if s["obj_id"] == o]
        if ys:
            p0[6 + 3 * k: 9 + 3 * k] = np.mean(np.stack(ys), axis=0)
    return residuals, p0, unpack, free_obj


def solve_refinement(samples, ground_truths, X0):
    residuals, p0, unpack, free_obj = make_residuals(samples, ground_truths, X0)
    r0 = residuals(p0)
    cost0 = 0.5 * float(r0 @ r0)
    print(f"\n[solver] residuals: {len(r0)}   params: {len(p0)}   "
          f"objects: {len(set(s['obj_id'] for s in samples))} "
          f"({len(ground_truths)} with GT, {len(free_obj)} free mu)")
    print(f"[solver] initial cost: {cost0:.6e}   "
          f"rms={np.sqrt(np.mean(r0 ** 2)) * 1000.0:.3f} mm")

    # 'lm' is unconstrained Levenberg-Marquardt; works well here.
    res = least_squares(residuals, p0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=2000)
    rN = residuals(res.x)
    costN = 0.5 * float(rN @ rN)
    print(f"[solver] final cost:   {costN:.6e}   "
          f"rms={np.sqrt(np.mean(rN ** 2)) * 1000.0:.3f} mm  "
          f"({res.message})")

    X_refined, mu_map = unpack(res.x)
    return X_refined, mu_map, res


# ===========================================================================
# Reporting
# ===========================================================================
def per_object_stats(samples, ground_truths, X):
    obj_ids = sorted({s["obj_id"] for s in samples})
    rows = []
    for o in obj_ids:
        ys = np.stack([y_for_sample(s, X) for s in samples if s["obj_id"] == o], axis=0)
        anchor = ground_truths[o] if o in ground_truths else ys.mean(axis=0)
        d = np.linalg.norm(ys - anchor, axis=1)
        rows.append({
            "obj_id": o,
            "n_views": int(ys.shape[0]),
            "has_gt": o in ground_truths,
            "anchor": np.asarray(anchor, dtype=np.float64),
            "trans_std_per_axis_mm": (ys.std(axis=0) * 1000.0).tolist(),
            "rms_mm": float(d.mean() * 1000.0),
            "max_mm": float(d.max() * 1000.0),
        })
    return rows


def print_report(label, rows):
    print(f"\n=== {label} ===")
    print(f"{'obj':>4}  {'views':>5}  {'gt':>3}  "
          f"{'anchor (m)':>32}  {'std xyz (mm)':>22}  "
          f"{'rms (mm)':>8}  {'max (mm)':>8}")
    for r in rows:
        a = r["anchor"]
        sx, sy, sz = r["trans_std_per_axis_mm"]
        print(f"{r['obj_id']:>4}  {r['n_views']:>5}  "
              f"{('yes' if r['has_gt'] else 'no'):>3}  "
              f"({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})  "
              f"({sx:5.1f},{sy:5.1f},{sz:5.1f})  "
              f"{r['rms_mm']:8.2f}  {r['max_mm']:8.2f}")


def print_x_delta(X0, X1):
    dT = np.linalg.inv(X0) @ X1
    dt_mm = dT[:3, 3] * 1000.0
    drpy_deg = R.from_matrix(dT[:3, :3]).as_euler("xyz", degrees=True)
    print(f"\n[delta] X_refined relative to X_initial (in X_initial frame):")
    print(f"        translation (mm):  ({dt_mm[0]:+.2f}, {dt_mm[1]:+.2f}, {dt_mm[2]:+.2f})")
    print(f"        rotation rpy (deg): ({drpy_deg[0]:+.3f}, {drpy_deg[1]:+.3f}, {drpy_deg[2]:+.3f})")


def print_x(label, X):
    t_mm = X[:3, 3] * 1000.0
    rpy_deg = R.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True)
    print(f"\n[{label}] T_flange_camera (m):\n{X}")
    print(f"[{label}] translation (mm):  ({t_mm[0]:+.2f}, {t_mm[1]:+.2f}, {t_mm[2]:+.2f})")
    print(f"[{label}] euler xyz   (deg): ({rpy_deg[0]:+.3f}, {rpy_deg[1]:+.3f}, {rpy_deg[2]:+.3f})")


# ===========================================================================
# Saving
# ===========================================================================
def save_refined(out_path, X_refined, X_initial, samples, ground_truths,
                 K, D, euler_conv, parent_file, mu_map, solver_stats):
    obj_ids = sorted({s["obj_id"] for s in samples})

    sample_T_bf = np.stack([s["T_bf"] for s in samples], axis=0) if samples else np.zeros((0, 4, 4))
    sample_c_cam = np.stack([s["c_cam"] for s in samples], axis=0) if samples else np.zeros((0, 3))
    sample_c_base = np.stack([s["c_base"] for s in samples], axis=0) if samples else np.zeros((0, 3))
    sample_obj_id = np.array([s["obj_id"] for s in samples], dtype=np.int32)

    gt_obj_ids = np.array(sorted(ground_truths.keys()), dtype=np.int32)
    gt_q = (np.stack([ground_truths[o] for o in gt_obj_ids], axis=0)
            if len(gt_obj_ids) else np.zeros((0, 3)))
    mu_obj_ids = np.array(sorted(mu_map.keys()), dtype=np.int32) if mu_map else np.zeros((0,), dtype=np.int32)
    mu_arr = (np.stack([np.asarray(mu_map[o]) for o in mu_obj_ids], axis=0)
              if len(mu_obj_ids) else np.zeros((0, 3)))

    np.savez(
        out_path,
        T_flange_camera=X_refined,
        T_flange_camera_initial=X_initial,
        method="refined",
        parent_file=str(parent_file),
        K=K, D=D,
        euler_convention=euler_conv,
        obj_ids=np.array(obj_ids, dtype=np.int32),
        sample_T_bf=sample_T_bf,
        sample_c_cam=sample_c_cam,
        sample_c_base=sample_c_base,
        sample_obj_id=sample_obj_id,
        ground_truth_obj_ids=gt_obj_ids,
        ground_truth_q=gt_q,
        mu_obj_ids=mu_obj_ids,
        mu_q=mu_arr,
        solver_stats=np.array(str(solver_stats), dtype=object),
    )
    print(f"\nSaved -> {out_path}")


# ===========================================================================
# DETR + SAM2 capture
# ===========================================================================
def load_models(weights_arg, pretrained, resolution, enable_sam):
    """Load RF-DETR (and optionally SAM2). Returns:

        model, class_names, sam_predictor, sam_device, sam_autocast_dtype
    """
    if weights_arg:
        weights = Path(weights_arg)
    elif pretrained:
        weights = PRETRAINED_WEIGHTS
    else:
        weights = FINETUNED_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"Loading RF-DETR Medium (resolution={resolution}, "
          f"weights={weights.name}) ...")
    model = RFDETRMedium(pretrain_weights=str(weights), resolution=resolution)
    model.optimize_for_inference()
    if pretrained and not weights_arg:
        from rfdetr.util.coco_classes import COCO_CLASSES
        class_names = COCO_CLASSES
    else:
        class_names = model.class_names
    print(f"Classes: {class_names}")
    print("Warming up RF-DETR ...")
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), threshold=0.5)
    print("Warmup done.")

    sam_predictor = None
    sam_device = "cpu"
    sam_autocast_dtype = None
    if enable_sam:
        if not SAM2_CKPT.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CKPT}")
        import torch
        sam_device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_autocast_dtype = _pick_autocast_dtype()
        print(f"Loading SAM2 ({SAM2_CKPT.name}) on {sam_device} ...")
        sam_predictor = load_sam2_predictor(SAM2_CKPT, SAM2_CFG, sam_device)
        print("SAM2 ready.")

    return model, class_names, sam_predictor, sam_device, sam_autocast_dtype


def perform_capture(pipe, align, filters, depth_scale, color_intr,
                    model, sam_predictor, sam_device, sam_autocast_dtype,
                    robot, X_current, euler_conv,
                    conf, max_det, depth_stride):
    """Grab one RGBD frame, run RF-DETR + SAM2, extract centroids per object.

    Returns a 'frozen' state dict or None if the capture is unusable
    (no detections, no rect fits, robot unreachable, ...).
    """
    color, depth_metres = grab_rgbd(pipe, align, filters, depth_scale)
    if color is None or depth_metres is None:
        print("  [capture] failed to grab RGBD frame")
        return None

    T_bf, raw_pose = get_flange_pose_matrix(robot, euler_conv)
    T_bc = T_bf @ X_current

    dets = detect_2d(model, color, conf, max_det, roi=None)
    if not dets:
        print("  [capture] RF-DETR returned no detections")
        return None

    masks, per_obj_pts_base, per_obj_rect = extract_objects_3d(
        color, depth_metres, color_intr, T_bc, dets,
        sam_predictor=sam_predictor,
        sam_device=sam_device,
        sam_autocast_dtype=sam_autocast_dtype,
        depth_stride=depth_stride,
    )

    centroids = []
    for det_idx, rect in enumerate(per_obj_rect):
        if rect is None:
            continue
        c_base = np.asarray(rect["centroid"], dtype=np.float64)
        c_cam = np.asarray(base_to_cam(c_base, T_bc), dtype=np.float64)
        centroids.append({
            "det_idx": det_idx,
            "c_cam": c_cam,
            "c_base_at_capture": c_base,
        })

    if not centroids:
        print("  [capture] no usable centroids (no rect fits passed RANSAC)")
        return None

    # Number the centroids along a short path instead of RF-DETR's arbitrary
    # detection order, so consecutive labels (0->1->2->...) are physically
    # adjacent and the user only has to jog a little between touches. Start the
    # walk from the current gripper tip so the first move is short too.
    tip_base = (T_bf @ T_FLANGE_TIP)[:3, 3]
    order = order_by_nearest_neighbour(
        [c["c_base_at_capture"] for c in centroids], start=tip_base)
    centroids = [centroids[i] for i in order]
    for label_idx, ci in enumerate(centroids):
        ci["label_idx"] = label_idx

    print(f"  [capture] {len(centroids)} usable centroid(s):")
    for ci in centroids:
        c = ci["c_base_at_capture"]
        print(f"    #{ci['label_idx']}  base=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) m")

    return {
        "base_img": color.copy(),
        "T_bf_capture": T_bf,
        "T_bc_capture": T_bc,
        "raw_pose_capture": raw_pose,
        "dets": dets,
        "masks": masks,
        "per_obj_pts_base": per_obj_pts_base,
        "per_obj_rect": per_obj_rect,
        "centroids": centroids,
        "status": ["pending"] * len(centroids),
    }


def run_touch_phase(frozen, robot, euler_conv, K_live, class_names,
                    samples, ground_truths, history, next_obj_id,
                    topdown_X, draw_topdown_fn, display, jog):
    """Interactive touch loop on a frozen frame.

    For each centroid the user can press 'g' to record the gripper-tip
    GT, 'k' to skip, '['/']' to cycle, 'x' to finish, 'b' to abandon.

    Returns the (possibly updated) next_obj_id. Mutates samples /
    ground_truths / history in place.
    """
    centroids = frozen["centroids"]
    if not centroids:
        return next_obj_id

    selected_idx = 0
    last_status = ""

    display.named_window(WINDOW_FROZEN, frozen["base_img"].shape[1],
                         frozen["base_img"].shape[0] + 84)

    try:
        while True:
            # Live flange pose so we can show the gripper tip on the
            # frozen image and so 'g' uses the latest robot pose.
            try:
                T_bf_now, _raw = get_flange_pose_matrix(robot, euler_conv)
                tip_base = (T_bf_now @ T_FLANGE_TIP)[:3, 3]
            except Exception as e:
                T_bf_now = None
                tip_base = None
                last_status = f"[robot] {e}"

            overlay = draw_frozen_overlay(frozen, K_live, selected_idx,
                                          last_status, tip_base=tip_base,
                                          class_names=class_names)
            display.imshow(WINDOW_FROZEN, overlay)
            display.imshow(WINDOW_TOPDOWN,
                           draw_topdown_fn(samples, ground_truths, topdown_X))

            key = display.wait_key(20) & 0xFF
            if key == 255:
                continue

            # Keyboard jog: position the gripper tip on the centroid without a
            # teach pendant. Handled before the touch keys; on a successful jog
            # we loop so the next frame re-projects the tip marker.
            try:
                jstat = handle_jog_key(robot, key, jog)
            except Exception as e:
                last_status = f"[jog] {e}"
                continue
            if jstat is not None:
                last_status = jstat
                continue

            if key == ord('x'):
                last_status = "leaving frozen frame."
                print("  [touch] done with this freeze.")
                break

            if key == ord('b'):
                print("  [touch] abandoning this freeze (no samples saved).")
                break

            if key == ord('['):
                selected_idx = (selected_idx - 1) % len(centroids)
                last_status = f"selected #{selected_idx}"
                continue
            if key == ord(']'):
                selected_idx = (selected_idx + 1) % len(centroids)
                last_status = f"selected #{selected_idx}"
                continue

            if key in (ord('k'),):
                if frozen["status"][selected_idx] != "touched":
                    frozen["status"][selected_idx] = "skipped"
                    last_status = f"skipped #{selected_idx}"
                    print(f"  [touch] skipped centroid #{selected_idx}")
                # Auto-advance to next pending.
                for d in range(1, len(centroids) + 1):
                    j = (selected_idx + d) % len(centroids)
                    if frozen["status"][j] == "pending":
                        selected_idx = j
                        break
                continue

            if key == ord('g'):
                if T_bf_now is None:
                    last_status = "robot pose unavailable; can't record GT"
                    continue
                if frozen["status"][selected_idx] == "touched":
                    last_status = f"#{selected_idx} already touched (use u to undo)"
                    continue

                ci = centroids[selected_idx]
                q_base = (T_bf_now @ T_FLANGE_TIP)[:3, 3]
                obj_id = int(next_obj_id)
                next_obj_id += 1

                sample = {
                    "obj_id": obj_id,
                    "T_bf": frozen["T_bf_capture"].copy(),
                    "raw_pose": frozen["raw_pose_capture"],
                    "c_base": ci["c_base_at_capture"].copy(),
                    "c_cam": ci["c_cam"].copy(),
                }
                samples.append(sample)
                ground_truths[obj_id] = np.asarray(q_base, dtype=np.float64)
                history.append(("touch", obj_id, selected_idx, frozen))
                frozen["status"][selected_idx] = "touched"

                err_mm = float(np.linalg.norm(
                    ci["c_base_at_capture"] - q_base) * 1000.0)
                last_status = (f"touched #{selected_idx}  err to c_base "
                               f"(initial X)={err_mm:.1f}mm")
                print(f"  [touch] obj_id={obj_id}  centroid #{selected_idx}  "
                      f"q_base={np.round(q_base, 4).tolist()}  "
                      f"|c_base - q_base|={err_mm:.1f} mm")

                # Auto-advance to next pending centroid; if none, hint to leave.
                advanced = False
                for d in range(1, len(centroids) + 1):
                    j = (selected_idx + d) % len(centroids)
                    if frozen["status"][j] == "pending":
                        selected_idx = j
                        advanced = True
                        break
                if not advanced:
                    last_status = ("all centroids resolved -- press 'x' "
                                   "to return to live preview")
                continue
    finally:
        display.destroy_window(WINDOW_FROZEN)

    return next_obj_id


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default=ROBOT_IP)
    ap.add_argument("--input", default=INPUT_FILE,
                    help="initial T_flange_camera npz (from realsense_calib_eye_in_hand.py)")
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--pretrained", action="store_true",
                    help="Use the COCO pretrained RF-DETR instead of fine-tuned weights")
    ap.add_argument("--weights", type=str, default=None,
                    help="Custom RF-DETR checkpoint path (overrides --pretrained)")
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--no-sam", action="store_true",
                    help="Skip SAM2 (centroids will be unavailable -- mostly a debug flag)")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="RF-DETR confidence threshold")
    ap.add_argument("--max-det", type=int, default=30,
                    help="Max detections per frame")
    ap.add_argument("--depth-stride", type=int, default=1,
                    help="Subsample SAM mask pixels by this stride before deprojection")
    ap.add_argument("--tip-offset", type=float, default=None,
                    help="Override gripper tip offset along flange +Z (m). "
                         "Default 0.23 = 23 cm (set in T_FLANGE_TIP).")
    ap.add_argument("--xyz-step", type=float, default=JOG_XYZ_STEP_MM,
                    help="Keyboard jog translation step in mm (w/s/a/d/e/f).")
    ap.add_argument("--rpy-step", type=float, default=JOG_RPY_STEP_DEG,
                    help="Keyboard jog rotation step in deg (t/y/h/n/j/m).")
    ap.add_argument("--jog-vel", type=float, default=JOG_VEL_PCT,
                    help="MoveJ velocity (%% of max) used for keyboard jogs.")
    ap.add_argument("--web", action="store_true",
                    help="Serve the windows in a browser instead of opening "
                         "native OpenCV windows. Useful over SSH: forward the "
                         "port and open http://localhost:<port>/.")
    ap.add_argument("--web-host", default="0.0.0.0",
                    help="Interface the web viewer binds to (default 0.0.0.0).")
    ap.add_argument("--web-port", type=int, default=7000,
                    help="Port for the web viewer (default 7000).")
    ap.add_argument("--web-quality", type=int, default=80,
                    help="JPEG quality 1-100 for the web stream (default 80).")
    args = ap.parse_args()

    if args.tip_offset is not None:
        T_FLANGE_TIP[:3, 3] = [0.0, 0.0, float(args.tip_offset)]

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run realsense_calib_eye_in_hand.py first.")
        sys.exit(1)
    cal = np.load(in_path, allow_pickle=True)
    X0 = np.asarray(cal["T_flange_camera"], dtype=np.float64).copy()
    K_npz = np.asarray(cal["K"], dtype=np.float64) if "K" in cal.files else None
    D_npz = np.asarray(cal["D"], dtype=np.float64) if "D" in cal.files else None
    euler_conv = str(cal["euler_convention"]) if "euler_convention" in cal.files else "xyz"

    print("=" * 70)
    print("Refine eye-in-hand calibration  (RF-DETR + SAM2 + gripper-tip GT)")
    print("=" * 70)
    print(f"[input]   {in_path}")
    print(f"[output]  {args.output}")
    print(f"[euler]   {euler_conv}")
    print(f"[tip]     T_flange_tip translation = "
          f"{T_FLANGE_TIP[:3, 3].tolist()} m  (override with --tip-offset)")
    print_x("initial", X0)

    print(f"\nConnecting to robot at {args.robot_ip} ...")
    robot = Robot.RPC(args.robot_ip)
    time.sleep(1.0)
    err, pose = robot.GetActualToolFlangePose(flag=1)
    if err != 0:
        print(f"ERROR: cannot read flange pose (err={err}). Aborting.")
        sys.exit(1)
    print(f"Robot OK. Initial flange pose: {[f'{v:.2f}' for v in pose]}")

    # Enable manual mode + power so the keyboard jog (IK + MoveJ) can move the
    # arm. If a safety stop is latched this raises and we abort early.
    try:
        setup_robot_for_motion(robot)
        print(f"[jog] enabled. xyz-step={args.xyz_step} mm, "
              f"rpy-step={args.rpy_step} deg, vel={args.jog_vel}%")
    except Exception as e:
        print(f"ERROR: could not enable robot for jogging: {e}")
        sys.exit(1)

    # Mutable jog state, shared by the live and frozen loops; the step-adjust
    # keys mutate it in place.
    jog = {"xyz": float(args.xyz_step),
           "rpy": float(args.rpy_step),
           "vel": float(args.jog_vel)}

    model, class_names, sam_predictor, sam_device, sam_autocast_dtype = load_models(
        weights_arg=args.weights,
        pretrained=args.pretrained,
        resolution=args.resolution,
        enable_sam=not args.no_sam,
    )

    pipe, align, color_intr, depth_scale, filters = init_realsense_rgbd()

    K_live = np.array([
        [color_intr['fx'], 0.0, color_intr['cx']],
        [0.0, color_intr['fy'], color_intr['cy']],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    print(f"[camera] live color K =\n{K_live}")
    if K_npz is not None:
        print("[camera] (calib npz K was for a different resolution and is "
              "not used for projection)")

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
    display.named_window(WINDOW_LIVE, CAMERA_WIDTH, CAMERA_HEIGHT)
    display.named_window(WINDOW_TOPDOWN, TOPDOWN_WINDOW_PX, TOPDOWN_WINDOW_PX)

    print("\nReady. Move the arm so the camera sees the workspace, then SPACE.")
    print("In the frozen frame: g=touch, k=skip, [ ]=cycle, x=done, b=abandon.\n")

    samples = []
    ground_truths = {}
    history = []          # for undo: list of ("touch", obj_id, label_idx, frozen)
    next_obj_id = 0
    last_status = "ready"
    do_compute = False

    try:
        while True:
            color, _depth = grab_rgbd(pipe, align, filters, depth_scale)
            if color is None:
                # No RealSense frame this cycle. Still update the windows with
                # a status image so the (web) view renders fully and the user
                # can tell the camera -- not the viewer -- is the problem.
                wait_img = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 24,
                                   dtype=np.uint8)
                cv2.putText(wait_img, "waiting for RealSense frame...",
                            (20, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                            cv2.LINE_AA)
                display.imshow(WINDOW_LIVE, wait_img)
                display.imshow(WINDOW_TOPDOWN,
                               draw_topdown(samples, ground_truths, X0))
                if (display.wait_key(30) & 0xFF) == ord('q'):
                    print("Quitting without solving.")
                    break
                continue

            live = draw_live_overlay(color, samples, last_status)
            display.imshow(WINDOW_LIVE, live)
            display.imshow(WINDOW_TOPDOWN,
                           draw_topdown(samples, ground_truths, X0))

            key = display.wait_key(1) & 0xFF
            if key == 255:
                continue

            # Keyboard jog also works in the live preview, so you can frame the
            # workspace before freezing without touching a teach pendant.
            try:
                jstat = handle_jog_key(robot, key, jog)
            except Exception as e:
                last_status = f"[jog] {e}"
                continue
            if jstat is not None:
                last_status = jstat
                continue

            if key == ord('q'):
                print("Quitting without solving.")
                break

            if key in (13, 10, ord(' ')):  # ENTER or SPACE
                print("\n[capture] running RF-DETR + SAM2 ...")
                frozen = perform_capture(
                    pipe, align, filters, depth_scale, color_intr,
                    model, sam_predictor, sam_device, sam_autocast_dtype,
                    robot, X0, euler_conv,
                    conf=args.conf, max_det=args.max_det,
                    depth_stride=args.depth_stride,
                )
                if frozen is None:
                    last_status = "capture failed (no centroids)"
                    continue
                next_obj_id = run_touch_phase(
                    frozen, robot, euler_conv, K_live, class_names,
                    samples, ground_truths, history, next_obj_id,
                    topdown_X=X0, draw_topdown_fn=draw_topdown,
                    display=display, jog=jog,
                )
                last_status = (f"freeze produced {sum(1 for s in frozen['status'] if s == 'touched')} "
                               f"new touches; total={len(samples)}")
                continue

            if key == ord('u'):
                if not history:
                    last_status = "nothing to undo"
                    print("[undo] nothing to undo.")
                    continue
                action = history.pop()
                if action[0] == "touch":
                    _, obj_id, label_idx, frozen_ref = action
                    samples[:] = [s for s in samples if s["obj_id"] != obj_id]
                    ground_truths.pop(obj_id, None)
                    if frozen_ref is not None and 0 <= label_idx < len(frozen_ref["status"]):
                        frozen_ref["status"][label_idx] = "pending"
                    last_status = f"undid touch (obj_id={obj_id})"
                    print(f"[undo] removed touch obj_id={obj_id}. "
                          f"total samples: {len(samples)}")
                continue

            if key == ord('r'):
                samples.clear()
                ground_truths.clear()
                history.clear()
                next_obj_id = 0
                last_status = "reset"
                print("[reset] all samples / GT cleared.")
                continue

            if key == ord('c'):
                do_compute = True
                break

    finally:
        display.destroy_all()
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            robot.CloseRPC()
        except Exception:
            pass

    if not do_compute:
        return

    if len(samples) == 0:
        print("\nERROR: no samples captured.")
        return
    if len(samples) < 3:
        print(f"\nWARNING: only {len(samples)} samples. The 6-DOF refinement "
              f"is severely under-constrained. Capture more, or proceed with caution.")

    print("\n" + "=" * 70)
    print(f"REFINEMENT  ({len(samples)} samples)")
    print("=" * 70)
    print_report("BEFORE (X = initial)", per_object_stats(samples, ground_truths, X0))

    X_refined, mu_map, res = solve_refinement(samples, ground_truths, X0)

    print_report("AFTER  (X = refined)", per_object_stats(samples, ground_truths, X_refined))
    print_x("refined", X_refined)
    print_x_delta(X0, X_refined)

    save_refined(
        Path(args.output),
        X_refined=X_refined,
        X_initial=X0,
        samples=samples,
        ground_truths=ground_truths,
        K=K_npz if K_npz is not None else K_live,
        D=D_npz if D_npz is not None else np.zeros((1, 5)),
        euler_conv=euler_conv,
        parent_file=in_path,
        mu_map=mu_map,
        solver_stats={
            "message": str(res.message),
            "nfev": int(res.nfev),
            "cost": float(res.cost),
        },
    )

    print("\nUse the new file by passing --flange-file flange_camera_transform_refined.npz")
    print("to validate_eye_in_hand.py / live_detect_3d.py / cluster.py.")


if __name__ == "__main__":
    main()

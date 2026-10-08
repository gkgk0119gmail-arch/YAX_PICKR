#!/usr/bin/env python3
"""
Live RF-DETR + SAM2 + 3D segmentation viewer.

Builds on `live_detect.py`:
  * Adds an aligned depth stream (rs.align to color).
  * Reads the live Fairino flange pose and combines it with
    `flange_camera_transform.npz` to get T_base_camera every frame.
  * For each SAM mask, deprojects the masked pixels into a per-object
    3D point cloud in the robot base frame.
  * Live 3D viewer in meshcat (browser tab): the base-frame triad, a
    live camera-frame triad placed at T_base_camera, and one coloured
    point cloud per SAM detection.

Camera + filter parameters are imported from ``cluster.py`` so this
script stays in sync with ``grasp_detection_server.py`` and
``utils/realsense_pc.py``.

NOTE: this script intentionally does NOT fit a rectangle / top-face /
yaw -- that proved too unstable on noisy depth. See the chat history
or the README for next-step ideas to add a stable fit (work in 2D
mask pixels first, then lift to 3D; RANSAC plane + OBB on inliers;
template fitting against known box dimensions; etc.).

Run:
    python live_detect_3d.py --sam

Importable API (for other scripts that want the same per-object
masks / point clouds / rect fits without launching the live viewer):

    from live_detect_3d import (
        detect_2d,             # RF-DETR with optional ROI cropping
        extract_objects_3d,    # SAM2 + deproject + base-frame transform + rect fit
        detect_objects_3d,     # one-shot: detect_2d + extract_objects_3d
        deproject_mask_to_camera, transform_to_base,
        fit_top_rectangle,
        load_flange_camera_npz, connect_robot, get_T_base_flange,
        init_realsense_rgbd,
    )

Controls (extends live_detect.py):
    Left-drag/Right-click  ROI / clear ROI
    t                      toggle temporal averaging
    s                      toggle SAM2 overlay (also drives 3D)
    p                      toggle meshcat viewer updates
    q                      quit
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# Reuse all the 2D infrastructure already in live_detect.py.
from live_detect import (
    RoiState,
    DetectionAverager,
    load_sam2_predictor,
    run_sam2_on_dets,
    _pick_autocast_dtype,
    run_detection,
    draw_detections,
    PRETRAINED_WEIGHTS,
    FINETUNED_WEIGHTS,
    SAM2_CKPT,
    SAM2_CFG,
    WINDOW_NAME,
)

# Camera / filter parameters are sourced from cluster.py so this script
# stays bit-identical with grasp_detection_server.py and utils/realsense_pc.py.
import cluster
from cluster import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    DEPTH_UNITS,
    LASER_POWER,
    FILTER_DECIMATION_ENABLED,
    FILTER_DISPARITY_ENABLED,
    FILTER_SPATIAL_ENABLED,
    FILTER_TEMPORAL_ENABLED,
    FILTER_HOLE_FILLING_ENABLED,
    FILTER_DECIMATION_MAGNITUDE,
    FILTER_SPATIAL_ALPHA,
    FILTER_SPATIAL_DELTA,
    FILTER_TEMPORAL_ALPHA,
    FILTER_TEMPORAL_DELTA,
)

from cluster import fit_plane_ransac_from_depth, fit_rectangle_to_plane_points

from rfdetr import RFDETRMedium


PERCEPTION_ROOT = Path(__file__).resolve().parent
DEFAULT_FLANGE_FILE = PERCEPTION_ROOT / "flange_camera_transform.npz"
DEFAULT_ROBOT_IP = "192.168.58.2"
FAIRINO_SDK_PATH = "/home/jeong/fr5sdk/linux"

# Which RealSense to open. Sourced from the project-root config.py (single
# source of truth) so the pick pipeline always pins the D415 even when two
# cameras are attached. Importable both when launched via detection_server.py
# (project root already on sys.path) and when this script runs standalone.
_PROJECT_ROOT = PERCEPTION_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from config import CAMERA_SERIAL
except Exception:
    CAMERA_SERIAL = None

# Mask deprojection cleanup
DEPTH_Z_MIN = 0.05  # m
DEPTH_Z_MAX = 2.5   # m

# Per-object dominant-plane fit (used to fit a min-area rectangle to the
# visible flat face). We do NOT assume a (0,0,1) top-face normal — many
# objects sit tilted on the table and forcing the rectangle into the XY
# shadow gives skewed boxes. RANSAC recovers the actual face normal and
# inlier set, then fit_rectangle_to_plane_points uses that normal as the
# 2D projection basis (mirrors cluster.py's per-subplane handling).
RECT_RANSAC_MAX_ITERS = 300
RECT_RANSAC_INLIER_THRESH = 0.005  # m, depth-noise tolerance on the face
RECT_MIN_INLIER_FRAC = 0.3         # reject if dominant plane is too sparse
RECT_MIN_POINTS = 30               # absolute minimum points for a fit

# Meshcat viewer ------------------------------------------------------------
# Max number of detections we will draw per frame.
MESHCAT_MAX_DETS = 16

# All per-object SAM point clouds are merged into a single PointCloud and
# pushed as ONE websocket message per frame. This is the only reliable way
# we found to keep all clouds visible at once: per-object PointClouds add up
# fast (10 objects * ~45 k points = ~11 MB float32 / frame) and saturate
# the meshcat-server -> browser websocket, which silently clips the
# trailing clouds in each burst. Merging plus an aggressive total-points
# cap keeps the per-frame payload to <1 MB and makes the update atomic.
MESHCAT_MAX_TOTAL_POINTS = 20000

# Point render size in meters.
MESHCAT_POINT_SIZE = 0.005

# Only push to meshcat every Nth camera frame. The OpenCV view stays at
# full FPS; ~15 Hz in the 3D viewer is plenty and gives the browser room
# to actually drain the websocket between updates.
MESHCAT_UPDATE_EVERY_N_FRAMES = 2

# How often (seconds) to print T_base_flange / T_flange_camera / T_base_camera
# in the main loop.
TRANSFORM_PRINT_INTERVAL_S = 1.0

# Per-detection color palette matching live_detect.COLOR_PALETTE (hex list)
_PALETTE_HEX = [
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
]


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
                    dtype=np.float32)


MESHCAT_PALETTE = np.stack([_hex_to_rgb01(h) for h in _PALETTE_HEX], axis=0)


def _fmt_T(T):
    """Compact 4-line string for a 4x4 transform."""
    T = np.asarray(T, dtype=np.float64)
    lines = []
    for row in T:
        lines.append("  [{:+8.4f} {:+8.4f} {:+8.4f} | {:+8.4f}]".format(*row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RealSense init (RGBD aligned to color)
# ---------------------------------------------------------------------------
def get_device():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    if CAMERA_SERIAL:
        for d in devs:
            if d.get_info(rs.camera_info.serial_number) == CAMERA_SERIAL:
                return d
        attached = ", ".join(
            f"{d.get_info(rs.camera_info.name)}"
            f"({d.get_info(rs.camera_info.serial_number)})"
            for d in devs
        )
        raise RuntimeError(
            f"Configured CAMERA_SERIAL={CAMERA_SERIAL} not found. "
            f"Attached devices: {attached}"
        )
    return devs[0]


def init_realsense_rgbd():
    """Start color + depth, align depth to color, configure high-accuracy depth.

    Returns (pipe, align, color_intrinsics_dict, depth_scale_m, filters_dict).
    """
    print("Initializing RealSense camera (RGBD aligned to color) ...")
    cfg = rs.config()
    if CAMERA_SERIAL:
        cfg.enable_device(CAMERA_SERIAL)
        print(f"Pinning RealSense device by serial: {CAMERA_SERIAL}")
    cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT,
                      rs.format.z16, CAMERA_FPS)
    cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT,
                      rs.format.bgr8, CAMERA_FPS)
    while True:
        pipe = None
        try:
            pipe = rs.pipeline()
            profile = pipe.start(cfg)

            color_profile = profile.get_stream(rs.stream.color)\
                .as_video_stream_profile()
            color_intr = color_profile.get_intrinsics()
            color_intrinsics = {
                "fx": color_intr.fx, "fy": color_intr.fy,
                "cx": color_intr.ppx, "cy": color_intr.ppy,
            }
            print(f"Color intrinsics: fx={color_intr.fx:.2f} fy={color_intr.fy:.2f} "
                  f"cx={color_intr.ppx:.2f} cy={color_intr.ppy:.2f}")

            depth_sensor = profile.get_device().first_depth_sensor()
            try:
                depth_sensor.set_option(
                    rs.option.visual_preset,
                    rs.rs400_visual_preset.high_accuracy,
                )
            except Exception as e:
                print(f"[warn] visual_preset: {e}")
            try:
                depth_sensor.set_option(rs.option.depth_units, DEPTH_UNITS)
            except Exception as e:
                print(f"[warn] depth_units: {e}")
            try:
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            except Exception:
                pass
            if depth_sensor.supports(rs.option.laser_power):
                try:
                    lp_range = depth_sensor.get_option_range(rs.option.laser_power)
                    lp_value = float(min(max(LASER_POWER, lp_range.min),
                                         lp_range.max))
                    depth_sensor.set_option(rs.option.laser_power, lp_value)
                except Exception as e:
                    print(f"[warn] laser_power: {e}")

            for s in profile.get_device().sensors:
                if s.get_info(rs.camera_info.name) == "RGB Camera":
                    try:
                        s.set_option(rs.option.enable_auto_exposure, 1)
                    except Exception:
                        pass
                    break

            depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth scale: {depth_scale:.6f} m/unit")

            align = rs.align(rs.stream.color)

            # Build the same filter chain as cluster.py / grasp_detection_server.py.
            #
            # Constraint: this script indexes depth pixel-wise against the
            # color-aligned mask, so any filter that changes depth shape after
            # align would break that. Decimation only changes shape when
            # magnitude > 1; cluster.py currently uses magnitude=1 (a no-op for
            # shape). If FILTER_DECIMATION_MAGNITUDE is ever raised above 1,
            # we skip the decimation filter here and warn — the rest of the
            # chain still matches cluster.py exactly.
            filters = {
                "decimation": None,
                "depth_to_disparity": None,
                "spatial": None,
                "temporal": None,
                "disparity_to_depth": None,
                "hole_filling": None,
            }
            if FILTER_DECIMATION_ENABLED:
                if FILTER_DECIMATION_MAGNITUDE == 1:
                    dec = rs.decimation_filter()
                    dec.set_option(rs.option.filter_magnitude,
                                   FILTER_DECIMATION_MAGNITUDE)
                    filters["decimation"] = dec
                    print(f"Decimation filter: magnitude={FILTER_DECIMATION_MAGNITUDE}")
                else:
                    print(f"[warn] FILTER_DECIMATION_MAGNITUDE="
                          f"{FILTER_DECIMATION_MAGNITUDE} > 1 would break depth/color "
                          f"pixel alignment; skipping decimation in live_detect_3d.py.")
            if FILTER_DISPARITY_ENABLED:
                filters["depth_to_disparity"] = rs.disparity_transform(True)
                filters["disparity_to_depth"] = rs.disparity_transform(False)
                print("Disparity transform enabled")
            if FILTER_SPATIAL_ENABLED:
                sp = rs.spatial_filter()
                sp.set_option(rs.option.filter_smooth_alpha, FILTER_SPATIAL_ALPHA)
                sp.set_option(rs.option.filter_smooth_delta, FILTER_SPATIAL_DELTA)
                filters["spatial"] = sp
                print(f"Spatial filter: alpha={FILTER_SPATIAL_ALPHA}, "
                      f"delta={FILTER_SPATIAL_DELTA}")
            if FILTER_TEMPORAL_ENABLED:
                t = rs.temporal_filter()
                t.set_option(rs.option.filter_smooth_alpha, FILTER_TEMPORAL_ALPHA)
                t.set_option(rs.option.filter_smooth_delta, FILTER_TEMPORAL_DELTA)
                filters["temporal"] = t
                print(f"Temporal filter: alpha={FILTER_TEMPORAL_ALPHA}, "
                      f"delta={FILTER_TEMPORAL_DELTA}")
            if FILTER_HOLE_FILLING_ENABLED:
                filters["hole_filling"] = rs.hole_filling_filter()
                print("Hole filling filter enabled")

            print("Camera ready.")
            return pipe, align, color_intrinsics, depth_scale, filters

        except RuntimeError as e:
            msg = str(e)
            print(f"Init error: {msg}")
            if pipe:
                try:
                    pipe.stop()
                except Exception:
                    pass
            recoverable = [
                "get_xu(ctrl=1)", "Device or resource busy",
                "failed to set power state", "Couldn't resolve requests",
            ]
            if any(err in msg for err in recoverable):
                print("Hardware reset...")
                try:
                    get_device().hardware_reset()
                    time.sleep(2)
                except Exception:
                    time.sleep(2)
            else:
                raise


# ---------------------------------------------------------------------------
# Robot + flange transform
# ---------------------------------------------------------------------------
def load_flange_camera_npz(path):
    """Load T_flange_camera (4x4 m) and the Euler convention used at calib."""
    print(f"Loading flange calibration from {path}")
    data = np.load(str(path), allow_pickle=True)
    T_fc = np.asarray(data["T_flange_camera"], dtype=np.float64)
    euler_conv = (str(data["euler_convention"])
                  if "euler_convention" in data.files else "xyz")
    print(f"T_flange_camera:\n{T_fc}")
    print(f"Euler convention: '{euler_conv}'")
    return T_fc, euler_conv


def connect_robot(ip):
    """Open Fairino RPC. Returns Robot or None on failure (caller falls back)."""
    if FAIRINO_SDK_PATH not in sys.path:
        sys.path.insert(0, FAIRINO_SDK_PATH)
    try:
        from fairino import Robot
    except ImportError as e:
        print(f"[robot] Fairino SDK not importable ({e}); skipping.")
        return None
    try:
        print(f"[robot] Connecting to {ip} ...")
        robot = Robot.RPC(ip)
        time.sleep(1.0)
        err, pose = robot.GetActualToolFlangePose(flag=1)
        if err != 0 or pose is None:
            print(f"[robot] GetActualToolFlangePose at startup failed (err={err})")
            try:
                robot.CloseRPC()
            except Exception:
                pass
            return None
        print(f"[robot] Connected. Initial flange pose: {pose}")
        return robot
    except Exception as e:
        print(f"[robot] Connection error: {e}")
        return None


def get_T_base_flange(robot, euler_conv):
    """Return (T_base_flange 4x4 in metres, raw_pose) or (None, None)."""
    from scipy.spatial.transform import Rotation as R
    try:
        err, pose = robot.GetActualToolFlangePose(flag=1)
    except Exception as e:
        print(f"[robot] read error: {e}")
        return None, None
    if err != 0 or pose is None:
        return None, None
    T = np.eye(4)
    T[:3, 3] = np.asarray(pose[:3], dtype=np.float64) / 1000.0
    T[:3, :3] = R.from_euler(euler_conv, pose[3:], degrees=True).as_matrix()
    return T, pose


# ---------------------------------------------------------------------------
# Mask deprojection (per-detection)
# ---------------------------------------------------------------------------
def deproject_mask_to_camera(mask_uint8, depth_metres, color_intr,
                             stride=1, z_min=DEPTH_Z_MIN, z_max=DEPTH_Z_MAX):
    """Deproject mask pixels into 3D camera-frame points using the pinhole model.

    Parameters:
        mask_uint8      : (H, W) uint8 binary mask aligned to the color image.
        depth_metres    : (H, W) float depth in metres, aligned to color.
        color_intr      : dict with fx, fy, cx, cy.
        stride          : pixel subsample factor (>=1) for speed.
        z_min, z_max    : drop points outside this depth window.

    Returns (K, 3) float32 in camera frame.
    """
    H, W = mask_uint8.shape
    if stride > 1:
        sub = np.zeros_like(mask_uint8)
        sub[::stride, ::stride] = mask_uint8[::stride, ::stride]
        ys, xs = np.nonzero(sub)
    else:
        ys, xs = np.nonzero(mask_uint8)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_metres[ys, xs]
    valid = (z > z_min) & (z < z_max) & np.isfinite(z)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    fx = float(color_intr["fx"]); fy = float(color_intr["fy"])
    cx = float(color_intr["cx"]); cy = float(color_intr["cy"])
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def transform_to_base(pts_cam, T_base_camera):
    if pts_cam is None or pts_cam.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts_h = np.hstack([
        pts_cam.astype(np.float64),
        np.ones((len(pts_cam), 1), dtype=np.float64),
    ])
    return (pts_h @ T_base_camera.T)[:, :3]


# ---------------------------------------------------------------------------
# Rectangle / centroid fit per segmented object
# ---------------------------------------------------------------------------
def fit_top_rectangle(pts_base,
                      max_iters=RECT_RANSAC_MAX_ITERS,
                      inlier_thresh=RECT_RANSAC_INLIER_THRESH,
                      min_inlier_frac=RECT_MIN_INLIER_FRAC,
                      min_points=RECT_MIN_POINTS):
    """Fit a min-area rectangle to the dominant flat face of an object cloud.

    Uses ``cluster.fit_plane_ransac_from_depth`` to find the largest planar
    region without assuming a particular orientation, so tilted faces fit
    correctly. The rectangle is then fit on the inliers using the *fitted*
    plane normal as the 2D projection basis (this mirrors what cluster.py
    does for each detected subplane in ``detect_grasp_planes``).

    Parameters:
        pts_base       : (N, 3) array in base frame.
        max_iters      : RANSAC iteration budget per object.
        inlier_thresh  : meters; point-to-plane distance tolerance.
        min_inlier_frac: reject when the dominant plane covers < this fraction
                         of the SAM cloud (likely a non-planar object).
        min_points     : absolute minimum inlier count.

    Returns dict with:
        - 'centroid' : (3,) mean of plane inliers
        - 'center'   : (3,) min-area-rect center
        - 'corners'  : (4, 3) rectangle corners in 3D
        - 'width'    : meters
        - 'height'   : meters
        - 'angle'    : degrees (cv2.minAreaRect convention)
        - 'normal'   : (3,) fitted face normal (oriented to +Z in base frame)
        - 'd'        : float; plane offset so n.x + d = 0
        - 'top_pts'  : (M, 3) plane inliers used for the fit
    or None if no usable fit was found.
    """
    if pts_base is None or len(pts_base) < min_points:
        return None

    try:
        n_fit, d_fit, inlier_mask = fit_plane_ransac_from_depth(
            np.asarray(pts_base, dtype=np.float64),
            max_iters=max_iters,
            inlier_thresh=inlier_thresh,
        )
    except RuntimeError:
        return None

    inliers = pts_base[inlier_mask]
    if len(inliers) < min_points:
        return None
    if (len(inliers) / float(len(pts_base))) < min_inlier_frac:
        return None

    # Orient the face normal "upward" (+Z in base frame) for visual / yaw
    # consistency across frames. fit_plane_ransac_from_depth already nudges
    # toward +z but the input here is in *base* frame, so re-check.
    n_fit = np.asarray(n_fit, dtype=np.float64)
    if n_fit[2] < 0:
        n_fit = -n_fit
        d_fit = -float(d_fit)

    rect = fit_rectangle_to_plane_points(inliers, n_fit)
    if rect is None:
        return None
    rect = dict(rect)
    rect["centroid"] = inliers.mean(axis=0)
    rect["normal"] = n_fit
    rect["d"] = float(d_fit)
    rect["top_pts"] = inliers
    # print('rect centroid: ', rect["centroid"])
    return rect


# ---------------------------------------------------------------------------
# High-level detection pipeline (importable from other files)
# ---------------------------------------------------------------------------
def detect_2d(model, img_bgr, conf, max_det, roi=None):
    """Run RF-DETR on a frame, optionally restricted to an ROI.

    When ``roi=(x1, y1, x2, y2)`` is provided the image is cropped before
    inference and the returned boxes are remapped back to full-image
    coordinates -- mirrors the ROI handling in ``main()``.

    Returns the list-of-dicts produced by ``live_detect.run_detection``.
    """
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        crop = img_bgr[ry1:ry2, rx1:rx2].copy()
        dets = run_detection(model, crop, conf, max_det)
        for d in dets:
            bx1, by1, bx2, by2 = d["xyxy"]
            d["xyxy"] = [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1]
        return dets
    return run_detection(model, img_bgr, conf, max_det)


def extract_objects_3d(
    img_bgr, depth_metres, color_intr, T_base_camera, dets,
    *,
    sam_predictor=None, sam_device="cpu", sam_autocast_dtype=None,
    depth_stride=1, fit_rect=True,
):
    """Lift 2D detections into per-object 3D point clouds + rectangle fits.

    Given a precomputed detection list this:
      1. Runs SAM2 (if a predictor is supplied) to get per-detection masks.
      2. Deprojects each mask through ``color_intr`` into camera-frame points.
      3. Transforms the points into the base frame via ``T_base_camera``.
      4. Optionally fits a top-face min-area rectangle with
         ``fit_top_rectangle`` (centroid + corners + normal).

    Parameters mirror the inline pipeline in ``main()`` so external callers
    get bit-identical results.

    Returns ``(masks, per_obj_pts_base, per_obj_rect)``:
        masks            : list[ndarray uint8 (H, W)] or None.
        per_obj_pts_base : list[ndarray (Ki, 3) float] in base frame.
        per_obj_rect     : list[dict | None]; each entry aligns with ``dets``.
    Empty lists are returned when SAM is disabled or yields no usable masks.
    """
    masks = None
    if sam_predictor is not None and dets:
        try:
            masks = run_sam2_on_dets(
                sam_predictor, img_bgr, dets, sam_device, sam_autocast_dtype,
            )
        except Exception as e:
            print(f"[SAM2] error: {e}")
            masks = None

    per_obj_pts_base = []
    per_obj_rect = []
    if masks is not None and len(masks) == len(dets):
        stride = max(1, int(depth_stride))
        for m in masks:
            pts_cam = deproject_mask_to_camera(
                m, depth_metres, color_intr, stride=stride,
            )
            pts_base = transform_to_base(pts_cam, T_base_camera)
            per_obj_pts_base.append(pts_base)
            per_obj_rect.append(fit_top_rectangle(pts_base) if fit_rect else None)
    return masks, per_obj_pts_base, per_obj_rect


def detect_objects_3d(
    model, img_bgr, depth_metres, color_intr, T_base_camera,
    *,
    conf=0.5, max_det=30, roi=None,
    sam_predictor=None, sam_device="cpu", sam_autocast_dtype=None,
    depth_stride=1, fit_rect=True,
):
    """End-to-end RF-DETR + SAM2 + 3D extraction for a single RGB-D frame.

    Convenience wrapper around :func:`detect_2d` and
    :func:`extract_objects_3d` for callers that don't need temporal
    averaging or any other glue between the two stages.

    Returns ``(dets, masks, per_obj_pts_base, per_obj_rect)``.
    """
    dets = detect_2d(model, img_bgr, conf, max_det, roi=roi)
    masks, per_obj_pts_base, per_obj_rect = extract_objects_3d(
        img_bgr, depth_metres, color_intr, T_base_camera, dets,
        sam_predictor=sam_predictor,
        sam_device=sam_device,
        sam_autocast_dtype=sam_autocast_dtype,
        depth_stride=depth_stride,
        fit_rect=fit_rect,
    )
    return dets, masks, per_obj_pts_base, per_obj_rect


# ---------------------------------------------------------------------------
# Meshcat live viewer
# ---------------------------------------------------------------------------
class LiveMeshcatViewer:
    """3D viewer: one merged point cloud + per-object rectangles.

    Scene layout (all in base frame):
        base_frame             -- RGB triad at the robot base origin
        camera_frame           -- triad placed at T_base_camera
        clouds                 -- ONE PointCloud holding every object's points
        rect_<i>/centroid      -- green sphere at top-face centroid
        rect_<i>/rect_center   -- yellow sphere at min-area-rect center
        rect_<i>/corner_<j>    -- 4 small corner spheres (palette colour)
        rect_<i>/edge_<j>      -- 4 rectangle edges (palette colour)

    Why one merged cloud? Per-object PointClouds add up to ~11 MB float32
    per frame at 10 objects * ~45 k points each, which saturates the
    meshcat-server -> browser websocket and causes the trailing clouds in
    each burst to silently disappear from the scene. Merging gives a
    single atomic websocket message that either fully arrives or doesn't,
    and combined with a total-points cap stays well under any drain rate.
    """

    def __init__(self, max_dets=MESHCAT_MAX_DETS,
                 max_total_points=MESHCAT_MAX_TOTAL_POINTS,
                 point_size=MESHCAT_POINT_SIZE,
                 base_triad_size=0.15, camera_triad_size=0.08):
        import meshcat
        import meshcat.geometry as g
        self.g = g
        self.max_dets = int(max_dets)
        self.max_total_points = int(max_total_points)
        self.point_size = float(point_size)
        self._rng = np.random.default_rng(0xC10D5)

        self.vis = meshcat.Visualizer()
        try:
            self.vis.open()
        except Exception as e:
            print(f"[meshcat] open() failed (browser will need to load url): {e}")
        try:
            print(f"[meshcat] viewer at: {self.vis.url()}")
        except Exception:
            pass

        self.vis.delete()
        self.vis["base_frame"].set_object(g.triad(base_triad_size))
        self.vis["camera_frame"].set_object(g.triad(camera_triad_size))
        self._last_n_rects = 0

    @staticmethod
    def _translation_T(p):
        T = np.eye(4)
        T[:3, 3] = np.asarray(p, dtype=np.float64)
        return T

    def _build_merged_cloud(self, per_obj_pts_base, n):
        """Concatenate all object clouds into one (3, N) positions + colors.

        Total points across all objects is bounded by ``max_total_points``.
        Each object gets at most ``max_total_points // k`` points (with a
        200-point floor so small objects stay visible), where k is the
        number of non-empty objects this frame. Returns (positions, colors)
        as (3, N) float32 arrays, or None if every object is empty.
        """
        nonempty = [(i, p) for i, p in enumerate(per_obj_pts_base[:n])
                    if p is not None and len(p) > 0]
        if not nonempty:
            return None

        per_cloud_cap = max(200, self.max_total_points // len(nonempty))

        pos_chunks = []
        col_chunks = []
        for i, pts in nonempty:
            if len(pts) > per_cloud_cap:
                idx = self._rng.choice(len(pts),
                                       size=per_cloud_cap, replace=False)
                pts = pts[idx]
            pos_chunks.append(pts.astype(np.float32))
            rgb = MESHCAT_PALETTE[i % len(MESHCAT_PALETTE)]
            col_chunks.append(np.broadcast_to(rgb, (len(pts), 3))
                              .astype(np.float32))

        positions = np.concatenate(pos_chunks, axis=0).T
        colors = np.concatenate(col_chunks, axis=0).T
        return positions, colors

    def _draw_rectangle(self, grp_path, rect, palette_rgb):
        """Centroid + rect center + 4 corner spheres + 4 edges under grp_path."""
        g = self.g
        color_hex = (int(palette_rgb[0] * 255) << 16
                     | int(palette_rgb[1] * 255) << 8
                     | int(palette_rgb[2] * 255))

        centroid = np.asarray(rect["centroid"], dtype=np.float64)
        self.vis[f"{grp_path}/centroid"].set_object(
            g.Sphere(0.012), g.MeshBasicMaterial(color=0x00FF00),
        )
        self.vis[f"{grp_path}/centroid"].set_transform(self._translation_T(centroid))

        if rect.get("corners") is None:
            return

        rect_center = np.asarray(rect["center"], dtype=np.float64)
        self.vis[f"{grp_path}/rect_center"].set_object(
            g.Sphere(0.008), g.MeshBasicMaterial(color=0xFFFF00),
        )
        self.vis[f"{grp_path}/rect_center"].set_transform(self._translation_T(rect_center))

        corners = np.asarray(rect["corners"], dtype=np.float64)
        for j in range(4):
            self.vis[f"{grp_path}/corner_{j}"].set_object(
                g.Sphere(0.005), g.MeshBasicMaterial(color=color_hex),
            )
            self.vis[f"{grp_path}/corner_{j}"].set_transform(
                self._translation_T(corners[j])
            )
            seg = np.stack([corners[j], corners[(j + 1) % 4]],
                           axis=1).astype(np.float32)
            self.vis[f"{grp_path}/edge_{j}"].set_object(
                g.Line(g.PointsGeometry(seg),
                       g.LineBasicMaterial(color=color_hex, linewidth=3)),
            )

    def update(self, per_obj_pts_base, per_obj_rect=None, T_base_camera=None):
        g = self.g
        n = min(len(per_obj_pts_base), self.max_dets)

        # 1) Merged cloud -- one atomic websocket message for all points.
        merged = self._build_merged_cloud(per_obj_pts_base, n)
        if merged is not None:
            positions, colors = merged
            self.vis["clouds"].set_object(
                g.PointCloud(position=positions, color=colors,
                             size=self.point_size),
            )
        else:
            self.vis["clouds"].delete()

        # 2) Per-object rectangles. Prune slots that disappeared, then
        # delete + redraw each surviving slot. Rect children are tiny
        # (single spheres + 2-vertex lines) so the delete/recreate
        # micro-race here is harmless even if it occasionally flickers.
        for i in range(n, self._last_n_rects):
            self.vis[f"rect_{i}"].delete()

        for i in range(n):
            grp_path = f"rect_{i}"
            rect = (per_obj_rect[i]
                    if per_obj_rect is not None and i < len(per_obj_rect)
                    else None)
            self.vis[grp_path].delete()
            if rect is None:
                continue
            self._draw_rectangle(grp_path, rect,
                                 MESHCAT_PALETTE[i % len(MESHCAT_PALETTE)])

        # 3) Camera triad transform (geometry is created in __init__).
        if T_base_camera is not None:
            self.vis["camera_frame"].set_transform(
                np.asarray(T_base_camera, dtype=np.float64)
            )

        self._last_n_rects = n

    def close(self):
        # The browser holds the viewer; no resources to release on our side.
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Live RF-DETR + SAM2 + 3D segmentation (meshcat)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use the COCO pretrained RF-DETR instead of the fine-tuned weights")
    parser.add_argument("--weights", type=str, default=None,
                        help="Custom RF-DETR checkpoint path (overrides --pretrained)")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--sam", action="store_true",
                        help="Enable SAM2 (required to get masks for 3D segmentation)")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--flange-file", default=str(DEFAULT_FLANGE_FILE))
    parser.add_argument("--depth-stride", type=int, default=1,
                        help="Subsample mask pixels by this stride before deprojecting (>=1)")
    args = parser.parse_args()

    if not args.sam:
        print("[note] --sam is required to populate the 3D view with masks. "
              "Running anyway (2D detections only).")

    if args.weights:
        weights = Path(args.weights)
    elif args.pretrained:
        weights = PRETRAINED_WEIGHTS
    else:
        weights = FINETUNED_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"Loading RF-DETR Medium (resolution={args.resolution}, "
          f"weights={weights.name}) ...")
    model = RFDETRMedium(pretrain_weights=str(weights),
                         resolution=args.resolution)
    model.optimize_for_inference()
    if args.pretrained and not args.weights:
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
    if args.sam:
        if not SAM2_CKPT.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CKPT}")
        import torch
        sam_device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_autocast_dtype = _pick_autocast_dtype()
        print(f"Loading SAM2 ({SAM2_CKPT.name}) on {sam_device} ...")
        sam_predictor = load_sam2_predictor(SAM2_CKPT, SAM2_CFG, sam_device)
        print("SAM2 ready.")
    use_sam = args.sam

    # Flange transform + robot connection (with safe fallback) --------------
    flange_path = Path(args.flange_file)
    if not flange_path.exists():
        raise FileNotFoundError(f"Flange calibration not found: {flange_path}")
    T_flange_camera, euler_conv = load_flange_camera_npz(flange_path)
    print("\n=== T_flange_camera (static, from calibration npz) ===")
    print(_fmt_T(T_flange_camera))

    robot = connect_robot(args.robot_ip)
    if robot is None:
        print("[warn] Robot unreachable; falling back to fixed "
              "cluster.T_base_camera (eye-to-hand assumption).")
        print("\n=== T_base_camera (fallback, fixed) ===")
        print(_fmt_T(cluster.T_base_camera))
    T_base_camera_fallback = cluster.T_base_camera.copy()
    last_T_base_camera = T_base_camera_fallback
    last_T_base_flange = None
    pose_age = None
    last_transform_print = 0.0

    # Camera ---------------------------------------------------------------
    pipe, align, color_intr, depth_scale, filters = init_realsense_rgbd()

    # Meshcat --------------------------------------------------------------
    mc_viewer = LiveMeshcatViewer()
    use_3d = True

    # OpenCV viewer + trackbars (UX matches live_detect.py) ----------------
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)
    cv2.createTrackbar("Conf x100",     WINDOW_NAME, 50, 100, lambda v: None)
    cv2.createTrackbar("Max Det",       WINDOW_NAME, 30, 100, lambda v: None)
    cv2.createTrackbar("Avg Window",    WINDOW_NAME,  5,  30, lambda v: None)
    cv2.createTrackbar("Min Hit% x100", WINDOW_NAME, 40, 100, lambda v: None)

    roi_state = RoiState()
    cv2.setMouseCallback(WINDOW_NAME, roi_state.mouse_cb)

    averager = DetectionAverager(window=5, iou_thresh=0.3, min_hit_ratio=0.4)
    use_avg = False

    fps_filter = 0.0
    prev_time = time.time()
    mc_frame_counter = 0

    print("\n--- Controls ---")
    print("  Left-drag   : ROI       Right-click : clear ROI")
    print("  t/s/q       : toggle averaging / SAM2 / quit")
    print("  p           : toggle meshcat viewer updates\n")

    try:
        while True:
            try:
                frames = pipe.wait_for_frames()
            except RuntimeError as e:
                print(f"[camera] {e}")
                continue
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Apply the same filter chain order as cluster.py:
            #   decimation -> [disparity] -> spatial -> temporal ->
            #   [disparity_back] -> hole_filling
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
            depth_frame = depth

            img = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_metres = depth_raw * float(depth_scale)
            H, W = img.shape[:2]

            # Live flange pose -> T_base_camera
            if robot is not None:
                T_bf, _pose = get_T_base_flange(robot, euler_conv)
                if T_bf is not None:
                    last_T_base_flange = T_bf
                    last_T_base_camera = T_bf @ T_flange_camera
                    pose_age = 0
                else:
                    pose_age = (pose_age + 1) if pose_age is not None else 1
                    if pose_age == 1 or pose_age % 30 == 0:
                        print(f"[robot] stale flange pose (age={pose_age})")
            T_base_camera = last_T_base_camera

            # Throttled transform dump
            now_t = time.time()
            if (now_t - last_transform_print) >= TRANSFORM_PRINT_INTERVAL_S:
                last_transform_print = now_t
                tag = "live" if (robot is not None and pose_age == 0) else "fallback"
                print(f"\n--- Transforms [{tag}] (m) ---")
                if last_T_base_flange is not None:
                    print("T_base_flange (live, from Fairino):")
                    print(_fmt_T(last_T_base_flange))
                print("T_flange_camera (static, from npz):")
                print(_fmt_T(T_flange_camera))
                print("T_base_camera = T_base_flange @ T_flange_camera:")
                print(_fmt_T(T_base_camera))

            # Trackbar params
            conf = max(cv2.getTrackbarPos("Conf x100", WINDOW_NAME), 1) / 100.0
            max_det = max(cv2.getTrackbarPos("Max Det", WINDOW_NAME), 1)
            avg_win = max(cv2.getTrackbarPos("Avg Window", WINDOW_NAME), 1)
            min_hit_pct = max(cv2.getTrackbarPos("Min Hit% x100", WINDOW_NAME), 1) / 100.0
            averager.window = avg_win
            averager.min_hit_ratio = min_hit_pct
            averager.max_miss = avg_win

            # RF-DETR detect, ROI-aware (matches live_detect.py:466-482).
            roi = roi_state.roi
            dets = detect_2d(model, img, conf, max_det, roi=roi)

            if use_avg:
                dets = averager.update(dets)
            else:
                averager.reset()

            # SAM2 + per-mask deprojection + transform to base + rect fit.
            # Temporal averaging happens between detect_2d and this call,
            # which is why we don't use the all-in-one detect_objects_3d
            # here -- but external callers can.
            masks, per_obj_pts_base, per_obj_rect = extract_objects_3d(
                img, depth_metres, color_intr, T_base_camera, dets,
                sam_predictor=sam_predictor if use_sam else None,
                sam_device=sam_device,
                sam_autocast_dtype=sam_autocast_dtype,
                depth_stride=args.depth_stride,
            )
            for _idx, _pts_base in enumerate(per_obj_pts_base):
                print(f"pts_base.shape for idx {_idx}: {_pts_base.shape}")
            print('len per_obj_pts_base: ', len(per_obj_pts_base))
            print('len per_obj_rect: ', len(per_obj_rect))

            # Rate-limited meshcat push: the OpenCV view stays at full FPS,
            # but the 3D viewer runs at ~CAM_FPS / N to give the browser
            # room to drain the websocket between updates.
            mc_frame_counter += 1
            if use_3d and (mc_frame_counter % MESHCAT_UPDATE_EVERY_N_FRAMES == 0):
                mc_viewer.update(per_obj_pts_base,
                                 per_obj_rect=per_obj_rect,
                                 T_base_camera=T_base_camera)

            # OpenCV 2D view
            vis = img.copy()
            if roi is not None:
                cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]),
                              (0, 255, 255), 2)
            if roi_state.drag_preview is not None:
                dp = roi_state.drag_preview
                overlay = vis.copy()
                cv2.rectangle(overlay, (dp[0], dp[1]), (dp[2], dp[3]),
                              (0, 255, 255), 2)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            vis = draw_detections(vis, dets, class_names, masks=masks)

            # FPS / HUD
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps_filter = 0.9 * fps_filter + 0.1 * (1.0 / dt if dt > 0 else 0)

            n_pcd = sum(1 for p in per_obj_pts_base if p is not None and len(p) > 0)
            n_rect = sum(1 for r in per_obj_rect if r is not None)
            sam_str = "SAM:on" if (use_sam and sam_predictor is not None) else \
                      ("SAM:off" if sam_predictor is not None else "SAM:--")
            if robot is None:
                pose_str = "pose:fb"
            elif pose_age == 0:
                pose_str = "pose:live"
            else:
                pose_str = f"pose:stale({pose_age})"
            hud = (f"FPS {fps_filter:.1f} | {len(dets)} det | "
                   f"{n_pcd} clouds | {n_rect} rects | {sam_str} | "
                   f"{pose_str} | 3D:{'on' if use_3d else 'off'}")
            cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)

            help_txt = ("Left-drag: ROI | Right-click: clear | "
                        "t: avg | s: SAM | p: 3D | q: quit")
            cv2.putText(vis, help_txt, (10, CAMERA_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, help_txt, (10, CAMERA_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                use_avg = not use_avg
                if not use_avg:
                    averager.reset()
                print(f"Temporal averaging: {'ON' if use_avg else 'OFF'}")
            elif key == ord("s"):
                if sam_predictor is None:
                    print("[SAM2] not loaded - relaunch with --sam to enable.")
                else:
                    use_sam = not use_sam
                    print(f"SAM2: {'ON' if use_sam else 'OFF'}")
            elif key == ord("p"):
                use_3d = not use_3d
                print(f"Meshcat updates: {'ON' if use_3d else 'OFF'}")

    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            mc_viewer.close()
        except Exception:
            pass
        if robot is not None:
            try:
                robot.CloseRPC()
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    main()

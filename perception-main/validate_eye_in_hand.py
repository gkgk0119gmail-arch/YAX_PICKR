"""
Validate an eye-in-hand calibration (T_flange_camera) using the robot.

Idea
----
Put ONE static object on the table. From several different arm poses, capture a
point cloud, transform it to base frame using

    T_base_camera = T_base_flange(now) @ T_flange_camera

and run `cluster.detect_grasp_planes` to extract the object's centroid in the
base frame. Because the object is physically static, every captured centroid
should land at (approximately) the same 3D point. The spread across samples is
a direct measure of the calibration quality.

Usage
-----
    python validate_eye_in_hand.py [--robot-ip 192.168.58.2] \
                                   [--flange-file flange_camera_transform.npz]

Controls (window or terminal):
    ENTER  capture a sample at the current arm pose
    u      undo last sample
    r      reset all samples
    q      quit and print the final report

Pass --vis to pop up a window showing the color image with the ROI bbox and
detection centroids overlaid (projected from base frame back into pixels).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

import cluster

# --- Fairino SDK -------------------------------------------------------------
_FAIRINO_SDK_PATH = "/home/jeong/fr5sdk/linux"
if _FAIRINO_SDK_PATH not in sys.path:
    sys.path.insert(0, _FAIRINO_SDK_PATH)
from fairino import Robot  # noqa: E402

# --- RealSense settings (match cluster.py) -----------------------------------
CAMERA_WIDTH = cluster.CAMERA_WIDTH
CAMERA_HEIGHT = cluster.CAMERA_HEIGHT
CAMERA_FPS = cluster.CAMERA_FPS
DEPTH_UNITS = cluster.DEPTH_UNITS
LASER_POWER = cluster.LASER_POWER

# Match-radius between captures: a centroid in a new sample is matched to the
# anchor centroid (sample 0) if it lies within this distance. Larger than
# expected calibration error, smaller than typical inter-object spacing.
MATCH_RADIUS_M = 0.08


def _get_first_device():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]


def _print_device_info(profile):
    """Print model, serial, firmware, USB type. Warn on USB 2.x."""
    dev = profile.get_device()

    def _safe(info):
        try:
            return dev.get_info(info)
        except Exception:
            return "unknown"

    name = _safe(rs.camera_info.name)
    serial = _safe(rs.camera_info.serial_number)
    fw = _safe(rs.camera_info.firmware_version)
    usb = _safe(rs.camera_info.usb_type_descriptor)
    print(f"[device] {name}  serial={serial}  fw={fw}  usb={usb}")
    if isinstance(usb, str) and usb.startswith("2."):
        print(
            "[device] WARNING: USB 2.x detected. 1280x720 @ 30 fps RGB+depth will not "
            "fit USB 2 bandwidth and wait_for_frames() will time out.\n"
            "         Either plug into a USB 3 port / cable, or lower resolution with "
            "--width 848 --height 480."
        )


def _warm_up(pipe, n_frames=30, frame_timeout_ms=15000, max_consec_failures=3):
    """Pull n_frames; tolerate a few startup misses (D405 sometimes drops the
    first few frames after preset/option changes)."""
    consec_fail = 0
    got = 0
    for i in range(n_frames):
        try:
            pipe.wait_for_frames(timeout_ms=frame_timeout_ms)
            consec_fail = 0
            got += 1
        except RuntimeError as e:
            consec_fail += 1
            print(f"[warm-up] frame {i}: {e}  (consec fail={consec_fail})")
            if consec_fail >= max_consec_failures:
                raise
    return got


def setup_realsense(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
    """Configure the RealSense camera with the same reconnection / hardware
    reset retry mechanism used by cluster.py and grasp_detection_server.py."""
    print(f"Initializing RealSense camera ({width}x{height} @ {fps} fps)...")
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipe = None
    profile = None
    color_intrinsics = None
    depth_to_color_ext = None
    while True:
        try:
            pipe = rs.pipeline()
            profile = pipe.start(cfg)
            _print_device_info(profile)

            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            ci = color_profile.get_intrinsics()
            color_intrinsics = {'fx': ci.fx, 'fy': ci.fy, 'cx': ci.ppx, 'cy': ci.ppy}
            d2c = depth_profile.get_extrinsics_to(color_profile)
            depth_to_color_ext = {
                'R': np.array(d2c.rotation).reshape(3, 3),
                't': np.array(d2c.translation),
            }

            depth_sensor = profile.get_device().first_depth_sensor()
            try:
                depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            except Exception as e:
                print(f"[warn] could not set visual_preset=high_accuracy: {e}")
            if depth_sensor.supports(rs.option.depth_units):
                try:
                    du_range = depth_sensor.get_option_range(rs.option.depth_units)
                    du_value = float(min(max(DEPTH_UNITS, du_range.min), du_range.max))
                    depth_sensor.set_option(rs.option.depth_units, du_value)
                except Exception as e:
                    print(f"[warn] could not set depth_units: {e}")
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            # laser_power isn't supported on every device (e.g. D405)
            if depth_sensor.supports(rs.option.laser_power):
                try:
                    lp_range = depth_sensor.get_option_range(rs.option.laser_power)
                    lp_value = float(min(max(LASER_POWER, lp_range.min), lp_range.max))
                    depth_sensor.set_option(rs.option.laser_power, lp_value)
                except Exception as e:
                    print(f"[warn] could not set laser_power: {e}")
            else:
                print("[info] depth sensor does not support laser_power option (normal on D405)")
            for s in profile.get_device().sensors:
                if s.get_info(rs.camera_info.name) == "RGB Camera":
                    try:
                        s.set_option(rs.option.enable_auto_exposure, 1)
                    except Exception as e:
                        print(f"[warn] could not enable RGB auto-exposure: {e}")
                    break

            print("Camera initialized successfully.")
            break

        except RuntimeError as e:
            msg = str(e)
            print(f"Initialization error: {msg}")

            recoverable_errors = [
                "get_xu(ctrl=1)",
                "Device or resource busy",
                "failed to set power state",
                "Couldn't resolve requests",
            ]

            if any(err in msg for err in recoverable_errors):
                print("Camera busy or in bad state, doing hardware reset...")
                if pipe:
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                try:
                    dev = _get_first_device()
                    dev.hardware_reset()
                    print("Reset triggered, waiting 2s for re-enumeration...")
                    time.sleep(2)
                except Exception as reset_err:
                    print(f"Failed to reset device: {reset_err}")
                    time.sleep(2)
            else:
                print(f"Fatal error initializing camera: {e}")
                sys.exit(1)

    decimation = temporal = None
    if cluster.FILTER_DECIMATION_ENABLED:
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, cluster.FILTER_DECIMATION_MAGNITUDE)
    if cluster.FILTER_TEMPORAL_ENABLED:
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, cluster.FILTER_TEMPORAL_ALPHA)
        temporal.set_option(rs.option.filter_smooth_delta, cluster.FILTER_TEMPORAL_DELTA)

    print("[camera] warming up ...")
    try:
        _warm_up(pipe)
    except RuntimeError as e:
        print(
            f"[camera] warm-up failed: {e}\n"
            "  Likely causes:\n"
            "    - USB 2.x cable/port (see [device] line above)\n"
            "    - Requested {w}x{h} @ {f}fps too high for this link\n"
            "  Retry with smaller stream:  --width 848 --height 480 --fps 30"
            .format(w=width, h=height, f=fps)
        )
        try:
            pipe.stop()
        except Exception:
            pass
        sys.exit(1)
    return pipe, decimation, temporal, color_intrinsics, depth_to_color_ext


def capture_pts_cam(pipe, decimation, temporal):
    """Capture one filtered point cloud in the camera frame."""
    pc = rs.pointcloud()
    frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not depth:
        return None, None
    if decimation:
        depth = decimation.process(depth)
    if temporal:
        depth = temporal.process(depth)
    points = pc.calculate(depth)
    pts_cam = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    mask = (pts_cam[:, 2] > 0) & np.isfinite(pts_cam[:, 2])
    color_image = np.asanyarray(color.get_data()) if color else None
    return pts_cam[mask], color_image


def get_flange_pose_matrix(robot, euler_conv):
    """Return current T_base_flange (m) and the raw [x,y,z,rx,ry,rz] (mm, deg)."""
    err, pose = robot.GetActualToolFlangePose(flag=1)
    if err != 0 or pose is None:
        raise RuntimeError(f"GetActualToolFlangePose failed: err={err}")
    T = np.eye(4)
    T[:3, 3] = np.asarray(pose[:3], dtype=np.float64) / 1000.0
    T[:3, :3] = R.from_euler(euler_conv, pose[3:], degrees=True).as_matrix()
    return T, pose


def detect_object_centroid_base(pts_cam, T_base_camera):
    """
    Run cluster's detector with the live T_base_camera and return the list of
    detected centroids in base frame (one per subplane). Bounding box, plane,
    and other params come from cluster module defaults.
    """
    result = cluster.detect_grasp_planes(
        pts_cam,
        T_base_camera_override=T_base_camera,
    )
    centers = [np.asarray(c, dtype=np.float64) for c in result["centers"]]
    return centers, result


def _project_base_to_pixel(pts_base, T_base_camera, color_intrinsics, d2c_ext):
    """Project (N,3) base-frame points to color-image pixel coordinates."""
    pts_base = np.atleast_2d(np.asarray(pts_base, dtype=np.float64))
    if pts_base.size == 0:
        return np.zeros((0, 2))
    T_camera_base = np.linalg.inv(T_base_camera)
    pts_h = np.hstack([pts_base, np.ones((len(pts_base), 1))])
    pts_dcam = (pts_h @ T_camera_base.T)[:, :3].astype(np.float32)
    return cluster.project_pts_to_color_pixels(pts_dcam, color_intrinsics, d2c_ext)


def visualize_capture(window, color_image, T_base_camera, centroids, plane,
                      color_intrinsics, d2c_ext, bbox_xy, sample_idx):
    """Draw ROI bbox + centroids over the color image and imshow it."""
    if color_image is None or color_intrinsics is None:
        return
    img = color_image.copy()
    H, W = img.shape[:2]

    # ROI bbox: 4 base-frame corners on the dominant plane
    if bbox_xy is not None and plane is not None:
        (xmn, xmx), (ymn, ymx) = bbox_xy
        n = np.asarray(plane.get('n'), dtype=np.float64)
        d = float(plane.get('d', 0.0))
        nz = float(n[2]) if abs(n[2]) > 1e-6 else 1e-6

        def _z(x, y):
            return (-d - n[0] * x - n[1] * y) / nz

        corners_base = np.array([
            [xmn, ymn, _z(xmn, ymn)],
            [xmx, ymn, _z(xmx, ymn)],
            [xmx, ymx, _z(xmx, ymx)],
            [xmn, ymx, _z(xmn, ymx)],
        ], dtype=np.float64)
        corners_px = _project_base_to_pixel(corners_base, T_base_camera, color_intrinsics, d2c_ext)
        for i in range(4):
            p1 = tuple(np.round(corners_px[i]).astype(int))
            p2 = tuple(np.round(corners_px[(i + 1) % 4]).astype(int))
            cv2.line(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "ROI (bbox_xy)", tuple(np.round(corners_px[0]).astype(int) + np.array([4, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Centroids
    if len(centroids) > 0:
        cent_arr = np.stack(centroids, axis=0)
        cent_px = _project_base_to_pixel(cent_arr, T_base_camera, color_intrinsics, d2c_ext)
        for i, c in enumerate(centroids):
            px = int(round(cent_px[i, 0]))
            py = int(round(cent_px[i, 1]))
            in_img = (0 <= px < W) and (0 <= py < H)
            color = (0, 0, 255) if in_img else (0, 165, 255)
            cv2.drawMarker(img, (px, py), color, markerType=cv2.MARKER_CROSS,
                           markerSize=22, thickness=2, line_type=cv2.LINE_AA)
            cv2.circle(img, (px, py), 9, color, 2, cv2.LINE_AA)
            label = f"#{i} ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) m"
            tx = max(5, min(W - 280, px + 14))
            ty = max(20, min(H - 5, py - 14))
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)

    banner = f"sample #{sample_idx}  detections={len(centroids)}"
    cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window, img)
    cv2.waitKey(1)


def report(samples):
    """Print spread statistics for matched centroids across samples."""
    if len(samples) < 2:
        print("\n[report] Need at least 2 samples to compute spread.")
        return

    anchor_centroids = samples[0]["centroids"]
    if not anchor_centroids:
        print("\n[report] First sample had no detections; cannot anchor.")
        return

    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT  ({len(samples)} samples)")
    print("=" * 70)

    for ai, anchor in enumerate(anchor_centroids):
        matched = [anchor]
        match_info = [(0, anchor)]
        for si in range(1, len(samples)):
            cents = samples[si]["centroids"]
            if not cents:
                continue
            dists = [np.linalg.norm(c - anchor) for c in cents]
            j = int(np.argmin(dists))
            if dists[j] <= MATCH_RADIUS_M:
                matched.append(cents[j])
                match_info.append((si, cents[j]))

        if len(matched) < 2:
            print(f"\nAnchor #{ai} at {anchor.round(4)}: no matches in other samples.")
            continue

        M = np.stack(matched, axis=0)
        mean = M.mean(axis=0)
        residuals = M - mean
        per_axis_std_mm = residuals.std(axis=0) * 1000.0
        per_sample_err_mm = np.linalg.norm(residuals, axis=1) * 1000.0

        print(f"\n--- Anchor #{ai} ---")
        print(f"  matched across {len(matched)}/{len(samples)} samples")
        print(f"  mean centroid (base, m): "
              f"({mean[0]:+.4f}, {mean[1]:+.4f}, {mean[2]:+.4f})")
        print(f"  std per axis  (mm)     : "
              f"x={per_axis_std_mm[0]:.2f}  y={per_axis_std_mm[1]:.2f}  z={per_axis_std_mm[2]:.2f}")
        print(f"  per-sample 3D error (mm): "
              f"mean={per_sample_err_mm.mean():.2f}  "
              f"max={per_sample_err_mm.max():.2f}")
        for (si, c), e in zip(match_info, per_sample_err_mm):
            fp = samples[si]["flange_pose_raw"]  
            print(f"    sample {si}: centroid=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  "
                  f"err={e:5.2f}mm  flange(mm,deg)=[{fp[0]:.1f},{fp[1]:.1f},{fp[2]:.1f},"
                  f"{fp[3]:.1f},{fp[4]:.1f},{fp[5]:.1f}]")

    print("\nInterpretation:")
    print("  < 5 mm   excellent")
    print("  5-15 mm  acceptable for most pick tasks")
    print("  > 15 mm  re-check calibration (wrong Euler convention? noisy poses?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.58.2")
    ap.add_argument("--flange-file", default="flange_camera_transform.npz")
    ap.add_argument("--width", type=int, default=CAMERA_WIDTH,
                    help=f"Camera stream width (default: {CAMERA_WIDTH}). "
                         "Try 848 if using D405 on USB 2 or seeing 'Frame didn't arrive'.")
    ap.add_argument("--height", type=int, default=CAMERA_HEIGHT,
                    help=f"Camera stream height (default: {CAMERA_HEIGHT}). Try 480 to pair with --width 848.")
    ap.add_argument("--fps", type=int, default=CAMERA_FPS,
                    help=f"Camera FPS (default: {CAMERA_FPS}).")
    ap.add_argument("--vis", action="store_true",
                    help="Show a window with the color image, ROI bbox, and detection centroids "
                         "overlaid after each capture.")
    args = ap.parse_args()

    cal_path = Path(args.flange_file)
    if not cal_path.exists():
        print(f"ERROR: {cal_path} not found. Run realsense_calib_eye_in_hand.py first.")
        sys.exit(1)

    cal = np.load(cal_path, allow_pickle=True)
    T_flange_camera = cal["T_flange_camera"]
    euler_conv = str(cal["euler_convention"]) if "euler_convention" in cal.files else "xyz"
    print(f"[cal] Loaded {cal_path}")
    print(f"[cal] T_flange_camera (m):\n{T_flange_camera}")
    print(f"[cal] euler convention: '{euler_conv}'")

    print(f"[robot] Connecting to {args.robot_ip} ...")
    robot = Robot.RPC(args.robot_ip)
    time.sleep(1.0)

    pipe, decimation, temporal, color_intrinsics, d2c_ext = setup_realsense(
        args.width, args.height, args.fps
    )

    vis_window = "validate_eye_in_hand" if args.vis else None
    if vis_window:
        cv2.namedWindow(vis_window, cv2.WINDOW_NORMAL)

    samples = []
    print("\nControls: ENTER capture | u undo | r reset | q quit & report")
    try:
        while True:
            line = input(f"[{len(samples)} samples] > ").strip().lower()
            if line in ("q", "quit", "exit"):
                break
            if line == "u":
                if samples:
                    samples.pop()
                    print(f"[undo] now {len(samples)} samples")
                continue
            if line == "r":
                samples = []
                print("[reset]")
                continue

            t0 = time.time()
            T_base_flange, raw_pose = get_flange_pose_matrix(robot, euler_conv)
            T_base_camera = T_base_flange @ T_flange_camera
            print('t_base_flange: ', T_base_flange)
            print('t_flange_camera: ', T_flange_camera)
            print('t_base_camera: ', T_base_camera)

            pts_cam, color_image = capture_pts_cam(pipe, decimation, temporal)
            if pts_cam is None or len(pts_cam) < 1000:
                print("[capture] no/too few points, skipping")
                continue

            centroids, result = detect_object_centroid_base(pts_cam, T_base_camera)

            print(f"[capture] flange (mm,deg) "
                  f"[{raw_pose[0]:.1f}, {raw_pose[1]:.1f}, {raw_pose[2]:.1f}, "
                  f"{raw_pose[3]:.1f}, {raw_pose[4]:.1f}, {raw_pose[5]:.1f}]  "
                  f"{len(centroids)} detection(s)  ({(time.time()-t0)*1000:.0f}ms)")
            for i, c in enumerate(centroids):
                print(f"    centroid[{i}] base (m): "
                      f"({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")

            samples.append({
                "flange_pose_raw": np.asarray(raw_pose, dtype=np.float64),
                "T_base_flange": T_base_flange,
                "T_base_camera": T_base_camera,
                "centroids": centroids,
            })

            if vis_window:
                visualize_capture(
                    vis_window,
                    color_image,
                    T_base_camera,
                    centroids,
                    result.get("plane"),
                    color_intrinsics,
                    d2c_ext,
                    result.get("bbox_xy"),
                    sample_idx=len(samples) - 1,
                )
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            robot.CloseRPC()
        except Exception:
            pass
        cv2.destroyAllWindows()

    report(samples)


if __name__ == "__main__":
    main()

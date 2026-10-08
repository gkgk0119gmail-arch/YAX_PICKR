from __future__ import annotations

import argparse
import json
import datetime as _dt
import os
import sys
from typing import Optional, Tuple

try:
    import pyrealsense2 as rs  # Intel RealSense SDK Python bindings
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: pyrealsense2 is not installed or failed to load.\n"
        "Install with one of:\n"
        "  - pip install pyrealsense2-macosx   # on macOS\n"
        "  - pip install pyrealsense2          # on Linux/Windows\n"
        "On macOS you may also need: brew install librealsense"
    )
    raise

try:
    import imageio.v2 as imageio  # Preferred for saving 16-bit PNG
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

import numpy as np


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_depth_scale(active_profile: rs.pipeline_profile) -> float:
    device = active_profile.get_device()

    # Preferred: get the first depth sensor directly if available
    try:
        depth_sensor = device.first_depth_sensor()
        return float(depth_sensor.get_depth_scale())
    except Exception:
        pass

    # Fallback: iterate sensors and cast to depth_sensor
    for sensor in device.query_sensors():
        try:
            depth_sensor = sensor.as_depth_sensor()
            return float(depth_sensor.get_depth_scale())
        except Exception:
            # Last-resort fallback: check for depth_units option
            try:
                if sensor.supports(rs.option.depth_units):
                    return float(sensor.get_option(rs.option.depth_units))
            except Exception:
                pass
            continue

    # Fallback typical scale (meters per unit) for D4xx is 0.001
    return 0.001


def _list_connected_devices() -> list[str]:
    ctx = rs.context()
    devices = ctx.query_devices()
    names: list[str] = []
    for d in devices:
        try:
            name = d.get_info(rs.camera_info.name)
            serial = d.get_info(rs.camera_info.serial_number)
            fw = d.get_info(rs.camera_info.firmware_version)
            names.append(f"{name} (S/N {serial}, FW {fw})")
        except Exception:
            names.append("<unknown device>")
    return names


def capture_single_depth_frame(
    *,
    output_dir: str,
    resolution: Tuple[int, int] = (848, 480),
    fps: int = 30,
    warmup_frames: int = 5,
    frame_timeout_s: float = 5.0,
    basename: Optional[str] = None,
    save_png: bool = True,
    save_npy: bool = True,
    save_viz: bool = True,
) -> Tuple[str, Optional[str]]:
    """Capture one depth frame from an Intel RealSense and save it.

    Returns (npy_path, png_path). png_path may be None if PNG saving is disabled
    or imageio is unavailable.
    """

    width, height = int(resolution[0]), int(resolution[1])

    device_list = _list_connected_devices()
    if not device_list:
        raise RuntimeError(
            "No device connected.\n"
            "Troubleshooting:\n"
            "- Use a USB 3.1/3.2 cable and a direct Mac port (avoid some hubs).\n"
            "- Try a different cable/port; the D455 needs sufficient power.\n"
            "- Confirm macOS sees it: system_profiler SPUSBDataType | grep -i realsense | cat\n"
            "- Verify SDK sees it: rs-enumerate-devices | cat (from brew install librealsense).\n"
            "- Prefer a Python venv, not conda base, to avoid libusb conflicts.\n"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(fps))

    npy_path: Optional[str] = None
    png_path: Optional[str] = None
    viz_path: Optional[str] = None

    try:
        try:
            profile = pipeline.start(config)
        except Exception as start_err:
            print(
                f"Requested profile {width}x{height}@{fps} failed ({start_err}). "
                "Falling back to default device profile."
            )
            profile = pipeline.start()  # let SDK choose defaults
        depth_scale = _get_depth_scale(profile)
        # Fetch intrinsics for the active depth stream
        try:
            depth_stream_profile = (
                profile.get_stream(rs.stream.depth).as_video_stream_profile()
            )
            intr = depth_stream_profile.get_intrinsics()
            intr_meta = {
                "width": int(intr.width),
                "height": int(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
                "model": int(intr.model) if hasattr(intr, "model") else None,
                "coeffs": list(getattr(intr, "coeffs", [])) if hasattr(intr, "coeffs") else [],
                "depth_scale_m_per_unit": float(depth_scale),
            }
        except Exception:
            intr_meta = {
                "width": width,
                "height": height,
                "fx": width / 2.0,
                "fy": height / 2.0,
                "ppx": (width - 1) / 2.0,
                "ppy": (height - 1) / 2.0,
                "model": None,
                "coeffs": [],
                "depth_scale_m_per_unit": float(depth_scale),
            }

        # Warm-up for auto-exposure stability
        for _ in range(max(0, int(warmup_frames))):
            pipeline.wait_for_frames()

        # Wait for a frame set containing depth
        timeout_ms = int(max(0.0, float(frame_timeout_s)) * 1000)
        frames = pipeline.wait_for_frames(timeout_ms)
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("Failed to acquire depth frame within timeout")

        depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())  # uint16 in sensor units
        if depth_raw.dtype != np.uint16:
            depth_raw = depth_raw.astype(np.uint16, copy=False)

        _ensure_dir(output_dir)
        stamp = _now_stamp()
        base = basename if basename else f"realsense_depth_{stamp}"

        # Always save .npy for fidelity regardless of PNG availability
        npy_path = os.path.join(output_dir, f"{base}.npy")
        np.save(npy_path, depth_raw)

        # Save intrinsics/metadata JSON for later back-projection
        meta_path = os.path.join(output_dir, f"{base}_intrinsics.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(intr_meta, f, indent=2)
        except Exception as meta_exc:
            print(f"WARNING: Failed to write intrinsics JSON: {meta_exc}")

        # Optionally save 16-bit PNG (requires imageio)
        if save_png and _HAS_IMAGEIO:
            png_path = os.path.join(output_dir, f"{base}.png")
            imageio.imwrite(png_path, depth_raw)
        elif save_png and not _HAS_IMAGEIO:
            print(
                "WARNING: imageio not installed, skipping PNG save. Install with: pip install imageio"
            )

        # Optionally save a colorized 8-bit visualization using RealSense colorizer
        if save_viz:
            try:
                colorizer = rs.colorizer()
                colorized = colorizer.process(depth_frame)
                color_np = np.asanyarray(colorized.get_data())  # uint8, HxWx3
                if _HAS_IMAGEIO:
                    viz_path = os.path.join(output_dir, f"{base}_viz.png")
                    imageio.imwrite(viz_path, color_np)
                else:
                    print(
                        "WARNING: imageio not installed, skipping colorized PNG save. Install with: pip install imageio"
                    )
            except Exception as viz_exc:
                print(f"WARNING: Failed to create colorized depth visualization: {viz_exc}")

        # Report simple stats
        nonzero = depth_raw[depth_raw > 0]
        near_m = float(nonzero.min() * depth_scale) if nonzero.size else float("nan")
        far_m = float(nonzero.max() * depth_scale) if nonzero.size else float("nan")
        print(
            f"Saved depth frame: shape={depth_raw.shape} dtype={depth_raw.dtype} scale={depth_scale:.6f} m/unit\n"
            f"Nearest(nonzero)≈{near_m:.3f} m, Farthest(nonzero)≈{far_m:.3f} m\n"
            f"NPY: {npy_path}\n"
            f"INTR: {meta_path}"
        )
        if png_path:
            print(f"PNG: {png_path}")
        if viz_path:
            print(f"VIZ: {viz_path}")

        return npy_path, png_path

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a depth image from Intel RealSense and save it.")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"), help="Output directory")
    parser.add_argument("--width", type=int, default=848, help="Depth stream width")
    parser.add_argument("--height", type=int, default=480, help="Depth stream height")
    parser.add_argument("--fps", type=int, default=30, help="Depth stream FPS")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Frames to discard for auto-exposure warmup")
    parser.add_argument("--timeout", type=float, default=5.0, help="Frame acquisition timeout in seconds")
    parser.add_argument("--basename", type=str, default=None, help="Base filename without extension")
    parser.add_argument("--no-png", action="store_true", help="Do not write PNG (only .npy)")
    parser.add_argument("--no-npy", action="store_true", help="Do not write .npy (only PNG if available)")
    parser.add_argument("--no-viz", action="store_true", help="Do not write colorized depth visualization")
    parser.add_argument("--list-devices", action="store_true", help="List connected RealSense devices and exit")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # if args.list-devices:
    #     devices = _list_connected_devices()
    #     if not devices:
    #         print("No RealSense devices found.")
    #         return 1
    #     print("Connected RealSense devices:")
    #     for d in devices:
    #         print("-", d)
    #     return 0

    if args.no_png and args.no_npy:
        print("Nothing to save: both --no-png and --no-npy are set.")
        return 2

    try:
        capture_single_depth_frame(
            output_dir=args.out_dir,
            resolution=(args.width, args.height),
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            frame_timeout_s=args.timeout,
            basename=args.basename,
            save_png=not args.no_png,
            save_npy=not args.no_npy,
            save_viz=not args.no_viz,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



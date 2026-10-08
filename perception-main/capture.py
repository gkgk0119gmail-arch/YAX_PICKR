#!/usr/bin/env python3
"""
Capture aligned RGB + depth frames from a RealSense camera.
Left-drag to draw an ROI, right-click to clear it.
Press SPACE to save the ROI crop (or full frame if no ROI).
Saves to perception/data/captures/ with timestamps.
"""

import time
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs

SAVE_DIR = Path(__file__).parent / "data" / "captures"
WIDTH = 1280
HEIGHT = 720
FPS = 30
WINDOW_NAME = "RealSense Capture"


class RoiState:
    def __init__(self):
        self.drawing = False
        self.pt1 = None
        self.roi = None
        self.drag_preview = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1 = (x, y)
            self.drag_preview = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x1 = min(self.pt1[0], x); y1 = min(self.pt1[1], y)
            x2 = max(self.pt1[0], x); y2 = max(self.pt1[1], y)
            self.drag_preview = (x1, y1, x2, y2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.pt1 is not None:
                x1 = min(self.pt1[0], x); y1 = min(self.pt1[1], y)
                x2 = max(self.pt1[0], x); y2 = max(self.pt1[1], y)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.roi = (x1, y1, x2, y2)
                    print(f"ROI set: ({x1},{y1})-({x2},{y2})  "
                          f"size {x2-x1}x{y2-y1}")
                else:
                    self.roi = None
            self.drag_preview = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.roi = None
            self.drag_preview = None
            print("ROI cleared — will save full frame")


def get_device():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]


def init_realsense():
    print("Initializing RealSense camera...")
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    while True:
        try:
            pipe = rs.pipeline()
            profile = pipe.start(cfg)
            for sensor in profile.get_device().sensors:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                    break
            align = rs.align(rs.stream.color)
            print("Camera initialized successfully.")
            return pipe, align
        except RuntimeError as e:
            msg = str(e)
            print(f"Initialization error: {msg}")
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


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pipe, align = init_realsense()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    roi_state = RoiState()
    cv2.setMouseCallback(WINDOW_NAME, roi_state.mouse_cb)

    count = 0
    print("\n--- Controls ---")
    print("  Left-drag    : draw ROI")
    print("  Right-click  : clear ROI (full frame)")
    print("  SPACE        : save ROI crop (RGB + depth)")
    print("  q            : quit\n")

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            vis = color_img.copy()
            roi = roi_state.roi

            if roi is not None:
                cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]),
                              (0, 255, 255), 2)

            if roi_state.drag_preview is not None:
                dp = roi_state.drag_preview
                overlay = vis.copy()
                cv2.rectangle(overlay, (dp[0], dp[1]), (dp[2], dp[3]),
                              (0, 255, 255), 2)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

            roi_str = (f"ROI {roi[2]-roi[0]}x{roi[3]-roi[1]}"
                       if roi else "FULL FRAME")
            info = f"Captures: {count} | {roi_str} | SPACE=save  Q=quit"
            cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                if roi is not None:
                    rx1, ry1, rx2, ry2 = roi
                    save_rgb = color_img[ry1:ry2, rx1:rx2]
                    save_depth = depth_img[ry1:ry2, rx1:rx2]
                else:
                    save_rgb = color_img
                    save_depth = depth_img

                ts = time.strftime("%Y%m%d_%H%M%S")
                rgb_path = SAVE_DIR / f"{ts}_rgb.png"
                depth_path = SAVE_DIR / f"{ts}_depth.png"
                cv2.imwrite(str(rgb_path), save_rgb)
                cv2.imwrite(str(depth_path), save_depth)
                count += 1
                sz = f"{save_rgb.shape[1]}x{save_rgb.shape[0]}"
                print(f"[{count}] Saved {rgb_path.name} + {depth_path.name}  ({sz})")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print(f"Done — {count} pairs saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()

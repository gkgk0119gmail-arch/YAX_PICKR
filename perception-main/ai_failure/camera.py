"""Top RealSense D435i read directly over V4L2 (no librealsense needed)."""
import subprocess
import sys

import cv2
import numpy as np

COLOR_DEV, DEPTH_DEV = 4, "/dev/video0"
DW, DH = 848, 480


def grab_color(n_warmup=30):
    cap = cv2.VideoCapture(COLOR_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for _ in range(n_warmup):               # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        sys.exit("color camera read failed (is another program using /dev/video4?)")
    return frame


def grab_depth_median_m(n=10):
    subprocess.run(["v4l2-ctl", "-d", DEPTH_DEV, f"--set-fmt-video=width={DW},height={DH},pixelformat=0"],
                   check=True, capture_output=True)
    raw = subprocess.run(["v4l2-ctl", "-d", DEPTH_DEV, "--stream-mmap", f"--stream-count={n}", "--stream-to=-"],
                         check=True, capture_output=True).stdout
    d = np.frombuffer(raw, np.uint16)[: n * DW * DH].reshape(n, DH, DW).astype(np.float32) / 1000.0
    c = d[:, 140:340, 250:600]
    return float(np.median(c[c > 0]))       # table/box distance near the image centre

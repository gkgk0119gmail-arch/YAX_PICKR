"""Live top-down RealSense view (color | depth) shown via ffplay (OpenCV here is headless)."""
import subprocess
import threading

import cv2
import numpy as np

COLOR_DEV, DEPTH_DEV = 4, "/dev/video0"
DW, DH = 848, 480
OUT_W, OUT_H = 848 * 2, 480

# Depth: raw Z16 frames from v4l2-ctl on stdout
subprocess.run(["v4l2-ctl", "-d", DEPTH_DEV,
                f"--set-fmt-video=width={DW},height={DH},pixelformat=0"], check=True)
depth_proc = subprocess.Popen(["v4l2-ctl", "-d", DEPTH_DEV, "--stream-mmap",
                               "--stream-count=0", "--stream-to=-"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
latest_depth = np.zeros((DH, DW), np.uint16)


def read_depth():
    global latest_depth
    n = DW * DH * 2
    while True:
        buf = depth_proc.stdout.read(n)
        if len(buf) < n:
            return
        latest_depth = np.frombuffer(buf, np.uint16).reshape(DH, DW)


threading.Thread(target=read_depth, daemon=True).start()

cap = cv2.VideoCapture(COLOR_DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

player = subprocess.Popen(["ffplay", "-loglevel", "error", "-window_title",
                           "Top camera: color | depth", "-f", "rawvideo",
                           "-pixel_format", "bgr24", "-video_size", f"{OUT_W}x{OUT_H}",
                           "-framerate", "30", "-fflags", "nobuffer", "-i", "-"],
                          stdin=subprocess.PIPE)

try:
    while player.poll() is None:
        ok, color = cap.read()
        if not ok:
            continue
        color = cv2.resize(color, (DW, DH))
        d = latest_depth.astype(np.float32) / 1000.0  # metres
        vis = cv2.applyColorMap((255 * np.clip((d - 0.6) / 0.4, 0, 1)).astype(np.uint8),
                                cv2.COLORMAP_JET)
        vis[d == 0] = 0
        cy, cx = DH // 2, DW // 2
        c = d[cy - 3:cy + 4, cx - 3:cx + 4]
        c = c[c > 0]
        txt = f"center {np.median(c):.3f} m" if c.size else "center n/a"
        cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        player.stdin.write(np.hstack([color, vis]).tobytes())
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    cap.release()
    depth_proc.kill()
    if player.poll() is None:
        player.kill()

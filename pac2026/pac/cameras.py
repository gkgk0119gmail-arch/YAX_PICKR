"""Camera access. Live: top D435i over V4L2 (colour stays open, depth grabbed on demand).
Offline: a saved colour image (+ optional raw Z16 depth) so the whole pipeline runs without hardware.
The wrist D405 is not wired in yet (CameraManager.wrist() returns None)."""
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .types import Frame


class CameraManager:
    def __init__(self, cfg, image=None, depth=None):
        c = cfg["camera"]
        self.dw, self.dh = c["depth_size"]
        self.unit = c["depth_unit_m"]
        self.nd = c["depth_frames"]
        self.depth_dev = c["depth_device"]
        self.static = image is not None
        if self.static:
            self._color = cv2.imread(str(image))
            if self._color is None:
                raise FileNotFoundError(image)
            self._depth = self._load_raw(depth) if depth else None
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(c["color_device"], cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(30):                       # auto-exposure settles
                self.cap.read()
            self._lock = threading.Lock()
            self._latest = None
            self._run = True
            threading.Thread(target=self._grab_loop, daemon=True).start()

    def _grab_loop(self):
        """Keep the newest colour frame so the pipeline and the dashboard never wait on the camera."""
        while self._run:
            ok, img = self.cap.read()
            if ok:
                with self._lock:
                    self._latest = img
            else:
                time.sleep(0.05)

    def latest(self):
        if self.static:
            return self._color
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def _load_raw(self, path):
        d = np.fromfile(path, np.uint16)
        d = d[: len(d) // (self.dw * self.dh) * self.dw * self.dh].reshape(-1, self.dh, self.dw)[-self.nd:]
        d = d.astype(np.float32) * self.unit
        d[d == 0] = np.nan
        with np.errstate(all="ignore"):
            return np.nanmedian(d, 0)

    def _grab_depth(self):
        subprocess.run(["v4l2-ctl", "-d", self.depth_dev,
                        f"--set-fmt-video=width={self.dw},height={self.dh},pixelformat=0"],
                       check=True, capture_output=True)
        raw = subprocess.run(["v4l2-ctl", "-d", self.depth_dev, "--stream-mmap",
                              f"--stream-count={self.nd}", "--stream-to=-"], check=True, capture_output=True).stdout
        tmp = Path("/tmp/pac_depth.raw")
        tmp.write_bytes(raw)
        return self._load_raw(tmp)

    def top(self, with_depth=True) -> Frame:
        if self.static:
            return Frame(self._color.copy(), None if self._depth is None else self._depth.copy())
        time.sleep(0.15)                                # make sure the frame is newer than the last motion
        img = self.latest()
        if img is None:
            raise RuntimeError("top colour camera read failed")
        return Frame(img, self._grab_depth() if with_depth else None)

    def wrist(self):
        return None                                    # TODO: D405 (pyrealsense2) once mounted

    def close(self):
        if self.cap is not None:
            self._run = False
            time.sleep(0.1)
            self.cap.release()

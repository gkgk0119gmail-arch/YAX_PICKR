"""Camera access. Live: RealSense over pyrealsense2 — top D435i and wrist D405, picked by serial
(/dev/videoN numbers move with plug order). Depth is aligned to the colour image, so depth_m[v, u]
is the distance at colour pixel (u, v), and each Frame carries the colour intrinsics.
Offline: a saved colour image (+ optional raw Z16 depth) so the whole pipeline runs without hardware."""
import threading
import time
import warnings

import cv2
import numpy as np

from .types import Frame


class RealSenseCam:
    """One RealSense: a background thread keeps the newest colour frame; depth is grabbed on
    demand from the *next* `n_depth` framesets, aligned to colour, and median-combined."""

    def __init__(self, serial, color_size, depth_size, fps, n_depth, profile=None):
        import pyrealsense2 as rs

        self.serial = str(serial)
        self.nd = n_depth
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, fps)
        self.pipe = rs.pipeline()
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()   # D435i 1 mm, D405 0.1 mm
        if profile:
            self._apply_profile(rs, prof.get_device(), profile)
        self.align = rs.align(rs.stream.color)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
                           "cx": i.ppx, "cy": i.ppy, "model": str(i.model), "coeffs": list(i.coeffs)}
        self._lock = threading.Lock()
        self._latest = None
        self._acc = None                       # depth frames being collected for a pending grab
        self._done = threading.Event()
        self._run = True
        for _ in range(30):                    # auto-exposure settles / manual settings take effect
            self.pipe.wait_for_frames(5000)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _apply_profile(rs, dev, p):
        """Grey-card exposure / white balance (manual, auto off) on the sensor that makes colour.
        On the D405 that is the stereo module, so its depth uses this exposure too."""
        sensor = next(s for s in dev.query_sensors()
                      if any(sp.stream_type() == rs.stream.color for sp in s.get_stream_profiles()))
        for name in ("enable_auto_exposure", "enable_auto_white_balance", "exposure", "gain", "white_balance"):
            if name not in p:
                continue
            for attempt in range(3):                   # USB control transfers can time out right after start
                try:
                    sensor.set_option(getattr(rs.option, name), float(p[name]))
                    break
                except RuntimeError as e:
                    if attempt == 2:
                        print(f"camera profile: could not set {name} ({e}) - keeping the camera default", flush=True)
                    time.sleep(0.3)

    def _loop(self):
        while self._run:
            try:
                fs = self.pipe.wait_for_frames(2000)
            except RuntimeError:
                continue
            cf = fs.get_color_frame()
            if cf:
                with self._lock:
                    self._latest = np.asanyarray(cf.get_data()).copy()
            if self._acc is not None and len(self._acc) < self.nd:
                df = self.align.process(fs).get_depth_frame()
                if df:
                    self._acc.append(np.asanyarray(df.get_data()).copy())
                    if len(self._acc) >= self.nd:
                        self._done.set()

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def depth(self, timeout=5.0):
        """Median of the next n_depth aligned depth frames, metres, NaN where there is no depth."""
        self._done.clear()
        self._acc = []
        if not self._done.wait(timeout):
            self._acc = None
            raise RuntimeError(f"RealSense {self.serial}: depth timed out")
        stack, self._acc = np.stack(self._acc), None
        d = stack.astype(np.float32) * self.scale
        d[stack == 0] = np.nan
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # pixels with no depth in any frame stay NaN
            return np.nanmedian(d, 0)

    def grab(self, with_depth=True) -> Frame:
        img = self.latest()
        if img is None:
            raise RuntimeError(f"RealSense {self.serial}: no colour frame")
        return Frame(img, self.depth() if with_depth else None, self.intrinsics)

    def close(self):
        self._run = False
        self._thread.join(timeout=3)
        self.pipe.stop()


class CameraManager:
    def __init__(self, cfg, image=None, depth=None):
        c = cfg["camera"]
        self.static = image is not None
        self._top = self._wrist = None
        if self.static:
            self.dw, self.dh = c["offline_depth_size"]
            self.unit = c["offline_depth_unit_m"]
            self.nd = c["depth_frames"]
            self._color = cv2.imread(str(image))
            if self._color is None:
                raise FileNotFoundError(image)
            self._depth = self._load_raw(depth) if depth else None
            return
        import pyrealsense2 as rs
        import yaml

        from .paths import ROOT                         # pac2026/, where config.yaml lives

        prof_path = ROOT / c.get("profile", "camera_profile.yaml")
        profiles = (yaml.safe_load(prof_path.read_text()) or {}) if prof_path.exists() else {}
        connected = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
        for role in ("top", "wrist"):
            r = c[role]
            if str(r["serial"]) not in connected:
                if role == "top":
                    raise RuntimeError(f"top camera {r['serial']} not connected (found: {sorted(connected)})")
                print(f"wrist camera {r['serial']} not connected — wrist() will return None", flush=True)
                continue
            p = profiles.get(role)
            if p and str(p.get("serial")) != str(r["serial"]):
                print(f"{role}: camera_profile.yaml is for another camera ({p.get('serial')}) — not applied", flush=True)
                p = None
            elif not p:
                print(f"{role}: no grey-card profile — auto exposure / white balance", flush=True)
            cam = RealSenseCam(r["serial"], r["color_size"], r["depth_size"], r["fps"], c["depth_frames"], p)
            setattr(self, f"_{role}", cam)

    def latest(self):
        return self._color if self.static else self._top.latest()

    def _load_raw(self, path):
        """Offline raw Z16 dump (from the old V4L2 capture): depth pixels, NOT aligned to colour."""
        d = np.fromfile(path, np.uint16)
        d = d[: len(d) // (self.dw * self.dh) * self.dw * self.dh].reshape(-1, self.dh, self.dw)[-self.nd:]
        d = d.astype(np.float32) * self.unit
        d[d == 0] = np.nan
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # pixels with no depth in any frame stay NaN
            return np.nanmedian(d, 0)

    def top(self, with_depth=True) -> Frame:
        if self.static:
            return Frame(self._color.copy(), None if self._depth is None else self._depth.copy())
        time.sleep(0.15)                                # make sure the frame is newer than the last motion
        return self._top.grab(with_depth)

    def wrist(self, with_depth=True):
        if self._wrist is None:
            return None
        time.sleep(0.15)
        return self._wrist.grab(with_depth)

    def close(self):
        for cam in (self._top, self._wrist):
            if cam is not None:
                cam.close()

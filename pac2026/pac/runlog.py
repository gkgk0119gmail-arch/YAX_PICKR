"""One folder per run (messages, step images, json) and, optionally, live updates to the dashboard."""
import json
import time
from pathlib import Path

import cv2
import numpy as np


class RunLog:
    def __init__(self, root: Path, dash=None):
        self.dir = Path(root) / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._file = open(self.dir / "log.txt", "a", encoding="utf-8")
        self.n = 0
        self.dash = dash
        if dash:
            dash.start_run(self.dir.name)

    # --- messages
    def info(self, msg, level="info"):
        line = f"[{time.time() - self.t0:7.2f}s] {msg}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()
        if self.dash:
            self.dash.log(time.time() - self.t0, msg, level)

    def warn(self, msg):
        self.info(msg, "warn")

    # --- images: saved to the run folder; `panel` also shows it on a dashboard card
    def image(self, name, img, panel=None, caption=None):
        self.n += 1
        cv2.imwrite(str(self.dir / f"{self.n:03d}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if self.dash and panel:
            self.dash.set_panel(panel, img, caption)

    def json(self, name, obj):
        def default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return str(o)
        (self.dir / f"{name}.json").write_text(json.dumps(obj, indent=1, default=default, ensure_ascii=False))

    # --- dashboard-only state (no-ops without a dashboard)
    def step(self, n, status, note=""):
        if self.dash:
            self.dash.set_step(n, status, note)

    def metric(self, **kv):
        if self.dash:
            self.dash.set_metrics(**kv)

    def overlay(self, **kv):
        if self.dash:
            self.dash.set_overlay(**kv)

    @property
    def stop_requested(self):
        return bool(self.dash and self.dash.stop_flag.is_set())

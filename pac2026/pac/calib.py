"""Colour-pixel -> robot-base XY on the table plane (homography), plus image -> robot yaw.

Real use: build pixel_to_robot.json on site (robot places a block at known XY, the top camera
finds it -> >= 4 pairs -> cv2.findHomography). Without the file a MOCK mapping is used so the
pipeline can run end to end (positions are then meaningless for a real robot)."""
import json
from pathlib import Path

import cv2
import numpy as np


class PixelToRobot:
    def __init__(self, H, table_z_m, mock=False):
        self.H = np.asarray(H, np.float64)
        self.table_z = float(table_z_m)
        self.mock = mock

    @classmethod
    def load_or_mock(cls, path: Path, table_z_m, image_shape):
        if path.exists():
            d = json.loads(path.read_text())
            return cls(d["H"], d.get("table_z_robot_m", table_z_m))
        h, w = image_shape[:2]
        s = 0.0006                                         # 0.6 mm per pixel, robot ~0.45 m in front of image centre
        H = [[0, s, 0.45 - s * h / 2], [s, 0, -s * w / 2], [0, 0, 1]]
        return cls(H, table_z_m, mock=True)

    @staticmethod
    def fit(pixels, robot_xy):
        H, _ = cv2.findHomography(np.float64(pixels), np.float64(robot_xy), 0)
        return H

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"H": self.H.tolist(), "table_z_robot_m": self.table_z}, indent=1))

    def to_robot(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def yaw_to_robot(self, u, v, deg_img):
        """Map an image direction at (u, v) to a yaw angle (rad) in the robot base frame."""
        a = np.radians(deg_img)
        x0, y0 = self.to_robot(u, v)
        x1, y1 = self.to_robot(u + 10 * np.cos(a), v + 10 * np.sin(a))
        return float(np.arctan2(y1 - y0, x1 - x0))

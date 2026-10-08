"""Every model is loaded once and reused (loading VGGT etc. takes seconds)."""
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image

from . import paths  # noqa: F401


class Vggt:
    def __init__(self, model_id):
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained(model_id).to("cuda").eval()

    def points(self, img_bgr):
        """Camera-frame point map (VGGT units, HxWx3) on the VGGT grid for one colour image."""
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            cv2.imwrite(f.name, img_bgr)
            images = load_and_preprocess_images([f.name]).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(images[None])
        extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        depth = pred["depth"][0].float().cpu().numpy()
        return unproject_depth_map_to_point_map(depth, extri[0].float().cpu().numpy(),
                                                intri[0].float().cpu().numpy())[0]


class CellClassifier:
    def __init__(self, path):
        from train_classifier import Dinov3Features
        b = joblib.load(path)
        self.clf, self.classes = b["clf"], b["classes"]
        self.feat = Dinov3Features()

    def __call__(self, crops_bgr):
        X = self.feat([Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr])
        p = self.clf.predict_proba(X)
        names = [self.classes[k] for k in self.clf.classes_]
        return [(names[int(np.argmax(r))], float(np.max(r))) for r in p]


class ModelHub:
    def __init__(self, cfg, base: Path):
        self.cfg, self.base, self._m = cfg, base, {}

    def _path(self, key):
        return (self.base / self.cfg["models"][key]).resolve()

    @property
    def scene(self):
        if "scene" not in self._m:
            from scene_state import SceneState
            self._m["scene"] = SceneState()
        return self._m["scene"]

    @property
    def vggt(self):
        if "vggt" not in self._m:
            self._m["vggt"] = Vggt(self.cfg["models"]["vggt"])
        return self._m["vggt"]

    @property
    def grasp(self):
        if "grasp" not in self._m:
            from detect_grasp import BlockGraspDetector
            self._m["grasp"] = BlockGraspDetector(self._path("rfdetr_weights"),
                                                  resolution=self.cfg["models"]["rfdetr_resolution"],
                                                  threshold=self.cfg["perception"]["block_threshold"])
        return self._m["grasp"]

    @property
    def cells(self):
        if "cells" not in self._m:
            self._m["cells"] = CellClassifier(self._path("cell_classifier"))
        return self._m["cells"]

    def warmup(self, log):
        for name in ("scene", "vggt", "grasp", "cells"):
            log.info(f"loading model: {name}")
            getattr(self, name)

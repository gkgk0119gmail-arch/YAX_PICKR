#!/usr/bin/env python3
"""
Cell-state classifier for insertion-failure detection: frozen DINOv3 features + logistic regression.

Each crop (from label_round.py) -> DINOv3 ViT-B/16 -> [CLS | mean of centre patches] (1536-d)
-> standardised -> logistic regression. Only a few seconds to train, ~10 ms per cell to run.

Evaluation is leave-one-shot-out: all crops of one photo are held out together, because
cells of the same photo share lighting and neighbours and would leak.

Usage (in ai_bubble/.venv, which has transformers + sklearn):
    HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python train_classifier.py
Writes model/cell_classifier.joblib and model/report.txt.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

HERE = Path(__file__).resolve().parent
BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
CLASSES = ["empty", "seated", "straddle", "collapsed"]


class Dinov3Features:
    """CLS token + mean of the patch tokens that cover the cell itself (the inner part of
    the crop; label_round.py adds a 25 % margin on each side)."""

    def __init__(self, device="cuda"):
        self.proc = AutoImageProcessor.from_pretrained(BACKBONE)
        self.model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
        self.device = device
        self.n_reg = getattr(self.model.config, "num_register_tokens", 0)

    @torch.no_grad()
    def __call__(self, images):
        inp = self.proc(images=images, return_tensors="pt").to(self.device)
        h = self.model(**inp).last_hidden_state                 # B x (1 + reg + P) x C
        cls = h[:, 0]
        patches = h[:, 1 + self.n_reg:]
        g = int(round(patches.shape[1] ** 0.5))
        patches = patches.reshape(len(images), g, g, -1)
        lo, hi = round(g * 0.25 / 1.5), round(g * 1.25 / 1.5)  # cell region inside the margin
        centre = patches[:, lo:hi, lo:hi].mean((1, 2))
        f = torch.cat([cls, centre], 1)
        return torch.nn.functional.normalize(f, dim=1).cpu().numpy()


def load_dataset(root):
    paths, labels, groups = [], [], []
    for ci, name in enumerate(CLASSES):
        for p in sorted((root / name).glob("*.jpg")):
            paths.append(p)
            labels.append(ci)
            groups.append(re.sub(r"_cell\d+$", "", p.stem))      # photo id
    return paths, np.array(labels), np.array(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=HERE / "dataset", type=Path)
    ap.add_argument("--out", default=HERE / "model", type=Path)
    args = ap.parse_args()

    paths, y, groups = load_dataset(args.data)
    present = sorted(set(y))
    print("crops per class:", {CLASSES[k]: v for k, v in sorted(Counter(y).items())},
          "| photos:", len(set(groups)))

    feat = Dinov3Features()
    X = np.concatenate([feat([Image.open(p).convert("RGB") for p in paths[i:i + 32]])
                        for i in range(0, len(paths), 32)])

    def make():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5,
                                                                  class_weight="balanced"))

    # leave-one-photo-out
    pred = np.empty_like(y)
    for g in sorted(set(groups)):
        te = groups == g
        clf = make().fit(X[~te], y[~te])
        pred[te] = clf.predict(X[te])
    names = [CLASSES[k] for k in present]
    report = classification_report(y, pred, labels=present, target_names=names, digits=3, zero_division=0)
    cm = confusion_matrix(y, pred, labels=present)
    acc = (pred == y).mean()
    wrong = [f"  {p.parent.name}/{p.name} -> predicted {CLASSES[q]}" for p, t, q in zip(paths, y, pred) if t != q]
    text = (f"leave-one-photo-out accuracy {acc:.3f} ({(pred == y).sum()}/{len(y)})\n\n{report}\n"
            f"confusion (rows=true, cols=pred, order {names}):\n{cm}\n\nmistakes:\n" + "\n".join(wrong))
    print(text)

    args.out.mkdir(parents=True, exist_ok=True)
    final = make().fit(X, y)
    joblib.dump({"clf": final, "classes": CLASSES, "backbone": BACKBONE}, args.out / "cell_classifier.joblib")
    (args.out / "report.txt").write_text(text)
    print(f"\nsaved {args.out / 'cell_classifier.joblib'}")


if __name__ == "__main__":
    main()

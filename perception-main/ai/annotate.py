#!/usr/bin/env python3
"""
Bounding-box annotation tool for box detection training data.

Usage:
    python annotate.py [--captures-dir DIR] [--annotations FILE]

Controls:
    Left-drag     Draw a bounding box
    Left/Right    Previous / next image  (A/D also work)
    Z             Undo last box on current image
    X             Delete box nearest to mouse cursor
    S             Save annotations to disk
    Q / Escape    Save and quit
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Annotate Boxes"
BOX_COLOR = (0, 255, 0)
BOX_COLOR_DRAW = (0, 200, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


class AnnotationState:
    def __init__(self, captures_dir: str, annotations_path: str):
        self.captures_dir = Path(captures_dir)
        self.annotations_path = Path(annotations_path)

        self.images = sorted(
            p.name for p in self.captures_dir.glob("*_rgb.png")
        )
        if not self.images:
            sys.exit(f"No *_rgb.png files found in {self.captures_dir}")

        self.annotations: dict[str, list[list[int]]] = {}
        if self.annotations_path.exists():
            with open(self.annotations_path) as f:
                self.annotations = json.load(f)
            print(f"Loaded existing annotations from {self.annotations_path}")

        for name in self.images:
            self.annotations.setdefault(name, [])

        self.idx = 0
        self.drawing = False
        self.pt1 = None
        self.pt2 = None
        self.mouse_pos = (0, 0)
        self.dirty = False

    @property
    def current_name(self) -> str:
        return self.images[self.idx]

    @property
    def current_boxes(self) -> list[list[int]]:
        return self.annotations[self.current_name]

    def load_image(self) -> np.ndarray:
        path = self.captures_dir / self.current_name
        img = cv2.imread(str(path))
        if img is None:
            sys.exit(f"Failed to read {path}")
        return img

    def save(self):
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.annotations_path, "w") as f:
            json.dump(self.annotations, f, indent=2)
        self.dirty = False
        print(f"Saved annotations to {self.annotations_path}")

    def add_box(self, x1: int, y1: int, x2: int, y2: int):
        bx1, by1 = min(x1, x2), min(y1, y2)
        bx2, by2 = max(x1, x2), max(y1, y2)
        if (bx2 - bx1) > 5 and (by2 - by1) > 5:
            self.current_boxes.append([bx1, by1, bx2, by2])
            self.dirty = True

    def undo(self):
        if self.current_boxes:
            self.current_boxes.pop()
            self.dirty = True

    def delete_nearest(self, mx: int, my: int):
        if not self.current_boxes:
            return
        best_i, best_dist = -1, float("inf")
        for i, (x1, y1, x2, y2) in enumerate(self.current_boxes):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            d = (mx - cx) ** 2 + (my - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_i = i
        if best_i >= 0:
            self.current_boxes.pop(best_i)
            self.dirty = True

    def mouse_cb(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1 = (x, y)
            self.pt2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.pt1 is not None:
                self.add_box(self.pt1[0], self.pt1[1], x, y)
            self.drawing = False
            self.pt1 = None
            self.pt2 = None


def render(state: AnnotationState) -> np.ndarray:
    img = state.load_image()

    for i, (x1, y1, x2, y2) in enumerate(state.current_boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        cv2.putText(img, str(i), (x1 + 4, y1 + 20), FONT, 0.6,
                    BOX_COLOR, 2, cv2.LINE_AA)

    if state.drawing and state.pt1 and state.pt2:
        cv2.rectangle(img, state.pt1, state.pt2, BOX_COLOR_DRAW, 2)

    h, w = img.shape[:2]
    n_boxes = len(state.current_boxes)
    save_indicator = " *" if state.dirty else ""
    hud = (f"[{state.idx + 1}/{len(state.images)}] "
           f"{state.current_name}  |  {n_boxes} box(es){save_indicator}")
    cv2.putText(img, hud, (10, 28), FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, hud, (10, 28), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    help_txt = "L/R: nav | Z: undo | X: delete nearest | S: save | Q: quit"
    cv2.putText(img, help_txt, (10, h - 12), FONT, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, help_txt, (10, h - 12), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser(description="Bounding box annotation tool")
    parser.add_argument("--captures-dir",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "data", "captures"))
    parser.add_argument("--annotations",
                        default=os.path.join(os.path.dirname(__file__),
                                             "annotations.json"))
    args = parser.parse_args()

    state = AnnotationState(args.captures_dir, args.annotations)
    print(f"Found {len(state.images)} images to annotate.")
    print(f"Annotations file: {state.annotations_path}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, state.mouse_cb)

    while True:
        vis = render(state)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # Q or Escape
            if state.dirty:
                state.save()
            break
        elif key == ord("s"):
            state.save()
        elif key == ord("z"):
            state.undo()
        elif key == ord("x"):
            state.delete_nearest(*state.mouse_pos)
        elif key in (ord("d"), 83, 3):  # D, Right arrow
            state.idx = (state.idx + 1) % len(state.images)
        elif key in (ord("a"), 81, 2):  # A, Left arrow
            state.idx = (state.idx - 1) % len(state.images)

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

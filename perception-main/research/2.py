from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, QTimer

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =========================
# CONFIG
# =========================
@dataclass
class Config:
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30

    DET_WEIGHTS: Path = Path("/home/jeong/perception/models/rtdetr-l.pt")
    CONF: float = 0.25
    IOU: float = 0.7
    MAX_DET: int = 50

    SAM2_CKPT: Path = Path("/home/jeong/perception/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    SAM2_CFG: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    ASSOC_MIN_IOU: float = 0.15
    TOGGLE_SAME_IOU: float = 0.40

    ROI_PAD_RATIO: float = 0.20
    SAM2_EVERY: int = 1

    MIN_MASK_DEPTH_PIXELS: int = 300
    DEPTH_NEIGHBORHOOD: int = 3

    OVERLAY_ALPHA: float = 0.45
    NEAR_BRIGHTER: bool = True
    DEPTH_CLIP_PCT: tuple[float, float] = (5.0, 95.0)

    EDGE_KERNEL: int = 3
    EDGE_DILATE_ITERS: int = 1
    EDGE_INNER_COLOR: tuple[int, int, int] = (255, 255, 255)
    EDGE_OUTER_COLOR: tuple[int, int, int] = (0, 0, 0)

    PC_STRIDE: int = 8
    PC_MAX_POINTS: int = 5000
    DRAW_PC_DOTS: bool = True
    DOT_SIZE: int = 1


CFG = Config()


def pick_autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def load_sam2_predictor(ckpt: Path, model_cfg: str, device: str) -> SAM2ImagePredictor:
    model = build_sam2(model_cfg, str(ckpt), device=device)
    return SAM2ImagePredictor(model)


def init_realsense(cfg: Config) -> tuple[rs.pipeline, rs.align, float, rs.intrinsics]:
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, cfg.WIDTH, cfg.HEIGHT, rs.format.bgr8, cfg.FPS)
    rs_cfg.enable_stream(rs.stream.depth, cfg.WIDTH, cfg.HEIGHT, rs.format.z16, cfg.FPS)

    profile = pipeline.start(rs_cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    align = rs.align(rs.stream.color)

    for _ in range(10):
        pipeline.wait_for_frames()

    return pipeline, align, depth_scale, intr


def get_aligned_color_depth(pipeline: rs.pipeline, align: rs.align) -> tuple[np.ndarray | None, np.ndarray | None]:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color = aligned.get_color_frame()
    depth = aligned.get_depth_frame()
    if not color or not depth:
        return None, None
    return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())


def run_detection(det_model: YOLO, img_bgr: np.ndarray, conf: float, iou: float, max_det: int, device: str) -> list[dict]:
    results = det_model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        device=0 if device == "cuda" else "cpu",
    )
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    xyxy = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    cfs = r0.boxes.conf.detach().cpu().numpy().astype(float)
    cls = r0.boxes.cls.detach().cpu().numpy().astype(int)

    dets: list[dict] = []
    for i in range(len(xyxy)):
        dets.append({"xyxy": xyxy[i], "conf": float(cfs[i]), "cls": int(cls[i]), "name": det_model.names[int(cls[i])]})

    dets.sort(key=lambda d: d["conf"], reverse=True)
    for j, d in enumerate(dets):
        d["idx"] = j
    return dets


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def choose_box_by_click(dets: list[dict], x: int, y: int) -> int | None:
    candidates = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = float((x2 - x1) * (y2 - y1))
            candidates.append((area, -d["conf"], d["idx"]))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def associate_target_box(dets: list[dict], prev_box: np.ndarray, target_cls: int, min_iou: float) -> np.ndarray | None:
    best_iou = 0.0
    best_box = None
    for d in dets:
        if d["cls"] != target_cls:
            continue
        iou_val = xyxy_iou(prev_box, d["xyxy"])
        if iou_val > best_iou:
            best_iou = iou_val
            best_box = d["xyxy"]
    return None if (best_box is None or best_iou < min_iou) else best_box


def roi_from_box(frame_bgr: np.ndarray, box_xyxy: np.ndarray, pad_ratio: float) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray] | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = int(max(bw, bh) * pad_ratio)

    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad)
    ry2 = min(h, y2 + pad)
    if rx2 <= rx1 + 1 or ry2 <= ry1 + 1:
        return None

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    roi_box = np.array([x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1], dtype=np.float32)
    return roi, (rx1, ry1, rx2, ry2), roi_box


def sam2_segment_on_roi(
    predictor: SAM2ImagePredictor,
    roi_bgr: np.ndarray,
    roi_box_xyxy: np.ndarray,
    device: str,
    autocast_dtype: torch.dtype,
) -> tuple[np.ndarray, float]:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    box = roi_box_xyxy.astype(np.float32)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                predictor.set_image(roi_rgb)
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        else:
            predictor.set_image(roi_rgb)
            masks, scores, _ = predictor.predict(box=box, multimask_output=False)

    return (masks[0] > 0).astype(np.uint8), float(scores[0])


def paste_roi_mask_to_full(full_shape_hw: tuple[int, int], roi_xyxy: tuple[int, int, int, int], roi_mask01: np.ndarray) -> np.ndarray:
    h, w = full_shape_hw
    rx1, ry1, rx2, ry2 = roi_xyxy
    full = np.zeros((h, w), dtype=np.uint8)
    full[ry1:ry2, rx1:rx2] = roi_mask01
    return full


def bbox_center_xy(box_xyxy: np.ndarray, shape_hw: tuple[int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box_xyxy
    cx = int((x1 + x2) * 0.5)
    cy = int((y1 + y2) * 0.5)
    h, w = shape_hw
    return int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))


def mask_centroid_xy(mask01: np.ndarray, fallback_xy: tuple[int, int]) -> tuple[int, int]:
    if mask01 is None or mask01.max() == 0:
        return fallback_xy
    mom = cv2.moments(mask01.astype(np.uint8), binaryImage=True)
    if mom["m00"] <= 1e-6:
        return fallback_xy
    return int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"])


def depth_median_in_neighborhood(depth_u16: np.ndarray, depth_scale: float, x: int, y: int, k: int) -> float | None:
    h, w = depth_u16.shape
    x1, y1 = max(0, x - k), max(0, y - k)
    x2, y2 = min(w, x + k + 1), min(h, y + k + 1)
    patch = depth_u16[y1:y2, x1:x2].reshape(-1)
    patch = patch[patch > 0]
    return None if patch.size == 0 else float(np.median(patch) * depth_scale)


def depth_stats_in_mask(
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    depth_scale: float,
    min_pixels: int,
    clip_pct: tuple[float, float],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    m = mask01.astype(bool)
    if not np.any(m):
        return None, None, None, None, None

    d = depth_u16[m].astype(np.float32)
    d = d[d > 0]
    if d.size == 0:
        return None, None, None, None, None

    z = d * float(depth_scale)
    zmin, zmax, zmed = float(np.min(z)), float(np.max(z)), float(np.median(z))

    if d.size < min_pixels:
        return zmed, zmin, zmax, None, None

    lo = float(np.percentile(z, clip_pct[0]))
    hi = float(np.percentile(z, clip_pct[1]))
    if hi <= lo + 1e-6:
        return zmed, zmin, zmax, None, None
    return zmed, zmin, zmax, lo, hi


def deproject_xyz(intr: rs.intrinsics, x: int, y: int, depth_m: float) -> tuple[float, float, float]:
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return float(X), float(Y), float(Z)


def point_cloud_from_mask(
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    depth_scale: float,
    intr: rs.intrinsics,
    stride: int,
    max_points: int,
) -> np.ndarray:
    fx, fy, ppx, ppy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    ds_mask = mask01[::stride, ::stride]
    ds_depth = depth_u16[::stride, ::stride]
    valid = (ds_mask > 0) & (ds_depth > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    rr, cc = np.nonzero(valid)
    if rr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if rr.size > max_points:
        idx = np.random.choice(rr.size, size=max_points, replace=False)
        rr, cc = rr[idx], cc[idx]

    v = (rr * stride).astype(np.float32)
    u = (cc * stride).astype(np.float32)
    z = (ds_depth[rr, cc].astype(np.float32) * float(depth_scale))

    x = (u - ppx) / fx * z
    y = (v - ppy) / fy * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def apply_depth_shaded_green_overlay(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    alpha: float,
    near_brighter: bool,
    clip_lo: float | None,
    clip_hi: float | None,
) -> np.ndarray:
    out = img_bgr.copy()
    m = mask01.astype(bool)
    if not np.any(m):
        return out

    overlay = out.copy()

    if clip_lo is None or clip_hi is None or clip_hi <= clip_lo + 1e-6:
        overlay[m] = (0, 255, 0)
        return cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0.0)

    z = depth_u16.astype(np.float32) * float(depth_scale)
    zn = np.clip((z - clip_lo) / (clip_hi - clip_lo), 0.0, 1.0)
    if near_brighter:
        zn = 1.0 - zn

    g = (40.0 + 215.0 * zn).astype(np.uint8)
    overlay[..., 0][m] = 0
    overlay[..., 2][m] = 0
    overlay[..., 1][m] = g[m]

    return cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0.0)


def add_sharp_edge_outline(img_bgr: np.ndarray, mask01: np.ndarray, cfg: Config) -> np.ndarray:
    out = img_bgr
    m = mask01.astype(np.uint8)
    if m.max() == 0:
        return out

    k = cfg.EDGE_KERNEL if cfg.EDGE_KERNEL % 2 == 1 else cfg.EDGE_KERNEL + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
    edge = (edge > 0).astype(np.uint8)

    outer = cv2.dilate(edge, kernel, iterations=max(1, cfg.EDGE_DILATE_ITERS))
    outer_bool = outer.astype(bool)
    edge_bool = edge.astype(bool)

    out = out.copy()
    out[outer_bool] = cfg.EDGE_OUTER_COLOR
    out[edge_bool] = cfg.EDGE_INNER_COLOR
    return out


def draw_point_dots(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    stride: int,
    clip_lo: float | None,
    clip_hi: float | None,
    near_brighter: bool,
    dot_size: int,
) -> np.ndarray:
    out = img_bgr
    ds_mask = mask01[::stride, ::stride]
    ds_depth = depth_u16[::stride, ::stride]
    valid = (ds_mask > 0) & (ds_depth > 0)
    if not np.any(valid):
        return out

    rr, cc = np.nonzero(valid)
    if rr.size == 0:
        return out

    v = (rr * stride).astype(np.int32)
    u = (cc * stride).astype(np.int32)
    z = (ds_depth[rr, cc].astype(np.float32) * float(depth_scale))

    if clip_lo is not None and clip_hi is not None and clip_hi > clip_lo + 1e-6:
        zn = np.clip((z - clip_lo) / (clip_hi - clip_lo), 0.0, 1.0)
        if near_brighter:
            zn = 1.0 - zn
        g = (40.0 + 215.0 * zn).astype(np.uint8)
    else:
        g = np.full((z.shape[0],), 220, dtype=np.uint8)

    if dot_size <= 1:
        out[v, u] = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=1)
        return out

    for i in range(u.shape[0]):
        cv2.circle(out, (int(u[i]), int(v[i])), dot_size, (0, int(g[i]), 0), -1, lineType=cv2.LINE_AA)
    return out


def draw_ui(
    img_bgr: np.ndarray,
    dets: list[dict],
    target: dict | None,
    dist_m: float | None,
    xyz: tuple[float, float, float] | None,
    pc: np.ndarray,
    zmin: float | None,
    zmax: float | None,
    sam2_score: float | None,
    fps: float,
    det_ms: float,
    seg_ms: float,
) -> np.ndarray:
    vis = img_bgr.copy()

    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, f'{d["idx"]}:{d["name"]} {d["conf"]:.2f}', (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if target is not None and target.get("box") is not None:
        x1, y1, x2, y2 = map(int, target["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        name = str(target["name"]).upper()
        dist_txt = "(no depth)" if dist_m is None else f"{dist_m:.2f}m"
        cv2.putText(vis, f"TARGET: {name}  {dist_txt}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2, cv2.LINE_AA)

        pc_n = int(pc.shape[0]) if pc is not None else 0
        zrange = "(no depth)" if (zmin is None or zmax is None) else f"Zrange={zmin:.2f}~{zmax:.2f}m"
        if xyz is not None:
            X, Y, Z = xyz
            cv2.putText(vis, f"XYZ(m): {X:+.3f}, {Y:+.3f}, {Z:+.3f}   PC={pc_n}   {zrange}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, f"XYZ(m): (no depth)   PC={pc_n}   {zrange}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if sam2_score is not None:
            cv2.putText(vis, f"SAM2 score={sam2_score:.3f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(vis, f"FPS={fps:.1f}  det(ms)={det_ms:.1f}  seg(ms)={seg_ms:.1f}  dets={len(dets)}",
                (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


class VideoWidget(QLabel):
    """QLabel subclass that handles mouse clicks for bbox selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.click_callback = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.click_callback:
            pos = event.position()
            self.click_callback(int(pos.x()), int(pos.y()))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RT-DETR + SAM2 + Depth-shaded Mask + PointCloud")

        # Initialize models and camera
        print("[Init] Setting up PyTorch...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if not CFG.DET_WEIGHTS.exists():
            raise FileNotFoundError(f"DET_WEIGHTS not found: {CFG.DET_WEIGHTS}")
        if not CFG.SAM2_CKPT.exists():
            raise FileNotFoundError(f"SAM2_CKPT not found: {CFG.SAM2_CKPT}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast_dtype = pick_autocast_dtype()

        print("[Init] Loading detection model...")
        self.det_model = YOLO(str(CFG.DET_WEIGHTS))

        print("[Init] Loading SAM2...")
        self.predictor = load_sam2_predictor(CFG.SAM2_CKPT, CFG.SAM2_CFG, self.device)

        print("[Init] Starting RealSense...")
        self.pipeline, self.align, self.depth_scale, self.intr = init_realsense(CFG)

        print(f"[Info] device={self.device}, autocast={self.autocast_dtype}")
        print(f"[Info] RealSense mode={CFG.WIDTH}x{CFG.HEIGHT}@{CFG.FPS} depth_scale={self.depth_scale}")

        # State
        self.state = {
            "dets": [],
            "target": None,
            "fps": 0.0,
            "t_prev": time.time(),
            "det_ms": 0.0,
            "seg_ms": 0.0,
            "frame_id": 0,
            "last_roi_mask": None,
            "last_roi_shape": None,
            "last_sam2_score": None,
            "last_pc": np.zeros((0, 3), dtype=np.float32),
        }

        # UI
        self.video_label = VideoWidget()
        self.video_label.click_callback = self.on_click
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)
        self.setCentralWidget(container)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(1)  # ~1ms interval, limited by processing speed

        print("[UI] Click bbox to select target, Q to quit, C to clear")

    def on_click(self, x: int, y: int):
        """Handle mouse click for bbox selection."""
        sel = choose_box_by_click(self.state["dets"], x, y)
        if sel is None:
            return

        clicked = self.state["dets"][sel]

        if self.state["target"] is not None and self.state["target"].get("box") is not None:
            if int(clicked["cls"]) == int(self.state["target"]["cls"]):
                if xyxy_iou(clicked["xyxy"], self.state["target"]["box"]) >= CFG.TOGGLE_SAME_IOU:
                    self.state["target"] = None
                    self.state["last_roi_mask"] = None
                    self.state["last_roi_shape"] = None
                    self.state["last_sam2_score"] = None
                    self.state["last_pc"] = np.zeros((0, 3), dtype=np.float32)
                    print("[UI] target OFF")
                    return

        self.state["target"] = {"cls": int(clicked["cls"]), "name": str(clicked["name"]), "box": clicked["xyxy"].copy()}
        self.state["last_roi_mask"] = None
        self.state["last_roi_shape"] = None
        self.state["last_sam2_score"] = None
        print(f'[Target] ON: {self.state["target"]["name"]}  box={self.state["target"]["box"]}')

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Q or key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_C:
            self.state["target"] = None
            self.state["last_roi_mask"] = None
            self.state["last_roi_shape"] = None
            self.state["last_sam2_score"] = None
            self.state["last_pc"] = np.zeros((0, 3), dtype=np.float32)
            print("[UI] target cleared")

    def process_frame(self):
        """Process one frame from RealSense."""
        frame_bgr, depth_u16 = get_aligned_color_depth(self.pipeline, self.align)
        if frame_bgr is None or depth_u16 is None:
            return

        self.state["frame_id"] += 1

        t_now = time.time()
        dt = t_now - self.state["t_prev"]
        self.state["t_prev"] = t_now
        if dt > 0:
            self.state["fps"] = 0.9 * self.state["fps"] + 0.1 * (1.0 / dt) if self.state["fps"] > 0 else (1.0 / dt)

        t0 = time.time()
        dets = run_detection(self.det_model, frame_bgr, CFG.CONF, CFG.IOU, CFG.MAX_DET, self.device)
        self.state["det_ms"] = (time.time() - t0) * 1000.0
        self.state["dets"] = dets

        out = frame_bgr
        dist_m = None
        xyz = None
        zmin = None
        zmax = None
        self.state["seg_ms"] = 0.0
        sam2_score = self.state["last_sam2_score"]
        self.state["last_pc"] = np.zeros((0, 3), dtype=np.float32)

        if self.state["target"] is not None:
            new_box = associate_target_box(dets, self.state["target"]["box"], self.state["target"]["cls"], CFG.ASSOC_MIN_IOU)
            if new_box is not None:
                self.state["target"]["box"] = new_box

                roi_pack = roi_from_box(frame_bgr, new_box, CFG.ROI_PAD_RATIO)
                if roi_pack is not None:
                    roi_bgr, roi_xyxy, roi_box = roi_pack
                    roi_h, roi_w = roi_bgr.shape[:2]

                    need_update = (self.state["frame_id"] % CFG.SAM2_EVERY == 0) or (self.state["last_roi_mask"] is None)

                    if need_update:
                        t1 = time.time()
                        roi_mask, sam2_score = sam2_segment_on_roi(self.predictor, roi_bgr, roi_box, self.device, self.autocast_dtype)
                        self.state["seg_ms"] = (time.time() - t1) * 1000.0

                        self.state["last_roi_mask"] = roi_mask
                        self.state["last_roi_shape"] = (roi_h, roi_w)
                        self.state["last_sam2_score"] = sam2_score
                    else:
                        roi_mask = self.state["last_roi_mask"]
                        if roi_mask is None:
                            roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                        else:
                            prev_shape = self.state["last_roi_shape"]
                            if prev_shape is None or prev_shape != (roi_h, roi_w):
                                roi_mask = cv2.resize(roi_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

                    mask01 = paste_roi_mask_to_full(frame_bgr.shape[:2], roi_xyxy, roi_mask)

                    zmed, zmin, zmax, clip_lo, clip_hi = depth_stats_in_mask(
                        depth_u16, mask01, self.depth_scale, CFG.MIN_MASK_DEPTH_PIXELS, CFG.DEPTH_CLIP_PCT
                    )
                    dist_m = zmed

                    cx0, cy0 = bbox_center_xy(new_box, depth_u16.shape)
                    cx, cy = mask_centroid_xy(mask01, (cx0, cy0))
                    if dist_m is None:
                        dist_m = depth_median_in_neighborhood(depth_u16, self.depth_scale, cx, cy, CFG.DEPTH_NEIGHBORHOOD)
                    if dist_m is not None:
                        xyz = deproject_xyz(self.intr, cx, cy, dist_m)

                    out = apply_depth_shaded_green_overlay(
                        out, mask01, depth_u16, self.depth_scale,
                        alpha=CFG.OVERLAY_ALPHA,
                        near_brighter=CFG.NEAR_BRIGHTER,
                        clip_lo=clip_lo,
                        clip_hi=clip_hi,
                    )
                    out = add_sharp_edge_outline(out, mask01, CFG)

                    pc = point_cloud_from_mask(depth_u16, mask01, self.depth_scale, self.intr, CFG.PC_STRIDE, CFG.PC_MAX_POINTS)
                    self.state["last_pc"] = pc
                    self.state["last_sam2_score"] = sam2_score

                    if CFG.DRAW_PC_DOTS:
                        out = draw_point_dots(
                            out, mask01, depth_u16, self.depth_scale,
                            stride=CFG.PC_STRIDE,
                            clip_lo=clip_lo,
                            clip_hi=clip_hi,
                            near_brighter=CFG.NEAR_BRIGHTER,
                            dot_size=CFG.DOT_SIZE,
                        )
                else:
                    cv2.putText(out, "ROI failed (click again)", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    self.state["last_roi_mask"] = None
                    self.state["last_roi_shape"] = None
                    self.state["last_sam2_score"] = None
            else:
                cv2.putText(out, "TARGET LOST (click again)", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                self.state["last_roi_mask"] = None
                self.state["last_roi_shape"] = None
                self.state["last_sam2_score"] = None

        vis = draw_ui(
            out, dets, self.state["target"],
            dist_m, xyz, self.state["last_pc"], zmin, zmax,
            sam2_score, self.state["fps"], self.state["det_ms"], self.state["seg_ms"]
        )

        # Convert BGR to RGB for Qt
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        print("[Cleanup] Stopping...")
        self.timer.stop()
        self.pipeline.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(CFG.WIDTH, CFG.HEIGHT)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

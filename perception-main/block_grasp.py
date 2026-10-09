"""Blocks to pick, from the top D435i: Grounding DINO -> SAM2 inside each block box -> RANSAC rectangle
-> block centre + jaw direction in the robot base frame. Containers (wooden tray, basket with dividers)
are detected separately so they are never mistaken for a block.

  objects  = Grounding DINO hits of BLOCK_PROMPTS and CONTAINER_PROMPTS, hits of one object merged
  label    = container if a container prompt scores higher or it is bigger than MAX_BLOCK_M (metric size
             from depth); block if 40-150 mm; blocks whose centre lies in a container are "in_container"
  grasp    = rectangle centre at mid-height of the block, jaws closing across the short side

  HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 perception-main/ai_bubble/.venv/bin/python perception-main/block_grasp.py
      [--image data/.../color.jpg]   (uses rs_depth_m.npy + intrinsics.json next to it)
Outputs: perception-main/data/blocks/<stamp>/ (result.png, blocks.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "research" / "box_cells"), str(HERE / "ai_failure"), str(HERE)]
from rect_fit import ransac_rect, refine_rect  # noqa: E402

VGGT_ID = "facebook/VGGT-1B-Commercial"

BLOCK_PROMPTS = ["bubble wrap block.", "styrofoam block.", "box wrapped in bubble wrap."]
CONTAINER_PROMPTS = ["wooden tray.", "basket with dividers.", "grid of dividers.", "basket.", "tray."]
CONTAINER_WORDS = ("tray", "basket", "grid", "divider")
GDINO_THR = 0.10                 # box + text threshold (low: few misses; DBSCAN on the 3D points checks every hit)
GDINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam2.1-hiera-large"
MIN_BLOCK_M, MAX_BLOCK_M = 0.04, 0.15
MIN_BLOCK_SCORE = 0.10           # with the Qwen3-VL prior (the scan): Qwen decides which hits are blocks
GDINO_THR_FAST, MIN_BLOCK_SCORE_FAST = 0.20, 0.20   # fast checks during the run (no prior): fewer, surer hits
CONTAINER_M = (0.20, 0.50)      # tray / basket footprint; bigger = a table, smaller = not a container
CONTAINER_WALL_M = 0.02         # rim must stand this much above the table (a flat ChArUco board does not)
GRIPPER_MAX_M = 0.100            # on-site gripper stroke (max_range_config 100 mm)
CALIB_TOP = HERE.parent / "pac2026" / "calib" / "top_cam_to_base.json"
REACH_M = 0.626


def plane_ransac(Q, tol=0.004, iters=1000, max_tilt_deg=20, seed=0, refit_tol=0.007):
    """Near-horizontal plane through Nx3 points; returns (unit normal up, point on it, inlier mask)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < np.cos(np.radians(max_tilt_deg)):
            continue
        inl = np.abs((Q - a) @ n) < tol
        if best is None or inl.sum() > best[1].sum():
            best = (n, inl)
    n, inl = best
    for _ in range(4):                      # refit on the inliers, re-select: converges to the same set
        c = Q[inl].mean(0)                  # whatever sample the RANSAC started from
        _, _, vt = np.linalg.svd(Q[inl] - c, full_matrices=False)
        n = vt[2] * np.sign(vt[2][2])
        new = np.abs((Q - c) @ n) < refit_tol   # wider than tol: a bulging bubble-wrap top is one face
        if (new == inl).all():
            break
        inl = new
    return n, c, inl


def long_side_grasp(corners):
    """Rectangle (4x2, in order) -> grasp: the two LONG sides are where the fingers press; the centre line
    runs between them (from the midpoint of one short side to the other), and the grasp point is its middle.
    Returns (centre xy, jaw closing direction (unit, across the long sides), short len, long len, centre line 2x2)."""
    c = np.asarray(corners, float)
    e = [c[(i + 1) % 4] - c[i] for i in range(4)]
    L = [np.linalg.norm(v) for v in e]
    if L[0] + L[2] >= L[1] + L[3]:          # sides 0 and 2 are the long ones
        long_a, long_b = (c[0], c[1]), (c[2], c[3])
        short_mids = ((c[1] + c[2]) / 2, (c[3] + c[0]) / 2)
    else:
        long_a, long_b = (c[1], c[2]), (c[3], c[0])
        short_mids = ((c[0] + c[1]) / 2, (c[2] + c[3]) / 2)
    ma, mb = np.mean(long_a, 0), np.mean(long_b, 0)          # midpoints of the long sides
    centre = (ma + mb) / 2                                    # middle of the centre line
    jaw = (mb - ma) / max(np.linalg.norm(mb - ma), 1e-9)      # fingers close from one long side to the other
    short = float(np.linalg.norm(mb - ma))
    long_ = float(np.linalg.norm(short_mids[0] - short_mids[1]))
    return centre, jaw, short, long_, np.array(short_mids)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


class BlockFinder:
    def __init__(self, device="cuda", use_vggt=True):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.device = device
        self.gd_proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.gd = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(SAM_ID)
        self.sam = Sam2Model.from_pretrained(SAM_ID).to(device).eval()
        self.T = np.array(json.loads(CALIB_TOP.read_text())["T_base_cam"])
        self.use_vggt = use_vggt
        self.vggt = None

    # ---------------------------------------------------------------- detection
    def _gdino(self, img, prompts, thr=0.25):
        W, H = img.size
        hits = []
        for p in prompts:
            inp = self.gd_proc(images=img, text=p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                r = self.gd_proc.post_process_grounded_object_detection(self.gd(**inp), inp.input_ids, threshold=thr,
                                                                        text_threshold=thr, target_sizes=[(H, W)])[0]
            hits += [(float(s), [float(v) for v in b], p) for b, s in zip(r["boxes"].tolist(), r["scores"].tolist())
                     if (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * W * H]          # table-sized hits dropped
        return hits

    def _gdino_batched(self, img, thr=GDINO_THR):
        """All block and container prompts in ONE forward pass (~0.2 s instead of one pass per prompt).
        The phrase(s) a hit matched decide its group: any container word -> container, else block."""
        W, H = img.size
        text = " ".join(BLOCK_PROMPTS + CONTAINER_PROMPTS)
        inp = self.gd_proc(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            r = self.gd_proc.post_process_grounded_object_detection(self.gd(**inp), inp.input_ids, threshold=thr,
                                                                    text_threshold=thr, target_sizes=[(H, W)])[0]
        out = []
        for b, sc, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), r["text_labels"]):
            if (b[2] - b[0]) * (b[3] - b[1]) >= 0.5 * W * H:
                continue                                         # table-sized
            group = "container" if any(w in lab for w in CONTAINER_WORDS) else "block"
            out.append((group, float(sc), [float(v) for v in b], lab))
        return out

    def _objects(self, img):
        """Merge block and container hits that cover the same object; keep the best score of each group."""
        objs = []
        for group, s, b, p in self._gdino_batched(img, GDINO_THR if getattr(self, "prior", None) is not None
                                                  else GDINO_THR_FAST):
            for o in objs:
                if iou(o["box"], b) > 0.7:
                    if s > o["score"][group]:
                        o["score"][group], o["prompt"][group] = s, p
                    break
            else:
                o = {"box": b, "score": {"block": 0.0, "container": 0.0}, "prompt": {"block": None, "container": None}}
                o["score"][group], o["prompt"][group] = s, p
                objs.append(o)
        return objs

    # ---------------------------------------------------------------- geometry
    def _deproject(self, u, v, z, K):
        return np.array([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z])

    def _to_base(self, p_cam):
        return self.T[:3, :3] @ p_cam + self.T[:3, 3]

    def _pix_to_base(self, u, v, z, K):
        """Many pixels at once (arrays u, v, depth z) -> base-frame points Nx3."""
        P = np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], 1)
        return P @ self.T[:3, :3].T + self.T[:3, 3]

    def _rise(self, depth, K, box_mask):
        """How far the inside of a box rises above the table around it (m): p95 height inside - table ring."""
        H, W = depth.shape
        ys, xs = np.nonzero(box_mask)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        g = max(10, int(0.15 * max(x2 - x1, y2 - y1)))
        ring = np.zeros((H, W), bool)
        ring[max(y1 - g, 0):y2 + g, max(x1 - g, 0):x2 + g] = True
        ring &= ~box_mask

        def zs(mask):
            v, u = np.nonzero(mask)
            v, u = v[::5], u[::5]
            d = depth[v, u]
            ok = np.isfinite(d) & (d > 0.1)
            P = np.stack([(u[ok] - K["cx"]) / K["fx"] * d[ok], (v[ok] - K["cy"]) / K["fy"] * d[ok], d[ok]], 1)
            return (P @ self.T[:3, :3].T + self.T[:3, 3])[:, 2]

        zi, zr = zs(box_mask), zs(ring)
        if zi.size < 50 or zr.size < 50:
            return 0.0
        rim = np.percentile(zi, 95)
        # a container is hollow: its inside sits well below the rim. A bag / solid object is not.
        hollow = rim - np.median(zi) > 0.5 * (rim - np.median(zr))
        return float(rim - np.median(zr)) if hollow else 0.0

    def _gdino_crop(self, img, box, grow=0.35):
        """Grounding DINO on a crop around a rough box: the best block hit there, in full-picture px, or None."""
        W, H = img.size
        x1, y1, x2, y2 = box
        gx, gy = grow * (x2 - x1), grow * (y2 - y1)
        cx1, cy1, cx2, cy2 = (int(max(0, x1 - gx)), int(max(0, y1 - gy)), int(min(W, x2 + gx)), int(min(H, y2 + gy)))
        hits = [h for h in self._gdino_batched(img.crop((cx1, cy1, cx2, cy2))) if h[0] == "block"]
        if not hits:
            return None
        g, sc, b, lab = max(hits, key=lambda h: h[1] * iou([b + o for b, o in zip(h[2], (cx1, cy1, cx1, cy1))], box))
        b = [b[0] + cx1, b[1] + cy1, b[2] + cx1, b[3] + cy1]
        return (sc, b, lab) if iou(b, box) > 0.2 else None

    def _match_prior(self, img, objs, prior):
        """Qwen3-VL's rough boxes decide HOW MANY blocks there are and roughly where; Grounding DINO gives the
        clean box of each: the hit overlapping the rough box most (any score), else Grounding DINO on a crop
        around it, else the rough box itself. Hits that match no rough box are not blocks."""
        cands = [o for o in objs if o["label"] in ("block", "other") and o["score"]["block"] > 0]
        used, blocks, rep = set(), [], dict(qwen=len(prior), gdino=0, crop=0, qwen_box=0)
        for p in prior:
            best = max(((iou(o["box"], p["box"]), i) for i, o in enumerate(cands) if i not in used), default=(0, None))
            if best[1] is not None and best[0] >= 0.4:
                used.add(best[1])
                o = cands[best[1]]
                o.update(label="block", qwen_pose=p.get("pose"), why=None)
                blocks.append(o)
                rep["gdino"] += 1
                continue
            c = self._gdino_crop(img, p["box"])
            if c is not None:
                sc, b, lab = c
                blocks.append(dict(box=b, score={"block": sc, "container": 0.0}, prompt={"block": lab, "container": None},
                                   label="block", qwen_pose=p.get("pose"), from_crop=True))
                rep["crop"] += 1
            else:
                blocks.append(dict(box=list(p["box"]), score={"block": 0.0, "container": 0.0},
                                   prompt={"block": "Qwen", "container": None}, label="block", qwen_pose=p.get("pose"),
                                   qwen_box=True))
                rep["qwen_box"] += 1
        rest = []
        for i, o in enumerate(cands):
            if i not in used and o["label"] == "block":
                o["label"], o["why"] = "other", "no Qwen block here"
        for o in objs:
            if not any(o is b for b in blocks):
                rest.append(o)
        return blocks + rest, rep

    def _median_depth(self, depth, mask):
        z = depth[mask]
        z = z[np.isfinite(z) & (z > 0.1)]
        return float(np.median(z)) if z.size > 20 else None

    def __call__(self, color_bgr, depth_m, K):
        img = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        H, W = depth_m.shape
        objs = self._objects(img)
        min_score = MIN_BLOCK_SCORE if getattr(self, "prior", None) is not None else MIN_BLOCK_SCORE_FAST
        for o in objs:                                   # metric size from the box and the depth inside it
            x1, y1, x2, y2 = (int(round(v)) for v in o["box"])
            m = np.zeros((H, W), bool)
            m[max(y1, 0):y2, max(x1, 0):x2] = True
            z = self._median_depth(depth_m, m)
            o["size_m"] = None if z is None else [(x2 - x1) * z / K["fx"], (y2 - y1) * z / K["fy"]]
            size = max(o["size_m"]) if o["size_m"] else 0.0
            o["wall_m"] = self._rise(depth_m, K, m)
            if o["score"]["container"] > o["score"]["block"] or size > MAX_BLOCK_M:
                ok = CONTAINER_M[0] <= size <= CONTAINER_M[1] and o["wall_m"] >= CONTAINER_WALL_M
                o["label"] = "container" if ok else "other"
                o["why"] = None if ok else (f"size {size * 100:.0f} cm" if not CONTAINER_M[0] <= size <= CONTAINER_M[1]
                                            else "flat or solid (no walls around a hollow inside)")
            elif size < MIN_BLOCK_M or o["score"]["block"] < min_score:
                o["label"] = "other"
                o["why"] = f"block score {o['score']['block']:.2f}" if size >= MIN_BLOCK_M else "too small"
            else:
                o["label"] = "block"
        prior, self.prior = getattr(self, "prior", None), None     # Qwen3-VL boxes, for this call only
        self.last_prior = None
        if prior is not None:                            # block count / rough places from Qwen, clean boxes
            try:                                         # from Grounding DINO
                objs, self.last_prior = self._match_prior(img, objs, prior)
            except Exception as e:                       # never lose the Grounding DINO result for this
                self.last_prior = dict(error=str(e))
        containers = [o for o in objs if o["label"] == "container"]
        blocks = [o for o in objs if o["label"] == "block"]
        for b in blocks:
            cx, cy = (b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2
            b["in_container"] = any(c["box"][0] <= cx <= c["box"][2] and c["box"][1] <= cy <= c["box"][3]
                                    for c in containers)
        if blocks:
            self._segment_and_fit(img, blocks, depth_m, K)
            if self.use_vggt:
                self._vggt_top_faces(color_bgr, blocks, depth_m, K)
        return objs

    # ---------------------------------------------------------------- VGGT top face
    def _vggt_points(self, color_bgr, depth_m, K):
        """Whole-frame VGGT depth, metric via RealSense (median ratio), unprojected with the real
        intrinsics, in the robot base frame. (A tight crop around one block flattens VGGT's depth.)"""
        import tempfile
        sys.path.insert(0, str(HERE / "vggt"))
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        if self.vggt is None:
            self.vggt = VGGT.from_pretrained(VGGT_ID).to(self.device).eval()
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            cv2.imwrite(f.name, color_bgr)
            images = load_and_preprocess_images([f.name]).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            d = self.vggt(images[None])["depth"][0, 0, ..., 0].float().cpu().numpy()
        vh, vw = d.shape
        H, W = depth_m.shape
        rs = cv2.resize(np.nan_to_num(depth_m, nan=0.0), (vw, vh), interpolation=cv2.INTER_NEAREST)
        ok = rs > 0.1
        d = d * float(np.median(rs[ok] / d[ok]))
        u = (np.arange(vw) + 0.5) * W / vw - 0.5
        v = (np.arange(vh) + 0.5) * H / vh - 0.5
        uu, vv = np.meshgrid(u, v)
        P = np.stack([(uu - K["cx"]) / K["fx"] * d, (vv - K["cy"]) / K["fy"] * d, d], -1).reshape(-1, 3)
        return (P @ self.T[:3, :3].T + self.T[:3, 3]).reshape(vh, vw, 3)

    def _vggt_top_faces(self, color_bgr, blocks, depth_m, K):
        """Per block: the high part of the VGGT points (top 2 cm) -> plane RANSAC (top height) ->
        rectangle RANSAC on the top face seen from above (centre, size, jaw direction).
        Height = VGGT top plane - VGGT table ring; absolute z = RealSense table + that height."""
        try:
            Pv = self._vggt_points(color_bgr, depth_m, K)
        except Exception as e:                                  # keep the RealSense estimate
            print(f"VGGT top face skipped: {e}")
            return
        vh, vw = Pv.shape[:2]
        for b in blocks:
            if "mask" not in b or "robot" not in b:
                continue
            m = cv2.resize(b["mask"].astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST).astype(bool)
            Q = Pv[m]
            if len(Q) < 80:
                continue
            ring = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & \
                ~cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            tz_v = float(np.median(Pv[ring][:, 2]))
            hi = Q[Q[:, 2] > np.percentile(Q[:, 2], 95) - 0.02]
            n, c, inl = plane_ransac(hi)
            top = hi[inl]
            xy = top[:, :2] * 1000
            o = xy.min(0) - 10
            g = np.zeros((int(np.ptp(xy[:, 1])) + 21, int(np.ptp(xy[:, 0])) + 21), np.uint8)
            uv = (xy - o).astype(int)
            g[uv[:, 1], uv[:, 0]] = 1
            g = cv2.morphologyEx(cv2.dilate(g, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            cnt = max(cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0, :]
            _, _, _, ratio, corners = ransac_rect(cnt, iters=1000, tol=2.0)
            ref = refine_rect(cnt, corners)                     # right-angle least squares on the inliers
            if ref is not None:
                corners = ref[4]
            corners = (corners + o) / 1000
            centre, jaw, l_short, l_long, cline = long_side_grasp(corners)
            l1, l2 = l_short, l_long
            r = b["robot"]
            height = float(c[2] - tz_v)
            table_z = r["table_z_m"]                                # RealSense: absolute table height
            top_xyz = np.array([centre[0], centre[1], table_z + height])
            r.update(top_xyz_m=top_xyz.round(4).tolist(), height_m=round(height, 4),
                     grasp_xyz_m=[round(centre[0], 4), round(centre[1], 4), round(table_z + 0.5 * height, 4)],
                     size_m=[round(float(min(l1, l2)), 4), round(float(max(l1, l2)), 4)],
                     jaw_yaw_deg=round(float(np.degrees(np.arctan2(jaw[1], jaw[0]))) % 180, 1),
                     reach_m=round(float(np.hypot(*centre)), 3),
                     fits_gripper=bool(min(l1, l2) < GRIPPER_MAX_M - 0.005),
                     method="vggt top plane + rect RANSAC",
                     top_plane_tilt_deg=round(float(np.degrees(np.arccos(abs(n[2])))), 1),
                     top_inliers=int(inl.sum()), rect_fit=round(float(ratio), 2))
            b["top_face_corners_m"] = np.c_[corners, np.full(4, top_xyz[2])].round(4).tolist()
            b["centerline_m"] = np.c_[cline, np.full(2, top_xyz[2])].round(4).tolist()

    def _segment_and_fit(self, img, blocks, depth_m, K):
        boxes = [[float(v) for v in b["box"]] for b in blocks]
        sin = self.sam_proc(images=img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            so = self.sam(**sin, multimask_output=False)
        masks = self.sam_proc.post_process_masks(so.pred_masks.cpu(), sin["original_sizes"])[0][:, 0].numpy()
        H, W = depth_m.shape
        for b, m in zip(blocks, masks):
            x1, y1, x2, y2 = (int(round(v)) for v in b["box"])
            inside = np.zeros((H, W), bool)
            inside[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = True
            m = m.astype(bool) & inside
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
            if n < 2:
                continue
            m = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))     # largest piece
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(cnts, key=cv2.contourArea)[:, 0, :]
            center, size, ang, ratio, corners = ransac_rect(cnt, iters=1000)
            ref = refine_rect(cnt, corners)
            if ref is not None:
                center, size, ang, _, corners = ref
            b["mask"] = m
            b["rect_px"] = dict(center=[float(v) for v in center], corners=corners.tolist(), inlier_ratio=float(ratio))
            # 3D: top surface = depth on the inner part of the mask; table = a ring around the block
            core = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            z_top = self._median_depth(depth_m, core)
            ring = cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & \
                ~cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            if z_top is None:
                continue
            top = self._to_base(self._deproject(*center, z_top, K))
            rv, ru = np.nonzero(ring)
            rd = depth_m[rv, ru]
            rok = np.isfinite(rd) & (rd > 0.1)
            ring_pts = self._pix_to_base(ru[rok][::7], rv[rok][::7], rd[rok][::7], K)
            table_z = float(np.median(ring_pts[:, 2])) if len(ring_pts) else float("nan")
            C = np.array([self._to_base(self._deproject(u, v, z_top, K)) for u, v in corners])   # rect corners at top
            e1, e2 = C[1] - C[0], C[3] - C[0]
            l1, l2 = np.linalg.norm(e1[:2]), np.linalg.norm(e2[:2])
            jaw = e1 if l1 <= l2 else e2                     # jaws close across the short side
            # the camera also sees the block's sides (it is 10 cm tall), which inflates the image outline:
            # use only the top face in 3D (points within 15 mm of the highest part) for centre / size / angle
            vs, us = np.nonzero(m)
            ds = depth_m[vs, us]
            ok = np.isfinite(ds) & (ds > 0.1)
            P = self._pix_to_base(us[ok][::3], vs[ok][::3], ds[ok][::3], K)
            if len(P) > 100:
                zt = np.percentile(P[:, 2], 90)
                F = P[P[:, 2] > zt - 0.015]
                if len(F) > 60:
                    (cx, cy), (w, h), ang = cv2.minAreaRect((F[:, :2] * 1000).astype(np.float32))
                    top = np.array([cx / 1000, cy / 1000, zt])
                    a = np.radians(ang)
                    d1, d2 = np.array([np.cos(a), np.sin(a), 0]), np.array([-np.sin(a), np.cos(a), 0])
                    l1, l2 = w / 1000, h / 1000
                    jaw = d1 if l1 <= l2 else d2
                    b["top_face_pts"] = int(len(F))
            height = top[2] - table_z
            grasp = top.copy()
            grasp[2] = table_z + 0.5 * height if np.isfinite(table_z) else top[2] - 0.03
            b["robot"] = dict(top_xyz_m=top.round(4).tolist(), table_z_m=round(table_z, 4), height_m=round(height, 4),
                              grasp_xyz_m=grasp.round(4).tolist(),
                              size_m=[round(float(min(l1, l2)), 4), round(float(max(l1, l2)), 4)],
                              jaw_yaw_deg=round(float(np.degrees(np.arctan2(jaw[1], jaw[0]))) % 180, 1),
                              reach_m=round(float(np.hypot(top[0], top[1])), 3),
                              fits_gripper=bool(min(l1, l2) < GRIPPER_MAX_M - 0.005))


def draw(color, objs):
    vis = color.copy()
    for o in objs:
        x1, y1, x2, y2 = (int(round(v)) for v in o["box"])
        if o["label"] == "container":
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(vis, f"container {o['score']['container']:.2f}", (x1 + 4, y1 + 24), 0, 0.7, (255, 0, 255), 2)
        elif o["label"] == "block":
            col = (128, 128, 128) if o.get("in_container") else (0, 255, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
            if "mask" in o:
                vis[o["mask"]] = (0.6 * vis[o["mask"]] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
                cv2.polylines(vis, [np.array(o["rect_px"]["corners"], np.int32)], True, (0, 0, 255), 2)
                cx, cy = (int(v) for v in o["rect_px"]["center"])
                cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 3)
            r = o.get("robot")
            txt = f"block {o['score']['block']:.2f}" + (" (in container)" if o.get("in_container") else "")
            cv2.putText(vis, txt, (x1, y1 - 24), 0, 0.55, col, 2)
            if r:
                ok = r["reach_m"] < REACH_M - 0.07 and r["fits_gripper"]
                cv2.putText(vis, f"{r['size_m'][0] * 1000:.0f}x{r['size_m'][1] * 1000:.0f}mm h{r['height_m'] * 1000:.0f} "
                                 f"{r['reach_m'] * 100:.0f}cm", (x1, y1 - 6), 0, 0.55, (0, 200, 0) if ok else (0, 0, 255), 2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(vis, f"ignored: {o.get('why', '')}", (x1 + 4, y2 - 8), 0, 0.5, (160, 160, 160), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, help="colour image with rs_depth_m.npy + intrinsics.json next to it")
    args = ap.parse_args()
    out = HERE / "data" / "blocks" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    if args.image:
        color = cv2.imread(str(args.image))
        depth = np.load(args.image.parent / "rs_depth_m.npy")
        K = json.loads((args.image.parent / "intrinsics.json").read_text())
    else:
        sys.path.insert(0, str(HERE / "research" / "box_cells"))
        from gdino_vggt_lines import grab_top
        color, depth, K = grab_top()
    cv2.imwrite(str(out / "color.jpg"), color)
    np.save(out / "rs_depth_m.npy", depth.astype(np.float32))
    (out / "intrinsics.json").write_text(json.dumps(K))
    t0 = time.time()
    finder = BlockFinder()
    t1 = time.time()
    objs = finder(color, depth, K)
    t2 = time.time()
    cv2.imwrite(str(out / "result.png"), draw(color, objs))
    rows = []
    for o in objs:
        r = {k: v for k, v in o.items() if k != "mask"}
        rows.append(r)
        if o["label"] == "container":
            print(f"container  {o['score']['container']:.2f} '{o['prompt']['container']}'  "
                  f"~{o['size_m'][0] * 100:.0f}x{o['size_m'][1] * 100:.0f} cm, walls {o['wall_m'] * 1000:.0f} mm")
        elif o["label"] == "other":
            g = "container" if o["score"]["container"] > o["score"]["block"] else "block"
            print(f"ignored    {o['score'][g]:.2f} '{o['prompt'][g]}' as {g}: {o.get('why')}")
        elif o["label"] == "block":
            rb = o.get("robot")
            s = (f"block      {o['score']['block']:.2f} '{o['prompt']['block']}'"
                 + ("  [in container]" if o["in_container"] else ""))
            if rb:
                g = np.array(rb["grasp_xyz_m"]) * 1000
                s += (f"  grasp ({g[0]:.0f}, {g[1]:.0f}, {g[2]:.0f}) mm  jaw yaw {rb['jaw_yaw_deg']:.0f} deg  "
                      f"{rb['size_m'][0] * 1000:.0f}x{rb['size_m'][1] * 1000:.0f} mm h {rb['height_m'] * 1000:.0f}  "
                      f"reach {rb['reach_m'] * 1000:.0f} mm{'' if rb['fits_gripper'] else '  TOO WIDE'}")
            print(s)
    (out / "blocks.json").write_text(json.dumps(rows, indent=1))
    print(f"models {t1 - t0:.1f} s, detect+segment {t2 - t1:.1f} s -> {out}")


if __name__ == "__main__":
    main()

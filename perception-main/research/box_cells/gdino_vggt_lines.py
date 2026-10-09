"""Live: Grounding DINO box -> VGGT point cloud -> points inside the box -> Canny + RANSAC partition grid.

  top D435i colour (+ aligned RealSense depth, only for metric scale)
  -> Grounding DINO with several prompts, highest-confidence non-table hit -> box rectangle
  -> square crop around the box (box fills VGGT's 518 px input; --no-crop for the whole frame)
  -> VGGT-1B-Commercial depth / point map from the crop, scaled to metres with RealSense
  -> point cloud in a table frame (table z = 0, +z up; the camera looks down) cropped to the rectangle (PLY)
  -> box_lines.detect on the VGGT depth inside the rectangle (height slice, Canny, sequential RANSAC)

  HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 perception-main/ai_bubble/.venv/bin/python \
      perception-main/research/box_cells/gdino_vggt_lines.py [--image img.jpg]
Outputs: perception-main/data/vggt/live_<stamp>/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PM = HERE.parents[1]                                   # perception-main/
sys.path[:0] = [str(HERE), str(PM / "ai_failure"), str(PM / "vggt"), str(PM.parent / "pac2026")]
import box_lines as bl  # noqa: E402
from capture_cells import to_color_px, to_vggt_px  # noqa: E402
from box_height import ransac_plane, roi_masks  # noqa: E402
from label_points import write_ply  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


# Several descriptions of the box; each is run separately and the highest-confidence detection wins.
# Grounding DINO scores are text-image match, not probabilities (a perfect hit is often only ~0.5),
# so one prompt alone is fragile: "tray." or "partition box." pick the white tables on some frames.
PROMPTS = ["wooden tray.", "box with partitions.", "partitioned box.", "grid of dividers.",
           "partition box.", "cardboard box."]
MAX_AREA = 0.5          # a hit covering more than this fraction of the frame is a table, not the box
VGGT_ID = "facebook/VGGT-1B-Commercial"
_GDINO = {}
_VGGT = {}


def share_models(gd_proc=None, gd=None, vggt_model=None):
    """Use models a caller already has on the GPU (e.g. BlockFinder's) instead of loading a second copy."""
    if gd_proc is not None and gd is not None:
        _GDINO["proc"], _GDINO["model"] = gd_proc, gd
    if vggt_model is not None:
        _VGGT["model"] = vggt_model


def load_vggt():
    if "model" not in _VGGT:
        _VGGT["model"] = VGGT.from_pretrained(VGGT_ID).to("cuda").eval()
    return _VGGT["model"]


def find_box_multi(img_bgr, prompts=PROMPTS, threshold=0.25, grow=0.03):
    """Run every prompt, drop table-sized hits, return the highest-confidence box (colour px, grown a
    little) and a report: winning prompt/score and which other prompts found the same box (IoU > 0.8)."""
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    mid = "IDEA-Research/grounding-dino-base"
    if not _GDINO:
        _GDINO["proc"] = AutoProcessor.from_pretrained(mid)
        _GDINO["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda").eval()
    proc, model = _GDINO["proc"], _GDINO["model"]
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    W, H = img.size
    hits = []                                            # (score, box, matched phrase)
    inp = proc(images=img, text=" ".join(prompts), return_tensors="pt").to("cuda")   # one pass for all prompts
    with torch.no_grad():
        r = proc.post_process_grounded_object_detection(model(**inp), inp.input_ids, threshold=threshold,
                                                        text_threshold=threshold, target_sizes=[(H, W)])[0]
    for b, sc, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), r["text_labels"]):
        if (b[2] - b[0]) * (b[3] - b[1]) <= MAX_AREA * W * H:
            hits.append((sc, b, lab))
    if not hits:
        return None, {"prompts": prompts, "hits": 0}

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        i = ix * iy
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

    score, (x1, y1, x2, y2), prompt = max(hits, key=lambda h: h[0])
    agree = sorted({p for sc, b, p in hits if iou(b, [x1, y1, x2, y2]) > 0.8})
    gx, gy = grow * (x2 - x1), grow * (y2 - y1)
    report = {"prompt": prompt, "score": round(score, 3), "agreeing_prompts": agree,
              "all": [(p, round(sc, 3), [round(v) for v in b]) for sc, b, p in sorted(hits, reverse=True)]}
    return [x1 - gx, y1 - gy, x2 + gx, y2 + gy], report


def grab_top():
    import yaml
    from pac.cameras import CameraManager
    cfg = yaml.safe_load((PM.parent / "pac2026" / "config.yaml").read_text())
    cams = CameraManager(cfg)
    try:
        time.sleep(1.0)                                 # exposure settles
        f = cams.top(with_depth=True)
    finally:
        cams.close()
    return f.color, f.depth_m, f.intrinsics


def top_intrinsics():
    """Colour intrinsics of the top camera at 1280x720 (offline runs that did not save them)."""
    import pyrealsense2 as rs
    import yaml
    c = yaml.safe_load((PM.parent / "pac2026" / "config.yaml").read_text())["camera"]["top"]
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(str(c["serial"]))
    cfg.enable_stream(rs.stream.color, *c["color_size"], rs.format.bgr8, c["fps"])
    i = pipe.start(cfg).get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy}


def unproject(d_m, K, ox, oy, crop_w, crop_h):
    """VGGT depth (vh x vw grid over the crop) -> camera-frame points with the REAL colour intrinsics.
    VGGT's own focal guess is unreliable on a crop (the box comes out too big), its depth is fine."""
    vh, vw = d_m.shape
    u = (np.arange(vw) + 0.5) * crop_w / vw - 0.5 + ox
    v = (np.arange(vh) + 0.5) * crop_h / vh - 0.5 + oy
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - K["cx"]) / K["fx"] * d_m, (vv - K["cy"]) / K["fy"] * d_m, d_m], -1)


def vggt(img_path):
    model = load_vggt()
    images = load_and_preprocess_images([str(img_path)]).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images[None])
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"][0].float().cpu().numpy()                     # 1xHxWx1
    pts = unproject_depth_map_to_point_map(depth, extri[0].float().cpu().numpy(),
                                           intri[0].float().cpu().numpy())[0]
    rgb = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return depth[0, ..., 0], pts, rgb


def to_table_frame(pts, roi_v):
    """Camera looks down at the table: express the cloud in a table frame instead of the camera's.
    Table plane (RANSAC on a thin ring around the box) -> z = 0, +z up toward the camera, x along the
    image right edge, y toward the image top, origin under the box centre. z is then height above the table."""
    vh, vw = pts.shape[:2]
    table_m, box_m = roi_masks((vh, vw), roi_v)
    P = pts[table_m].reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    n, d, frac = ransac_plane(P)                       # n points toward the camera = up
    ex = np.array([1.0, 0, 0]) - n[0] * n              # camera x projected onto the table
    ex /= np.linalg.norm(ex)
    R = np.stack([ex, np.cross(n, ex), n])             # rows: table x, y, z in camera coordinates
    c = pts[box_m].reshape(-1, 3).mean(0)
    origin = c - (n @ c + d) * n                       # box centre dropped onto the table
    tilt = float(np.degrees(np.arccos(abs(n[2]))))
    return (pts - origin) @ R.T, dict(R=R, origin=origin, table_inliers=float(frac), camera_tilt_deg=tilt)


def split_box_partition(res, z, box_h, raised_frac=0.5, wall_band=0.04):
    """Separate the box from the partition among raised points (z = height above the table, VGGT grid).
    The outermost line group on each side is a box wall, the groups in between are partition dividers:
    a raised point belongs to whichever is nearer. Returns label map: 0 none, 1 box wall, 2 partition."""
    corners = res["corners"]
    tl, tr, bl_ = corners[0], corners[1], corners[3]
    ax = (tr - tl) / np.linalg.norm(tr - tl)
    ay = (bl_ - tl) / np.linalg.norm(bl_ - tl)
    vh, vw = z.shape
    yy, xx = np.mgrid[0:vh, 0:vw]
    P = np.stack([xx, yy], -1).astype(np.float32) - tl
    off = {0: P @ ay, 1: P @ ax}                       # h groups are spaced along ay, v groups along ax
    d_wall = np.full((vh, vw), np.inf)
    d_div = np.full((vh, vw), np.inf)
    for axis, groups in ((0, res["h_groups"]), (1, res["v_groups"])):
        cen = [np.mean([l.offset for l in g]) for g in groups]
        for k, c in enumerate(cen):
            d = np.abs(off[axis] - c)
            if k in (0, len(cen) - 1):
                d_wall = np.minimum(d_wall, d)
            else:
                d_div = np.minimum(d_div, d)
    m = np.zeros((vh, vw), np.uint8)
    cv2.fillConvexPoly(m, corners.astype(np.int32), 1)
    m = cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(bool)
    raised = m & (np.nan_to_num(z, nan=-1) > raised_frac * box_h)
    lab = np.zeros((vh, vw), np.uint8)
    side = min(np.linalg.norm(tr - tl), np.linalg.norm(bl_ - tl))
    wall = d_wall <= np.maximum(d_div, wall_band * side)  # where a divider meets the wall, the wall wins
    lab[raised & wall] = 1
    lab[raised & ~wall] = 2
    return lab


REACH_M = 0.626          # PiPER reach
CALIB_TOP = PM.parent / "pac2026" / "calib" / "top_cam_to_base.json"


def basket_spec():
    """Known basket (config perception.basket): rows, cols, wall height - or None."""
    import yaml
    c = yaml.safe_load((PM.parent / "pac2026" / "config.yaml").read_text()).get("perception", {}).get("basket")
    return c or None


def equal_cells(corners, rows, cols):
    """Split the box outline (tl, tr, br, bl) into rows x cols equal quads (fallback when Canny finds
    a different grid, e.g. because of a block standing in a cell)."""
    tl, tr, br, bot_l = (np.asarray(c, float) for c in corners)

    def pt(u, v):                                        # bilinear inside the outline, u along tl->tr, v along tl->bl
        return (1 - v) * ((1 - u) * tl + u * tr) + v * ((1 - u) * bot_l + u * br)
    cells = []
    for r_ in range(rows):
        for c_ in range(cols):
            q = np.array([pt(c_ / cols, r_ / rows), pt((c_ + 1) / cols, r_ / rows),
                          pt((c_ + 1) / cols, (r_ + 1) / rows), pt(c_ / cols, (r_ + 1) / rows)], np.float32)
            cells.append(bl.Cell(r_, c_, q, tuple(q.mean(0)), float("nan")))
    return cells


def cells_to_robot(quads, depth_m, K, roi_c, wall_h=None):
    """Cell centres in the robot base frame from RealSense depth (aligned to colour) + the top-camera
    calibration. Returns per cell: floor centre xyz (m), and the box rim height (m) for the whole box."""
    T = np.array(json.loads(CALIB_TOP.read_text())["T_base_cam"])
    H, W = depth_m.shape
    vv, uu = np.mgrid[0:H, 0:W]
    d = np.nan_to_num(depth_m, nan=0.0)

    def base_pts(mask):
        z = d[mask]
        ok = z > 0.1
        u, v, z = uu[mask][ok], vv[mask][ok], z[ok]
        P = np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], 1)
        return P @ T[:3, :3].T + T[:3, 3]

    # walls / dividers only: a thin band along the cell outlines (dividers lie between neighbouring cells,
    # the outer wall just outside them). Not the whole detector box: a block standing next to the basket
    # or sitting in a cell must not count as the rim.
    cells_m = np.zeros((H, W), np.uint8)
    for q in quads:
        cv2.fillConvexPoly(cells_m, np.asarray(q, np.int32), 1)
    band = cv2.dilate(cells_m, np.ones((21, 21), np.uint8)).astype(bool) & \
        ~cv2.erode(cells_m, np.ones((9, 9), np.uint8)).astype(bool)
    if wall_h is not None:                               # known basket: rim = table next to the basket + wall
        x1, y1, x2, y2 = (int(round(v)) for v in roi_c)
        g = int(0.12 * max(x2 - x1, y2 - y1))
        ring = np.zeros((H, W), bool)
        ring[max(y1 - g, 0):y2 + g, max(x1 - g, 0):x2 + g] = True
        ring[max(y1, 0):y2, max(x1, 0):x2] = False
        rim_z = float(np.median(base_pts(ring)[:, 2])) + wall_h
    else:
        zb = base_pts(band)[:, 2]
        zb = zb[zb < np.median(zb) + 0.06]              # drop blocks standing in cells (taller than the walls)
        rim_z = float(np.percentile(zb, 90))             # wall / divider tops
    out = []
    for q in quads:
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, np.asarray(q, np.int32), 1)
        core = cv2.erode(m, np.ones((15, 15), np.uint8)).astype(bool)      # away from the walls
        P = base_pts(core)
        if len(P) < 30:
            out.append(None)
            continue
        floor_z = float(np.median(P[:, 2]))
        c = np.asarray(q, float).mean(0)
        zc = float(np.median(d[core][d[core] > 0.1]))
        pc = np.array([(c[0] - K["cx"]) / K["fx"] * zc, (c[1] - K["cy"]) / K["fy"] * zc, zc])
        xyz = T[:3, :3] @ pc + T[:3, 3]
        xyz[2] = floor_z
        out.append(dict(floor_xyz_m=xyz.round(4).tolist(), reach_m=round(float(np.hypot(xyz[0], xyz[1])), 3)))
    return out, rim_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, help="offline colour image (then --scale is used, no RealSense)")
    ap.add_argument("--prompt", action="append", help="Grounding DINO prompt(s); default: all of PROMPTS, best score wins")
    ap.add_argument("--no-crop", action="store_true", help="feed the whole frame to VGGT (old behaviour)")
    ap.add_argument("--slice", type=float, nargs=2, metavar=("LO", "HI"),
                    help="Canny height slice as fractions of the wall height (default 0.3 0.7; lower e.g. 0.1 0.35)")
    ap.add_argument("--margin", type=float, default=0.25, help="crop margin around the box, fraction of its size")
    ap.add_argument("--scale-ref-m", type=float, help="offline: camera-to-table distance for metric scale")
    args = ap.parse_args()
    summary, out = scan(args.image, args.prompt, args.no_crop, args.slice, args.margin, args.scale_ref_m)
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=1))
    print("saved", out)


def scan(image=None, prompts=None, no_crop=False, slice_frac=None, margin=0.25, scale_ref_m=None):
    """The whole pipeline on one frame (image path, or a live top-camera grab when None).
    Returns (summary dict, output dir). Raises SystemExit when Grounding DINO finds no box."""
    args = argparse.Namespace(image=None if image is None else Path(image), prompt=prompts, no_crop=no_crop,
                              slice=slice_frac, margin=margin, scale_ref_m=scale_ref_m)
    out = PM / "data" / "vggt" / f"live_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    t = {}
    t0 = time.time()
    if args.image:
        color = cv2.imread(str(args.image))
        dpath = args.image.parent / "rs_depth_m.npy"          # saved by an earlier live run
        rs_depth = np.load(dpath) if dpath.exists() else None
        kpath = args.image.parent / "intrinsics.json"
        K = json.loads(kpath.read_text()) if kpath.exists() else top_intrinsics()
    else:
        color, rs_depth, K = grab_top()
    (out / "intrinsics.json").write_text(json.dumps(K))
    cv2.imwrite(str(out / "color.jpg"), color)
    if rs_depth is not None:
        np.save(out / "rs_depth_m.npy", rs_depth.astype(np.float32))
    t["capture"] = time.time() - t0

    t0 = time.time()
    roi_c, gd = find_box_multi(color, args.prompt or PROMPTS)
    t["grounding_dino"] = time.time() - t0
    if roi_c is None:
        raise SystemExit("Grounding DINO: no box found")
    print(f"Grounding DINO: '{gd['prompt']}' {gd['score']:.2f}, same box from {len(gd['agreeing_prompts'])} prompt(s): "
          f"{gd['agreeing_prompts']}")

    # VGGT sees 518 px: crop a square around the box (with some table for the plane fit) so the box
    # fills the input instead of ~150 px of a full frame -> thin dividers survive
    H, W = color.shape[:2]
    if args.no_crop:
        ox, oy, img, dep = 0, 0, color, rs_depth
    else:
        bw, bh = roi_c[2] - roi_c[0], roi_c[3] - roi_c[1]
        side = int(min(max(bw, bh) * (1 + 2 * args.margin), H, W))
        ox = int(np.clip((roi_c[0] + roi_c[2]) / 2 - side / 2, 0, W - side))
        oy = int(np.clip((roi_c[1] + roi_c[3]) / 2 - side / 2, 0, H - side))
        img = color[oy:oy + side, ox:ox + side]
        dep = None if rs_depth is None else rs_depth[oy:oy + side, ox:ox + side]
    cv2.imwrite(str(out / "vggt_input.jpg"), img)
    roi_in = [roi_c[0] - ox, roi_c[1] - oy, roi_c[2] - ox, roi_c[3] - oy]
    off = np.array([ox, oy], np.float32)

    t0 = time.time()
    d_v, pts, rgb = vggt(out / "vggt_input.jpg")
    t["vggt"] = time.time() - t0
    vh, vw = d_v.shape
    # metric scale: median ratio RealSense / VGGT over all pixels with valid RealSense depth
    if dep is not None:
        rs_small = cv2.resize(np.nan_to_num(dep, nan=0).astype(np.float32), (vw, vh),
                              interpolation=cv2.INTER_NEAREST)
        ok = rs_small > 0.1
        scale = float(np.median(rs_small[ok] / d_v[ok]))
    else:
        scale = (args.scale_ref_m or 1.0) / float(np.median(d_v))
    d_m = d_v * scale
    pts = unproject(d_m, K, ox, oy, img.shape[1], img.shape[0])   # real intrinsics, not VGGT's guess

    roi_v = to_vggt_px(roi_in, (vh, vw), img.shape[:2])
    x1, y1, x2, y2 = (int(round(v)) for v in roi_v)
    x1, y1 = max(x1, 0), max(y1, 0)
    inside = np.zeros((vh, vw), bool)
    inside[y1:y2 + 1, x1:x2 + 1] = True
    pts, frame = to_table_frame(pts, roi_v)             # z = height above the table, +z up
    np.savez(out / "table_frame.npz", **frame)
    zb = pts[inside][:, 2]
    print(f"table frame: camera tilt {frame['camera_tilt_deg']:.1f} deg, table inliers {frame['table_inliers']:.0%}, "
          f"box points z p50 {np.nanpercentile(zb, 50) * 100:.1f} / p99 {np.nanpercentile(zb, 99) * 100:.1f} cm")
    write_ply(out / "box_points.ply", pts[inside], rgb[inside],
              np.zeros(inside.sum(), np.uint8), np.full(inside.sum(), -1, np.int16))
    write_ply(out / "all_points.ply", pts.reshape(-1, 3), rgb.reshape(-1, 3),
              np.zeros(vh * vw, np.uint8), np.full(vh * vw, -1, np.int16))

    t0 = time.time()
    spec = basket_spec()
    res = bl.detect(d_m, roi_v, box_height=spec["wall_height_m"] if spec else None, slice_frac=args.slice)
    if spec and (res["rows"], res["cols"]) != (spec["rows"], spec["cols"]):
        print(f"grid {res['rows']}x{res['cols']} from Canny/RANSAC != known basket {spec['rows']}x{spec['cols']}: "
              f"using the box outline split into equal cells")
        res["cells"] = equal_cells(res["corners"], spec["rows"], spec["cols"])
        res["rows"], res["cols"] = spec["rows"], spec["cols"]
        res["equal_split"] = True
    t["canny_ransac"] = time.time() - t0

    box_h = float(np.nanpercentile(zb, 99))             # p99 height in the table frame (3D, tilt removed)

    cv2.imwrite(str(out / "lines_vggt.png"), bl.render(res))

    # box and partition as separate objects (3D): outer wall groups vs inner divider groups
    z = pts[..., 2]
    lab = (np.zeros(z.shape, np.uint8) if res.get("equal_split")      # equal split: no wall / divider groups
           else split_box_partition(res, z, box_h))
    P3 = pts.reshape(-1, 3)
    for k, name in ((1, "box_wall"), (2, "partition")):
        sel = (lab == k).reshape(-1)
        write_ply(out / f"{name}.ply", P3[sel], rgb.reshape(-1, 3)[sel],
                  np.full(sel.sum(), k, np.uint8), np.full(sel.sum(), -1, np.int16))
    sep = rgb[..., ::-1].copy() // 3
    sep[lab == 1] = (0, 0, 255)
    sep[lab == 2] = (0, 220, 0)
    sep = cv2.resize(sep, (vw * 2, vh * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(sep, "box wall", (10, 25), 0, 0.7, (0, 0, 255), 2)
    cv2.putText(sep, "partition", (10, 52), 0, 0.7, (0, 220, 0), 2)
    cv2.imwrite(str(out / "box_vs_partition.png"), sep)
    bw_pts, pt_pts = P3[(lab == 1).reshape(-1)], P3[(lab == 2).reshape(-1)]
    ext = lambda Q: [round(float(np.percentile(Q[:, i], 99) - np.percentile(Q[:, i], 1)) * 100, 1) for i in (0, 1)]
    parts = {"box_wall_points": int(len(bw_pts)), "partition_points": int(len(pt_pts)),
             "box_outer_cm": ext(bw_pts) if len(bw_pts) > 50 else None,
             "partition_extent_cm": ext(pt_pts) if len(pt_pts) > 50 else None,
             "box_wall_height_cm": round(float(np.percentile(bw_pts[:, 2], 95)) * 100, 1) if len(bw_pts) > 50 else None,
             "partition_height_cm": round(float(np.percentile(pt_pts[:, 2], 95)) * 100, 1) if len(pt_pts) > 50 else None,
             "dividers": [len(res["h_groups"]) - 2, len(res["v_groups"]) - 2]}
    print("box / partition:", parts)
    vis = color.copy()
    cv2.rectangle(vis, (int(roi_c[0]), int(roi_c[1])), (int(roi_c[2]), int(roi_c[3])), (0, 255, 255), 2)
    cv2.rectangle(vis, (ox, oy), (ox + img.shape[1], oy + img.shape[0]), (255, 128, 0), 1)   # VGGT input
    cells = []
    for i, c in enumerate(res["cells"], 1):
        q = to_color_px(c.corners_px, (vh, vw), img.shape[:2]) + off
        cv2.polylines(vis, [q.astype(np.int32)], True, (0, 255, 0), 2)
        cx, cy = q.mean(0)
        cv2.putText(vis, str(i), (int(cx) - 8, int(cy) + 8), 0, 0.9, (0, 0, 255), 2)
        cells.append({"id": i, "row": c.row, "col": c.col, "quad_px": q.round(1).tolist()})
    robot = None
    if rs_depth is not None and CALIB_TOP.exists():
        # rim = table next to the basket + wall height (known, else measured on the outer walls in this scan;
        # RealSense depth on the thin wall tops alone reads ~2-3 cm too low)
        rc, rim_z = cells_to_robot([c["quad_px"] for c in cells], rs_depth, K, roi_c,
                                   wall_h=spec["wall_height_m"] if spec else float(res["box_height"]))
        robot = {"rim_z_m": round(rim_z, 4), "cells": []}
        print(f"robot frame (top_cam_to_base): box rim at z {rim_z * 1000:.0f} mm")
        for c, r in zip(cells, rc):
            c["robot"] = r
            if r is None:
                print(f"  cell {c['id']}: no depth")
                continue
            x, y, z = (v * 1000 for v in r["floor_xyz_m"])
            ok = r["reach_m"] < REACH_M - 0.05
            print(f"  cell {c['id']} (r{c['row']} c{c['col']}): centre x {x:6.1f} y {y:6.1f} mm, floor z {z:5.1f} mm, "
                  f"{r['reach_m'] * 1000:4.0f} mm from base {'OK' if ok else 'OUT OF REACH'}")
            q = np.array(c["quad_px"])
            cv2.putText(vis, f"{r['reach_m'] * 100:.0f}cm", (int(q[:, 0].mean()) - 26, int(q[:, 1].mean()) + 34), 0, 0.6,
                        (0, 200, 0) if ok else (0, 0, 255), 2)
        robot["cells"] = rc
    for L in res["lines"]:
        p = to_color_px(np.stack([L.p - 2000 * L.d, L.p + 2000 * L.d]), (vh, vw), img.shape[:2]) + off
        cv2.line(vis, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (255, 0, 255), 1)
    cv2.putText(vis, f"{res['rows']}x{res['cols']} cells, box h {box_h * 100:.1f} cm",
                (20, 40), 0, 1.0, (0, 255, 255), 2)
    cv2.putText(vis, f"GDINO '{gd['prompt']}' {gd['score']:.2f} ({len(gd['agreeing_prompts'])} prompts agree)",
                (20, 75), 0, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(out / "result.png"), vis)
    summary = dict(rows=res["rows"], cols=res["cols"], box_height_m=round(box_h, 4),
                   wall_height_m=round(float(spec["wall_height_m"] if spec else res["box_height"]), 4), vggt_scale=scale,
                   vggt_input_px=[ox, oy, img.shape[1], img.shape[0]], vggt_size=[vw, vh],
                   roi_color_px=[round(v, 1) for v in roi_c], grounding_dino=gd, box_partition=parts, n_lines=len(res["lines"]),
                   n_box_points=int(inside.sum()), times_s={k: round(v, 2) for k, v in t.items()}, cells=cells,
                   robot=robot)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary, out


if __name__ == "__main__":
    main()

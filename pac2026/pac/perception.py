"""Vision steps of the task, as plain functions over a Frame. Each returns a small dataclass, saves an
image to the run folder and (if the dashboard is on) updates its card, metrics and live overlay.
They wrap the code in perception-main (box_lines, box_height, scene_state, detect_grasp, ...)."""
import time

import cv2
import numpy as np
from PIL import Image

from . import paths  # noqa: F401
from .types import BoxInfo, CellInfo, Grasp, Partition

import box_grid                                             # noqa: E402
import box_height as bh                                     # noqa: E402
import box_lines as bl                                      # noqa: E402
from capture_cells import to_color_px, to_vggt_px           # noqa: E402
from detect_grasp import draw as draw_grasps                # noqa: E402
from label_round import crop_cell                           # noqa: E402
from predict_cells import visible_fraction                  # noqa: E402
from scene_state import draw as draw_scene                  # noqa: E402

STATE_COLORS = {"empty": (180, 180, 180), "seated": (0, 200, 0), "straddle": (0, 0, 255),
                "uncertain": (0, 200, 255), "out_of_view": (255, 0, 255), "unknown": (255, 255, 255)}
DEFAULT_TABLE_M = 0.85     # used for the VGGT scale only when no RealSense depth is available


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000)


def _height_view(img, h_m, roi):
    """Colour-mapped VGGT height (0-20 cm) blended over the image, ROI drawn."""
    hm = cv2.resize(np.nan_to_num(h_m).astype(np.float32), (img.shape[1], img.shape[0]))
    vis = cv2.applyColorMap((255 * np.clip(hm / 0.20, 0, 1)).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.4, vis, 0.6, 0)
    x1, y1, x2, y2 = map(int, roi)
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return out


def scan_box(frame, hub, cfg, log) -> BoxInfo:
    """Step 1: box / partition state (Grounding DINO) + box rim height (RealSense p99, VGGT fallback)."""
    t0 = time.perf_counter()
    st = hub.scene(frame.color)
    t_gd = _ms(t0)
    sc = lambda x: f"{x[0]:.2f}" if x else "-"
    log.image("1_scene", draw_scene(frame.color, st), panel="scene",
              caption=f"판정: {st['state']} · 상자 {sc(st['box'])} · 파티션 {sc(st['partition'])} · {t_gd} ms")
    info = BoxInfo(state=st["state"],
                   box_xyxy=st["box"][1] if st["box"] else None,
                   partition_xyxy=st["partition"][1] if st["partition"] else None)
    log.overlay(box=info.box_xyxy or info.partition_xyxy)
    log.metric(timings_ms={"Grounding DINO": t_gd})
    if info.state != "partition_in_box":
        return info

    roi = info.box_xyxy or info.partition_xyxy
    t0 = time.perf_counter()
    pts = hub.vggt.points(frame.color)
    t_vg = _ms(t0)
    vh, vw = pts.shape[:2]
    roi_v = to_vggt_px(roi, (vh, vw), frame.color.shape[:2])
    ring, inside = bh.roi_masks((vh, vw), roi_v)
    n, d, frac = bh.ransac_plane(pts[ring].reshape(-1, 3))

    if frame.depth_m is not None:
        plane = box_grid.fit_table_plane(frame.depth_m)
        z_ref = float(plane[plane.shape[0] // 2, plane.shape[1] // 2])
    else:
        z_ref = DEFAULT_TABLE_M
        log.warn(f"no RealSense depth: VGGT scale from a default table distance {z_ref} m (heights approximate)")
    scale = z_ref / abs(d / n[2])
    h_v = (pts @ n + d) * scale
    info.height_vggt_m = float(np.nanpercentile(h_v[inside], 99))
    info.vggt_depth_m = pts[..., 2] * scale
    info.roi_vggt = roi_v

    if frame.depth_m is not None:
        h_rs = plane - frame.depth_m
        try:
            corners = box_grid.find_box_outline(h_rs, 0.05)
            m = np.zeros(h_rs.shape, np.uint8)
            cv2.fillConvexPoly(m, corners.astype(np.int32), 1)
            m = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool)
            info.height_m = float(np.nanpercentile(h_rs[m], 99))
        except RuntimeError as e:
            log.warn(f"RealSense box outline not found ({e}); using the VGGT height")
    if info.height_m is None:
        info.height_m = info.height_vggt_m

    log.image("1_height", _height_view(frame.color, h_v, roi), panel="height",
              caption=f"상자 높이 RealSense p99 {info.height_m*100:.1f} cm · VGGT {info.height_vggt_m*100:.1f} cm · "
                      f"테이블 평면 inlier {frac*100:.0f}% · 테이블 {z_ref:.3f} m · VGGT {t_vg} ms")
    log.metric(box_height_cm=round(info.height_m * 100, 1), height_vggt_cm=round(info.height_vggt_m * 100, 1),
               timings_ms={"VGGT": t_vg})
    return info


def classify_cells(frame, cells, hub, cfg):
    """DINOv3 cell state for each CellInfo (in place)."""
    p = cfg["perception"]
    crops = [crop_cell(frame.color, c.quad_px) for c in cells]
    for c, (name, conf) in zip(cells, hub.cells(crops)):
        if visible_fraction(c.quad_px, frame.color.shape) < p["cell_min_visible"]:
            name = "out_of_view"
        elif conf < p["cell_min_conf"]:
            name = "uncertain"
        c.state, c.conf = name, conf
    return cells


def draw_cells(img, cells, target=None):
    out = img.copy()
    for c in cells:
        col = STATE_COLORS.get(c.state, (255, 255, 255))
        cv2.polylines(out, [np.round(c.quad_px).astype(np.int32)], True, col, 4 if c.id == target else 2)
        x, y = map(int, c.center_px)
        cv2.putText(out, f"{c.id} {c.state}", (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(out, f"{c.conf:.2f}", (x - 18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return out


def cells_overlay(cells):
    return [{"id": c.id, "quad": c.quad_px, "state": c.state} for c in cells]


def split_partition(frame, box: BoxInfo, hub, cfg, log) -> Partition:
    """Step 2: n x m grid from VGGT depth (Canny + RANSAC lines) and the state of every cell."""
    t0 = time.perf_counter()
    res = bl.detect(box.vggt_depth_m, roi=box.roi_vggt)
    t_lines = _ms(t0)
    log.image("2_lines", bl.render(res), panel="lines",
              caption=f"RANSAC 직선 {len(res['lines'])}개 · 가로 그룹 {len(res['h_groups'])} · 세로 그룹 "
                      f"{len(res['v_groups'])} → {res['rows']}×{res['cols']} 칸 · {t_lines} ms")
    shape_v = box.vggt_depth_m.shape
    cells = []
    for i, c in enumerate(res["cells"]):
        q = to_color_px(c.corners_px, shape_v, frame.color.shape[:2])
        top = q[1] - q[0]
        cells.append(CellInfo(id=i + 1, row=c.row, col=c.col, quad_px=q.tolist(),
                              center_px=tuple(q.mean(0)), yaw_deg=float(np.degrees(np.arctan2(top[1], top[0])))))
    t0 = time.perf_counter()
    classify_cells(frame, cells, hub, cfg)
    t_cls = _ms(t0)
    counts = {}
    for c in cells:
        counts[c.state] = counts.get(c.state, 0) + 1
    log.image("2_cells", draw_cells(frame.color, cells), panel="cells",
              caption=" · ".join(f"{k} {v}" for k, v in counts.items()) + f" · DINOv3 {t_cls} ms ({len(cells)}칸)")
    log.metric(grid=f"{res['rows']}×{res['cols']}", empty_cells=counts.get("empty", 0),
               timings_ms={"Canny+RANSAC": t_lines, "DINOv3 칸 판정": t_cls})
    log.overlay(cells=cells_overlay(cells))
    return Partition(res["rows"], res["cols"], cells)


def find_blocks(frame, box: BoxInfo, hub, cfg, log):
    """Step 3: bubble-wrap blocks OUTSIDE the box (supply area) with grasp centre / jaw axis / width."""
    results, (t_det, t_seg) = hub.grasp(Image.fromarray(cv2.cvtColor(frame.color, cv2.COLOR_BGR2RGB)))

    def in_box(r):
        if not box.box_xyxy or "center" not in r:
            return False
        x1, y1, x2, y2 = box.box_xyxy
        x, y = r["center"]
        return x1 <= x <= x2 and y1 <= y <= y2

    grasps = [Grasp(tuple(r["center"]), r["jaw_axis_deg"], r["opening_px"], r["length_px"], r["score"], r["box"])
              for r in results if "center" in r and not in_box(r)]
    grasps.sort(key=lambda g: -g.score)
    log.image("3_blocks", draw_grasps(frame.color, results), panel="blocks",
              caption=f"검출 {len(results)} · 공급 영역 {len(grasps)} · RF-DETR {t_det*1e3:.0f} ms · SAM2 {t_seg*1e3:.0f} ms"
                      + (f" · 1순위 점수 {grasps[0].score:.2f}, 폭 {grasps[0].opening_px:.0f}px, "
                         f"각도 {grasps[0].jaw_deg:.0f}°" if grasps else ""))
    log.metric(blocks_supply=len(grasps), timings_ms={"RF-DETR": round(t_det * 1e3), "SAM2": round(t_seg * 1e3)})
    log.overlay(grasps=[{"center": g.center_px, "jaw": g.jaw_deg, "opening": g.opening_px, "box": g.box_xyxy}
                        for g in grasps], grasp_index=0)
    log.info(f"blocks: {len(results)} detected, {len(grasps)} in the supply area "
             f"(RF-DETR {t_det*1e3:.0f} ms, SAM2 {t_seg*1e3:.0f} ms)")
    return grasps


def cell_state(frame, cell: CellInfo, hub, cfg, log=None):
    """Step 7 check: state of one cell after placing."""
    c = CellInfo(cell.id, cell.row, cell.col, cell.quad_px, cell.center_px, cell.yaw_deg)
    t0 = time.perf_counter()
    classify_cells(frame, [c], hub, cfg)
    if log:
        log.image(f"7_verify_cell{c.id}", draw_cells(frame.color, [c], target=c.id), panel="verify",
                  caption=f"칸 {c.id}: {c.state} ({c.conf:.2f}) · {_ms(t0)} ms")
    return c.state, c.conf

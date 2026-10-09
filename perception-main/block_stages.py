"""Step-by-step pictures of the block pipeline in block_grasp.py:
  1_gdino.png   Grounding DINO: every merged object, block vs container scores, final label
  2_sam2.png    SAM2 mask inside each block box
  3_ransac.png  RANSAC rectangle on the mask outline (image) + the top face found in 3D, its centre and
                the jaw direction (the grasp actually used)
  stages.png    the three side by side

  HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 perception-main/ai_bubble/.venv/bin/python perception-main/block_stages.py
      [--image data/.../color.jpg]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "research" / "box_cells")]
from block_grasp import BlockFinder  # noqa: E402


def stage_images(color, objs, K, T_base_cam):
    Ti = np.linalg.inv(T_base_cam)

    def px(P):
        pc = Ti[:3, :3] @ np.asarray(P) + Ti[:3, 3]
        return int(round(pc[0] / pc[2] * K["fx"] + K["cx"])), int(round(pc[1] / pc[2] * K["fy"] + K["cy"]))

    def title(img, s):
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(img, s, (12, 31), 0, 0.95, (255, 255, 255), 2)
        return img

    g = color.copy()
    for o in objs:
        x1, y1, x2, y2 = (int(round(v)) for v in o["box"])
        col = {"block": (0, 255, 255), "container": (255, 0, 255)}.get(o["label"], (150, 150, 150))
        cv2.rectangle(g, (x1, y1), (x2, y2), col, 3 if o["label"] != "other" else 1)
        cv2.putText(g, f"{o['label']}", (x1 + 4, y1 + 24), 0, 0.7, col, 2)
        cv2.putText(g, f"blk {o['score']['block']:.2f} / cnt {o['score']['container']:.2f}", (x1 + 4, y1 + 48), 0, 0.55,
                    col, 2)
    title(g, "1  Grounding DINO: block vs container prompts")

    s = color.copy()
    for o in objs:
        if o["label"] != "block" or "mask" not in o:
            continue
        m = o["mask"]
        s[m] = (0.45 * s[m] + 0.55 * np.array([0, 210, 0])).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(s, cnts, -1, (0, 120, 0), 2)
        x1, y1, x2, y2 = (int(round(v)) for v in o["box"])
        cv2.rectangle(s, (x1, y1), (x2, y2), (0, 255, 255), 1)
    title(s, "2  SAM2 mask inside each block box")

    r = color.copy()
    for o in objs:
        if o["label"] != "block" or "rect_px" not in o:
            continue
        cv2.polylines(r, [np.array(o["rect_px"]["corners"], np.int32)], True, (0, 0, 255), 2)       # image outline fit
        rb = o.get("robot")
        if not rb:
            continue
        top = np.array(rb["top_xyz_m"])
        a = np.radians(rb["jaw_yaw_deg"])
        jaw = np.array([np.cos(a), np.sin(a), 0.0])
        side = np.array([-np.sin(a), np.cos(a), 0.0])
        w, l = rb["size_m"]
        if "top_face_corners_m" in o:                       # VGGT top face: refined RANSAC rectangle
            corners = [np.array(c) for c in o["top_face_corners_m"]]
        else:
            corners = [top + sx * w / 2 * jaw + sy * l / 2 * side for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.polylines(r, [np.array([px(c) for c in corners], np.int32)], True, (255, 0, 0), 3)   # top face (3D)
        if "centerline_m" in o:                             # long-side centre line
            cl = o["centerline_m"]
            cv2.line(r, px(cl[0]), px(cl[1]), (0, 200, 255), 2)
        c = px(top)
        cv2.drawMarker(r, c, (255, 0, 0), cv2.MARKER_CROSS, 26, 3)
        for sgn in (-1, 1):                                                                     # jaws
            cv2.arrowedLine(r, px(top + sgn * (w / 2 + 0.03) * jaw), px(top + sgn * (w / 2 - 0.005) * jaw),
                            (255, 0, 255), 3, tipLength=0.35)
        cv2.putText(r, f"{w * 1000:.0f}x{l * 1000:.0f} mm  h {rb['height_m'] * 1000:.0f}", (c[0] - 70, c[1] + 70), 0, 0.6,
                    (255, 0, 0), 2)
    title(r, "3  top face: VGGT plane + RANSAC + right-angle refine (blue), centre line (orange), jaws")
    return g, s, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    args = ap.parse_args()
    if args.image:
        color = cv2.imread(str(args.image))
        depth = np.load(args.image.parent / "rs_depth_m.npy")
        K = json.loads((args.image.parent / "intrinsics.json").read_text())
    else:
        from gdino_vggt_lines import grab_top
        color, depth, K = grab_top()
    out = HERE / "data" / "blocks" / f"stages_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "color.jpg"), color)
    np.save(out / "rs_depth_m.npy", depth.astype(np.float32))
    (out / "intrinsics.json").write_text(json.dumps(K))
    f = BlockFinder()
    objs = f(color, depth, K)
    imgs = stage_images(color, objs, K, f.T)
    for name, im in zip(("1_gdino", "2_sam2", "3_ransac"), imgs):
        cv2.imwrite(str(out / f"{name}.png"), im)
    cv2.imwrite(str(out / "stages.png"), np.hstack([cv2.resize(im, (640, 360)) for im in imgs]))
    crops = []                                               # zoom on each block in the stage-3 picture
    for o in objs:
        if o["label"] == "block" and "rect_px" in o:
            x1, y1, x2, y2 = (int(round(v)) for v in o["box"])
            m = int(0.35 * max(x2 - x1, y2 - y1))
            crop = imgs[2][max(y1 - m, 0):y2 + m, max(x1 - m, 0):x2 + m]
            crops.append(cv2.resize(crop, (360, int(360 * crop.shape[0] / crop.shape[1]))))
    if crops:
        h = max(c.shape[0] for c in crops)
        cv2.imwrite(str(out / "3_ransac_zoom.png"),
                    np.hstack([np.vstack([c, np.full((h - c.shape[0], 360, 3), 255, np.uint8)]) for c in crops]))
    print("saved", out)


if __name__ == "__main__":
    main()

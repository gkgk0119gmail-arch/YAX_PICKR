"""Go 5 cm above a block's top face (tool vertical, jaws across the long sides), check with the wrist D405
that the gripper is vertical to the top face and centred on it, save the picture, go back to rest.
Nothing is grasped.

  perception-main/ai_bubble/.venv/bin/python robot_piper/hover_check.py          # plan only
  perception-main/ai_bubble/.venv/bin/python robot_piper/hover_check.py --go
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "perception-main"), str(ROOT / "perception-main" / "research" / "box_cells")]
import arm_ik  # noqa: E402
import hover_test as ht  # noqa: E402
from board_hover import REST_Q, connect, enable  # noqa: E402
from pick_place import path_ok, yaw_for_jaw  # noqa: E402

HOVER_M = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()
    tcp = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]["tcp_offset_m"]
    p = connect()
    _, js, _ = ht.read(p)
    js[1] = (js[1] + 180) % 360 - 180
    if np.abs(js - REST_Q).max() > 12:
        raise SystemExit("arm is not at rest")
    from block_grasp import BlockFinder
    from block_stages import stage_images
    from gdino_vggt_lines import grab_top
    run = ROOT / "pac2026" / "runs" / f"hover_check_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    color, depth, K = grab_top()
    finder = BlockFinder()
    objs = finder(color, depth, K)
    for name, im in zip(("1_gdino", "2_sam2", "3_ransac"), stage_images(color, objs, K, finder.T)):
        cv2.imwrite(str(run / f"{name}.png"), im)
    blocks = [o for o in objs if o["label"] == "block" and o.get("robot") and not o.get("in_container")]
    cand = [b for b in blocks if 0.18 < b["robot"]["reach_m"] < 0.45]
    if not cand:
        raise SystemExit("no block within 18-45 cm")
    tgt = max(cand, key=lambda b: b["score"]["block"])
    r = tgt["robot"]
    top = np.array(r["top_xyz_m"])
    others = [np.array(b["robot"]["top_xyz_m"]) for b in blocks if b is not tgt]
    yaw = yaw_for_jaw(r["jaw_yaw_deg"])
    yaws = sorted({(yaw + k * 90) % 360 for k in range(4) if 60 <= (yaw + k * 90) % 360 <= 300},
                  key=lambda y: abs(y - 180))
    best = None
    for y in yaws:
        q, beta, e, a = arm_ik.ik_tip([top[0], top[1], top[2] + HOVER_M], y, tcp, max_tilt=15)
        if e < 1.5 and a < 0.7 and (best is None or abs(beta) < abs(best[1])):
            best = (q, beta, y)
    if best is None:
        raise SystemExit("hover pose not reachable")
    q_h, beta, yaw = best
    ok1, w1, why1 = path_ok(REST_Q, q_h, tcp, r["table_z_m"], others + [top], False, None)
    ok2, w2, why2 = path_ok(q_h, REST_Q, tcp, r["table_z_m"], others + [top], False, None)
    print(f"block top ({top[0] * 1000:.0f}, {top[1] * 1000:.0f}, {top[2] * 1000:.0f}) mm, "
          f"{r['size_m'][0] * 1000:.0f}x{r['size_m'][1] * 1000:.0f} mm, jaw {r['jaw_yaw_deg']:.0f} deg -> tool yaw "
          f"{yaw:.0f}, planned tool tilt {beta:+.1f} deg at {HOVER_M * 100:.0f} cm above the top")
    print(f"path rest->hover {'ok' if ok1 else 'FAIL'} ({w1 * 1000:.0f} mm, {why1}); back {'ok' if ok2 else 'FAIL'} "
          f"({w2 * 1000:.0f} mm, {why2}); pictures in {run}")
    if not (ok1 and ok2):
        raise SystemExit("unsafe path - not moving")
    if not args.go:
        print("DRY RUN - nothing sent")
        return
    from charuco_view import Cam
    from wrist_check import check
    cam = Cam("wrist")
    enable(p)
    try:
        okm, err, st = ht.move_j(p, q_h, args.speed)
        print(f"hover: {'ok' if okm else 'FAILED'} ({st.arm_status})")
        if okm:
            time.sleep(0.5)
            _, jn, _ = ht.read(p)
            jn[1] = (jn[1] + 180) % 360 - 180
            res, vis = check(cam, jn, tcp, top[:2], top[2], r["jaw_yaw_deg"])
            cv2.imwrite(str(run / "4_d405_check.png"), vis)
            (run / "d405_check.json").write_text(json.dumps(res, indent=1))
            print("D405 check:", json.dumps(res))
        okm, err, st = ht.move_j(p, REST_Q, args.speed)
        print(f"rest: {'ok' if okm else 'FAILED'}")
    finally:
        cam.pipe.stop()


if __name__ == "__main__":
    main()

"""Put blocks into basket cells, one cell per round: scan the basket + cells (gdino_vggt_lines: VGGT, Canny,
RANSAC) and the blocks (block_grasp: VGGT top face RANSAC), pick a block that is not in the basket, carry it
over the basket rim and let go 2 cm above the cell floor, aligned with the cell walls; back to rest.

Cell numbers are from the robot's point of view, looking at the basket:
  1 = far-left, 2 = far-right, 3 = near-left, 4 = near-right   (far = +x, left = +y of the base frame)

  perception-main/ai_bubble/.venv/bin/python robot_piper/fill_cells.py --cells 1 3          # plan only
  perception-main/ai_bubble/.venv/bin/python robot_piper/fill_cells.py --cells 1 3 --go
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
PM = ROOT / "perception-main"
sys.path[:0] = [str(ROOT / "robot_piper"), str(PM), str(PM / "research" / "box_cells")]
import arm_ik  # noqa: E402
import hover_test as ht  # noqa: E402
from board_hover import REST_Q, connect, enable  # noqa: E402
from pick_place import GRASP_ABOVE_TABLE, PLACE_DROP, RELEASE_MM, WristSeen, gripper, ik_any, yaw_for_jaw  # noqa: E402

CELL_DRIFT_M = 0.015          # going down into / up out of a cell: tip stays this close to the cell centre
STRAIGHT_TILT = 25.0          # vertical legs, high part: tilt allowed where the wrist cannot be vertical (deg)
VERTICAL_TILT = 0.5           # vertical legs, where the fingers are beside a block / inside a cell (deg)
HOVER_M = 0.06                # start of the straight descent above the block top (D405 check here)
LINE_S = 0.25                 # straight legs: seconds per cm (4 cm/s)
BLOCK_H_MIN = 0.12            # the blocks are ~90 x 90 x 120 mm: never plan with a lower block (camera may say less)
GRASP_BELOW_TOP = 0.055       # fingertips 5-6 cm below the block top (above the 5 cm basket walls when placing)
STEP_M = 0.01                 # vertical legs: one waypoint per cm
HOLD_MM = (70.0, 97.0)        # gripper width that means "a block is between the fingers" (90 mm block)


def straight(xy, z0, z1, yaw, tcp, q_seed, vertical_below=None):
    """Joint waypoints (1 cm apart) for a straight vertical line of the tool TIP from z0 to z1 above xy.
    The wrist cannot keep the tool vertical high up (j5 limit 70 deg), so the tilt there is only as large
    as needed (penalised); at or below vertical_below the tool must be vertical (<= VERTICAL_TILT).
    Returns (list of q, worst position error mm, [(z, tilt)]) or (None, ...) if unreachable."""
    zs = np.linspace(z0, z1, max(2, int(abs(z1 - z0) / STEP_M) + 1))
    qs, worst_e, prof = [], 0.0, []
    seed = q_seed
    for z in zs:
        must = vertical_below is not None and z <= vertical_below + 1e-6
        q, beta, e, a = arm_ik.ik_tip([xy[0], xy[1], z], yaw, tcp, q_seed=seed,
                                      max_tilt=VERTICAL_TILT if must else STRAIGHT_TILT)
        worst_e = max(worst_e, e)
        prof.append((float(z), float(beta)))
        if e > 1.5 or a > 0.7:
            return None, worst_e, prof
        qs.append(q)
        seed = q
    return qs, worst_e, prof


def tilt_table(prof, ref_z, label):
    return f"   {label}: " + ", ".join(f"{(z - ref_z) * 1000:+.0f}mm:{abs(b):.1f}deg" for z, b in prof[::2] + [prof[-1]])


def capture(out):
    from gdino_vggt_lines import grab_top
    color, depth, K = grab_top()
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "color.jpg"), color)
    np.save(out / "rs_depth_m.npy", depth.astype(np.float32))
    (out / "intrinsics.json").write_text(json.dumps(K))
    return color, depth, K


def load_finder():
    """BlockFinder (Grounding DINO, SAM2) + VGGT, loaded once; the basket scan uses the same models."""
    import gdino_vggt_lines as gvl
    from block_grasp import BlockFinder
    finder = BlockFinder()
    finder.vggt = gvl.load_vggt()
    return finder


def scan_basket(frame_dir, finder):
    """Runs gdino_vggt_lines on the saved frame in this process, with the finder's Grounding DINO and VGGT
    (same models and weights as a separate run, without loading them again); returns (summary, output dir)."""
    import gdino_vggt_lines as gvl
    if finder.vggt is None:
        finder.vggt = gvl.load_vggt()
    gvl.share_models(finder.gd_proc, finder.gd, finder.vggt)
    try:
        return gvl.scan(frame_dir / "color.jpg")
    except SystemExit as e:
        raise SystemExit(f"basket scan failed: {e}")


def number_cells(cells):
    """Robot-view numbering: far row = larger x, left column = larger y."""
    P = np.array([c["robot"]["floor_xyz_m"][:2] for c in cells])
    xm, ym = np.median(P[:, 0]), np.median(P[:, 1])
    out = {}
    for c, (x, y) in zip(cells, P):
        far, left = x > xm, y > ym
        out[{(True, True): 1, (True, False): 2, (False, True): 3, (False, False): 4}[(far, left)]] = c
    return out


def basket_geometry(cells):
    """Footprint polygon (base xy, expanded by half a cell + the wall) and the cell axis angle."""
    P = np.array([c["robot"]["floor_xyz_m"][:2] for c in cells])
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    pitch = np.median(np.sort(d, 1)[:, 1])                          # centre-to-centre of neighbouring cells
    hull = cv2.convexHull((P * 1000).astype(np.float32))[:, 0] / 1000
    c0 = P.mean(0)
    grow = pitch / 2 + 0.03
    poly = np.array([c0 + (p - c0) * (1 + grow / max(np.linalg.norm(p - c0), 1e-6)) for p in hull])
    i = np.argsort(d[0])[1]                                          # a neighbour of cell 0 -> one wall axis
    v = P[i] - P[0]
    return poly, float(np.degrees(np.arctan2(v[1], v[0]))), float(pitch)


def inside(poly, xy, margin=0.0):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(xy[0]), float(xy[1])), True) > -margin


def check_leg(q_a, q_b, tcp, table_z, blocks, carrying, poly, rim_z, cell_xy=None, n=40):
    """Tip / carried block must clear the table, other blocks and the basket rim. A leg into or out of a
    cell (cell_xy) may go below the rim, but only straight above that cell."""
    worst, why = 1.0, ""
    for a in np.linspace(0, 1, n):
        tip, _ = arm_ik.tip_of((1 - a) * np.asarray(q_a) + a * np.asarray(q_b), tcp)
        low = tip[2] - (GRASP_ABOVE_TABLE if carrying else 0.0)
        cands = [(low - table_z + 0.003, "table")]
        for o in blocks:
            if np.hypot(*(tip[:2] - o[:2])) < (0.12 if carrying else 0.06):
                cands.append((low - (o[2] + (0.02 if carrying else 0.01)), "block"))
        over = inside(poly, tip[:2], 0.045 if carrying else 0.0)
        if over:
            if cell_xy is not None and np.hypot(*(tip[:2] - cell_xy)) < CELL_DRIFT_M:
                pass                                                  # straight down into the target cell
            else:
                cands.append((low - (rim_z + 0.01), "basket rim"))
        for c, w in cands:
            if c < worst:
                worst, why = c, w
    return worst >= 0.0, worst, why


SOFT_CLOSE_S = 1.6           # close the gripper over this long (a gentle squeeze of the bubble wrap)


def soft_close(p, effort, secs=SOFT_CLOSE_S):
    """Close step by step from the current opening to 0 instead of one jump, then hold."""
    w0 = p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
    n = max(10, int(secs / 0.03))
    for w in np.linspace(w0, 0.0, n):
        p.GripperCtrl(int(round(w * 1000)), effort, 0x01, 0)
        time.sleep(secs / n)
    t0 = time.time()
    while time.time() - t0 < 0.8:
        p.GripperCtrl(0, effort, 0x01, 0)
        time.sleep(0.02)
    return p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


class Replan(Exception):
    """The fast top-camera check saw the scene change before the move: plan again."""


def basket_xy_from(objs, rim_z, K):
    """Basket position (base xy) from its Grounding DINO box: the box centre projected onto the rim plane.
    Only used to compare two views made the same way (a shift), not as an absolute position."""
    from observer import pixel_to_plane
    cont = [o for o in objs if o["label"] == "container"]
    if not cont:
        return None
    c = max(cont, key=lambda o: o["score"]["container"])
    x1, y1, x2, y2 = c["box"]
    return pixel_to_plane((x1 + x2) / 2, (y1 + y2) / 2, rim_z, K)


def update_scene_from_obs(scene, res, carried=None):
    """Compare a fast top-camera check with what the scene remembers: blocks pushed on the table, the
    basket shifted, cells that got a block. The scene is updated in place; returns what changed."""
    ch = dict(moved=[], basket_shift_mm=None, newly_occupied=[])
    if not res or res.get("error"):
        if res:
            print(f"   [check] skipped: {res['error']}")
        return ch
    dets = res["blocks"]
    xy_of = lambda o: np.array(o["robot"]["top_xyz_m"][:2])
    # basket shift
    bxy = basket_xy_from(res["containers"], scene["rim_z"], res["K"])
    if bxy is not None and scene.get("basket_ref_xy") is not None:
        sft = bxy - scene["basket_ref_xy"]
        if 0.012 < np.hypot(*sft) < 0.15:
            for c in scene["num"].values():
                c["robot"]["floor_xyz_m"][0] += float(sft[0])
                c["robot"]["floor_xyz_m"][1] += float(sft[1])
            scene["poly"] = scene["poly"] + sft
            for pl in scene["placed"]:
                pl["xy"] = pl["xy"] + sft
            scene["basket_ref_xy"] = bxy
            ch["basket_shift_mm"] = np.round(sft * 1000).tolist()
    # blocks on the table
    for b in scene["free"]:
        if b is carried:
            continue
        p0 = xy_of(b)
        near = [d for d in dets if not inside(scene["poly"], xy_of(d)) and np.hypot(*(xy_of(d) - p0)) < 0.08]
        if not near:
            continue                                        # not seen (covered): keep the old pose
        d = min(near, key=lambda o: np.hypot(*(xy_of(o) - p0)))
        dist = np.hypot(*(xy_of(d) - p0))
        dyaw = abs(((d["robot"]["jaw_yaw_deg"] - b["robot"]["jaw_yaw_deg"]) + 45) % 90 - 45)
        if dist > 0.015 or dyaw > 8:
            b["robot"]["top_xyz_m"][:2] = xy_of(d).tolist()
            b["robot"]["jaw_yaw_deg"] = d["robot"]["jaw_yaw_deg"]
            b["robot"]["reach_m"] = float(np.hypot(*xy_of(d)))
            ch["moved"].append((np.round(p0 * 1000).tolist(), round(dist * 1000), round(dyaw)))
    # cells that hold a block now
    for k, c in scene["num"].items():
        if k in scene["occupied"]:
            continue
        cxy = np.array(c["robot"]["floor_xyz_m"][:2])
        hit = [d for d in dets if np.hypot(*(xy_of(d) - cxy)) < 0.45 * scene["pitch"]]
        if hit:
            scene["occupied"].add(k)
            scene["placed"].append(dict(xy=cxy.copy(), yaw=scene["axis"], top=c["robot"]["floor_xyz_m"][2] + BLOCK_H_MIN))
            ch["newly_occupied"].append(k)
    if ch["moved"] or ch["basket_shift_mm"] or ch["newly_occupied"]:
        print(f"   [check {res['secs']:.1f} s] blocks moved {ch['moved']}, basket shift {ch['basket_shift_mm']} mm, "
              f"newly filled cells {ch['newly_occupied']}")
    else:
        print(f"   [check {res['secs']:.1f} s] no change")
    return ch


def plan_place(scene, cell_no, beta, tip_above_bottom, tcp, q_seed=None, cell_xy=None):
    """Place lines for a held block (tilt beta fixed): straight down into the cell, as high as reachable."""
    cell = scene["num"][cell_no]["robot"]
    cxy = np.asarray(cell_xy if cell_xy is not None else cell["floor_xyz_m"][:2], float)
    floor = cell["floor_xyz_m"][2]
    tops = [pl["top"] for pl in scene["placed"]] + [b["robot"]["table_z_m"] + BLOCK_H_MIN for b in scene["free"]]
    z_carry = scene["rim_z"] + 0.02 + tip_above_bottom
    z_max = min(max(tops + [scene["rim_z"]]) + 0.04 + tip_above_bottom, 0.35)
    z_place = floor + tip_above_bottom + PLACE_DROP
    for yaw_p in sorted([((scene["axis"] + 90 + k * 90 + 180) % 360) - 180 for k in range(4)],
                        key=lambda y: abs(abs(y) - 180)):
        PL = arm_ik.pose_line(cxy, z_carry, z_place, yaw_p, beta, tcp, q_seed=q_seed)
        if PL is not None:
            return PL, extend_up(cxy, z_carry, z_max, yaw_p, beta, tcp, PL[0]), yaw_p, cxy, floor
    return None


class Skip(Exception):
    """This cell cannot be done now (planning / perception); the arm is at rest - go on with the next."""


class Halt(Exception):
    """Stop everything (the arm may not be at rest)."""


def hands_in_view(finder, color, thr=0.35):
    """Grounding DINO (one pass): hands / arms over the table spoil every scan - wait until they are gone."""
    import torch
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    inp = finder.gd_proc(images=img, text="hand. arm.", return_tensors="pt").to(finder.device)
    with torch.no_grad():
        r = finder.gd_proc.post_process_grounded_object_detection(finder.gd(**inp), inp.input_ids, threshold=thr,
                                                                  text_threshold=thr, target_sizes=[img.size[::-1]])[0]
    return [(round(float(s_), 2), l) for s_, l in zip(r["scores"].tolist(), r["text_labels"])]


WAIT_HANDS = False            # --wait-hands: wait until no hand / arm is in view before the scan


def capture_clear(out, finder, wait_s=90):
    """Open the cameras ONCE, wait until no hand / arm is over the table, then take colour + depth.
    (Reopening the cameras every few seconds makes their USB control calls time out.)"""
    import gdino_vggt_lines  # noqa: F401  (puts pac2026 on the path)
    from pac.cameras import CameraManager
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())
    cams = CameraManager(cfg)
    try:
        t0 = time.time()
        while True:
            time.sleep(0.5)
            f = cams.top(with_depth=False)
            hands = hands_in_view(finder, f.color)
            if not hands:
                break
            if not WAIT_HANDS:                           # tight space on site: warn and scan anyway
                print(f"[scene] warning: hands in view {hands} - scanning anyway", flush=True)
                break
            if time.time() - t0 > wait_s:
                raise Halt("hands stayed in view - not scanning")
            print(f"[scene] hands in view {hands} - waiting for a clear table...", flush=True)
            time.sleep(1.5)
        f = cams.top(with_depth=True)
    finally:
        cams.close()
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "color.jpg"), f.color)
    np.save(out / "rs_depth_m.npy", f.depth_m.astype(np.float32))
    (out / "intrinsics.json").write_text(json.dumps(f.intrinsics))
    return f.color, f.depth_m, f.intrinsics


def scan_scene(finder, tag="scene"):
    """ONE scan at the start: basket + cells (VGGT, Canny, RANSAC) and the blocks (VGGT top face).
    Returns the scene; later rounds work from it (blocks move only when the robot moves them)."""
    run = ROOT / "pac2026" / "runs" / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    color, depth, K = capture_clear(run, finder)
    t0 = time.time()
    summ, scan_dir = scan_basket(run, finder)
    cells = [c for c in summ["cells"] if c.get("robot")]
    num = number_cells(cells)
    poly, axis_deg, pitch = basket_geometry(cells)
    rim_z = summ["robot"]["rim_z_m"]
    objs = finder(color, depth, K)
    blocks = [o for o in objs if o["label"] == "block" and o.get("robot")]
    occupied = {k for k, c in num.items() if any(
        np.hypot(*(np.array(b["robot"]["top_xyz_m"][:2]) - np.array(c["robot"]["floor_xyz_m"][:2]))) < 0.45 * pitch
        for b in blocks)}
    free = [b for b in blocks if not inside(poly, b["robot"]["top_xyz_m"][:2])]
    in_basket = [dict(xy=np.array(b["robot"]["top_xyz_m"][:2]), yaw=b["robot"]["jaw_yaw_deg"],
                      top=max(b["robot"]["top_xyz_m"][2], b["robot"]["table_z_m"] + BLOCK_H_MIN))
                 for b in blocks if inside(poly, b["robot"]["top_xyz_m"][:2])]
    print(f"[scene] {summ['rows']}x{summ['cols']} basket, rim z {rim_z * 1000:.0f} mm; {len(free)} block(s) on the "
          f"table, cells occupied {sorted(occupied)}; scan {time.time() - t0:.0f} s -> {run}")
    for k in sorted(num):
        xy = np.array(num[k]["robot"]["floor_xyz_m"][:2]) * 1000
        print(f"   cell {k}: ({xy[0]:.0f}, {xy[1]:.0f}) mm, {np.hypot(*xy):.0f} mm from the base"
              f"{'  [occupied]' if k in occupied else ''}")
    for b in free:
        t = np.array(b["robot"]["top_xyz_m"]) * 1000
        print(f"   block ({t[0]:.0f}, {t[1]:.0f}) mm, h {b['robot']['height_m'] * 1000:.0f}, reach {np.hypot(*t[:2]):.0f} mm")
    return dict(run=run, K=K, scan_dir=scan_dir, num=num, poly=poly, axis=axis_deg, pitch=pitch, rim_z=rim_z,
                free=free, occupied=occupied, placed=list(in_basket),
                basket_ref_xy=basket_xy_from(objs, rim_z, K))


def run_cell(cell_no, scene, p, finder, args, tcp, effort, open_mm):
    """Put one of the remembered free blocks into cell_no. Placed blocks are obstacles. The arm starts
    wherever the previous round left it (scene["q_now"]), not from the rest pose."""
    num, poly = scene["num"], scene["poly"]
    if cell_no not in num:
        raise Skip(f"cell {cell_no} not found")
    cell = num[cell_no]["robot"]
    cell_xy, cell_floor = np.array(cell["floor_xyz_m"][:2]), cell["floor_xyz_m"][2]
    run = ROOT / "pac2026" / "runs" / f"fill_cell{cell_no}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    print(f"[cell {cell_no}] centre ({cell_xy[0] * 1000:.0f}, {cell_xy[1] * 1000:.0f}) mm, "
          f"{np.hypot(*cell_xy) * 1000:.0f} mm from the base")
    cand = [b for b in scene["free"] if 0.15 < b["robot"]["reach_m"] < 0.55 and b["robot"]["fits_gripper"]]
    if not cand:
        far = ", ".join(f"({b['robot']['top_xyz_m'][0] * 1000:.0f}, {b['robot']['top_xyz_m'][1] * 1000:.0f}) "
                        f"{b['robot']['reach_m'] * 100:.0f} cm" for b in scene["free"])
        raise Skip(f"no block left on the table within 15-55 cm (blocks: {far or 'none'})")
    last_err = None
    # nearest block to the robot first: short reach, and the carry does not pass over the others
    import collide
    walls = collide.basket_walls([c["robot"]["floor_xyz_m"][:2] for c in num.values()], scene["axis"], scene["pitch"])
    for tgt in sorted(cand, key=lambda b: b["robot"]["reach_m"]) * 2:   # 2nd pass after a Replan
        if tgt not in scene["free"]:
            continue
        obst = [dict(xy=np.array(b["robot"]["top_xyz_m"][:2]), yaw=b["robot"]["jaw_yaw_deg"],
                     top=max(b["robot"]["top_xyz_m"][2], b["robot"]["table_z_m"] + BLOCK_H_MIN)) for b in scene["free"]]
        obst += scene["placed"]
        scene_c = collide.Scene(float(tgt["robot"]["table_z_m"]), obst, walls, scene["rim_z"])
        try:
            res = plan_and_move(tgt, scene_c, scene["free"].index(tgt), cell_no, cell_xy, cell_floor, num, poly,
                                scene["axis"], scene["rim_z"], p, finder, scene["K"], scene["scan_dir"], run, args,
                                tcp, effort, open_mm, scene["q_now"], scene)
        except Replan as e:
            print(f"   re-planning: {e}")
            continue
        except Skip as e:
            last_err = e
            print(f"   block at {np.round(np.array(tgt['robot']['top_xyz_m'][:2]) * 1000).tolist()} mm: {e}")
            continue
        scene["q_now"] = res.pop("end_q", scene["q_now"])
        placed_cell = res.pop("placed_cell", cell_no)      # the cell may have changed after the check
        placed_xy = np.asarray(res.pop("placed_xy", cell_xy))
        if res["status"] in ("done", "planned"):            # remember: block gone from the table, now in the cell
            scene["free"].remove(tgt)
            scene["placed"].append(dict(xy=placed_xy.copy(), yaw=scene["axis"],
                                        top=cell_floor + max(tgt["robot"]["height_m"], BLOCK_H_MIN)))
            scene["occupied"].add(placed_cell)
        return dict(cell=placed_cell, run=str(run), **res)
    raise Skip(f"no block could be planned ({last_err})")


BETAS = [0, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 24, -24, 28, -28, 32, -32]   # tool tilt about the jaw axis
_PICK_CACHE = {}              # (block, beta, yaw, scene version) -> pick-side result; the pick does not depend on the cell


def extend_up(xy, z_from, z_max, yaw, beta, tcp, q_from):
    """Same tool orientation, straight up from z_from in 1 cm steps for as long as the wrist reaches
    (at most z_max). Returns [q_from, ...] - the last one is the highest reachable point."""
    qs, z, seed = [q_from], z_from, q_from
    while z + STEP_M <= z_max + 1e-9:
        z += STEP_M
        q, pe, re = arm_ik.ik_pose([xy[0], xy[1], z], yaw, beta, tcp, seed)
        if pe > 1.5 or re > 0.5:
            break
        qs.append(q)
        seed = q
    return qs


FORCE_MAX = 0.04              # a plan that only touches other blocks by up to this much may still run (forced)


def plan_and_move(tgt, scene_c, tgt_idx, cell_no, cell_xy, cell_floor, num, poly, axis_deg, rim_z,
                  p, finder, K, scan_dir, run, args, tcp, effort, open_mm, start_q=REST_Q, scene=None):
    """Tilted top grasp: the tool may lean by beta about the jaw axis (fingers stay flat on the block
    faces). The tip moves on the vertical line through the block centre, orientation fixed; the block is
    released with the SAME beta, so it stands upright again. Every leg is checked in 3D (gripper body,
    fingers, D405, carried block) against the table, the other blocks and the basket walls."""
    import collide
    r = tgt["robot"]
    table_z, H = float(r["table_z_m"]), max(r["height_m"], BLOCK_H_MIN)
    pick_xy = np.array(r["top_xyz_m"][:2])
    top_z = table_z + H
    z_grasp = top_z - GRASP_BELOW_TOP
    tip_above_bottom = z_grasp - table_z                  # the held block hangs this far below the tip
    z_hover = top_z + HOVER_M
    tops = [b["top"] for i, b in enumerate(scene_c.blocks) if i != tgt_idx] + [rim_z]
    # straight lines only as high as needed: at the block up to the D405 hover height, at the cell from
    # "carried block 2 cm above the rim"; the carry in between is a joint move checked in 3D against
    # the other blocks (carrying high above everything would put the wrist out of reach)
    z_carry_low = rim_z + 0.02 + tip_above_bottom                 # carried block 2 cm above the basket rim
    z_carry_high = max(tops) + 0.02 + tip_above_bottom            # ... or 2 cm above every other block
    z_carries = [z_carry_low]
    z_carry = z_carry_low
    z_max = min(max(tops) + 0.04 + tip_above_bottom, 0.35)      # as high as useful: held block 4 cm above all
    z_line_top = z_hover
    z_place = cell_floor + tip_above_bottom + PLACE_DROP
    yaws_g = [((r["jaw_yaw_deg"] + 90 + k * 90 + 180) % 360) - 180 for k in range(4)]
    yaws_p = [((axis_deg + 90 + k * 90 + 180) % 360) - 180 for k in range(4)]
    blk_world = collide.block_points(pick_xy, r["jaw_yaw_deg"], table_z, H)
    print(f"   block ({pick_xy[0] * 1000:.0f}, {pick_xy[1] * 1000:.0f}) mm, h {H * 1000:.0f}, "
          f"{np.hypot(*pick_xy) * 1000:.0f} mm from the base; carry tip z {z_carry * 1000:.0f} mm")
    plan, forced, reasons = None, None, []
    for beta in BETAS:
        for yaw_g in sorted(yaws_g, key=lambda y: abs(abs(y) - 180)):
            key = (id(tgt), beta, yaw_g, len(scene_c.blocks), round(z_line_top, 3))
            if key not in _PICK_CACHE:
                L = arm_ik.pose_line(pick_xy, z_line_top, z_grasp, yaw_g, beta, tcp)
                if L is None:
                    _PICK_CACHE[key] = None
                else:
                    c_app, w_app = collide.check_path(L, tcp, 0.10, scene_c, skip_block=tgt_idx)
                    P, Tg = collide.body_world(L[-1], tcp, 0.08)
                    carried = (blk_world - Tg[:3, 3]) @ Tg[:3, :3]          # block in the flange frame
                    # lift: straight up past the hover point for as long as the wrist reaches, THEN move
                    Lup = extend_up(pick_xy, z_line_top, z_max, yaw_g, beta, tcp, L[0])
                    c_lift, w_lift = collide.check_path(L[::-1] + Lup[1:], tcp, 0.08, scene_c, carried,
                                                        skip_block=tgt_idx)
                    _PICK_CACHE[key] = (L, Lup, carried, c_app, w_app, c_lift, w_lift)
            if _PICK_CACHE[key] is None:
                continue
            L, Lup, carried, c_app, w_app, c_lift, w_lift = _PICK_CACHE[key]
            pick_checks = [("going down", c_app, w_app), ("lifting", c_lift, w_lift)]
            if forced is not None and min(c_app, c_lift) < forced["worst"]:
                continue                                    # cannot beat the best forced plan found so far
            for z_carry, yaw_p in [(zc, yp) for zc in z_carries
                                   for yp in sorted(yaws_p, key=lambda y: abs(abs(y) - 180))]:
                PL = arm_ik.pose_line(cell_xy, z_carry, z_place, yaw_p, beta, tcp)
                if PL is None:
                    continue
                # the cell side as high as the wrist reaches too: carry high, then straight down
                PLup = extend_up(cell_xy, z_carry, z_max, yaw_p, beta, tcp, PL[0])
                down_cell = PLup[::-1] + PL[1:]
                legs = [("to above block", [start_q, L[0]], 0.10, None, None),
                        ("carry to cell (high)", [Lup[-1], PLup[-1]], 0.08, carried, tgt_idx),
                        ("down into cell", down_cell, 0.08, carried, tgt_idx),
                        ("up out of cell", down_cell[::-1], 0.10, None, tgt_idx)]
                checks = pick_checks + [(name, *collide.check_path(qs, tcp, op, scene_c, car, skip_block=skip))
                                        for name, qs, op, car, skip in legs]
                bad = [(n, c, w) for n, c, w in checks if c < 0]
                cand = dict(beta=beta, yaw_g=yaw_g, yaw_p=yaw_p, L=L, Lup=Lup, PL=PL, PLup=PLup, carried=carried,
                            legs=checks,
                            worst=min(c for _, c, _ in checks))
                if not bad:
                    plan = cand
                    break
                reasons.append(f"beta {beta}: " + ", ".join(f"{n} hits {w} ({c * 1000:.0f} mm)" for n, c, w in bad))
                # table / basket walls never; other blocks (bubble wrap) only a little
                hard = any(w == "table" for _, _, w in bad)    # gripper vs table never; blocks / walls soft
                if not hard and cand["worst"] >= -FORCE_MAX and (forced is None or cand["worst"] > forced["worst"]):
                    forced = dict(cand, bad=bad)
            if plan:
                break
        if plan:
            break
    if plan is None and forced is not None:
        plan = forced
        print("   NO collision-free plan - running the least-bad one (touches other blocks): "
              + ", ".join(f"{n} {w} {c * 1000:.0f} mm" for n, c, w in forced["bad"]))
    if plan is None:
        why = reasons[-1] if reasons else "no wrist pose reaches it (too far / too close)"
        raise Skip(f"no tilt up to 32 deg works for this block ({np.hypot(*pick_xy) * 1000:.0f} mm from the "
                   f"base): {why}")
    is_forced = "bad" in plan
    L, PL, beta = plan["L"], plan["PL"], plan["beta"]
    print(f"   grasp tilt {beta:+d} deg about the jaw axis (released with the same tilt -> block upright), "
          f"tool yaw pick {plan['yaw_g']:.0f} / place {plan['yaw_p']:.0f} deg")
    z_lift = arm_ik.tip_of(plan["Lup"][-1], tcp)[0][2]
    z_cell = arm_ik.tip_of(plan["PLup"][-1], tcp)[0][2]
    print(f"   lift straight up to tip z {z_lift * 1000:.0f} mm (held block bottom {(z_lift - tip_above_bottom - table_z) * 1000:.0f}"
          f" mm above the table), carry over the cell at tip z {z_cell * 1000:.0f} mm (wanted {z_max * 1000:.0f})")
    for name, c, w in plan["legs"]:
        print(f"   3D check {name:20s} clearance {(c + collide.MARGIN) * 1000:5.0f} mm ({w})")
    T = finder.T
    Ti = np.linalg.inv(T)

    def px(P3):
        pc = Ti[:3, :3] @ np.asarray(P3) + Ti[:3, 3]
        return int(pc[0] / pc[2] * K["fx"] + K["cx"]), int(pc[1] / pc[2] * K["fy"] + K["cy"])
    vis = cv2.imread(str(scan_dir / "result.png"))
    for p0, p1 in scene_c.walls:
        cv2.line(vis, px([*p0, rim_z]), px([*p1, rim_z]), (255, 0, 255), 2)
    for k, c in num.items():
        cv2.putText(vis, f"#{k}", px([*c["robot"]["floor_xyz_m"][:2], rim_z]), 0, 1.2,
                    (255, 0, 0) if k == cell_no else (255, 255, 255), 3)
    cv2.arrowedLine(vis, px([*pick_xy, top_z]), px([*cell_xy, rim_z]), (255, 0, 0), 3, tipLength=0.06)
    cv2.putText(vis, f"tilt {beta:+d} deg", px([*pick_xy, top_z + 0.03]), 0, 0.8, (255, 0, 0), 2)
    cv2.imwrite(str(run / "plan.png"), vis)
    print(f"   plan picture: {run / 'plan.png'}")
    if not args.go:
        print("   DRY RUN - nothing sent")
        return dict(status="planned", beta=beta, forced=is_forced, block_mm=np.round(pick_xy * 1000).tolist(),
                    end_q=plan["PLup"][-1])

    obs = scene.get("obs") if scene else None
    if obs is not None and obs.pending():                  # the check started after the last placement
        ch = update_scene_from_obs(scene, obs.wait(), carried=None)
        if ch["moved"] or ch["basket_shift_mm"] or ch["newly_occupied"]:
            raise Replan("the table changed since the scan")
    try:
        wrist = WristSeen()
    except Exception:
        wrist = None
    enable(p)

    def mv(name, q):
        ok, err, st = ht.move_j(p, q, args.speed)
        print(f"   {name}: {'ok' if ok else 'FAILED'} ({st.arm_status})", flush=True)
        if not ok:
            raise Halt(f"move '{name}' failed ({st.arm_status}) - the arm holds its pose")

    def line(name, qs):
        """Straight up / down without stopping: waypoints 4x densified, streamed at LINE_S per cm."""
        dense = [qs[0]]
        for a, b in zip(qs[:-1], qs[1:]):
            dense += [a + (b - a) * f for f in (0.25, 0.5, 0.75, 1.0)]
        dt = LINE_S / 4
        for q in dense[1:]:
            qi = (np.asarray(q) * 1000).round().astype(int)
            t_end = time.time() + dt
            while time.time() < t_end:
                p.MotionCtrl_2(0x01, 0x01, args.line_speed, 0x00)
                p.JointCtrl(*[int(v) for v in qi])
                time.sleep(0.01)
        mv(name, qs[-1])

    try:
        mv("above block (6 cm)", L[0])
        gripper(p, RELEASE_MM, effort)                       # fully open (100 mm) before going down
        d405 = None
        if wrist:                                            # D405: centred? (tilt = beta is expected)
            from wrist_check import check
            _, jn, _ = ht.read(p)
            jn[1] = (jn[1] + 180) % 360 - 180
            d405, vis = check(wrist.cam, jn, tcp, pick_xy, top_z, r["jaw_yaw_deg"])
            if vis is not None:
                cv2.imwrite(str(run / "d405_check.png"), vis)
            (run / "d405_check.json").write_text(json.dumps(d405, indent=1))
            if d405.get("ok"):
                off = np.hypot(*d405["offset_mm"])
                print(f"   D405 at 6 cm: gripper vs top normal {d405['vertical_err_deg']} deg (planned tilt {beta}), "
                      f"centre offset {d405['offset_mm']} mm, jaw diff {d405['jaw_diff_deg']} deg")
                if off > 20 or d405["jaw_diff_deg"] > 15:
                    mv("rest", REST_Q)
                    raise Skip(f"D405 disagrees with the top camera (offset {off:.0f} mm, jaw "
                               f"{d405['jaw_diff_deg']} deg)")
                if off > 3:                                  # same tilt / yaw, line through the D405 centre
                    L2 = arm_ik.pose_line(np.array(d405["centre_xy_m"]), z_line_top, z_grasp, plan["yaw_g"], beta,
                                          tcp, q_seed=jn)
                    if L2 is not None and collide.check_path(L2, tcp, 0.10, scene_c,
                                                             skip_block=tgt_idx)[0] >= min(0.0, plan["worst"]):
                        print(f"   corrected the grasp line by {off:.0f} mm")
                        mv("above block (corrected)", L2[0])
                        L = L2
            else:
                print(f"   D405 check skipped: {d405.get('why')}")
        line("down to block", L)
        w = soft_close(p, effort)
        print(f"   gripper closed at {w:.0f} mm")
        if not HOLD_MM[0] <= w <= HOLD_MM[1]:
            gripper(p, RELEASE_MM, effort)
            line("back up", L[::-1])
            mv("rest", REST_Q)
            raise Skip(f"gripper at {w:.0f} mm: {'did not close on the block' if w > HOLD_MM[1] else 'missed'}")
        Lup = plan["Lup"] if L is plan["L"] else extend_up(np.array(d405["centre_xy_m"]), z_line_top, z_max,
                                                           plan["yaw_g"], beta, tcp, L[0])
        line("lift straight up", L[::-1] + Lup[1:])
        print(f"   lifted straight up to tip z {arm_ik.tip_of(Lup[-1], tcp)[0][2] * 1000:.0f} mm")
        if wrist:
            h, nf = wrist.holding()
            print(f"   holding check: near fraction {nf:.2f} -> {'HOLDING' if h else 'not seen'}")
        PLup = plan["PLup"]
        _, jn, _ = ht.read(p)
        jn[1] = (jn[1] + 180) % 360 - 180
        if obs is not None:                                  # arm is at the pick spot: the basket is in view
            obs.start(jn, tcp, holding=True)
        mv("above cell (high)", PLup[-1])
        placed_cell, place_xy = cell_no, np.asarray(cell_xy, float)
        if obs is not None:
            ch = update_scene_from_obs(scene, obs.wait(), carried=tgt)
            if ch["basket_shift_mm"] or cell_no in scene["occupied"]:
                if cell_no in scene["occupied"]:             # someone filled it: take the next empty cell
                    free_cells = sorted([k for k in scene["num"] if k not in scene["occupied"]],
                                        key=lambda k: -np.hypot(*scene["num"][k]["robot"]["floor_xyz_m"][:2]))
                    if not free_cells:
                        raise Halt("no empty cell left while holding a block - the arm holds it")
                    placed_cell = free_cells[0]
                _, jn, _ = ht.read(p)
                jn[1] = (jn[1] + 180) % 360 - 180
                pp = plan_place(scene, placed_cell, beta, tip_above_bottom, tcp, q_seed=jn)
                if pp is None:
                    raise Halt(f"cell {placed_cell} not reachable after the change - the arm holds the block")
                PL, PLup, _, place_xy, cell_floor = pp
                print(f"   re-planned the placement into cell {placed_cell}")
                mv("above cell (updated)", PLup[-1])
        if wrist:                                            # D405: where is the cell exactly?
            from wrist_check import check_cell
            _, jn, _ = ht.read(p)
            jn[1] = (jn[1] + 180) % 360 - 180
            cc, vis = check_cell(wrist.cam, jn, tcp, place_xy, cell_floor, rim_z, scene["pitch"] if scene else 0.145,
                                 axis_deg)
            if vis is not None:
                cv2.imwrite(str(run / "d405_cell_check.png"), vis)
            (run / "d405_cell_check.json").write_text(json.dumps(cc, indent=1))
            if cc.get("ok"):
                sh = np.hypot(*cc["shift_mm"])
                print(f"   D405 over the cell: centre shift {cc['shift_mm']} mm (walls {cc['sides_found']}), "
                      f"held block offset {cc.get('held_block_offset_mm')} mm")
                if 3 < sh < 30:
                    pp = plan_place(scene, placed_cell, beta, tip_above_bottom, tcp, q_seed=jn,
                                    cell_xy=cc["centre_xy_m"])
                    if pp is not None:
                        PL, PLup, _, place_xy, cell_floor = pp
                        print(f"   corrected the cell centre by {sh:.0f} mm")
                        mv("above cell (D405-corrected)", PLup[-1])
            else:
                print(f"   D405 cell check skipped: {cc.get('why')}")
        line("down into cell", PLup[::-1] + PL[1:])
        gripper(p, RELEASE_MM, effort, secs=1.5)
        time.sleep(1.0)
        line("up out of cell", PL[::-1] + PLup[1:])          # stay up here: the next round starts from this pose
        _, q_end, _ = ht.read(p)
        q_end[1] = (q_end[1] + 180) % 360 - 180
        if obs is not None:                                  # look at the table while the next pick is planned
            obs.start(q_end, tcp, holding=False)
        print(f"[cell {placed_cell}] done{' (forced: may have touched another block)' if is_forced else ''}")
        return dict(status="done", grip_mm=round(w), beta=beta, forced=is_forced, d405=d405, end_q=q_end,
                    placed_cell=placed_cell, placed_xy=place_xy)
    finally:
        if wrist:
            wrist.close()


def main(argv=None, finder=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["auto"],
                    help="cell numbers (robot view: 1 far-left, 2 far-right, 3 near-left, 4 near-right) or 'auto' "
                         "(all empty cells); always filled farthest from the robot first")
    ap.add_argument("--max", type=int, default=16, help="at most this many cells")
    ap.add_argument("--speed", type=int, default=30, help="moves between poses, percent")
    ap.add_argument("--line-speed", type=int, default=40, help="controller speed while streaming the straight legs")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--wait-hands", action="store_true", help="wait until no hand is in view before scanning")
    ap.add_argument("--collide", action="store_true", help="3D collision checks while planning (off by default)")
    args = ap.parse_args(argv)
    import collide
    collide.ENABLED = args.collide
    global WAIT_HANDS
    WAIT_HANDS = args.wait_hands
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]
    tcp, effort, open_mm = cfg["tcp_offset_m"], int(cfg["gripper_effort"]), float(cfg["gripper_open_mm"])
    finder = finder or load_finder()
    p = connect()
    try:
        scene = scan_scene(finder)                          # the only scan: coordinates are remembered
    except (Halt, SystemExit) as e:
        print(f"[scene] {e}")
        return [dict(cell="-", status="HALT", why=str(e))]
    scene["q_now"] = REST_Q.copy()
    scene["obs"] = None
    if args.go:
        from observer import Observer
        try:
            scene["obs"] = Observer(finder)                 # top camera stays open for the fast checks
        except Exception as e:
            print(f"[check] top camera checks unavailable: {e}")
    num = scene["num"]
    want = sorted(num) if args.cells == ["auto"] else [int(c) for c in args.cells]

    def order(sc):
        t = [k for k in want if k in sc["num"] and k not in sc["occupied"]]
        return sorted(t, key=lambda k: -np.hypot(*sc["num"][k]["robot"]["floor_xyz_m"][:2]))   # farthest first
    todo = order(scene)[:args.max]
    print(f"[plan] cells in order (farthest first): {todo}")
    results = []
    for k in [c for c in want if c not in num or c in scene["occupied"]]:
        results.append(dict(cell=k, status="skipped", why="already has a block" if k in num else "no such cell"))
    tried = set()

    def to_rest():
        if not args.go:
            scene["q_now"] = REST_Q.copy()
            return
        from board_hover import go_rest
        enable(p)
        go_rest(p, args.speed)
        scene["q_now"] = REST_Q.copy()

    while todo and len(tried) < args.max:
        cell_no = todo.pop(0)
        tried.add(cell_no)
        try:
            r = run_cell(cell_no, scene, p, finder, args, tcp, effort, open_mm)
            results.append(r)
            todo = [k for k in order(scene) if k not in tried]       # cells may have been filled meanwhile
        except Skip as e:
            print(f"[cell {cell_no}] SKIPPED: {e}")
            results.append(dict(cell=cell_no, status="skipped", why=str(e)))
            if "no block left" in str(e):
                break
        except Halt as e:
            print(f"[cell {cell_no}] HALT: {e}")
            results.append(dict(cell=cell_no, status="HALT", why=str(e)))
            break
    if scene.get("obs") is not None:
        update_scene_from_obs(scene, scene["obs"].wait(), carried=None)
        scene["obs"].close()
    if args.go and np.abs(np.asarray(scene["q_now"]) - REST_Q).max() > 1:
        to_rest()                                           # only at the very end
        try:
            w = WristSeen()
            h, nf = w.holding()
            w.close()
            print(f"[end] at rest: wrist near fraction {nf:.2f} -> {'BLOCK STILL ON THE GRIPPER' if h else 'gripper empty'}")
        except Exception:
            pass
        gripper(p, 0, effort)
    print("\n===== summary =====")
    for r in results:
        extra = r.get("why") or (f"grip {r.get('grip_mm')} mm" if r["status"] == "done" else "")
        if r.get("forced"):
            extra += "  [forced: collision with a block was expected]"
        print(f"  cell {r['cell']}: {r['status']:8s} {extra}")
    return results


if __name__ == "__main__":
    main()

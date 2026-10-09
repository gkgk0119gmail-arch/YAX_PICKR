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
import live_view  # noqa: E402
import runlog  # noqa: E402
from pick_place import GRASP_ABOVE_TABLE, PLACE_DROP, RELEASE_MM, WristSeen, gripper, ik_any, yaw_for_jaw  # noqa: E402

CELL_DRIFT_M = 0.015          # going down into / up out of a cell: tip stays this close to the cell centre
STRAIGHT_TILT = 25.0          # vertical legs, high part: tilt allowed where the wrist cannot be vertical (deg)
VERTICAL_TILT = 0.5           # vertical legs, where the fingers are beside a block / inside a cell (deg)
HOVER_M = 0.06                # start of the straight descent above the block top (D405 check here)
HOVERS = (HOVER_M, 0.045, 0.03, 0.02, 0.012)   # lower starts where the wrist cannot reach 6 cm above this block
LINE_S = 0.12                 # straight legs: seconds per cm (~8 cm/s)
BLOCK_H_MIN = 0.12            # the blocks are ~90 x 90 x 120 mm: never plan with a lower block (camera may say less)
BLOCK_H_LYING = 0.09          # a block lying on its side: 90 x 120 mm top, 90 mm high


def block_nominal(r):
    """Nominal top-face sides (short, long) in m: standing 90 x 90, lying on a side 90 x 120."""
    import collide
    return (0.09, 0.09) if collide.is_square(r.get("size_m")) else (0.09, 0.12)


def block_height(r):
    """Planning height: square top = standing (120 mm), rectangular top = lying on a side (90 mm). The top
    camera often reads blocks too low, so its height only counts when it is larger."""
    import collide
    return max(r["height_m"], BLOCK_H_MIN if collide.is_square(r.get("size_m")) else BLOCK_H_LYING)
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
    except (SystemExit, RuntimeError, ValueError, IndexError) as e:   # one bad frame: the next one is tried
        raise SystemExit(f"basket scan failed: {e}")


def number_cells(cells):
    """Robot-view numbering for any grid (2x2 ... 3x5): rows from far (larger x) to near, in each row from
    left (larger y) to right; 1 = far-left. Uses the detector's row / col indices, so cells are never merged."""
    P = {(c["row"], c["col"]): np.array(c["robot"]["floor_xyz_m"][:2]) for c in cells}
    R, C = max(r for r, _ in P) + 1, max(c for _, c in P) + 1
    def step(axis):                                      # mean base-xy step when that index grows by one
        d = [P[(r, c)] - P[(r - (axis == 0), c - (axis == 1))] for (r, c) in P
             if (r - (axis == 0), c - (axis == 1)) in P]
        return np.mean(d, 0) if d else np.array([0.0, 0.0])
    s_row, s_col = step(0), step(1)
    # the detector axis that runs more along x gives the robot rows (far / near)
    if abs(s_row[0]) * np.hypot(*s_col) >= abs(s_col[0]) * np.hypot(*s_row):
        far_axis, lr_axis, n_far, n_lr = 0, 1, R, C
        far_flip, lr_flip = s_row[0] > 0, s_col[1] > 0
    else:
        far_axis, lr_axis, n_far, n_lr = 1, 0, C, R
        far_flip, lr_flip = s_col[0] > 0, s_row[1] > 0
    out = {}
    for c in cells:
        idx = (c["row"], c["col"])
        i = idx[far_axis] if not far_flip else n_far - 1 - idx[far_axis]     # 0 = far row
        j = idx[lr_axis] if not lr_flip else n_lr - 1 - idx[lr_axis]         # 0 = leftmost
        out[i * n_lr + j + 1] = c
    return out


GRID_AGREE_M = 0.012          # two basket detections agree when every cell centre is within this


def grid_match(a, b):
    """Pair the cells of two basket detections; None if the grids differ (size or any centre > GRID_AGREE_M)."""
    if (a["rows"], a["cols"]) != (b["rows"], b["cols"]):
        return None
    ca = [c for c in a["cells"] if c.get("robot")]
    cb = [c for c in b["cells"] if c.get("robot")]
    if len(ca) != len(cb):
        return None
    pairs = []
    for c in ca:
        p = np.array(c["robot"]["floor_xyz_m"][:2])
        m = min(cb, key=lambda o: np.hypot(*(np.array(o["robot"]["floor_xyz_m"][:2]) - p)))
        if np.hypot(*(np.array(m["robot"]["floor_xyz_m"][:2]) - p)) > GRID_AGREE_M:
            return None
        pairs.append((c, m))
    return pairs


def scan_basket_twice(run, finder):
    """Basket + cells from two frames taken ~1.5 s apart (a third one breaks a tie). The grid is kept only
    when two detections agree (same rows x cols, every centre within 12 mm); the centres are averaged."""
    dirs = [run] + [d for d in (run / "b", run / "c") if (d / "color.jpg").exists()]
    res = []
    for k, d in enumerate(dirs):
        try:
            res.append(scan_basket(d, finder))
        except SystemExit as e:
            print(f"[scene] basket detection {k + 1}: {e}")
            res.append(None)
        if k >= 1:
            ok = [r for r in res if r is not None]
            for i in range(len(ok)):
                for j in range(i + 1, len(ok)):
                    pairs = grid_match(ok[i][0], ok[j][0])
                    if pairs:
                        summ, scan_dir = ok[i]
                        dmax = max(np.hypot(*(np.array(a["robot"]["floor_xyz_m"][:2]) -
                                              np.array(b["robot"]["floor_xyz_m"][:2]))) for a, b in pairs)
                        for a, b in pairs:
                            a["robot"]["floor_xyz_m"] = ((np.array(a["robot"]["floor_xyz_m"]) +
                                                          np.array(b["robot"]["floor_xyz_m"])) / 2).round(4).tolist()
                        print(f"[scene] basket {summ['rows']}x{summ['cols']} confirmed by 2 of {len(res)} detections "
                              f"(centres within {dmax * 1000:.0f} mm, averaged)")
                        runlog.event("basket_confirmed", rows=summ["rows"], cols=summ["cols"], n=len(res),
                                     max_diff_mm=dmax * 1000)
                        return summ, scan_dir
            grids = [f"{r[0]['rows']}x{r[0]['cols']}" if r else "none" for r in res]
            print(f"[scene] basket detections disagree so far: {grids}")
    ok = [r for r in res if r is not None]
    if not ok:
        raise SystemExit("basket scan failed in every frame")
    print("[scene] WARNING: no two basket detections agree - using the first one (check the cells picture)")
    runlog.event("basket_unconfirmed", grids=[(r[0]["rows"], r[0]["cols"]) if r else None for r in res])
    return ok[0]


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


SOFT_CLOSE_S = 1.2           # close the gripper over this long (a gentle squeeze of the bubble wrap)
OPEN_EFFORT = 3000            # opening only: a higher effort opens faster (a slow opening dragged the block);
                              # closing keeps config gripper_effort (soft squeeze)
CELL_LOOK = False             # empty-gripper D405 look at the next cell before fetching a block (off: the
                              # calibration is good enough; --cell-look turns it on)


def soft_close(p, effort, secs=SOFT_CLOSE_S):
    """Close step by step from the current opening to 0 instead of one jump, then hold."""
    w0 = p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
    n = max(10, int(secs / 0.03))
    for w in np.linspace(w0, 0.0, n):
        p.GripperCtrl(int(round(w * 1000)), effort, 0x01, 0)
        time.sleep(secs / n)
    t0 = time.time()
    while time.time() - t0 < 0.4:
        p.GripperCtrl(0, effort, 0x01, 0)
        time.sleep(0.02)
    w = p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
    runlog.event("gripper_close", from_mm=w0, to_mm=w, secs=secs, effort=effort)
    return w


class Replan(Exception):
    """The fast top-camera check saw the scene change before the move: plan again."""


REGRID_TRIGGER_M = 0.012       # Grounding DINO basket box moved this much -> find the grid again
REGRID_MIN_M = 0.005           # grid moved less than this -> nothing to update


def regrid_basket(scene, res, guess):
    """The basket looks moved (Grounding DINO box, a rough measure). Find the grid again on that top-camera
    frame (same VGGT / Canny / RANSAC as the start) and move every cell, placed block and the footprint by
    the 2D rigid transform fitted old -> new cell centres. Returns (shift_mm, rot_deg) or None if the grid
    could not be matched (then nothing is changed)."""
    if res.get("depth") is None or scene.get("finder") is None:
        return None
    d = scene["run"] / f"regrid_{time.strftime('%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "color.jpg"), res["color"])
    np.save(d / "rs_depth_m.npy", res["depth"].astype(np.float32))
    (d / "intrinsics.json").write_text(json.dumps(res["K"]))
    try:
        summ, _ = scan_basket(d, scene["finder"])
    except SystemExit as e:
        print(f"   [basket] grid not found again ({e})")
        return None
    new = np.array([c["robot"]["floor_xyz_m"] for c in summ["cells"] if c.get("robot")])
    keys = sorted(scene["num"])
    old = np.array([scene["num"][k]["robot"]["floor_xyz_m"] for k in keys])
    # pair old cells (moved by the rough guess) with new ones; blocks standing in cells can make the new grid
    # miss or add cells, so only clear pairs count, and the fitted motion is applied to the OLD cells
    # (which keeps the D405 refinements)
    pairs = []
    for a in old[:, :2]:
        dist = np.hypot(*(new[:, :2] - (a + guess)).T)
        if len(dist) and dist.min() < 0.4 * scene["pitch"]:
            pairs.append((a, new[int(np.argmin(dist)), :2]))
    if not pairs:
        print(f"   [basket] no cell of the new grid ({len(new)} cells) near the old ones - not matched")
        return None
    A, B = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
    if len(pairs) >= 2:
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            return None
    else:
        R, ca, cb = np.eye(2), A[0], B[0]
    t = cb - R @ ca
    resid = np.hypot(*((A @ R.T + t) - B).T).max()
    if resid > 0.015:
        print(f"   [basket] grid fit residual {resid * 1000:.0f} mm over {len(pairs)} cells - not matched")
        return None
    move = np.hypot(*((old[:, :2] @ R.T + t) - old[:, :2]).T).max()
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    runlog.event("regrid", guess_mm=guess * 1000, move_mm=move * 1000, rot_deg=rot, resid_mm=resid * 1000,
                 pairs=len(pairs), new_cells=len(new), dir=str(d))
    if move < REGRID_MIN_M:
        return (0.0, 0.0)
    for k in keys:
        c = scene["num"][k]["robot"]["floor_xyz_m"]
        c[0], c[1] = (R @ np.array(c[:2]) + t).tolist()
    scene["poly"] = scene["poly"] @ R.T + t
    for pl in scene["placed"]:
        pl["xy"] = R @ pl["xy"] + t
        pl["yaw"] = pl["yaw"] + rot
    scene["axis"] = scene["axis"] + rot
    print(f"   [basket] fitted on {len(pairs)} cell(s) of the new grid (residual {resid * 1000:.0f} mm)")
    return (round(move * 1000), round(rot, 1))


CELL_BLOCK_RISE_M = 0.075     # a block in a cell stands 9-12 cm high: its top must be this far above the cell floor


def block_in_cell(det, cell, pitch):
    """A detection really is a block standing in this cell: near the cell centre AND tall (the woven basket /
    an empty cell floor can look like a bubble-wrapped block to Grounding DINO; it is not 9 cm high)."""
    r = det["robot"]
    near = np.hypot(*(np.array(r["top_xyz_m"][:2]) - np.array(cell["robot"]["floor_xyz_m"][:2]))) < 0.45 * pitch
    tall = r["top_xyz_m"][2] > cell["robot"]["floor_xyz_m"][2] + CELL_BLOCK_RISE_M
    return bool(near and tall and det.get("covered_by_arm", 0) < 0.05)


def remeasure(tgt, d, table_z):
    """Take the block pose from a D405 look (centre, jaw, top height, top size) when it looks like one of
    our blocks (short top side 70-110 mm). Returns True if the block was updated."""
    if not d.get("ok") or not d.get("size_mm"):
        return False
    s, l = sorted(d["size_mm"])
    if not (70 <= s <= 110 and l <= 140):
        return False
    r = tgt["robot"]
    old = np.array(r["top_xyz_m"][:2])
    r["top_xyz_m"][0], r["top_xyz_m"][1] = d["centre_xy_m"]
    r["top_xyz_m"][2] = d["top_z_m"]
    r["jaw_yaw_deg"] = d["jaw_deg"]
    r["height_m"] = d["top_z_m"] - table_z
    r["size_m"] = [s / 1000, l / 1000]
    r["reach_m"] = float(np.hypot(*d["centre_xy_m"]))
    runlog.event("remeasure", old_xy=old, new=r)
    return True


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


def update_scene_from_obs(scene, res, carried=None, blocks=True, basket=True):
    """Compare a fast top-camera check with what the scene remembers: blocks pushed on the table, the
    basket shifted, cells that got a block. The scene is updated in place; returns what changed."""
    import collide
    ch = dict(moved=[], basket_shift_mm=None, newly_occupied=[], new_blocks=[])
    if not res or res.get("error"):
        if res:
            print(f"   [check] skipped: {res['error']}")
        return ch
    dets = res["blocks"]
    xy_of = lambda o: np.array(o["robot"]["top_xyz_m"][:2])
    # basket shift
    bxy = basket_xy_from([o for o in res["containers"] if o.get("covered_by_arm", 0) < 0.03], scene["rim_z"], res["K"]) \
        if basket else None
    if bxy is not None and scene.get("basket_ref_xy") is not None:
        sft = bxy - scene["basket_ref_xy"]
        if REGRID_TRIGGER_M < np.hypot(*sft) < 0.20:         # the box moved: the grid itself tells by how much
            print(f"   [basket] Grounding DINO box moved {np.round(sft * 1000).tolist()} mm - finding the grid again")
            got = regrid_basket(scene, res, sft)
            if got is not None:
                scene["basket_ref_xy"] = bxy
                if got[0] > 0:
                    ch["basket_shift_mm"] = got
                    print(f"   [basket] grid moved {got[0]} mm, turned {got[1]} deg: all cells updated")
                else:
                    print("   [basket] grid did not move - no change")
            else:
                ch["basket_unsure"] = True
    # blocks on the table: ONE detection per remembered block (closest pairs first) - two labels must never
    # end up on the same block
    if blocks:
        clear = [d for d in dets if not inside(scene["poly"], xy_of(d)) and d.get("covered_by_arm", 0) < 0.05]
        rem = [b for b in scene["free"] if b is not carried]
        pairs = sorted((float(np.hypot(*(xy_of(d) - xy_of(b)))), bi, di)
                       for bi, b in enumerate(rem) for di, d in enumerate(clear))
        used_b, used_d = set(), set()
        for dist, bi, di in pairs:
            if dist >= 0.08:
                break
            if bi in used_b or di in used_d:
                continue
            used_b.add(bi)
            used_d.add(di)
            b, d = rem[bi], clear[di]
            if b.get("looked") == "ok":                     # the D405 look from above beats this frame
                continue
            p0 = xy_of(b)
            sym = collide.grip_step(d["robot"].get("size_m"))       # rectangular top: a 90 deg turn matters
            dyaw = abs(((d["robot"]["jaw_yaw_deg"] - b["robot"]["jaw_yaw_deg"]) + sym / 2) % sym - sym / 2)
            if dyaw > 30 and dist < 0.02:            # quick RealSense-only fit turned a still block: not trusted
                dyaw = 0.0
            if dist > 0.015 or dyaw > 8:
                b.pop("unpickable", None)                   # moved: worth another try
                b["robot"]["top_xyz_m"][:2] = xy_of(d).tolist()
                if dyaw > 8:
                    b["robot"]["jaw_yaw_deg"] = d["robot"]["jaw_yaw_deg"]
                if d["robot"].get("size_m"):
                    b["robot"]["size_m"] = d["robot"]["size_m"]
                b["robot"]["reach_m"] = float(np.hypot(*xy_of(d)))
                ch["moved"].append((np.round(p0 * 1000).tolist(), round(dist * 1000), round(dyaw)))
        # a clear, tall detection far from every remembered block: a block the scene lost - take it back
        for di, d in enumerate(clear):
            if di in used_d or d["robot"]["top_xyz_m"][2] - d["robot"].get("table_z_m", 0.0) < 0.06:
                continue
            if any(np.hypot(*(xy_of(d) - xy_of(b))) < 0.10 for b in scene["free"]):
                continue
            d["order_rank"] = max([b.get("order_rank", 0) for b in scene["free"]] + [0]) + 1
            scene["free"].append(d)
            ch["new_blocks"].append(np.round(xy_of(d) * 1000).tolist())
    # cells that hold a block now
    for k, c in scene["num"].items():
        if k in scene["occupied"]:
            continue
        cxy = np.array(c["robot"]["floor_xyz_m"][:2])
        hit = [d for d in dets if block_in_cell(d, c, scene["pitch"])]
        flat = [d for d in dets if np.hypot(*(xy_of(d) - cxy)) < 0.45 * scene["pitch"] and d not in hit]
        if flat and not hit:
            runlog.event("cell_flat_detection", cell=k, top_z=[d["robot"]["top_xyz_m"][2] for d in flat],
                         floor_z=c["robot"]["floor_xyz_m"][2])
        if hit:
            scene["occupied"].add(k)
            scene["placed"].append(dict(xy=cxy.copy(), yaw=scene["axis"], top=c["robot"]["floor_xyz_m"][2] + BLOCK_H_MIN))
            ch["newly_occupied"].append(k)
    if ch["moved"] or ch["basket_shift_mm"] or ch["newly_occupied"] or ch["new_blocks"]:
        print(f"   [check {res['secs']:.1f} s] blocks moved {ch['moved']}, basket shift {ch['basket_shift_mm']} mm, "
              f"newly filled cells {ch['newly_occupied']}, blocks found again {ch['new_blocks']}")
    else:
        print(f"   [check {res['secs']:.1f} s] no change")
    runlog.event("top_check", secs=res.get("secs"), changes=ch, blocks_seen=[xy_of(d) for d in dets],
                 q_now=scene.get("q_now"))
    return ch


D405_OUT_MIN_M = 0.01         # the D405 must sit at least this far outside (from the base) and above the tip


def d405_outside(q, tcp):
    """Team rule: the wrist D405 always on the OUTSIDE of the arm (farther from the base than the fingertips)
    and above them - never turned under / toward the robot."""
    import collide
    pos, R = arm_ik.fk(q)
    cam = R @ collide.T_FLANGE_D405[:3, 3] + pos / 1000
    tip = arm_ik.tip_of(q, tcp)[0]
    return bool(np.hypot(*cam[:2]) - np.hypot(*tip[:2]) > D405_OUT_MIN_M and cam[2] - tip[2] > D405_OUT_MIN_M)


def plan_place(scene, cell_no, beta, tip_above_bottom, tcp, q_seed=None, cell_xy=None):
    """Place lines for a held block (tilt beta fixed): straight down into the cell, as high as reachable."""
    cell = scene["num"][cell_no]["robot"]
    cxy = np.asarray(cell_xy if cell_xy is not None else cell["floor_xyz_m"][:2], float)
    floor = cell["floor_xyz_m"][2]
    tops = [pl["top"] for pl in scene["placed"]] + [b["robot"]["table_z_m"] + block_height(b["robot"]) for b in scene["free"]]
    z_carry = scene["rim_z"] + 0.02 + tip_above_bottom
    z_max = min(max(tops + [scene["rim_z"]]) + 0.04 + tip_above_bottom, 0.35)
    z_place = floor + tip_above_bottom + PLACE_DROP
    cands = []
    for yp in [((scene["axis"] + 90 + k * 90 + 180) % 360) - 180 for k in range(4)]:
        q_try = arm_ik.ik_pose([cxy[0], cxy[1], z_carry], yp, beta, tcp, q_seed)
        if q_try[1] < 1.5 and q_try[2] < 0.5 and d405_outside(q_try[0], tcp):
            cands.append((d405_score(q_try[0], tcp), yp))
    for _, yaw_p in sorted(cands, reverse=True):
        PL = arm_ik.pose_line(cxy, z_carry, z_place, yaw_p, beta, tcp, q_seed=q_seed)
        if PL is not None:
            return PL, extend_up(cxy, z_carry, z_max, yaw_p, beta, tcp, PL[0]), yaw_p, cxy, floor
    return None


class Skip(Exception):
    """This cell cannot be done now (planning / perception); the arm is at rest - go on with the next."""


class PickSkip(Skip):
    """The BLOCK cannot be picked (wrist cannot reach it as measured) - no other cell will change that."""


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
    live_view.set_source("top", cams._top.latest, cams._top.intrinsics)
    if cams._wrist is not None:
        live_view.set_source("wrist", cams._wrist.latest)
    live_view.status("scanning the table (VGGT)")
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
        for sub_ in ("b", "c"):                             # more frames for the basket double check
            time.sleep(1.5)
            try:
                g = cams.top(with_depth=True)
            except Exception as e:
                print(f"[scene] extra frame {sub_}: {e}")
                continue
            (out / sub_).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / sub_ / "color.jpg"), g.color)
            np.save(out / sub_ / "rs_depth_m.npy", g.depth_m.astype(np.float32))
            (out / sub_ / "intrinsics.json").write_text(json.dumps(g.intrinsics))
    finally:
        live_view.set_source("top", None)
        live_view.set_source("wrist", None)
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
    summ, scan_dir = scan_basket_twice(run, finder)
    cells = [c for c in summ["cells"] if c.get("robot")]
    num = number_cells(cells)
    poly, axis_deg, pitch = basket_geometry(cells)
    rim_z = summ["robot"]["rim_z_m"]
    objs = finder(color, depth, K)
    blocks = [o for o in objs if o["label"] == "block" and o.get("robot")]
    occupied = {k for k, c in num.items() if any(block_in_cell(b, c, pitch) for b in blocks)}
    free = [b for b in blocks if not inside(poly, b["robot"]["top_xyz_m"][:2])]
    import collide
    in_basket = [dict(xy=np.array(b["robot"]["top_xyz_m"][:2]), yaw=b["robot"]["jaw_yaw_deg"],
                      top=max(b["robot"]["top_xyz_m"][2], b["robot"]["table_z_m"] + block_height(b["robot"])),
                      half=collide.block_half(b["robot"].get("size_m")))
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
    sc = dict(run=run, K=K, scan_dir=scan_dir, num=num, poly=poly, axis=axis_deg, pitch=pitch, rim_z=rim_z,
              free=free, occupied=occupied, placed=list(in_basket),
              basket_ref_xy=basket_xy_from(objs, rim_z, K))
    live_view.scene(sc)                                     # dashboard: cells, blocks, progress (read live)
    return sc


def cell_halves(scene, k):
    """Half sizes of cell k along the basket axis (u) and across it (v), from its neighbours' centres
    (cells may be rectangular on a 3x5 basket); the pitch when it has no neighbour on an axis."""
    a = np.radians(scene["axis"])
    eu, ev = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array(scene["num"][k]["robot"]["floor_xyz_m"][:2])
    du, dv = [], []
    for j, o in scene["num"].items():
        if j == k:
            continue
        d = np.array(o["robot"]["floor_xyz_m"][:2]) - c
        pu, pv = abs(d @ eu), abs(d @ ev)
        if pv < 0.3 * pu:
            du.append(pu)
        elif pu < 0.3 * pv:
            dv.append(pv)
    return (min(du) if du else scene["pitch"]) / 2, (min(dv) if dv else scene["pitch"]) / 2


SURVEY_CAM_ABOVE_RIM = (0.24, 0.20, 0.28, 0.17)   # D405 height above the rim for the empty-gripper cell look


def look_pose(target_xyz, cam_zs, tcp, q_seed=None, min_tip_z=None, max_off_deg=18.0):
    """Joints that put the wrist D405 straight above target_xyz looking down at it (optical axis through the
    target), the camera at the first reachable height of cam_zs (base z, m); the gripper tip stays above
    min_tip_z. For a tool orientation (yaw, tilt) the camera pose follows from the hand-eye calibration, so
    the tip position that puts the camera on the right ray is computed directly: one IK per candidate.
    Returns (q, camera z) or None."""
    from wrist_check import T_FLANGE_CAM
    cam_f, zax_f = T_FLANGE_CAM[:3, 3], T_FLANGE_CAM[:3, 2]
    target = np.asarray(target_xyz, float)
    cands = []
    for beta in range(-32, 33, 4):
        for yaw in range(-180, 180, 15):
            R = arm_ik.tool_R(yaw, beta)
            zax = R @ zax_f
            if zax[2] > -np.cos(np.radians(max_off_deg)):   # camera must look (nearly) straight down
                continue
            cands.append((-zax[2], yaw, beta, R, zax))
    yaw_now = 0.0
    if q_seed is not None:                              # try the tool yaws near the current one first
        Rn = arm_ik.fk(q_seed)[1]
        yaw_now = float(np.degrees(np.arctan2(Rn[1, 0], Rn[0, 0])))
    cands.sort(key=lambda c: (round((1 - c[0]) * 200), abs((c[1] - yaw_now + 180) % 360 - 180)))
    for cz in cam_zs:
        tried = 0
        for _, yaw, beta, R, zax in cands:
            cam = target + (cz - target[2]) / -zax[2] * -zax           # on the ray through the target
            tip = cam - R @ (cam_f - np.array([0, 0, tcp]))
            if min_tip_z is not None and tip[2] < min_tip_z:
                continue
            # known before any IK: the D405 outside / above the tip, and the wrist within the arm's reach
            if not (np.hypot(*cam[:2]) - np.hypot(*tip[:2]) > D405_OUT_MIN_M and cam[2] - tip[2] > D405_OUT_MIN_M):
                continue
            flange = tip - R @ np.array([0, 0, tcp])
            if np.linalg.norm(flange - np.array([0, 0, 0.123])) > 0.60:
                continue
            tried += 1
            if tried > 12:
                break
            q, pe, re = arm_ik.ik_pose(tip, yaw, beta, tcp, q_seed)
            if pe <= 1.5 and re <= 0.5 and d405_outside(q, tcp):
                return q, cz
    return None


def survey_pose(cell_xy, floor_z, rim_z, tcp, q_seed=None):
    """Empty-gripper D405 look at a cell: camera 17-28 cm above the rim, looking at the cell floor centre.
    Returns (q, height above the rim) or None."""
    r = look_pose([cell_xy[0], cell_xy[1], floor_z], [rim_z + h for h in SURVEY_CAM_ABOVE_RIM], tcp, q_seed)
    return None if r is None else (r[0], r[1] - rim_z)


LOOK_CAM_ABOVE_TOP = (0.20, 0.17, 0.23, 0.15, 0.13)   # D405 height above the block top for the look from above


def look_at_block(tgt, scene, p, args, tcp):
    """Empty gripper, before planning the grasp: the D405 straight above the block, ~20 cm over its top, so
    the WHOLE top face is in the picture (at the 6 cm hover it runs off the picture / under the fingers and
    the centre is pulled to the visible part). The block's centre, jaw and top height are taken from it."""
    wrist = scene.get("wrist")
    if not args.go or wrist is None or tgt.get("looked"):
        return
    from wrist_check import check
    r = tgt["robot"]
    table_z = float(r["table_z_m"])
    top_z = table_z + block_height(r)
    xy = np.array(r["top_xyz_m"][:2])
    lp = look_pose([xy[0], xy[1], top_z], [top_z + h for h in LOOK_CAM_ABOVE_TOP], tcp,
                   np.asarray(scene["q_now"], float), min_tip_z=top_z + 0.04)
    tag = f"({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f})"
    if lp is None:
        print(f"[block {tag}] D405 look from above: no wrist pose - using the hover check")
        tgt["looked"] = "no pose"
        return
    q, cz = lp
    live_view.status(f"block {tag}: D405 look from above")
    enable(p)
    ok, err, st = ht.move_j(p, q, args.speed, name=f"look at block {tag}")
    if not ok:
        raise Halt(f"move to look at block {tag} failed ({st.arm_status})")
    _, jn, _ = ht.read(p)
    jn[1] = (jn[1] + 180) % 360 - 180
    scene["q_now"] = jn
    d, vis = check(wrist.cam, jn, tcp, xy, top_z, r["jaw_yaw_deg"], nominal=block_nominal(r))
    n_look = len(list(scene["run"].glob("d405_block_look_*.png"))) + 1
    if vis is not None:
        cv2.imwrite(str(scene["run"] / f"d405_block_look_{n_look}.png"), vis)
    runlog.event("d405_block_look", block=xy, cam_above_top=cz - top_z, q=jn, **d)
    if not d.get("ok"):
        print(f"[block {tag}] D405 look from above: {d.get('why')} - using the hover check")
        tgt["looked"] = "not seen"
        tgt["not_seen"] = d.get("why")                     # with an empty grasp next: a ghost label
        return
    tgt.pop("not_seen", None)
    off = np.hypot(*d["offset_mm"])
    nom = [v * 1000 for v in block_nominal(r)]
    if max(d.get("size_mm") or [0]) > nom[1] + 15:        # bigger than one block: tops of neighbours merged
        print(f"[block {tag}] D405 look from above: top {d.get('size_mm')} mm is larger than one block "
              f"{nom} - neighbours merged, keeping the top-camera pose")
        tgt["looked"] = "merged"
        return
    if d.get("truncation", {}).get("fix_mm"):                # partly hidden even from up here: nominal size
        d = dict(d, size_mm=[round(v * 1000) for v in block_nominal(r)])
    if off > 60 or not remeasure(tgt, d, table_z):
        print(f"[block {tag}] D405 look from above: offset {off:.0f} mm / size {d.get('size_mm')} looks wrong - "
              f"using the hover check")
        tgt["looked"] = "rejected"
        return
    tgt["looked"] = "ok"
    print(f"[block {tag}] D405 look from above ({(cz - top_z) * 100:.0f} cm over the top): centre moved "
          f"{d['offset_mm']} mm, jaw {d['jaw_deg']:.0f} deg, top z {d['top_z_m'] * 1000:.0f} mm, size {d['size_mm']} mm"
          + (f", partly hidden {d['truncation']}" if d.get("truncation", {}).get("fix_mm") else ""))


def refresh_blocks(scene):
    """Block positions from the top-camera check started when the arm left the last cell (arm over the basket,
    blocks in clear view). The basket itself is hidden by the arm in that frame: not used for it."""
    obs = scene.get("obs")
    if obs is not None and obs.pending():
        print("   [check] top camera, arm over the basket: block positions ...", end=" ")
        update_scene_from_obs(scene, obs.wait(), carried=None, basket=False)


def survey_cell(k, scene, p, args, tcp):
    """Empty gripper: move the D405 above cell k and measure its walls; the cell centre is corrected
    (once per cell) before a block is fetched for it. Nothing in the gripper hides the view here."""
    wrist = scene.get("wrist")
    c = scene["num"][k]
    if not args.go or wrist is None or c.get("surveyed"):
        return
    cxy, floor = np.array(c["robot"]["floor_xyz_m"][:2]), c["robot"]["floor_xyz_m"][2]
    sp = survey_pose(cxy, floor, scene["rim_z"], tcp, np.asarray(scene["q_now"], float))
    if sp is None:
        print(f"[cell {k}] D405 look-ahead: no wrist pose above this cell - skipped")
        c["surveyed"] = "no pose"
        return
    q, h = sp
    from wrist_check import check_cell
    live_view.status(f"cell {k}: D405 look at the empty cell")
    enable(p)
    ok, err, st = ht.move_j(p, q, args.speed, name=f"look at cell {k}")
    if not ok:
        raise Halt(f"move to look at cell {k} failed ({st.arm_status})")
    _, jn, _ = ht.read(p)
    jn[1] = (jn[1] + 180) % 360 - 180
    scene["q_now"] = jn
    halves = cell_halves(scene, k)
    obs = scene.get("obs")
    if obs is not None:                                  # arm over the basket, blocks in clear view: D435i now
        if obs.pending():
            update_scene_from_obs(scene, obs.wait(), carried=None)
        obs.start(jn, tcp, holding=False)
    cc, vis = check_cell(wrist.cam, jn, tcp, cxy, floor, scene["rim_z"], scene["pitch"], scene["axis"], halves)
    if obs is not None:
        print("   [check] top camera, arm over the basket: block positions ...", end=" ")
        update_scene_from_obs(scene, obs.wait(), carried=None, basket=False)
    if vis is not None:
        cv2.imwrite(str(scene["run"] / f"d405_cell{k}_look.png"), vis)
    runlog.event("d405_cell_look", cell=k, plan_xy=cxy, cam_above_rim=h, halves=halves, q=jn, **cc)
    if not cc.get("ok"):
        print(f"[cell {k}] D405 look-ahead: {cc.get('why')} - keeping the top-camera centre")
        c["surveyed"] = "not seen"
        return
    sh = np.hypot(*cc["shift_mm"])
    sides = cc["sides_found"]
    both = ("u+" in sides and "u-" in sides) or ("v+" in sides and "v-" in sides)
    if sh <= 40 and (both or sh <= 20):
        c["robot"]["floor_xyz_m"][0], c["robot"]["floor_xyz_m"][1] = cc["centre_xy_m"]
        c["surveyed"] = "corrected"
        print(f"[cell {k}] D405 look-ahead (empty gripper): walls {sides}, centre moved {cc['shift_mm']} mm "
              f"({sh:.0f} mm) -> ({cc['centre_xy_m'][0] * 1000:.0f}, {cc['centre_xy_m'][1] * 1000:.0f})")
    else:
        c["surveyed"] = "rejected"
        print(f"[cell {k}] D405 look-ahead: shift {sh:.0f} mm from walls {sides} looks wrong - keeping the "
              f"top-camera centre")


def far_point_on_axis(scene, margin_px=20):
    """O: on the robot's forward axis (base +x, y = 0) at table height, the farthest point the top camera still
    sees (inside the picture, margin_px from its edge)."""
    from observer import project
    K = scene["K"]
    W, H = K.get("width", 1280), K.get("height", 720)
    tz = float(np.median([b["robot"]["table_z_m"] for b in scene["free"]])) if scene["free"] else 0.0
    xs = np.arange(0.10, 1.50, 0.005)
    uv = project(np.c_[xs, np.zeros_like(xs), np.full_like(xs, tz)], K)
    inside = (uv[:, 0] > margin_px) & (uv[:, 0] < W - margin_px) & (uv[:, 1] > margin_px) & (uv[:, 1] < H - margin_px)
    return np.array([float(xs[inside].max()) if inside.any() else 0.5, 0.0])


def assign_block_order(scene):
    """Team rule for the blocks, fixed once per scan: block 1 = the one nearest O (far end of the robot's forward
    axis in the top view); then the others by distance from block 1."""
    O = far_point_on_axis(scene)
    bxy = lambda b: np.array(b["robot"]["top_xyz_m"][:2])
    blocks = list(scene["free"])
    if not blocks:
        return
    first = min(blocks, key=lambda b: np.hypot(*(bxy(b) - O)))
    p1 = bxy(first)
    for i, b in enumerate(sorted(blocks, key=lambda b: np.hypot(*(bxy(b) - p1))), 1):
        b["order_rank"] = i
    scene["order_O"] = O
    print(f"[order] O = ({O[0] * 1000:.0f}, {O[1] * 1000:.0f}) mm (far end of the forward axis in the top view); block 1 = "
          f"({p1[0] * 1000:.0f}, {p1[1] * 1000:.0f}), then by distance from it: "
          + ", ".join(f"{b['order_rank']}:({bxy(b)[0] * 1000:.0f},{bxy(b)[1] * 1000:.0f})"
                      for b in sorted(blocks, key=lambda b: b["order_rank"])))


def team_rule(scene, empty, blocks):
    """Team rule for one round. Blocks: the fixed order of assign_block_order (block 1 nearest O, then by
    distance from block 1). Cells: the robot's
    forward axis (base +x, straight ahead of the arm) and the line robot origin -> cell centre; the empty cell
    with the LARGEST angle between them is filled first (ties: the farther cell first). Returns (blocks, rank, C)."""
    cxy = {k: np.array(scene["num"][k]["robot"]["floor_xyz_m"][:2]) for k in empty}
    C = np.mean(list(cxy.values()), axis=0)                  # centre of the remaining cells' centres
    ax = np.array([1.0, 0.0])                                # the robot's forward axis (base +x)
    bxy = lambda b: np.array(b["robot"]["top_xyz_m"][:2])

    def angle(k):
        v = cxy[k] / max(np.linalg.norm(cxy[k]), 1e-9)
        return float(np.degrees(np.arccos(np.clip(v @ ax, -1, 1))))

    order = sorted(empty, key=lambda k: (-round(angle(k), 1), -float(np.hypot(*cxy[k]))))
    scene["cell_angles"] = {k: round(angle(k), 1) for k in empty}

    def rank(blk):                                           # the same for every block
        return list(order)
    if any("order_rank" not in b for b in scene.get("free", [])):
        assign_block_order(scene)
    return sorted(blocks, key=lambda b: b.get("order_rank", 1e9)), rank, C


def pickable(scene):
    return [b for b in scene["free"] if 0.15 < b["robot"]["reach_m"] < 0.55 and b["robot"]["fits_gripper"]
            and not b.get("unpickable")]


def planned_order(scene, want):
    """The order the main loop will follow if every round succeeds with its first choice: [(block, cell)]."""
    empty = [k for k in want if k in scene["num"] and k not in scene["occupied"]]
    blocks, out = pickable(scene), []
    while empty and blocks:
        ordered, rank, _ = team_rule(scene, empty, blocks)
        blk = ordered[0]
        k = rank(blk)[0]
        out.append((blk, k))
        empty.remove(k)
        blocks = [b for b in blocks if b is not blk]
    return out


def run_cell(cell_no, scene, p, finder, args, tcp, effort, open_mm, only_block=None):
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
    if only_block is not None:                              # the order given by the team
        cand = [only_block] if only_block in scene["free"] else []
    if not cand:
        far = ", ".join(f"({b['robot']['top_xyz_m'][0] * 1000:.0f}, {b['robot']['top_xyz_m'][1] * 1000:.0f}) "
                        f"{b['robot']['reach_m'] * 100:.0f} cm" for b in scene["free"])
        raise Skip(f"no block left on the table within 15-55 cm (blocks: {far or 'none'})")
    last_err = None
    # nearest block to the robot first: short reach, and the carry does not pass over the others
    import collide
    walls = collide.basket_walls([c["robot"]["floor_xyz_m"][:2] for c in num.values()], scene["axis"], scene["pitch"])
    gave_up = set()                                        # a Skip is final for this cell; only a Replan retries
    for tgt in sorted(cand, key=lambda b: b["robot"]["reach_m"]) * 3:   # more passes after a Replan
        if tgt not in scene["free"] or id(tgt) in gave_up:
            continue
        if cell_no in scene["occupied"]:
            raise Skip(f"cell {cell_no} got a block meanwhile")
        look_at_block(tgt, scene, p, args, tcp)              # empty gripper: whole block in the D405 view
        cell = scene["num"][cell_no]["robot"]                # re-read: the basket may have moved (re-detected)
        cell_xy, cell_floor = np.array(cell["floor_xyz_m"][:2]), cell["floor_xyz_m"][2]
        walls = collide.basket_walls([c["robot"]["floor_xyz_m"][:2] for c in num.values()], scene["axis"],
                                     scene["pitch"])
        obst = [dict(xy=np.array(b["robot"]["top_xyz_m"][:2]), yaw=b["robot"]["jaw_yaw_deg"],
                     top=max(b["robot"]["top_xyz_m"][2], b["robot"]["table_z_m"] + block_height(b["robot"])),
                     half=collide.block_half(b["robot"].get("size_m"))) for b in scene["free"]]
        obst += scene["placed"]
        scene_c = collide.Scene(float(tgt["robot"]["table_z_m"]), obst, walls, scene["rim_z"])
        try:
            res = plan_and_move(tgt, scene_c, scene["free"].index(tgt), cell_no, cell_xy, cell_floor, num, poly,
                                scene["axis"], scene["rim_z"], p, finder, scene["K"], scene["scan_dir"], run, args,
                                tcp, effort, open_mm, scene["q_now"], scene)
        except Replan as e:
            print(f"   re-planning: {e}")
            runlog.event("replan", cell=cell_no, why=str(e), q_now=scene["q_now"])
            continue
        except Skip as e:
            last_err = e
            gave_up.add(id(tgt))
            runlog.event("skip", cell=cell_no, block=tgt["robot"]["top_xyz_m"], why=str(e), q_now=scene["q_now"])
            print(f"   block at {np.round(np.array(tgt['robot']['top_xyz_m'][:2]) * 1000).tolist()} mm: {e}")
            continue
        scene["q_now"] = res.pop("end_q", scene["q_now"])
        runlog.event("cell_result", cell=cell_no, status=res.get("status"), q_now=scene["q_now"], run=str(run))
        placed_cell = res.pop("placed_cell", cell_no)      # the cell may have changed after the check
        placed_xy = np.asarray(res.pop("placed_xy", cell_xy))
        if res["status"] in ("done", "planned"):            # remember: block gone from the table, now in the cell
            scene["free"].remove(tgt)
            hl = max(collide.block_half(tgt["robot"].get("size_m")))      # turned by the place yaw: long side
            scene["placed"].append(dict(xy=placed_xy.copy(), yaw=scene["axis"], half=(hl, hl),
                                        top=cell_floor + block_height(tgt["robot"])))
            scene["occupied"].add(placed_cell)
        return dict(cell=placed_cell, run=str(run), **res)
    raise (PickSkip if isinstance(last_err, PickSkip) and only_block is not None else Skip)(
        f"no block could be planned ({last_err})")


BETAS = [0, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 24, -24, 28, -28, 32, -32]   # tool tilt about the jaw axis
_PICK_CACHE = {}              # (block, beta, yaw, scene version) -> pick-side result; the pick does not depend on the cell


def extend_up(xy, z_from, z_max, yaw, beta, tcp, q_from):
    """Same tool orientation, straight up from z_from in 1 cm steps for as long as the wrist reaches
    (at most z_max). Returns [q_from, ...] - the last one is the highest reachable point."""
    qs, z, seed = [q_from], z_from, q_from
    while z + STEP_M <= z_max + 1e-9:
        z += STEP_M
        q, pe, re = arm_ik.ik_pose([xy[0], xy[1], z], yaw, beta, tcp, seed, quick=True)
        if pe > 1.5 or re > 0.5:
            break
        qs.append(q)
        seed = q
    return qs


J5_LIM = 68.0                 # joint 5 limit used for the tuck (arm limit 70)


def draw_preview(scene):
    """Top-camera picture with the remembered blocks (B1..) and cells (#1..) numbered, for choosing an order."""
    from observer import project
    K = scene["K"]
    img = cv2.imread(str(scene["run"] / "color.jpg"))
    for k, c in scene["num"].items():
        xyz = np.array(c["robot"]["floor_xyz_m"])                      # the floor centre, not the rim (parallax)
        u, v = project(xyz[None], K)[0].astype(int)
        col = (128, 128, 128) if k in scene["occupied"] else (255, 0, 255)
        cv2.circle(img, (u, v), 30, col, 3)
        cv2.putText(img, f"#{k}", (u - 22, v + 12), 0, 1.1, col, 3)
    for i, b in enumerate(scene["free"], 1):
        u, v = project(np.array(b["robot"]["top_xyz_m"])[None], K)[0].astype(int)
        cv2.circle(img, (u, v), 26, (0, 200, 255), 3)
        cv2.putText(img, f"B{i}", (u - 22, v + 10), 0, 1.0, (0, 0, 255), 3)
        cv2.putText(img, f"{b['robot']['reach_m'] * 100:.0f}cm", (u - 26, v + 48), 0, 0.6, (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, "blocks B1..  cells #1 far-left  #2 far-right  #3 near-left  #4 near-right (robot view), grey = filled",
                (10, 28), 0, 0.65, (255, 255, 255), 2)
    return img


TUCK_CLEAR = 0.05             # held block bottom this far above the highest other object is enough


def tuck_poses(q_lift, q_cell, tcp, carried, need_z=None):
    """Lift the HELD BLOCK as high as possible by moving ONLY joint 5: try every joint-5 value within its
    limits and keep the one where the block's lowest point is highest. Carry with joint 5 held there,
    turn only joint 5 back above the cell. carried: block points in the flange frame (from the grasp).
    Returns (q at the lift point, q above the cell, joint-5 value, block bottom z before / after)."""
    q_lift, q_cell = np.asarray(q_lift, float), np.asarray(q_cell, float)

    def block_bottom(q):
        pos, R = arm_ik.fk(q)
        return float(np.min((carried @ R.T)[:, 2] + pos[2] / 1000))
    z0 = block_bottom(q_lift)
    tries = []
    for j5 in np.arange(-J5_LIM, J5_LIM + 0.1, 2.0):
        qa = q_lift.copy()
        qa[4] = j5
        tries.append((block_bottom(qa), float(j5)))
    # the smallest joint-5 turn that gets the block high enough (no needless swinging); else the highest
    enough = [t for t in tries if need_z is not None and t[0] >= need_z]
    best = (min(enough, key=lambda t: abs(t[1] - q_lift[4])) if enough
            else max(tries + [(z0, float(q_lift[4]))], key=lambda t: t[0]))
    qa, qb = q_lift.copy(), q_cell.copy()
    qa[4] = qb[4] = best[1]
    return qa, qb, best[1], z0, best[0]


def d405_score(q, tcp):
    """How well placed the wrist D405 is for a view down past the gripper: higher than the fingertips and
    outside the arm (farther from the base than the tip) is better (it is then not hidden by the block)."""
    import collide
    pos, R = arm_ik.fk(q)
    cam = R @ collide.T_FLANGE_D405[:3, 3] + pos / 1000
    tip = arm_ik.tip_of(q, tcp)[0]
    return float((cam[2] - tip[2]) + 0.5 * (np.hypot(*cam[:2]) - np.hypot(*tip[:2])))


ENTRY_CLEAR_M = 0.01          # the held block must pass this far above other blocks while it is carried in and
                              # joint 5 turns back over the cell (the turn swings it ~15 cm sideways and down)


def entry_clearance(q_lift, q_cell, tcp, carried, scene_c, skip_block, need_z):
    """Smallest clearance (m) between the held block and the other blocks on the way in: tuck at the lift point,
    carry with joint 5 tucked, joint 5 back above the cell (the joint-5 turn sweeps the block sideways while it
    sinks - this is where it hit placed blocks). Returns (clearance, tuck poses)."""
    tk = tuck_poses(q_lift, q_cell, tcp, carried, need_z)
    path = [tk[0] + (tk[1] - tk[0]) * f for f in np.linspace(0, 1, 8)] + \
           [tk[1] + (np.asarray(q_cell) - tk[1]) * f for f in np.linspace(0, 1, 10)]
    worst = np.inf
    for q in path:
        pos, R = arm_ik.fk(q)
        c, _ = scene_c.clearance(carried @ R.T + pos / 1000, skip_block=skip_block, walls=False)
        worst = min(worst, c)
    return worst, tk


D405_MAX_OFF_MM = 40          # up to this the D405 (close up) wins and the grasp line is corrected on the spot


FORCE_MAX = 0.04              # a plan that only touches other blocks by up to this much may still run (forced)


def plan_and_move(tgt, scene_c, tgt_idx, cell_no, cell_xy, cell_floor, num, poly, axis_deg, rim_z,
                  p, finder, K, scan_dir, run, args, tcp, effort, open_mm, start_q=REST_Q, scene=None):
    """Tilted top grasp: the tool may lean by beta about the jaw axis (fingers stay flat on the block
    faces). The tip moves on the vertical line through the block centre, orientation fixed; the block is
    released with the SAME beta, so it stands upright again. Every leg is checked in 3D (gripper body,
    fingers, D405, carried block) against the table, the other blocks and the basket walls."""
    import collide
    r = tgt["robot"]
    table_z, H = float(r["table_z_m"]), block_height(r)
    pick_xy = np.array(r["top_xyz_m"][:2])
    top_z = table_z + H
    z_grasp = top_z - GRASP_BELOW_TOP
    z_grasp_plan = z_grasp
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
    step = collide.grip_step(r.get("size_m"))            # rectangular top: only the 2 yaws across the short side
    yaws_g = [((r["jaw_yaw_deg"] + 90 + k * step + 180) % 360) - 180 for k in range(int(360 // step))]
    yaws_p = [((axis_deg + 90 + k * 90 + 180) % 360) - 180 for k in range(4)]
    blk_world = collide.block_points(pick_xy, r["jaw_yaw_deg"], table_z, H, collide.block_half(r.get("size_m")))
    print(f"   block ({pick_xy[0] * 1000:.0f}, {pick_xy[1] * 1000:.0f}) mm, h {H * 1000:.0f}, "
          f"{np.hypot(*pick_xy) * 1000:.0f} mm from the base; carry tip z {z_carry * 1000:.0f} mm")
    plan, forced, reasons = None, None, []
    pick_ok, place_ok = set(), set()                      # tilts that work at the block / at the cell
    best_entry = None                                      # least-bad way in, if none keeps the clearance
    d405_in = set()                                        # tilts dropped only because the D405 would be inside
    for beta in BETAS:
        feas = []
        for yaw_g in yaws_g:
            q_try = arm_ik.ik_pose([pick_xy[0], pick_xy[1], z_grasp + 0.02], yaw_g, beta, tcp)
            if q_try[1] < 1.5 and q_try[2] < 0.5:
                if not d405_outside(q_try[0], tcp):
                    d405_in.add(beta)
                    continue
                feas.append((d405_score(q_try[0], tcp), yaw_g))
        for _, yaw_g in sorted(feas, reverse=True):            # D405 up and outside the arm first
            key = (id(tgt), beta, yaw_g, len(scene_c.blocks), round(top_z, 3), tuple(np.round(pick_xy, 3)),
                   round(r["jaw_yaw_deg"], 1))
            if key not in _PICK_CACHE:
                # start of the straight descent: 6 cm above the top, lower (4.5 / 3 cm) where the wrist
                # cannot reach that high with this yaw / tilt
                for hov in HOVERS:
                    z_line_top = top_z + hov
                    L = arm_ik.pose_line(pick_xy, z_line_top, z_grasp, yaw_g, beta, tcp)
                    if L is not None:
                        break
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
                    _PICK_CACHE[key] = (L, Lup, carried, c_app, w_app, c_lift, w_lift, z_line_top)
            if _PICK_CACHE[key] is None:
                continue
            L, Lup, carried, c_app, w_app, c_lift, w_lift, z_top = _PICK_CACHE[key]
            pick_ok.add(beta)
            pick_checks = [("going down", c_app, w_app), ("lifting", c_lift, w_lift)]
            if forced is not None and min(c_app, c_lift) < forced["worst"]:
                continue                                    # cannot beat the best forced plan found so far
            place_order = []
            for yp in yaws_p:
                q_try = arm_ik.ik_pose([cell_xy[0], cell_xy[1], z_carry], yp, beta, tcp)
                if q_try[1] < 1.5 and q_try[2] < 0.5 and d405_outside(q_try[0], tcp):
                    place_order.append((d405_score(q_try[0], tcp), yp))
            for z_carry, yaw_p in [(zc, yp) for zc in z_carries for _, yp in sorted(place_order, reverse=True)]:
                PL = arm_ik.pose_line(cell_xy, z_carry, z_place, yaw_p, beta, tcp)
                if PL is None:
                    continue
                place_ok.add(beta)
                # the cell side as high as the wrist reaches too: carry high, then straight down
                PLup = extend_up(cell_xy, z_carry, z_max, yaw_p, beta, tcp, PL[0])
                # the way in: the joint-5 turn over the cell must keep the held block above the other blocks;
                # a larger grasp tilt lets the upright block stay higher over the cell (more room for the turn)
                e_clr, _ = entry_clearance(Lup[-1], PLup[-1], tcp, carried, scene_c, tgt_idx,
                                           max(tops) + TUCK_CLEAR)
                if e_clr < ENTRY_CLEAR_M:
                    reasons.append(f"tilt {beta}: joint 5 turning back over the cell brings the held block "
                                   f"{e_clr * 1000:.0f} mm from another block")
                    if best_entry is None or e_clr > best_entry[0]:
                        best_entry = (e_clr, dict(beta=beta, yaw_g=yaw_g, yaw_p=yaw_p, L=L, Lup=Lup, PL=PL,
                                                  PLup=PLup, carried=carried, z_top=z_top,
                                                  legs=pick_checks, worst=min(c for _, c, _ in pick_checks)))
                    continue
                down_cell = PLup[::-1] + PL[1:]
                legs = [("to above block", [start_q, L[0]], 0.10, None, None),
                        ("carry to cell (high)", [Lup[-1], PLup[-1]], 0.08, carried, tgt_idx),
                        ("down into cell", down_cell, 0.08, carried, tgt_idx),
                        ("up out of cell", down_cell[::-1], 0.10, None, tgt_idx)]
                checks = pick_checks + [(name, *collide.check_path(qs, tcp, op, scene_c, car, skip_block=skip))
                                        for name, qs, op, car, skip in legs]
                bad = [(n, c, w) for n, c, w in checks if c < 0]
                cand = dict(beta=beta, yaw_g=yaw_g, yaw_p=yaw_p, L=L, Lup=Lup, PL=PL, PLup=PLup, carried=carried,
                            z_top=z_top,
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
    if plan is None and forced is None and best_entry is not None:
        plan = best_entry[1]
        print(f"   NO way in keeps the held block {ENTRY_CLEAR_M * 1000:.0f} mm above the other blocks - using the best "
              f"one ({best_entry[0] * 1000:.0f} mm, tilt {plan['beta']:+d})")
    if plan is None and forced is not None:
        plan = forced
        print("   NO collision-free plan - running the least-bad one (touches other blocks): "
              + ", ".join(f"{n} {w} {c * 1000:.0f} mm" for n, c, w in forced["bad"]))
    if plan is None:
        if reasons:
            why = reasons[-1]
        elif not pick_ok and d405_in:
            why = (f"the block is reachable only with the D405 turned inside / under the gripper (tilts "
                   f"{sorted(d405_in)}) - not allowed")
        elif not pick_ok:
            why = f"the wrist cannot reach the block with the jaws across its short side (jaw {r['jaw_yaw_deg']:.0f} deg)"
        else:
            why = (f"the block can be picked with tilts {sorted(pick_ok)} but cell {cell_no} cannot be reached with "
                   f"any of them (it is released with the same tilt)")
        runlog.event("no_plan", cell=cell_no, block=pick_xy, jaw=r["jaw_yaw_deg"], size=r.get("size_m"), H=H,
                     pick_ok=sorted(pick_ok), place_ok=sorted(place_ok), why=why)
        raise (PickSkip if not pick_ok else Skip)(f"no tilt up to 32 deg works for this block ({np.hypot(*pick_xy) * 1000:.0f} mm from the "
                   f"base): {why}")
    is_forced = "bad" in plan
    L, PL, beta = plan["L"], plan["PL"], plan["beta"]
    z_line_top = plan["z_top"]
    hover_name = f"above block ({(z_line_top - top_z) * 100:.1f} cm)"
    need_z = max(tops) + TUCK_CLEAR
    tk = tuck_poses(plan["Lup"][-1], plan["PLup"][-1], tcp, plan["carried"], need_z)
    print(f"   carry: joint 5 only {plan['Lup'][-1][4]:.0f} -> {tk[2]:.0f} deg lifts the held block's lowest point "
          f"{tk[3] * 1000:.0f} -> {tk[4] * 1000:.0f} mm (needed {need_z * 1000:.0f}); joint 5 back to "
          f"{plan['PLup'][-1][4]:.0f} deg above the cell")
    print(f"   grasp tilt {beta:+d} deg about the jaw axis (released with the same tilt -> block upright), "
          f"tool yaw pick {plan['yaw_g']:.0f} / place {plan['yaw_p']:.0f} deg; D405 placement score pick "
          f"{d405_score(plan['L'][0], tcp) * 1000:+.0f} / place {d405_score(plan['PLup'][-1], tcp) * 1000:+.0f} mm "
          f"(+ = above / outside)")
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
        cv2.putText(vis, f"#{k}", px(c["robot"]["floor_xyz_m"]), 0, 1.2,
                    (255, 0, 0) if k == cell_no else (255, 255, 255), 3)
    cv2.arrowedLine(vis, px([*pick_xy, top_z]), px([*cell_xy, cell_floor]), (255, 0, 0), 3, tipLength=0.06)
    cv2.putText(vis, f"tilt {beta:+d} deg", px([*pick_xy, top_z + 0.03]), 0, 0.8, (255, 0, 0), 2)
    cv2.imwrite(str(run / "plan.png"), vis)
    print(f"   plan picture: {run / 'plan.png'}")
    runlog.event("plan", cell=cell_no, block=pick_xy, beta=beta, yaw_pick=plan["yaw_g"], yaw_place=plan["yaw_p"], start_q=start_q,
                 above_block_q=L[0], grasp_q=L[-1], lifted_q=plan["Lup"][-1], above_cell_q=plan["PLup"][-1], release_q=PL[-1],
                 tuck=tk, run=str(run))
    if not args.go:
        print("   DRY RUN - nothing sent")
        return dict(status="planned", beta=beta, forced=is_forced, block_mm=np.round(pick_xy * 1000).tolist(),
                    end_q=plan["PLup"][-1])

    looked = tgt.get("looked") == "ok"                     # centre measured by the D405 from above (whole block)
    line_xy = pick_xy
    obs = scene.get("obs") if scene else None
    if obs is not None and obs.pending():                  # the check started after the last placement
        ch = update_scene_from_obs(scene, obs.wait(), carried=None)
        if ch["moved"] or ch["basket_shift_mm"] or ch["newly_occupied"]:
            raise Replan("the table changed since the scan")
    own_wrist = not (scene and scene.get("wrist"))
    if own_wrist:
        try:
            wrist = WristSeen()
        except Exception:
            wrist = None
    else:
        wrist = scene["wrist"]
    live_view.targets(np.array([*pick_xy, top_z]), np.array([*cell_xy, cell_floor]))
    enable(p)

    def mv(name, q):
        live_view.status(f"cell {cell_no}: {name}")
        ok, err, st = ht.move_j(p, q, args.speed, name=name)
        print(f"   {name}: {'ok' if ok else 'FAILED'} ({st.arm_status})", flush=True)
        if not ok:
            raise Halt(f"move '{name}' failed ({st.arm_status}) - the arm holds its pose")

    def line(name, qs):
        """Straight up / down without stopping: waypoints 4x densified, streamed at LINE_S per cm."""
        dense = [qs[0]]
        for a, b in zip(qs[:-1], qs[1:]):
            dense += [a + (b - a) * f for f in (0.25, 0.5, 0.75, 1.0)]
        dt = LINE_S / 4
        live_view.status(f"cell {cell_no}: {name}")
        runlog.event("line", name=name, waypoints=len(dense), q_from=np.asarray(qs[0]), q_to=np.asarray(qs[-1]),
                     tip_from=arm_ik.tip_of(qs[0], tcp)[0], tip_to=arm_ik.tip_of(qs[-1], tcp)[0])
        for q in dense[1:]:
            qi = (np.asarray(q) * 1000).round().astype(int)
            t_end = time.time() + dt
            while time.time() < t_end:
                p.MotionCtrl_2(0x01, 0x01, args.line_speed, 0x00)
                p.JointCtrl(*[int(v) for v in qi])
                time.sleep(0.01)
        mv(name, qs[-1])

    try:
        mv(hover_name, L[0])
        if obs is not None:                                  # arm over the blocks, basket in view: has it moved?
            _, jh, _ = ht.read(p)
            jh[1] = (jh[1] + 180) % 360 - 180
            obs.start(jh, tcp, holding=False)
        gripper(p, RELEASE_MM, OPEN_EFFORT, secs=0.5)             # fully open (100 mm) before going down
        d405 = None
        if wrist:                                            # D405: centred? (tilt = beta is expected)
            from wrist_check import check
            _, jn, _ = ht.read(p)
            jn[1] = (jn[1] + 180) % 360 - 180
            d405, vis = check(wrist.cam, jn, tcp, pick_xy, top_z, r["jaw_yaw_deg"], nominal=block_nominal(r),
                              square=collide.is_square(r.get("size_m")))
            runlog.event("d405_block", plan_xy=pick_xy, plan_top_z=top_z, plan_jaw=r["jaw_yaw_deg"], q=jn, run=str(run),
                         **d405)
            if vis is not None:
                cv2.imwrite(str(run / "d405_check.png"), vis)
            (run / "d405_check.json").write_text(json.dumps(d405, indent=1))
            if d405.get("ok"):
                off = np.hypot(*d405["offset_mm"])
                print(f"   D405 above the block: top z {d405['top_z_m'] * 1000:.0f} mm (plan {top_z * 1000:.0f}), gripper vs top normal {d405['vertical_err_deg']} deg (planned tilt {beta}), "
                      f"centre offset {d405['offset_mm']} mm, jaw diff {d405['jaw_diff_deg']} deg, top pieces {d405.get('pieces')}")
                if d405.get("truncation", {}).get("fix_mm"):
                    print(f"   (top face partly hidden {d405['truncation']}: centre measured from the complete ends)")
                if off > D405_MAX_OFF_MM or d405["jaw_diff_deg"] > 15:
                    if scene is not None:                    # no trip to rest: the next plan starts from here
                        scene["q_now"] = jn
                    why = f"D405 disagrees with the top camera (offset {off:.0f} mm, jaw {d405['jaw_diff_deg']} deg)"
                    if looked:                               # the look from above was good: look again
                        tgt["looked"] = None
                        raise Replan(why + " - looking at the block again from above")
                    if remeasure(tgt, d405, table_z):        # the close-up view wins: plan again with it
                        raise Replan(why + " - re-planning with the D405 pose")
                    raise Skip(why)
                dz = d405["top_z_m"] - top_z                 # real top height (lying / standing, bubble wrap)
                if 0.008 < abs(dz) < 0.045:
                    z_grasp = d405["top_z_m"] - GRASP_BELOW_TOP
                    tip_above_bottom = z_grasp - table_z
                    print(f"   grasp height corrected by {dz * 1000:+.0f} mm (D405 top)")
                if looked and off > 3:
                    print("   (centre already measured from above with the whole block in view: hover offset only logged)")
                line_xy = pick_xy if looked else np.array(d405["centre_xy_m"])
                if (off > 3 and not looked) or z_grasp != z_grasp_plan:   # line through the D405 centre / height
                    L2 = arm_ik.pose_line(line_xy, z_line_top, z_grasp, plan["yaw_g"], beta, tcp, q_seed=jn)
                    if L2 is not None and collide.check_path(L2, tcp, 0.10, scene_c,
                                                             skip_block=tgt_idx)[0] >= min(0.0, plan["worst"]):
                        print(f"   corrected the grasp line by {0 if looked else off:.0f} mm")
                        mv("above block (corrected)", L2[0])
                        L = L2
                    elif z_grasp != z_grasp_plan:            # could not use the corrected line: keep the plan
                        z_grasp, tip_above_bottom = z_grasp_plan, z_grasp_plan - table_z
            else:
                print(f"   D405 check skipped: {d405.get('why')}")
        if obs is not None and obs.pending():                # arm over the block: basket only, blocks untouched
            ch = update_scene_from_obs(scene, obs.wait(), carried=None, blocks=False)
            if ch["basket_shift_mm"] or cell_no in scene["occupied"]:
                _, jn, _ = ht.read(p)
                jn[1] = (jn[1] + 180) % 360 - 180
                scene["q_now"] = jn
                raise Replan(f"the basket moved {ch['basket_shift_mm']} (mm, deg) - new cell centres, planning again")
        line("down to block", L)
        w = soft_close(p, effort)
        print(f"   gripper closed at {w:.0f} mm")
        if not HOLD_MM[0] <= w <= HOLD_MM[1]:
            gripper(p, RELEASE_MM, OPEN_EFFORT)
            line("back up", L[::-1])
            _, jn, _ = ht.read(p)
            jn[1] = (jn[1] + 180) % 360 - 180
            if scene is not None:                            # no trip to rest: re-plan from up here
                scene["q_now"] = jn
            tgt["looked"] = None                             # the next attempt looks from above again
            if w < 10 and tgt.get("not_seen") and scene is not None and tgt in scene["free"]:
                scene["free"].remove(tgt)                    # nothing seen there from above, nothing gripped
                runlog.event("ghost_removed", block=pick_xy, why=tgt["not_seen"])
                raise Skip(f"no block at ({pick_xy[0] * 1000:.0f}, {pick_xy[1] * 1000:.0f}): the D405 saw none from "
                           f"above and the gripper closed on nothing - label removed")
            if wrist and tgt.get("tries", 0) < 2:           # look again from up here, then plan with what is seen
                from wrist_check import check
                tgt["tries"] = tgt.get("tries", 0) + 1
                look_xy = np.array(d405["centre_xy_m"]) if d405 and d405.get("ok") else pick_xy
                d2, vis = check(wrist.cam, jn, tcp, look_xy, z_grasp + GRASP_BELOW_TOP, r["jaw_yaw_deg"],
                                nominal=block_nominal(r))
                runlog.event("d405_after_miss", q=jn, **d2)
                if vis is not None:
                    cv2.imwrite(str(run / f"d405_after_miss{tgt['tries']}.png"), vis)
                if d2.get("ok") and remeasure(tgt, d2, table_z):
                    raise Replan(f"gripper at {w:.0f} mm: missed - re-measured the block with the D405 "
                                 f"(moved {np.hypot(*d2['offset_mm']):.0f} mm, jaw {d2['jaw_deg']:.0f} deg, "
                                 f"top z {d2['top_z_m'] * 1000:.0f} mm)")
            raise Skip(f"gripper at {w:.0f} mm: {'did not close on the block' if w > HOLD_MM[1] else 'missed'}")
        Lup = plan["Lup"] if L is plan["L"] else extend_up(np.asarray(line_xy), z_line_top, z_max,
                                                           plan["yaw_g"], beta, tcp, L[0])
        line("lift straight up", L[::-1] + Lup[1:])
        print(f"   lifted straight up to tip z {arm_ik.tip_of(Lup[-1], tcp)[0][2] * 1000:.0f} mm")
        if wrist:
            h, nf = wrist.holding()
            print(f"   holding check: near fraction {nf:.2f} -> {'HOLDING' if h else 'not seen'}")
        PLup = plan["PLup"]
        _, jn, _ = ht.read(p)
        jn[1] = (jn[1] + 180) % 360 - 180
        if z_grasp != z_grasp_plan and scene is not None:  # the block hangs differently: release height again
            pp = plan_place(scene, cell_no, beta, tip_above_bottom, tcp, q_seed=plan["PLup"][-1], cell_xy=cell_xy)
            if pp is not None:
                PL, PLup = pp[0], pp[1]
        if obs is not None:                                  # arm is at the pick spot: the basket is in view
            obs.start(jn, tcp, holding=True)
        # joint 5 only, all the way up; travel with joint 5 fixed; joint 5 only back down above the cell
        _, q_up, _ = ht.read(p)
        q_up[1] = (q_up[1] + 180) % 360 - 180
        tk = tuck_poses(q_up, PLup[-1], tcp, plan["carried"], max(tops) + TUCK_CLEAR)
        print(f"   tuck: joint 5 {q_up[4]:.0f} -> {tk[2]:.0f} deg, held block bottom {tk[3] * 1000:.0f} -> "
              f"{tk[4] * 1000:.0f} mm, other joints unchanged")
        mv("tuck (joint 5 only)", tk[0])

        def put_back(why):
            """Holding a block that cannot go into the basket: carry it back (tucked), set it down where it
            was picked, open, go up; the round is skipped with the reason."""
            mv("back to the pick spot (tucked)", tk[0])
            mv("back to the pick spot", Lup[-1])
            line("put the block back", Lup[::-1] + L[1:])
            gripper(p, RELEASE_MM, OPEN_EFFORT, secs=0.7)
            time.sleep(0.5)
            line("up from the put-back block", L[::-1] + Lup[1:])
            _, jb, _ = ht.read(p)
            jb[1] = (jb[1] + 180) % 360 - 180
            if scene is not None:
                scene["q_now"] = jb
            runlog.event("put_back", why=why)
            raise Skip(f"put the block back: {why}")
        mv("carry tucked (joint 5 fixed)", tk[1])
        mv("above cell (high)", PLup[-1])
        placed_cell, place_xy = cell_no, np.asarray(cell_xy, float)
        if obs is not None:
            ch = update_scene_from_obs(scene, obs.wait(), carried=tgt, blocks=False)   # arm over the blocks
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
                if pp is None:                               # another empty cell, nearest first
                    for k in sorted([k for k in scene["num"] if k not in scene["occupied"] and k != placed_cell],
                                    key=lambda k: np.hypot(*(np.array(scene["num"][k]["robot"]["floor_xyz_m"][:2])
                                                             - np.asarray(cell_xy)))):
                        pp = plan_place(scene, k, beta, tip_above_bottom, tcp, q_seed=jn)
                        if pp is not None:
                            placed_cell = k
                            break
                if pp is None:                               # nowhere to go: back where it came from
                    put_back(f"no empty cell reachable after the basket moved")
                PL, PLup, _, place_xy, cell_floor = pp
                print(f"   re-planned the placement into cell {placed_cell}")
                mv("above cell (updated)", PLup[-1])
        if obs is not None:                                  # arm over the basket now: the table is in clear view -
            _, jo, _ = ht.read(p)                            # the next round's block positions, ready in time
            jo[1] = (jo[1] + 180) % 360 - 180
            obs.start(jo, tcp, holding=True)
        if wrist:                                            # D405: where is the cell exactly?
            from wrist_check import check_cell
            _, jn, _ = ht.read(p)
            jn[1] = (jn[1] + 180) % 360 - 180
            cc, vis = check_cell(wrist.cam, jn, tcp, place_xy, cell_floor, rim_z, scene["pitch"] if scene else 0.145,
                                 axis_deg, cell_halves(scene, placed_cell) if scene else None)
            if vis is not None:
                cv2.imwrite(str(run / "d405_cell_check.png"), vis)
            (run / "d405_cell_check.json").write_text(json.dumps(cc, indent=1))
            runlog.event("d405_cell", cell=placed_cell, plan_xy=place_xy, q=jn, **cc)
            if cc.get("ok"):
                sh = np.hypot(*cc["shift_mm"])
                print(f"   D405 over the cell: centre shift {cc['shift_mm']} mm (walls {cc['sides_found']}), "
                      f"held block offset {cc.get('held_block_offset_mm')} mm")
                looked = scene and scene["num"][placed_cell].get("surveyed") == "corrected"
                if looked:
                    print("   (the empty-gripper look already set this cell: not moved again)")
                elif not CELL_LOOK:                          # view mostly hidden by the held block: log only
                    print("   (cell check while holding: logged only, the cell centre is kept)")
                    looked = True
                if 3 < sh < 30 and not looked:
                    pp = plan_place(scene, placed_cell, beta, tip_above_bottom, tcp, q_seed=jn,
                                    cell_xy=cc["centre_xy_m"])
                    if pp is not None:
                        PL, PLup, _, place_xy, cell_floor = pp
                        print(f"   corrected the cell centre by {sh:.0f} mm")
                        mv("above cell (D405-corrected)", PLup[-1])
            else:
                print(f"   D405 cell check skipped: {cc.get('why')}")
        line("down into cell", PLup[::-1] + PL[1:])
        gripper(p, RELEASE_MM, OPEN_EFFORT, secs=0.7)
        time.sleep(0.3)
        line("up out of cell", PL[::-1] + PLup[1:])          # the way it came in, reversed: straight up ...
        _, q_end, _ = ht.read(p)
        q_end[1] = (q_end[1] + 180) % 360 - 180
        if abs(q_end[4] - tk[2]) > 3:                        # ... then joint 5 only back to the carry tuck,
            q_back = q_end.copy()                            # fingers raised before any sideways move
            q_back[4] = tk[2]
            mv("retreat: joint 5 back up (reverse of the entry)", q_back)
            q_end = q_back
        print(f"[cell {placed_cell}] done{' (forced: may have touched another block)' if is_forced else ''}")
        return dict(status="done", grip_mm=round(w), beta=beta, forced=is_forced, d405=d405, end_q=q_end,
                    placed_cell=placed_cell, placed_xy=place_xy)
    finally:
        if wrist and own_wrist:
            wrist.close()


def safe_rest(p, args):
    """Ctrl-C during a run: lift the gripper 8 cm straight up if it is low (between blocks / in a cell), then the
    rest pose at a gentle speed. A second Ctrl-C stops the arm where it is (damped stop)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot
    from board_hover import go_rest
    tcp = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]["tcp_offset_m"]
    print("\n[stop] Ctrl+C: going back to the rest pose (Ctrl+C again = stop right here)", flush=True)
    live_view.status("Ctrl+C: going back to rest")
    runlog.event("ctrl_c")
    try:
        enable(p)
        _, q, _ = ht.read(p)
        q[1] = (q[1] + 180) % 360 - 180
        tip = arm_ik.tip_of(q, tcp)[0]
        if np.abs(q - REST_Q).max() > 2 and tip[2] < 0.22:     # low: first straight up, same orientation
            R0 = arm_ik.fk(q)[1]
            target = tip + [0, 0, 0.08]

            lo, hi = arm_ik.JOINT_LIM[:, 0], arm_ik.JOINT_LIM[:, 1]
            for w_rot in (200, 20):                          # same orientation; else let the tool tilt a bit
                def resid(x, w=w_rot):
                    pos, R = arm_ik.fk(x)
                    t = (pos + R @ [0, 0, tcp * 1000]) / 1000
                    return np.r_[(t - target) * 1000, w * Rot.from_matrix(R0.T @ R).as_rotvec()]
                sol = least_squares(resid, np.clip(q, lo + 1e-3, hi - 1e-3), bounds=(lo, hi))
                if np.linalg.norm(resid(sol.x)[:3]) < 5:
                    break
            if np.linalg.norm(resid(sol.x)[:3]) < 5 and ht.path_min_tip_z(q, sol.x, tcp) > tip[2] * 1000 - 5:
                ok, err, st = ht.move_j(p, sol.x, 20, name="stop: lift 8 cm")
                print(f"[stop] lifted 8 cm: {'ok' if ok else 'not reached'}", flush=True)
        go_rest(p, 20)
        print("[stop] at rest. If a block is still in the gripper, take it out by hand.", flush=True)
    except KeyboardInterrupt:
        p.MotionCtrl_1(0x01, 0, 0)                                # damped stop where it is
        print("[stop] second Ctrl+C: arm stopped where it is (damped)", flush=True)
    except SystemExit as e:                                      # go_rest refused (path would dip)
        print(f"[stop] not moving to rest: {e} - move the arm by hand (teach mode)", flush=True)


def main(argv=None, finder=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["auto"],
                    help="cell numbers (robot view: 1 far-left, 2 far-right, 3 near-left, 4 near-right) or 'auto' "
                         "(all empty cells); always filled farthest from the robot first")
    ap.add_argument("--max", type=int, default=16, help="at most this many cells")
    ap.add_argument("--speed", type=int, default=60, help="moves between poses, percent")
    ap.add_argument("--line-speed", type=int, default=70, help="controller speed while streaming the straight legs")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--wait-hands", action="store_true", help="wait until no hand is in view before scanning")
    ap.add_argument("--collide", action="store_true", help="3D collision checks while planning (off by default)")
    ap.add_argument("--cell-look", action="store_true", help="D405 look at each empty cell before fetching a block")
    ap.add_argument("--preview", action="store_true", help="scan only: picture with block / cell numbers")
    ap.add_argument("--order", help='block->cell pairs from the last preview, e.g. "B2:2,B1:1,B3:4"')
    args = ap.parse_args(argv)
    import collide
    collide.ENABLED = args.collide
    global CELL_LOOK
    CELL_LOOK = args.cell_look
    global WAIT_HANDS
    WAIT_HANDS = args.wait_hands
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]
    tcp, effort, open_mm = cfg["tcp_offset_m"], int(cfg["gripper_effort"]), float(cfg["gripper_open_mm"])
    finder = finder or load_finder()
    p = connect()
    try:
        live_view.set_arm(p)                                    # dashboard: gripper opening
        try:
            scene = scan_scene(finder)                          # the only scan: coordinates are remembered
        except (Halt, SystemExit) as e:
            print(f"[scene] {e}")
            return [dict(cell="-", status="HALT", why=str(e))]
        preview_file = ROOT / "pac2026" / "runs" / "preview_latest.json"
        if args.preview:
            img = draw_preview(scene)
            out = ROOT / "pac2026" / "runs" / "preview_latest.png"
            cv2.imwrite(str(out), img)
            preview_file.write_text(json.dumps(dict(
                blocks={f"B{i}": b["robot"]["top_xyz_m"][:2] for i, b in enumerate(scene["free"], 1)},
                cells={str(k): c["robot"]["floor_xyz_m"][:2] for k, c in scene["num"].items()},
                occupied=sorted(scene["occupied"])), indent=1))
            print(f"[preview] {out}")
            for i, b in enumerate(scene["free"], 1):
                t = np.array(b["robot"]["top_xyz_m"]) * 1000
                print(f"   B{i}: ({t[0]:.0f}, {t[1]:.0f}) mm, {np.hypot(*t[:2]) / 10:.0f} cm from the base")
            return [dict(cell="-", status="preview", why=str(out))]
        pairs = None
        if args.order:                                          # "B2:2,B1:1" -> [(block obj, cell), ...]
            prev = json.loads(preview_file.read_text())
            pairs = []
            for item in args.order.replace(" ", "").split(","):
                bid, cell = item.split(":")
                pxy = np.array(prev["blocks"][bid.upper()])
                near = min(scene["free"], key=lambda b: np.hypot(*(np.array(b["robot"]["top_xyz_m"][:2]) - pxy)))
                if np.hypot(*(np.array(near["robot"]["top_xyz_m"][:2]) - pxy)) > 0.06:
                    raise SystemExit(f"{bid}: no block near its preview position any more - run --preview again")
                pairs.append((near, int(cell)))
            print("[order] " + ", ".join(f"{bid} -> cell {c}" for bid, c in
                                         (it.split(":") for it in args.order.replace(" ", "").split(","))))
        scene["q_now"] = REST_Q.copy()
        scene["obs"] = None
        scene["finder"] = finder                              # grid re-detection when the basket moves
        if args.go:
            from observer import Observer
            try:
                scene["obs"] = Observer(finder)                 # top camera stays open for the fast checks
                live_view.set_source("top", scene["obs"].cam.latest, scene["obs"].cam.intrinsics)
            except Exception as e:
                print(f"[check] top camera checks unavailable: {e}")
            try:
                scene["wrist"] = WristSeen()                    # D405 open for the whole run (checks + live view)
                live_view.set_source("wrist", scene["wrist"].latest)
            except Exception as e:
                scene["wrist"] = None
                print(f"[check] wrist D405 unavailable: {e}")
        num = scene["num"]
        want = sorted(num) if args.cells == ["auto"] else [int(c) for c in args.cells]

        def order(sc):
            t = [k for k in want if k in sc["num"] and k not in sc["occupied"]]
            return sorted(t, key=lambda k: -np.hypot(*sc["num"][k]["robot"]["floor_xyz_m"][:2]))   # farthest first
        todo = order(scene)[:args.max] if pairs is None else [c for _, c in pairs]
        forced_block = {c: b for b, c in pairs} if pairs else {}
        if pairs is None:                                       # what the team rule will do if every round succeeds
            po = planned_order(scene, want)
            print("[plan] fill order (block order from O -> cell at the largest angle from the robot's forward axis): "
                  + ", ".join(f"block ({b['robot']['top_xyz_m'][0] * 1000:.0f}, {b['robot']['top_xyz_m'][1] * 1000:.0f}) -> "
                              f"cell {k}" for b, k in po))
            out_of = [b for b in scene["free"] if b not in pickable(scene)]
            for b in out_of:
                t = np.array(b["robot"]["top_xyz_m"][:2]) * 1000
                why = (f"{b['robot']['reach_m'] * 100:.0f} cm from the base (reach 15-55 cm)"
                       if not 0.15 < b["robot"]["reach_m"] < 0.55 else "too wide for the gripper")
                print(f"[plan] block ({t[0]:.0f}, {t[1]:.0f}) is NOT used: {why} - move it closer if it is needed")
            left = [k for k in want if k in scene["num"] and k not in scene["occupied"] and k not in [c for _, c in po]]
            if left:
                print(f"[plan] cells {left} will stay EMPTY: not enough reachable blocks")
        else:
            print(f"[plan] cells in the given order: {todo}")
        live_view.plan(list(pairs) if pairs else planned_order(scene, want))   # dashboard numbers = this order
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

        rounds = 0
        fails = {}                                          # id(block) -> failed tries (tried again later)
        dead_cells = set()                                  # cells no block could go into
        while pairs is None and want and rounds < args.max:     # team rule: cells in their fixed order, each
            rounds += 1                                         # gets the next block in the block order
            empty = [k for k in want if k in scene["num"] and k not in scene["occupied"] and k not in dead_cells]
            blocks = pickable(scene)
            if not empty or not blocks:
                print(f"[order] done: empty cells {empty}, blocks left {len(blocks)}")
                break
            bxy = lambda b: np.array(b["robot"]["top_xyz_m"][:2])
            ordered, rank, C = team_rule(scene, empty, blocks)
            cell = rank(None)[0]                                # the next cell of the fill order
            # blocks in their order; one that failed goes behind the others and is tried again later
            ordered = sorted(ordered, key=lambda b: (fails.get(id(b), 0), b.get("order_rank", 1e9)))
            done_round = False
            for blk in ordered:
                if blk not in scene["free"]:                    # removed meanwhile (ghost / placed)
                    continue
                print(f"[order] cell {cell} ({scene['cell_angles'][cell]} deg from straight ahead) <- block "
                      f"{blk.get('order_rank')} at {np.round(bxy(blk) * 1000).tolist()} mm"
                      + (f" (retry {fails[id(blk)]})" if fails.get(id(blk)) else ""))
                try:
                    if CELL_LOOK:
                        survey_cell(cell, scene, p, args, tcp)  # empty gripper: measure the cell first
                    else:
                        refresh_blocks(scene)                   # top camera, taken with the arm over the basket
                    r = run_cell(cell, scene, p, finder, args, tcp, effort, open_mm, blk)
                    results.append(r)
                    done_round = True
                    break
                except Skip as e:                               # this block: the next one goes into this cell
                    fails[id(blk)] = fails.get(id(blk), 0) + 1
                    print(f"   block {blk.get('order_rank')} not done ({e}) - the next block goes into cell {cell}"
                          f"; this one is tried again later")
                    if fails[id(blk)] >= 2:
                        blk["unpickable"] = str(e)
                        print(f"   block {blk.get('order_rank')}: failed twice - left out (until it is seen moved)")
                except Halt as e:
                    print(f"[cell {cell}] HALT: {e}")
                    results.append(dict(cell=cell, status="HALT", why=str(e)))
                    want = []                                   # stop the whole run
                    done_round = True
                    break
            if not done_round:
                dead_cells.add(cell)
                print(f"[order] no block could go into cell {cell} - it stays empty, next cell")
                results.append(dict(cell=cell, status="skipped", why="no block could be placed in it"))
        while pairs is not None and todo and len(tried) < args.max:
            cell_no = todo.pop(0)
            tried.add(cell_no)
            try:
                if CELL_LOOK:
                    survey_cell(cell_no, scene, p, args, tcp)
                else:
                    refresh_blocks(scene)
                r = run_cell(cell_no, scene, p, finder, args, tcp, effort, open_mm, forced_block.get(cell_no))
                results.append(r)
                if pairs is None:
                    todo = [k for k in order(scene) if k not in tried]   # cells may have been filled meanwhile
            except Skip as e:
                print(f"[cell {cell_no}] SKIPPED: {e}")
                results.append(dict(cell=cell_no, status="skipped", why=str(e)))
                if "no block left" in str(e):
                    break
            except Halt as e:
                print(f"[cell {cell_no}] HALT: {e}")
                results.append(dict(cell=cell_no, status="HALT", why=str(e)))
                break
        live_view.targets()
        if scene.get("obs") is not None:
            update_scene_from_obs(scene, scene["obs"].wait(), carried=None)
        if args.go and np.abs(np.asarray(scene["q_now"]) - REST_Q).max() > 1:
            live_view.status("back to rest")
            to_rest()                                           # only at the very end
            try:
                w = scene.get("wrist") or WristSeen()
                h, nf = w.holding()
                print(f"[end] at rest: wrist near fraction {nf:.2f} -> {'BLOCK STILL ON THE GRIPPER' if h else 'gripper empty'}")
            except Exception:
                pass
            gripper(p, 0, effort)
        live_view.status(f"finished: {sum(r['status'] == 'done' for r in results)} cell(s) filled")
        time.sleep(0.3)
        live_view.set_source("top", None)
        live_view.set_source("wrist", None)
        if scene.get("obs") is not None:
            scene["obs"].close()
        if scene.get("wrist") is not None:
            scene["wrist"].close()
        print("\n===== summary =====")
        for r in results:
            extra = r.get("why") or (f"grip {r.get('grip_mm')} mm" if r["status"] == "done" else "")
            if r.get("forced"):
                extra += "  [forced: collision with a block was expected]"
            print(f"  cell {r['cell']}: {r['status']:8s} {extra}")
        return results
    except KeyboardInterrupt:                              # Ctrl-C: arm back to rest, then stop
        safe_rest(p, args)
        raise SystemExit("stopped by Ctrl+C")


if __name__ == "__main__":
    main()

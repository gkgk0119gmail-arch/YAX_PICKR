"""Planner IK spread over CPU processes.

plan_and_move tries every tool tilt x yaw one after the other, each with its own IK (a test pose, the straight
grasp line from the highest hover that works, the lift; the same at the cell). The candidates do not depend on
each other, so precompute() solves them all at once on a pool of worker processes with exactly the same
functions (arm_ik.ik_pose / pose_line / extend_up), and the planner then walks them in its usual order:
same plan, less waiting.

start() forks the workers; call it early (before camera / model threads), pickr.py does it first thing.
Any pool problem -> precompute() returns None and the planner solves one by one as before.
"""
import multiprocessing as mp
import os
import signal

import numpy as np

import arm_ik

USE_POOL = [False]
WORKERS = max(2, min(16, (os.cpu_count() or 4) - 4))
TIMEOUT_S = 30.0
_POOL = [None]


def extend_up(xy, z_from, z_max, yaw, beta, tcp, q_from, step):
    """Same tool orientation, straight up from z_from in steps for as long as the wrist reaches (at most z_max).
    Returns [q_from, ...] - the last one is the highest reachable point."""
    qs, z, seed = [q_from], z_from, q_from
    while z + step <= z_max + 1e-9:
        z += step
        q, pe, re = arm_ik.ik_pose([xy[0], xy[1], z], yaw, beta, tcp, seed, quick=True)
        if pe > 1.5 or re > 0.5:
            break
        qs.append(q)
        seed = q
    return qs


def pick_side(a):
    """One (tilt, yaw) at the block: test pose 2 cm above the grasp, grasp line from the highest hover that works,
    lift. Returns (q_try, L or None, line top z, Lup or None)."""
    xy, z_grasp, top_z, hovers, z_max, step, yaw, beta, tcp = a
    q_try = arm_ik.ik_pose([xy[0], xy[1], z_grasp + 0.02], yaw, beta, tcp)
    if not (q_try[1] < 1.5 and q_try[2] < 0.5):
        return q_try, None, None, None
    L, z_top = None, None
    for hov in hovers:
        z_top = top_z + hov
        L = arm_ik.pose_line(xy, z_top, z_grasp, yaw, beta, tcp)
        if L is not None:
            break
    if L is None:
        return q_try, None, z_top, None
    return q_try, L, z_top, extend_up(xy, z_top, z_max, yaw, beta, tcp, L[0], step)


def place_side(a):
    """One (tilt, yaw) at the cell: test pose at the carry height, line down to the release, lift.
    Returns (q_try, PL or None, PLup or None)."""
    xy, z_carry, z_place, z_max, step, yaw, beta, tcp = a
    q_try = arm_ik.ik_pose([xy[0], xy[1], z_carry], yaw, beta, tcp)
    if not (q_try[1] < 1.5 and q_try[2] < 0.5):
        return q_try, None, None
    PL = arm_ik.pose_line(xy, z_carry, z_place, yaw, beta, tcp)
    if PL is None:
        return q_try, None, None
    return q_try, PL, extend_up(xy, z_carry, z_max, yaw, beta, tcp, PL[0], step)


def _init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # Ctrl+C is the main process's (arm back to rest)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)                                 # one BLAS thread per worker
    except Exception:                                        # noqa: BLE001
        pass


def start():
    """Fork the worker processes (once)."""
    if _POOL[0] is None and USE_POOL[0]:
        try:
            _POOL[0] = mp.get_context("fork").Pool(WORKERS, initializer=_init)
        except Exception as e:                               # noqa: BLE001
            print(f"IK pool not started ({e}) - planning on one core")
            USE_POOL[0] = False
    return _POOL[0]


def precompute(betas, pick_xy, z_grasp, top_z, hovers, yaws_g, cell_xy, z_carry, z_place, z_max, yaws_p, tcp, step):
    """Every tilt x yaw at the block and at the cell, in parallel. Returns (pick, place) dicts keyed (beta, yaw),
    or None (no pool / it failed: the caller solves one by one)."""
    pool = start() if USE_POOL[0] else None
    if pool is None:
        return None
    kg = [(b, y) for b in betas for y in yaws_g]
    kp = [(b, y) for b in betas for y in yaws_p]
    xy_g, xy_p = tuple(map(float, pick_xy)), tuple(map(float, cell_xy))
    try:
        rg = pool.map_async(pick_side, [(xy_g, z_grasp, top_z, hovers, z_max, step, y, b, tcp) for b, y in kg],
                            chunksize=1)
        rp = pool.map_async(place_side, [(xy_p, z_carry, z_place, z_max, step, y, b, tcp) for b, y in kp],
                            chunksize=1)
        return dict(zip(kg, rg.get(TIMEOUT_S))), dict(zip(kp, rp.get(TIMEOUT_S)))
    except Exception as e:                                   # noqa: BLE001 - worker died / timeout: plan serially
        print(f"   IK pool failed ({e}) - planning on one core")
        try:
            pool.terminate()
        except Exception:                                    # noqa: BLE001
            pass
        _POOL[0] = None
        USE_POOL[0] = False
        return None

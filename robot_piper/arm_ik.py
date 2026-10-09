"""PC-side inverse kinematics for PiPER with a tilted top-down tool.

Tool pose = yaw about base z, then the tool z pointing down, then a tilt beta about the tool y axis
(the jaw-closing axis, so the fingers stay level). beta is a free variable in [-max_tilt, max_tilt]
(penalised, so the solver stays as vertical as the joint limits allow). Joints within JOINT_LIM.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from hover_test import JOINT_LIM, fk

SEEDS = [np.array(s, float) for s in ([0, 110, -80, 0, 60, 0], [0, 90, -60, 0, 40, 0], [0, 130, -100, 0, 65, 0],
                                      [0, 60, -40, 0, 30, 0], [0, 75, -95, 0, 50, 0])]


def rotvec(M):
    """Rotation vector of a rotation matrix (same as scipy's as_rotvec, much cheaper; used in every IK step)."""
    c = (M[0, 0] + M[1, 1] + M[2, 2] - 1.0) / 2.0
    ang = np.arccos(min(1.0, max(-1.0, c)))
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    if ang < 1e-6:
        return v / 2.0
    s = np.sin(ang)
    if s > 1e-6:
        return v * (ang / (2.0 * s))
    return Rot.from_matrix(M).as_rotvec()                   # ~180 deg: rare, exact fallback


def tool_R(yaw_deg, beta_deg):
    """Rotation of the tool (flange) frame: z down, jaws along tool y, tilted by beta about tool y."""
    return (Rot.from_euler("z", yaw_deg, degrees=True) * Rot.from_euler("x", 180, degrees=True)
            * Rot.from_euler("y", beta_deg, degrees=True)).as_matrix()


def ik_tip(tip_m, yaw_deg, tcp_m, q_seed=None, max_tilt=40.0, tilt_weight=0.3):
    """Joints (deg) putting the tool tip at tip_m with yaw_deg and the smallest tilt that works.
    Returns (q, beta_deg, pos_err_mm, rot_err_deg)."""
    lo, hi = JOINT_LIM[:, 0], JOINT_LIM[:, 1]
    tip = np.asarray(tip_m) * 1000

    def resid(x):
        q, beta = x[:6], x[6]
        pos, R = fk(q)
        Rt = tool_R(yaw_deg, beta)
        tip_q = pos + R @ [0, 0, tcp_m * 1000]
        return np.r_[tip_q - tip, 200 * rotvec(Rt.T @ R), tilt_weight * beta]

    best = None
    seeds = ([np.asarray(q_seed, float)] if q_seed is not None else []) + SEEDS
    for s in seeds:
        x0 = np.r_[np.clip(s, lo + 1e-3, hi - 1e-3), 0.0]
        sol = least_squares(resid, x0, bounds=(np.r_[lo, -max_tilt], np.r_[hi, max_tilt]))
        r = resid(sol.x)
        pe, re = float(np.linalg.norm(r[:3])), float(np.degrees(np.linalg.norm(r[3:6]) / 200))
        key = pe + 10 * re + 0.05 * abs(sol.x[6])
        if best is None or key < best[0]:
            best = (key, sol.x[:6], float(sol.x[6]), pe, re)
        if pe < 0.5 and re < 0.2 and abs(sol.x[6]) < 1:
            break
    return best[1], best[2], best[3], best[4]


def tip_of(q, tcp_m):
    pos, R = fk(q)
    return (pos + R @ [0, 0, tcp_m * 1000]) / 1000, R


def ik_pose(tip_m, yaw_deg, beta_deg, tcp_m, q_seed=None, quick=False):
    """Joints for the tool tip at tip_m with a FIXED orientation tool_R(yaw, beta) (no free tilt).
    quick: only from q_seed (a neighbouring point of a line: if that fails the pose is out of reach - trying
    all the other seeds only costs time). Returns (q, pos_err_mm, rot_err_deg)."""
    lo, hi = JOINT_LIM[:, 0], JOINT_LIM[:, 1]
    tip = np.asarray(tip_m) * 1000
    Rt = tool_R(yaw_deg, beta_deg)

    def resid(q):
        pos, R = fk(q)
        return np.r_[pos + R @ [0, 0, tcp_m * 1000] - tip, 200 * rotvec(Rt.T @ R)]

    best = None
    seeds = ([np.asarray(q_seed, float)] if q_seed is not None else []) + ([] if quick and q_seed is not None else SEEDS)
    for s in seeds:
        sol = least_squares(resid, np.clip(s, lo + 1e-3, hi - 1e-3), bounds=(lo, hi))
        r = resid(sol.x)
        pe, re = float(np.linalg.norm(r[:3])), float(np.degrees(np.linalg.norm(r[3:]) / 200))
        if best is None or pe + 10 * re < best[1] + 10 * best[2]:
            best = (sol.x, pe, re)
        if pe < 0.5 and re < 0.2:
            break
    return best


def pose_line(xy, z0, z1, yaw_deg, beta_deg, tcp_m, q_seed=None, step=0.01, tol_mm=1.5, tol_deg=0.5):
    """Tool tip on the vertical line above xy from z0 to z1, orientation fixed. List of q or None."""
    zs = np.linspace(z0, z1, max(2, int(abs(z1 - z0) / step) + 1))
    qs, seed = [], q_seed
    for z in zs:
        q, pe, re = ik_pose([xy[0], xy[1], z], yaw_deg, beta_deg, tcp_m, seed, quick=bool(qs))
        if pe > tol_mm or re > tol_deg:
            return None
        qs.append(q)
        seed = q
    return qs

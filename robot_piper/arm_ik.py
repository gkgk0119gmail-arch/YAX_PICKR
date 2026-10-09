"""PC-side inverse kinematics for PiPER with a tilted top-down tool.

Tool pose = yaw about base z, then the tool z pointing down, then a tilt beta about the tool y axis
(the jaw-closing axis, so the fingers stay level). beta is a free variable in [-max_tilt, max_tilt]
(penalised, so the solver stays as vertical as the joint limits allow). Joints within JOINT_LIM.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import hover_test as _ht
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


ANALYTIC_JAC = False          # exact Jacobian for ik_pose (2 FK per solver step instead of 7); False -> scipy's
                              # finite differences (the old way)
_D2R = np.pi / 180.0


def fk_axes(q_deg):
    """fk() plus every joint's axis (unit, base frame) and origin (mm): for the analytic Jacobian."""
    th = np.radians(np.asarray(q_deg, float)) + _ht._DH_THETA
    ct, st = np.cos(th), np.sin(th)
    T = np.eye(4)
    Z, O = np.empty((6, 3)), np.empty((6, 3))
    for i in range(6):
        ca, sa, a, d = _ht._CA[i], _ht._SA[i], _ht._DH_A[i], _ht._DH_D[i]
        T = T @ np.array([[ct[i], -st[i], 0.0, a],
                          [st[i] * ca, ct[i] * ca, -sa, -sa * d],
                          [st[i] * sa, ct[i] * sa, ca, ca * d],
                          [0.0, 0.0, 0.0, 1.0]])
        Z[i], O[i] = T[:3, 2], T[:3, 3]
    return T[:3, 3].copy(), T[:3, :3].copy(), Z, O


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _jl_inv(phi):
    """Inverse left Jacobian of SO(3) at rotation vector phi: d(rotvec) = Jl^-1 * (small rotation, left)."""
    t = np.linalg.norm(phi)
    P = _skew(phi)
    if t < 1e-6:
        return np.eye(3) - 0.5 * P + P @ P / 12.0
    return np.eye(3) - 0.5 * P + (1.0 / t ** 2 - (1.0 + np.cos(t)) / (2.0 * t * np.sin(t))) * (P @ P)


def pose_jac(q, Rt, tcp_mm):
    """d(ik_pose residual)/dq (6 x 6, per degree): tip position rows from the joint axes, rotation rows
    200 * Jl^-1(rotvec(Rt^T R)) Rt^T z_i."""
    pos, R, Z, O = fk_axes(q)
    phi = rotvec(Rt.T @ R)
    if np.linalg.norm(phi) > 2.8:                 # near 180 deg Jl^-1 blows up: finite differences there
        def res(x):
            p_, R_ = fk(x)
            return np.r_[p_ + R_[:, 2] * tcp_mm, 200 * rotvec(Rt.T @ R_)]
        r0 = res(q)
        return np.array([(res(q + e * 1e-6) - r0) / 1e-6 for e in np.eye(6)]).T
    tip = pos + R[:, 2] * tcp_mm
    Jp = np.cross(Z, tip - O).T
    Jr = 200.0 * _jl_inv(phi) @ (Rt.T @ Z.T)
    return np.vstack([Jp, Jr]) * _D2R


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
        sol = least_squares(resid, np.clip(s, lo + 1e-3, hi - 1e-3), bounds=(lo, hi),
                            jac=(lambda q: pose_jac(q, Rt, tcp_m * 1000)) if ANALYTIC_JAC else "2-point")
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

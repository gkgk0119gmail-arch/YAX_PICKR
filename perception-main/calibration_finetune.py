"""
T_base_camera fine-tuning.

How to use:
    1. Run: python detection_server.py --results (add --no_vis if headless).
    2. Place an object near a corner of BBOX_XY, wait for the [--results]
       block, copy the printed `before = (x, y, z)` line.
    3. Drive the EEF to that detected target (gripper-down, yaw to match the
       printed yaw). Manually jog ONLY in X/Y/Z until the tool tip touches
       the centroid of the object.
    4. Read the actual TCP position off the teach pendant /
       GetActualTCPPose() etc. -- only the (x, y, z) of the tip matters.
    5. Append the (before, after) XYZ pair to CORRESPONDENCES below.
    6. Repeat at >=3 well-spread XY locations across the workspace.
    7. Run: python perception/calibration_finetune.py
    8. Paste the printed T_BASE_CAMERA matrix into config.py.

Why only XYZ?
    The solver below ignores the 3x3 rotation block of any pose. It only
    reads the translation (tip position) of `before` and `after`, then fits
    a rigid 4x4 T_corr such that `T_corr @ before_xyz ~= after_xyz` (Kabsch
    /Umeyama SVD on point pairs). The corrected camera calibration is then
    `T_corr @ T_BASE_CAMERA`. The rotation of T_corr is constrained purely
    by how the before-points warp into the after-points across the
    workspace -- not by any rotation you record.

    => Yaw of the touch pose is irrelevant. Don't bother recording it.

Notes:
    - With 1 or 2 correspondences only the TRANSLATION is corrected.
    - With 3+ correspondences the full rotation + translation is fitted via
      SVD (Kabsch/Umeyama). This is what you need to fix tilt errors (e.g.
      "same object reports a different height depending on where I put it").
    - For good tilt correction, the 3+ correspondences MUST be spread
      across the workspace. Clustered points cannot constrain rotation.
      Recommended layout for BBOX_XY = ((0.25, 0.75), (-0.25, 0.25)):
            (0.30, -0.20)   (0.70, -0.20)
            (0.30, +0.20)   (0.70, +0.20)
      i.e. one correspondence near each corner of the detection area.
    - `before` is the gripper-tip XYZ (detection_server.py prints exactly
      that). `after` must ALSO be the gripper-tip XYZ in base frame. If
      your robot reports the FLANGE pose, push it out by the TCP offset:
            tip_pos = flange_pos + R_flange @ TCP_OFFSET
      with TCP_OFFSET = [0, 0, 0.14] for the current gripper.
"""

import os
import sys

import numpy as np

# Make the repo root importable when running this file directly from
# the perception/ directory (e.g. `cd perception && python calibration_finetune.py`).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import T_BASE_CAMERA  # noqa: E402  -- live calibration matrix


# =============================================================================
# EDIT BELOW: list of (before_xyz, after_xyz) correspondences.
# T_BC is loaded automatically from config.T_BASE_CAMERA so this script always
# fine-tunes whatever is currently deployed. To experiment with a different
# starting matrix, override T_BC right below.
# =============================================================================

T_BC = T_BASE_CAMERA

# Each entry is (before_xyz, after_xyz), both in base frame metres.
#   before_xyz : detected centroid, copy from `--results` output
#   after_xyz  : actual gripper-tip position when touching the centroid
#                = flange_pos + R_flange @ TCP_OFFSET    with
#                  TCP_OFFSET = config.TCP_OFFSET = [0, 0, 0.14]
# Add more pairs as you collect them at different XY positions.
CORRESPONDENCES = [
    # before                                       after (tip position, TCP_OFFSET applied)
    ((-0.0906, +0.5290, +0.0832), (-0.094838, +0.524255, +0.082154)),  # corner: -x, -y
    ((-0.1182, +0.7482, +0.0821), (-0.114403, +0.740707, +0.080364)),  # corner: -x, +y
    ((+0.1893, +0.7591, +0.0756), (+0.176406, +0.750549, +0.078203)),  # corner: +x, +y
    ((+0.1873, +0.5371, +0.0657), (+0.181046, +0.535095, +0.070925)),  # corner: +x, -y
]


# =============================================================================
# Solver
# =============================================================================

def _coerce_xyz(p) -> np.ndarray:
    """Accept (x,y,z) tuple/list/array, or a 4x4 pose (legacy) -> (3,) array."""
    arr = np.asarray(p, dtype=float)
    if arr.shape == (4, 4):
        return arr[:3, 3].copy()
    if arr.shape == (3,):
        return arr
    raise ValueError(
        f"expected (x,y,z) triple or 4x4 pose, got shape {arr.shape}"
    )


def _solve_rigid_transform_svd(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Kabsch/Umeyama: rigid 4x4 T such that tgt ~= T @ src for (N, 3) points."""
    assert src.shape == tgt.shape and src.shape[0] >= 3
    c_src = src.mean(axis=0)
    c_tgt = tgt.mean(axis=0)
    H = (src - c_src).T @ (tgt - c_tgt)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c_tgt - R @ c_src
    return T


def finetune_t_bc(T_bc: np.ndarray, correspondences: list) -> np.ndarray:
    """
    Correct T_base_camera from (before_xyz, after_xyz) pairs.

    - 1 or 2 pairs: translation-only correction (cannot solve for rotation).
    - 3+ pairs:    full rigid SVD correction (rotation + translation).
    """
    assert len(correspondences) >= 1, "Need at least 1 correspondence"

    before = np.array([_coerce_xyz(pair[0]) for pair in correspondences])
    after  = np.array([_coerce_xyz(pair[1]) for pair in correspondences])
    deltas = after - before
    dmags  = np.linalg.norm(deltas, axis=1)

    print(f"Loaded {len(correspondences)} correspondence(s).")
    print("Per-pair predicted -> actual error (mm):")
    for i, (b, a, d, m) in enumerate(zip(before, after, deltas, dmags)):
        print(f"  #{i+1:>2}: before xy=({b[0]:+.3f},{b[1]:+.3f}) z={b[2]:+.4f}"
              f"   delta=({d[0]*1000:+6.2f},{d[1]*1000:+6.2f},{d[2]*1000:+6.2f}) "
              f"  |delta|={m*1000:5.2f}")
    print(f"Mean |delta|: {dmags.mean()*1000:.2f} mm   Max: {dmags.max()*1000:.2f} mm")

    if len(correspondences) >= 3:
        print("\nMode: full 6-DOF SVD correction (rotation + translation).")
        T_corr = _solve_rigid_transform_svd(before, after)
    else:
        print("\nMode: translation-only (need 3+ correspondences for rotation).")
        T_corr = np.eye(4)
        T_corr[:3, 3] = deltas.mean(axis=0)

    # Residuals after correction.
    before_h = np.hstack([before, np.ones((len(before), 1))])
    corrected = (T_corr @ before_h.T).T[:, :3]
    res = np.linalg.norm(after - corrected, axis=1)
    angle_deg = float(np.degrees(np.arccos(
        np.clip((np.trace(T_corr[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    )))
    print(f"\nApplied correction:")
    print(f"  Rotation:    {angle_deg:.4f} deg")
    print(f"  Translation: ({T_corr[0,3]*1000:+.2f}, "
          f"{T_corr[1,3]*1000:+.2f}, {T_corr[2,3]*1000:+.2f}) mm")
    print(f"Residuals after correction (mm): "
          f"mean={res.mean()*1000:.2f}, max={res.max()*1000:.2f}")

    return T_corr @ T_bc


def format_matrix_for_code(T: np.ndarray, var_name: str = "T_base_camera") -> str:
    """Format 4x4 matrix as Python code for copy-paste."""
    lines = [f"{var_name} = np.array(["]
    for i in range(4):
        row = ", ".join(f"{T[i, j]:12.8f}" for j in range(4))
        lines.append(f"    [{row}],")
    lines.append("])")
    return "\n".join(lines)


# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("T_base_camera fine-tuning")
    print("=" * 70)
    print(f"\nCurrent T_base_camera (loaded from {_REPO_ROOT}/config.py):")
    print(T_BC)

    T_bc_new = finetune_t_bc(T_BC, CORRESPONDENCES)

    print("\n" + "=" * 70)
    print("CORRECTED T_base_camera:")
    print("=" * 70)
    print(T_bc_new)
    print("\nCopy-paste into config.py (replacing T_BASE_CAMERA):\n")
    print(format_matrix_for_code(T_bc_new, var_name="T_BASE_CAMERA"))

    np.save("t_bc_corrected.npy", T_bc_new)
    print("\nSaved to t_bc_corrected.npy")

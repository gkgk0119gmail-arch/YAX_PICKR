#!/usr/bin/env python3
"""Build and display the suction gripper mesh in meshcat.

Standalone visualization tool to design / inspect the end-effector geometry:

    - Suction cup (white cone) at the contact point (local origin)
    - Steel rod, 10 cm long, extending down from the body
    - Rectangular prism body (blue)
    - Cubical sensor bump on the lower side of the body (the "back")
    - RealSense camera mounted on the right (+Y) side via a ~6 cm protrusion plate

Local frame convention (matches create_gripper_mesh in view_clutter.py):
    - Origin = suction cup contact tip
    - +Z = approach axis (gripper body is at +Z, away from the object)

Usage:
    python build_gripper_mesh.py
    python build_gripper_mesh.py --meshcat-url tcp://127.0.0.1:6000
"""
from __future__ import annotations

import argparse
import time

import numpy as np


# ----------------------------------------------------------------------------
# Geometry parameters (meters). Tweak these to redesign the gripper.
# ----------------------------------------------------------------------------
CUP_BOTTOM_RADIUS = 0.020   # suction cup outer rim (4 cm diameter)
CUP_TOP_RADIUS = 0.008      # where the cup meets the rod
CUP_HEIGHT = 0.018          # cup lip height

ROD_RADIUS = 0.007          # steel rod (~14 mm diameter)
ROD_LENGTH = 0.10           # 10 cm rod

# Body: rectangular prism, 12 cm long with a 7 cm x 2.5 cm cross section.
#   width (X) = 7 cm, depth (Y) = 2.5 cm, height/length (Z) = 12 cm.
# The 2.5 cm faces are the "short sides" (normal to +/-X).
BODY_W = 0.070
BODY_D = 0.025
BODY_H = 0.120

# Cubical sensor bump on the short (2.5 cm) side, near the bottom. This +X face
# is the "back" of the gripper.
SENSOR_SIZE = (0.024, 0.022, 0.030)   # (X protrusion, Y, Z)

# RealSense camera on the right (+Y) side (right when the sensor / +X face is
# the back). It is an 8 cm bar mounted horizontally (long axis along X) so both
# ends are equidistant from the gripper centerline, held off the body by a
# ~6 cm protrusion plate.
CAM_BODY = (0.080, 0.020, 0.025)   # (X=length, Y, Z)
CAM_PLATE_LEN = 0.060              # protrusion plate length: body face -> camera (~6 cm)
CAM_CENTER_Z = 0.060               # camera center, measured above the body bottom
CAM_PLATE_XZ = (BODY_W, 0.012)     # protrusion plate cross section (X = full gripper width, Z)

# Bin: open-top container, 38 cm x 45 cm cross section, 40 cm tall.
BIN_X = 0.38
BIN_Y = 0.45
BIN_Z = 0.40
BIN_WALL = 0.010                   # wall / floor thickness
BIN_OFFSET = (0.50, 0.0, 0.0)      # where to place the bin relative to the gripper

# Colors
COL_CUP = 0xEEEEEE
COL_ROD = 0x9AA0A6
COL_BODY = 0x2348C8
COL_SENSOR = 0xE8E8E8
COL_CAM = 0x202124
COL_CAM_LENS = 0x3A6EA5
COL_BIN = 0xB7884A


def rot_x(theta: float) -> np.ndarray:
    """3x3 rotation about the X axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_z(theta: float) -> np.ndarray:
    """3x3 rotation about the Z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_T(translation, R: np.ndarray | None = None) -> np.ndarray:
    """Build a 4x4 homogeneous transform from translation (+ optional rotation)."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def build_gripper_parts():
    """Return a list of (name, geometry, material, transform) for the gripper.

    Geometries are created with meshcat primitives and positioned in the local
    gripper frame (origin at the suction cup tip, +Z = approach axis).
    """
    import meshcat.geometry as g

    parts = []

    # Z stack-up of the major components.
    cup_top_z = CUP_HEIGHT
    rod_bottom_z = cup_top_z
    rod_top_z = rod_bottom_z + ROD_LENGTH
    body_bottom_z = rod_top_z
    body_center_z = body_bottom_z + BODY_H / 2.0

    # meshcat Cylinder is aligned along +Y; rotate +90 deg about X to align +Z.
    z_align = rot_x(np.pi / 2.0)

    # --- Suction cup (cone widening toward the contact tip) -----------------
    cup = g.Cylinder(CUP_HEIGHT, radiusTop=CUP_TOP_RADIUS,
                     radiusBottom=CUP_BOTTOM_RADIUS)
    parts.append((
        "gripper/cup", cup,
        g.MeshLambertMaterial(color=COL_CUP),
        make_T([0, 0, CUP_HEIGHT / 2.0], z_align),
    ))

    # --- Steel rod (10 cm) ---------------------------------------------------
    rod = g.Cylinder(ROD_LENGTH, ROD_RADIUS)
    parts.append((
        "gripper/rod", rod,
        g.MeshLambertMaterial(color=COL_ROD, reflectivity=0.7),
        make_T([0, 0, rod_bottom_z + ROD_LENGTH / 2.0], z_align),
    ))

    # --- Body (rectangular prism) -------------------------------------------
    body = g.Box([BODY_W, BODY_D, BODY_H])
    parts.append((
        "gripper/body", body,
        g.MeshLambertMaterial(color=COL_BODY),
        make_T([0, 0, body_center_z]),
    ))

    # --- Sensor: cubical bump on the short (+X, 2.5 cm) side, near bottom ----
    sw, sd, sh = SENSOR_SIZE
    sensor_x = BODY_W / 2.0 + sw / 2.0 - 0.003   # slight overlap so it reads as attached
    sensor_z = body_bottom_z + sh / 2.0
    parts.append((
        "gripper/sensor", g.Box([sw, sd, sh]),
        g.MeshLambertMaterial(color=COL_SENSOR),
        make_T([sensor_x, 0.0, sensor_z]),
    ))

    # --- RealSense camera on the right (+Y) side ----------------------------
    cw, cd, ch = CAM_BODY
    cam_y = BODY_D / 2.0 + CAM_PLATE_LEN + cd / 2.0
    cam_z = body_bottom_z + CAM_CENTER_Z
    parts.append((
        "gripper/cam_body", g.Box([cw, cd, ch]),
        g.MeshLambertMaterial(color=COL_CAM),
        make_T([0.0, cam_y, cam_z]),
    ))

    # Protrusion plate: horizontal arm (~6 cm) joining the body face to the
    # camera, at the top of the camera bar (camera "mounts at the top").
    px, pz = CAM_PLATE_XZ
    body_face_y = BODY_D / 2.0
    cam_inner_y = cam_y - cd / 2.0
    plate_len = (cam_inner_y - body_face_y) + 0.008   # span + small overlaps
    plate_y = (body_face_y + cam_inner_y) / 2.0
    plate_z = cam_z + ch / 2.0 - pz / 2.0
    parts.append((
        "gripper/cam_plate", g.Box([px, plate_len, pz]),
        g.MeshLambertMaterial(color=COL_BODY),
        make_T([0.0, plate_y, plate_z]),
    ))

    # Two lenses on the outward face of the camera (pointing +Y), spaced along
    # the length of the bar (X). Cylinder defaults to +Y, so no rotation needed.
    lens_r = 0.006
    lens_y = cam_y + cd / 2.0
    for i, dx in enumerate((-0.022, 0.022)):
        parts.append((
            f"gripper/cam_lens_{i}", g.Cylinder(0.004, lens_r),
            g.MeshLambertMaterial(color=COL_CAM_LENS),
            make_T([dx, lens_y, cam_z]),
        ))

    return parts


def build_bin_parts(origin=(0.0, 0.0, 0.0), bin_x=BIN_X, bin_y=BIN_Y,
                    bin_z=BIN_Z, wall=BIN_WALL, color=COL_BIN, opacity=0.9):
    """Return (name, geometry, material, transform) for an open-top bin.

    The bin floor sits on the ``z=origin_z`` plane and the box is centered (in
    X/Y) at the given origin, all expressed in the bin's local frame. Dims and
    appearance default to the module constants so the standalone script is
    unchanged, but callers (e.g. ``bt_pick --vis``) can override them from
    ``config.py``. Modeled as a floor plus four walls of thickness ``wall``.
    """
    import meshcat.geometry as g

    ox, oy, oz = origin
    mat = g.MeshLambertMaterial(color=color, opacity=opacity, transparent=True)
    t = wall
    parts = []

    # Floor
    parts.append((
        "bin/floor", g.Box([bin_x, bin_y, t]),
        mat, make_T([ox, oy, oz + t / 2.0]),
    ))

    wall_z = oz + bin_z / 2.0
    # Walls along the +/-X faces (each spans the full Y extent).
    for sign in (-1, 1):
        parts.append((
            f"bin/wall_x{'p' if sign > 0 else 'n'}",
            g.Box([t, bin_y, bin_z]),
            mat, make_T([ox + sign * (bin_x / 2.0 - t / 2.0), oy, wall_z]),
        ))
    # Walls along the +/-Y faces.
    for sign in (-1, 1):
        parts.append((
            f"bin/wall_y{'p' if sign > 0 else 'n'}",
            g.Box([bin_x, t, bin_z]),
            mat, make_T([ox, oy + sign * (bin_y / 2.0 - t / 2.0), wall_z]),
        ))

    return parts


def main():
    parser = argparse.ArgumentParser(description="Build/show the gripper mesh in meshcat")
    parser.add_argument("--meshcat-url", type=str, default=None,
                        help="ZMQ URL for an existing meshcat server "
                             "(e.g. tcp://127.0.0.1:6000)")
    parser.add_argument("--triad-size", type=float, default=0.05,
                        help="Size of the coordinate frame triad")
    args = parser.parse_args()

    import meshcat
    import meshcat.geometry as g

    if args.meshcat_url:
        vis = meshcat.Visualizer(zmq_url=args.meshcat_url)
    else:
        vis = meshcat.Visualizer()

    try:
        vis.open()
    except Exception as e:
        print(f"[meshcat] open() failed (load the URL manually): {e}")

    try:
        print(f"[meshcat] viewer at: {vis.url()}")
    except Exception:
        pass

    vis.delete()

    # Coordinate frame at the suction cup tip (origin).
    vis["frame"].set_object(g.triad(args.triad_size))

    parts = build_gripper_parts() + build_bin_parts(BIN_OFFSET)
    for name, geom, mat, T in parts:
        vis[name].set_object(geom, mat)
        vis[name].set_transform(T)

    total_h = CUP_HEIGHT + ROD_LENGTH + BODY_H
    print(f"[gripper] total height = {total_h * 100:.1f} cm "
          f"(rod = {ROD_LENGTH * 100:.0f} cm)")
    print(f"[bin] {BIN_X * 100:.0f} x {BIN_Y * 100:.0f} cm, "
          f"{BIN_Z * 100:.0f} cm tall, placed at {BIN_OFFSET}")
    print(f"[scene] {len(parts)} parts displayed. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[gripper] bye")


if __name__ == "__main__":
    main()

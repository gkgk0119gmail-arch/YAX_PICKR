from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_realsense_json(d: Dict[str, Any]) -> "PinholeIntrinsics":
        return PinholeIntrinsics(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d.get("ppx", d.get("cx"))),
            cy=float(d.get("ppy", d.get("cy"))),
        )


def load_intrinsics_json(json_path: str) -> Tuple[PinholeIntrinsics, float]:
    """Load intrinsics and depth scale saved by the capture script.

    Returns (intrinsics, depth_scale_m_per_unit).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    intr = PinholeIntrinsics.from_realsense_json(meta)
    depth_scale = float(meta.get("depth_scale_m_per_unit", 0.001))
    return intr, depth_scale


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    depth_scale_m_per_unit: float = 1.0,
    valid_depth_range_m: Tuple[float, float] = (0.05, 20.0),
    remove_invalid: bool = True,
    return_xyz_image: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Back-project a depth image into a 3D point cloud using pinhole intrinsics.

    Parameters:
    - depth: HxW uint16/float32 depth image. If uint16, values are in sensor units.
    - intrinsics: pinhole parameters for the depth camera (must match depth size).
    - depth_scale_m_per_unit: meters per sensor unit (RealSense depth scale). If depth
      is already in meters, pass 1.0.
    - valid_depth_range_m: (min, max) range for valid depths after scaling.
    - remove_invalid: if True, filter out invalid/zero depths and return Nx3. If False,
      invalid points will be NaN in the image-shaped XYZ output.
    - return_xyz_image: if True, also return the HxWx3 XYZ image (NaN where invalid).

    Returns:
    - points: Nx3 array of 3D points in meters if remove_invalid=True, otherwise HxWx3.
    - xyz_image (optional): HxWx3 array if return_xyz_image=True, else None.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")

    height, width = depth.shape
    if intrinsics.width != width or intrinsics.height != height:
        raise ValueError(
            f"intrinsics size {intrinsics.width}x{intrinsics.height} does not match depth {width}x{height}"
        )

    # Convert to meters
    z_m = depth.astype(np.float32) * float(depth_scale_m_per_unit)

    # Build pixel grid (i: rows/y, j: cols/x)
    ii, jj = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    # Back-project
    X = (jj - intrinsics.cx) * z_m / intrinsics.fx
    Y = (ii - intrinsics.cy) * z_m / intrinsics.fy
    Z = z_m

    xyz_image = np.stack([X, Y, Z], axis=-1)

    # Mask invalid depths
    zmin, zmax = valid_depth_range_m
    valid = np.isfinite(Z) & (Z > float(zmin)) & (Z < float(zmax))

    if remove_invalid:
        pts = xyz_image[valid]
    else:
        # Fill invalid with NaNs to keep image layout
        xyz_image[~valid] = np.nan
        pts = xyz_image

    return (pts, xyz_image) if return_xyz_image else (pts, None)


def export_mesh_stl_from_depth(
    depth: np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    depth_scale_m_per_unit: float = 1.0,
    mesh_stride: int = 4,
    z_jump_thresh: float = 0.5,
    max_edge_len: float = 1.0,
    out_stl_path: str = "mesh.stl",
) -> bool:
    """Triangulate the depth into an STL mesh and save to disk.

    Returns True if export succeeded, False otherwise.
    """
    pts_img, xyz_img = depth_to_pointcloud(
        depth,
        intrinsics,
        depth_scale_m_per_unit=depth_scale_m_per_unit,
        remove_invalid=False,
        return_xyz_image=True,
    )

    if xyz_img is None:
        return False

    Z = xyz_img[..., 2]
    valid = np.isfinite(Z)

    rows = np.arange(0, intrinsics.height, mesh_stride)
    cols = np.arange(0, intrinsics.width, mesh_stride)
    Xs = xyz_img[rows][:, cols, 0]
    Ys = xyz_img[rows][:, cols, 1]
    Zs = xyz_img[rows][:, cols, 2]
    Val = valid[rows][:, cols]

    h_s, w_s = Xs.shape
    vertices = np.stack([Xs, Ys, Zs], axis=-1).reshape(-1, 3)
    faces = []

    def edge_ok(v0: np.ndarray, v1: np.ndarray) -> bool:
        return np.linalg.norm(v0 - v1) <= max_edge_len

    for r in range(h_s - 1):
        for c in range(w_s - 1):
            if not (Val[r, c] and Val[r, c + 1] and Val[r + 1, c] and Val[r + 1, c + 1]):
                continue
            idx00 = r * w_s + c
            idx01 = r * w_s + (c + 1)
            idx10 = (r + 1) * w_s + c
            idx11 = (r + 1) * w_s + (c + 1)

            v00 = vertices[idx00]
            v01 = vertices[idx01]
            v10 = vertices[idx10]
            v11 = vertices[idx11]

            z_patch = np.array([v00[2], v01[2], v10[2], v11[2]])
            if np.nanmax(z_patch) - np.nanmin(z_patch) > z_jump_thresh:
                continue

            if edge_ok(v00, v01) and edge_ok(v01, v10) and edge_ok(v10, v00):
                faces.append([idx00, idx01, idx10])
            if edge_ok(v11, v10) and edge_ok(v10, v01) and edge_ok(v01, v11):
                faces.append([idx11, idx10, idx01])

    faces = np.asarray(faces, dtype=np.int32)
    if faces.size == 0:
        return False

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(out_stl_path)
        return True
    except Exception:
        try:
            from stl import mesh as stl_mesh
            stl_data = np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype)
            stl_obj = stl_mesh.Mesh(stl_data)
            for i, (a, b, c) in enumerate(faces):
                stl_obj.vectors[i] = vertices[[a, b, c]]
            stl_obj.save(out_stl_path)
            return True
        except Exception:
            return False

def clean_point_cloud(points_xyz, voxel_size=0.005, nb_neighbors=20, std_ratio=1.5,
                      use_radius=False, radius=0.02, min_neighbors=8):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    if use_radius:
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    else:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

if __name__ == "__main__":
    import pyrealsense2 as rs
    import numpy as np
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pc = rs.pointcloud()
    pipeline.start(config)

    i = 0
    while i<5:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        print(depth_frame)
        color_frame = frames.get_color_frame()
        print(color_frame)
        i = i+1
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())
        points.export_to_ply("data/output{i}.ply", color_frame)
        print(type(points), points)
        print(type(vtx), vtx.shape, vtx)
        print(type(tex), tex.shape, tex)
    pipeline.stop()


# if __name__ == "__main__":
#     import argparse
#     import os

#     parser = argparse.ArgumentParser(description="Depth to point cloud and STL export")
#     parser.add_argument("--depth", required=True, help="Path to depth .npy (uint16 sensor units or float meters)")
#     parser.add_argument("--intrinsics", required=True, help="Path to intrinsics JSON")
#     parser.add_argument("--out-ply", default=None, help="Optional output PLY for point cloud (Nx3)")
#     parser.add_argument("--out-stl", default=None, help="Optional output STL mesh path")
#     parser.add_argument("--stride", type=int, default=4, help="Mesh stride for STL")
#     parser.add_argument("--z-jump", type=float, default=0.5, help="Z jump threshold (m) for meshing")
#     parser.add_argument("--max-edge", type=float, default=1.0, help="Max triangle edge len (m)")
#     args = parser.parse_args()

#     depth = np.load(args.depth)
#     intr, depth_scale = load_intrinsics_json(args.intrinsics)

#     # Detect if depth is already meters
#     scale = depth_scale if depth.dtype != np.float32 and depth.dtype != np.float64 else 1.0

#     points, _ = depth_to_pointcloud(depth, intr, depth_scale_m_per_unit=scale)
#     print(f"Points: {points.shape[0]}")

#     if args.out_ply:
#         try:
#             import plyfile
#             verts = np.core.records.fromarrays(points.T, names='x, y, z', formats='f4, f4, f4')
#             ply = plyfile.PlyData([plyfile.PlyElement.describe(verts, 'vertex')], text=True)
#             ply.write(args.out_ply)
#             print(f"Saved PLY: {args.out_ply}")
#         except Exception as e:
#             print(f"PLY export failed ({e}); skipping PLY.")

#     if args.out_stl:
#         ok = export_mesh_stl_from_depth(
#             depth,
#             intr,
#             depth_scale_m_per_unit=scale,
#             mesh_stride=args.stride,
#             z_jump_thresh=args.z_jump,
#             max_edge_len=args.max_edge,
#             out_stl_path=args.out_stl,
#         )
#         print(f"Saved STL: {args.out_stl}" if ok else "STL export failed or empty")

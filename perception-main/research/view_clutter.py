"""
Viewer for cluttered bin grasp detection results.

Loads saved viz_data.npz and displays the 3D visualization with:
- Colored segment pointclouds (objects)
- Grey background points (visible surfaces)
- Black occupancy volume (occluded space below visible surfaces)
- Red grasp cylinders at detected grasp poses

Usage:
    python view_clutter.py                          # View latest session
    python view_clutter.py data/grasp_20260127_*/   # View specific session
    python view_clutter.py --no-images              # Skip image display
    python view_clutter.py --no-occupancy           # Skip occupancy volume
    python view_clutter.py --voxel-size 0.005       # Finer occupancy resolution
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_viz_data(session_dir: Path) -> dict:
    """Load visualization data from npz file."""
    npz_path = session_dir / "viz_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No viz_data.npz found in {session_dir}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Handle depth image (may be empty array for older saves)
    depth_image = data.get('depth_image', np.array([]))
    if depth_image.size == 0:
        depth_image = None
    
    # Load intrinsics (with defaults for older saves)
    intrinsics = data.get('intrinsics', None)
    if intrinsics is not None:
        intrinsics = intrinsics.item()  # Convert from 0-d array
    else:
        # Default D455 intrinsics at 1280x720
        intrinsics = {'fx': 640.0, 'fy': 640.0, 'cx': 640.0, 'cy': 360.0, 'width': 1280, 'height': 720}
    
    # Load crop region
    crop = data.get('crop', None)
    if crop is not None:
        crop = crop.tolist() if hasattr(crop, 'tolist') else list(crop)
    
    # Load camera transform
    T_base_camera = data.get('T_base_camera', None)
    if T_base_camera is None:
        # Default transform (camera looking down at bin)
        T_base_camera = np.array([
            [-0.99870602, -0.05068679,  0.00413055,  0.46783590],
            [-0.05061085,  0.99857739,  0.01678431,  0.11403400],
            [-0.00497542,  0.01655354, -0.99985062,  1.02680489],
            [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        ])
    
    return {
        'segments': data['segments'].tolist(),
        'grasp_results': data['grasp_results'].tolist(),
        'background_points': data['background_points'],
        'background_colors': data['background_colors'],
        'suction_cup_diameter': float(data['suction_cup_diameter']),
        'color_image': data.get('color_image', None),
        'depth_image': depth_image,
        'depth_scale': float(data.get('depth_scale', 0.001)),
        'intrinsics': intrinsics,
        'crop': crop,
        'T_base_camera': T_base_camera,
    }


def build_occupancy_volume(data: dict, voxel_size: float = 0.005, floor_z: float = None, 
                           max_ray_depth: float = 0.15) -> np.ndarray:
    """Build occupancy volume via ray casting from camera through pointcloud.
    
    Uses the camera pose (T_base_camera) to cast rays through visible surface points.
    The occluded volume is the space BEHIND each surface point along the ray direction,
    from the surface down to the floor plane.
    
    Args:
        data: Loaded viz_data dict with pointcloud and T_base_camera
        voxel_size: Size of voxels in meters (default 1cm)
        floor_z: Z coordinate of floor in robot frame (auto-detected if None)
        max_ray_depth: Maximum depth to cast rays beyond surface (meters)
    
    Returns:
        Nx3 array of occupied voxel center points in robot base frame
    """
    T_base_camera = data['T_base_camera']
    
    # Camera position in world (base) frame
    camera_pos = T_base_camera[:3, 3]
    
    # Collect all surface points (already in world frame)
    surface_points = []
    
    # Background points
    bg_pts = data['background_points']
    if len(bg_pts) > 0:
        surface_points.append(bg_pts)
    
    # Segment points
    for seg in data['segments']:
        pts = np.array(seg['points'])
        if len(pts) > 0:
            surface_points.append(pts)
    
    if not surface_points:
        print("[Occupancy] No surface points available")
        return np.zeros((0, 3))
    
    all_surface_pts = np.vstack(surface_points)
    
    # Determine floor level from surface points if not specified
    if floor_z is None:
        # Floor is the minimum Z we observe (lowest visible surface)
        floor_z = np.percentile(all_surface_pts[:, 2], 5)
    
    # Downsample surface points to match output voxel resolution
    pcd_surface = o3d.geometry.PointCloud()
    pcd_surface.points = o3d.utility.Vector3dVector(all_surface_pts)
    pcd_surface = pcd_surface.voxel_down_sample(voxel_size)
    surface_pts = np.asarray(pcd_surface.points)
    
    print(f"[Occupancy] Ray casting: {len(surface_pts):,} surface pts, "
          f"camera at [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}], "
          f"floor_z={floor_z:.3f}, voxel={voxel_size*1000:.0f}mm")
    
    # Ray casting: for each surface point, cast ray from camera through it
    # and generate voxels in the occluded space beyond
    ray_dirs = surface_pts - camera_pos  # Rays from camera to surface
    ray_dists = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_dirs_normalized = ray_dirs / (ray_dists + 1e-8)
    
    # Number of steps along each ray beyond the surface
    n_steps = int(max_ray_depth / voxel_size)
    step_offsets = np.arange(1, n_steps + 1) * voxel_size  # Start just beyond surface
    
    # Vectorized: generate all occluded points at once
    # surface_pts: (N, 3), ray_dirs_normalized: (N, 3), step_offsets: (M,)
    # Result: (N, M, 3) -> (N*M, 3)
    N = len(surface_pts)
    M = len(step_offsets)
    
    # Broadcast: surface_pts[:, None, :] + ray_dirs[:, None, :] * offsets[None, :, None]
    occluded_pts = (surface_pts[:, None, :] + 
                   ray_dirs_normalized[:, None, :] * step_offsets[None, :, None])
    occluded_pts = occluded_pts.reshape(-1, 3)
    
    # Filter points above floor
    occluded_pts = occluded_pts[occluded_pts[:, 2] >= floor_z]
    
    if len(occluded_pts) > 0:
        occupancy_pts = occluded_pts
        
        # Voxelize to remove duplicates and regularize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occupancy_pts)
        pcd_voxel = pcd.voxel_down_sample(voxel_size)
        occupancy_pts = np.asarray(pcd_voxel.points)
        
        print(f"[Occupancy] Generated {len(occupancy_pts):,} voxels")
        return occupancy_pts
    
    return np.zeros((0, 3))


def create_gripper_mesh(position: np.ndarray, normal: np.ndarray, diameter: float = 0.03,
                        cup_height: float = 0.06, body_size: tuple = (0.06, 0.03, 0.13)):
    """Create gripper mesh with suction cup (cylinder) and body (rectangular prism).
    
    Args:
        position: Grasp position (suction cup contact point)
        normal: Surface normal at grasp point (gripper approach direction)
        diameter: Suction cup diameter in meters (default 3cm)
        cup_height: Height of suction cup cylinder (default 6cm)
        body_size: (width, depth, height) of gripper body prism (default 6cm x 3cm x 13cm)
    
    Returns:
        Open3D TriangleMesh of the gripper
    """
    # Create suction cup (cylinder) - 6cm long, 3cm diameter
    cup = o3d.geometry.TriangleMesh.create_cylinder(radius=diameter/2, height=cup_height)
    cup.paint_uniform_color([1.0, 0.3, 0.3])  # Light red for cup
    cup.translate([0, 0, cup_height/2])  # Position so bottom is at origin
    
    # Create gripper body (rectangular prism) - 13cm long, 6cm x 3cm cross section
    body_w, body_d, body_h = body_size
    body = o3d.geometry.TriangleMesh.create_box(width=body_w, height=body_d, depth=body_h)
    body.paint_uniform_color([0.4, 0.4, 0.4])  # Grey for body
    # Center horizontally, place on top of cup
    body.translate([-body_w/2, -body_d/2, cup_height])
    
    # Combine into single mesh
    gripper = cup + body
    
    # Compute rotation to align Z-axis with normal
    z = np.array([0, 0, 1])
    n = normal / np.linalg.norm(normal)
    v = np.cross(z, n)
    s, c = np.linalg.norm(v), np.dot(z, n)
    
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    else:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    
    gripper.rotate(R, center=[0, 0, 0])
    gripper.translate(position)
    return gripper



def visualize(data: dict, show_occupancy: bool = True, voxel_size: float = 0.01,
              max_ray_depth: float = 0.15):
    """Visualize segments with grasps, background, and occupancy volume.
    
    Args:
        data: Loaded viz_data dictionary
        show_occupancy: Whether to show occupancy volume as black points
        voxel_size: Voxel size for occupancy volume in meters
        max_ray_depth: Maximum ray depth beyond surface for occupancy
    """
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
    
    segments = data['segments']
    grasp_results = data['grasp_results']
    background_points = data['background_points']
    background_colors = data['background_colors']
    diameter = data['suction_cup_diameter']
    
    # Build grasp result lookup
    result_map = {r['segment_id']: r for r in grasp_results}
    
    # Add occupancy volume as black points (render first, so it's behind other points)
    occupancy_pts = None
    if show_occupancy:
        occupancy_pts = build_occupancy_volume(data, voxel_size=voxel_size, 
                                               max_ray_depth=max_ray_depth)
        if len(occupancy_pts) > 0:
            occ_pcd = o3d.geometry.PointCloud()
            occ_pcd.points = o3d.utility.Vector3dVector(occupancy_pts)
            # Black color for occupancy volume
            occ_pcd.colors = o3d.utility.Vector3dVector(
                np.zeros((len(occupancy_pts), 3))
            )
            geoms.append(occ_pcd)
    
    # Add background pointcloud in grey
    if len(background_points) > 0:
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(background_points)
        # Override to grey color for background
        grey_color = np.full((len(background_points), 3), 0.5)
        bg_pcd.colors = o3d.utility.Vector3dVector(grey_color)
        geoms.append(bg_pcd)
    
    # Add segment pointclouds with colors
    for seg in segments:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg['points'])
        pcd.colors = o3d.utility.Vector3dVector(seg['colors'])
        geoms.append(pcd)
        
        # Add gripper mesh for best candidate
        res = result_map.get(seg['segment_id'])
        if res and res['candidates']:
            best = res['candidates'][0]
            gripper = create_gripper_mesh(
                np.array(best['position']),
                np.array(best['normal']),
                diameter
            )
            geoms.append(gripper)
    
    n_occ = len(occupancy_pts) if occupancy_pts is not None else 0
    print(f"Displaying {len(segments)} segments, {len(background_points):,} background (grey), {n_occ:,} occupancy (black)")
    o3d.visualization.draw_geometries(geoms, window_name="Grasp Detection", width=1200, height=800)


def find_latest_session(data_dir: Path) -> Path | None:
    """Find the most recent grasp_* session directory."""
    sessions = sorted(data_dir.glob("grasp_*"), key=lambda p: p.name, reverse=True)
    for session in sessions:
        if (session / "viz_data.npz").exists():
            return session
    return None


def main():
    parser = argparse.ArgumentParser(description="View cluttered bin grasp detection results")
    parser.add_argument("session_dir", nargs="?", help="Path to session directory (default: latest)")
    parser.add_argument("--no-images", action="store_true", help="Skip showing color/depth images")
    parser.add_argument("--no-occupancy", action="store_true", help="Skip occupancy volume visualization")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Occupancy voxel size in meters (default: 0.01)")
    parser.add_argument("--ray-depth", type=float, default=0.15, help="Max ray depth beyond surface in meters (default: 0.15)")
    args = parser.parse_args()
    
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        # Find latest session
        data_dir = Path(__file__).parent / "data"
        session_dir = find_latest_session(data_dir)
        if session_dir is None:
            print("No sessions with viz_data.npz found in data/")
            sys.exit(1)
    
    print(f"Loading: {session_dir}")
    
    try:
        data = load_viz_data(session_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"  {len(data['segments'])} segments")
    print(f"  {len(data['grasp_results'])} grasp results")
    print(f"  Suction cup: {data['suction_cup_diameter']*1000:.0f}mm diameter")
    
    if data.get('color_image') is not None:
        print(f"  Color image: {data['color_image'].shape[:2]}")
    if data.get('depth_image') is not None:
        print(f"  Depth image: {data['depth_image'].shape[:2]}")
    

    
    # Show 3D visualization with occupancy volume
    visualize(data, show_occupancy=not args.no_occupancy, voxel_size=args.voxel_size, 
              max_ray_depth=args.ray_depth)


if __name__ == "__main__":
    main()

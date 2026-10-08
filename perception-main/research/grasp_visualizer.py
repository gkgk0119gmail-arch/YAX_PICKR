"""
Visualize labeled grasp points on meshes in different poses using meshcat.

Usage:
    python grasp_visualizer.py

Opens meshcat in browser showing all models in various poses with their
grasp points transformed to world frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as np
import open3d as o3d


# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR = Path("/home/jeong/perception/data")
LABELS_FILE = Path("/home/jeong/perception/app/grasp_labels.json")

# Gripper visualization dimensions
GRIPPER_CUP_DIAMETER = 0.025
GRIPPER_CUP_HEIGHT = 0.015
GRIPPER_ARROW_LENGTH = 0.05

# Grasp selection threshold
MIN_UPWARD_COMPONENT = 0.3  # Approach must point downward at least this much


@dataclass
class GraspLabel:
    """A labeled grasp point in object frame."""
    name: str
    position: np.ndarray  # [x, y, z] in object frame
    approach: np.ndarray  # Unit vector pointing INTO object
    quality: float = 1.0


@dataclass
class TransformedGrasp:
    """Grasp transformed to world frame."""
    name: str
    position: np.ndarray  # World frame position
    approach: np.ndarray  # World frame approach direction
    quality: float


# =============================================================================
# LOADING
# =============================================================================
def load_mesh(filepath: Path) -> o3d.geometry.TriangleMesh:
    """Load mesh and sample to point cloud."""
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def load_grasp_labels(labels_file: Path) -> dict[str, list[GraspLabel]]:
    """Load grasp labels from JSON."""
    if not labels_file.exists():
        print(f"No labels file found at {labels_file}")
        return {}
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    labels = {}
    for model_name, grasp_list in data.items():
        labels[model_name] = [
            GraspLabel(
                name=g["name"],
                position=np.array(g["position"]),
                approach=np.array(g["approach"]),
                quality=g.get("quality", 1.0),
            )
            for g in grasp_list
        ]
    
    return labels


def get_all_models(model_dir: Path) -> dict[str, Path]:
    """Find all OBJ files."""
    return {p.stem: p for p in model_dir.glob("*.obj")}


# =============================================================================
# POSE GENERATION
# =============================================================================
def create_pose(translation: list[float], rotation_axis: list[float], rotation_angle: float) -> np.ndarray:
    """Create a 4x4 transformation matrix."""
    T = np.eye(4)
    if rotation_angle != 0:
        T = tf.rotation_matrix(rotation_angle, rotation_axis)
    T[:3, 3] = translation
    return T


def generate_poses(model_name: str) -> list[tuple[str, np.ndarray]]:
    """
    Generate various poses for a model.
    Returns list of (pose_name, 4x4_transform) tuples.
    """
    poses = []
    
    # Identity (standing upright)
    poses.append(("upright", np.eye(4)))
    
    # Rotated around Y (yaw)
    poses.append(("rotated_45", create_pose([0, 0, 0], [0, 1, 0], np.pi/4)))
    poses.append(("rotated_90", create_pose([0, 0, 0], [0, 1, 0], np.pi/2)))
    
    # Lying on side (roll around X or Z)
    poses.append(("lying_x", create_pose([0, 0, 0], [1, 0, 0], np.pi/2)))
    poses.append(("lying_z", create_pose([0, 0, 0], [0, 0, 1], np.pi/2)))
    
    # Upside down
    poses.append(("upside_down", create_pose([0, 0, 0], [1, 0, 0], np.pi)))
    
    # Tilted
    poses.append(("tilted_30", create_pose([0, 0, 0], [1, 0, 0], np.pi/6)))
    
    return poses


# =============================================================================
# GRASP TRANSFORMATION
# =============================================================================
def transform_grasp(grasp: GraspLabel, object_pose: np.ndarray) -> TransformedGrasp:
    """
    Transform a grasp from object frame to world frame.
    
    Args:
        grasp: Grasp in object frame
        object_pose: 4x4 transform from object frame to world frame
        
    Returns:
        TransformedGrasp in world frame
    """
    # Transform position: p_world = T @ p_object
    pos_homogeneous = np.append(grasp.position, 1.0)
    pos_world = (object_pose @ pos_homogeneous)[:3]
    
    # Transform direction: only rotation applies (no translation)
    rotation = object_pose[:3, :3]
    approach_world = rotation @ grasp.approach
    approach_world = approach_world / np.linalg.norm(approach_world)
    
    return TransformedGrasp(
        name=grasp.name,
        position=pos_world,
        approach=approach_world,
        quality=grasp.quality,
    )


def select_best_grasp(
    grasps: list[GraspLabel],
    object_pose: np.ndarray,
    min_upward: float = MIN_UPWARD_COMPONENT,
) -> TransformedGrasp | None:
    """
    Select the best graspable grasp for a given pose.
    
    A grasp is "graspable" if the approach direction points downward enough
    (gripper can approach from above).
    
    Args:
        grasps: List of grasps in object frame
        object_pose: 4x4 transform from object to world frame
        min_upward: Minimum -Z component for approach (default 0.3)
        
    Returns:
        Best TransformedGrasp or None if no valid grasp
    """
    best_grasp: TransformedGrasp | None = None
    best_score = -float('inf')
    
    for grasp in grasps:
        transformed = transform_grasp(grasp, object_pose)
        
        # Score: how much does approach point downward (-Z)?
        # approach points INTO object, so for gripper from above, approach should be -Z
        upward_score = -transformed.approach[2]
        
        # Skip if not facing up enough
        if upward_score < min_upward:
            continue
        
        # Combined score
        score = upward_score * transformed.quality
        
        if score > best_score:
            best_score = score
            best_grasp = transformed
    
    return best_grasp


# =============================================================================
# VISUALIZATION
# =============================================================================
def mesh_to_points(mesh: o3d.geometry.TriangleMesh, num_points: int = 10000) -> np.ndarray:
    """Sample points from mesh surface."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)


def compute_gripper_pose(position: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """
    Compute 4x4 gripper pose matrix with Z-axis aligned to approach direction.
    """
    # Z-axis = approach direction (into object)
    z_axis = approach / np.linalg.norm(approach)
    
    # Choose X-axis perpendicular to Z
    # Use world Y as reference unless Z is parallel to Y
    world_y = np.array([0, 1, 0])
    if abs(np.dot(z_axis, world_y)) > 0.9:
        world_y = np.array([1, 0, 0])
    
    x_axis = np.cross(world_y, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y-axis completes the frame
    y_axis = np.cross(z_axis, x_axis)
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = position
    
    return T


def visualize_grasp(
    vis: meshcat.Visualizer,
    path: str,
    grasp: TransformedGrasp,
    color: int = 0x00ff00,
) -> None:
    """Visualize a single grasp as a cylinder suction cup."""
    # Compute gripper pose (Z pointing into object)
    gripper_pose = compute_gripper_pose(grasp.position, grasp.approach)
    
    # Suction cup cylinder - offset slightly back from contact point
    cup_offset = -GRIPPER_CUP_HEIGHT / 2
    cup_transform = gripper_pose.copy()
    cup_transform[:3, 3] += gripper_pose[:3, 2] * cup_offset  # Move back along approach
    
    vis[path]["cup"].set_object(
        geom.Cylinder(GRIPPER_CUP_HEIGHT, GRIPPER_CUP_DIAMETER / 2),
        geom.MeshLambertMaterial(color=color, opacity=0.85)
    )
    # Cylinder in meshcat is Y-up, rotate to align with Z (approach direction)
    cup_rotation = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    vis[path]["cup"].set_transform(cup_transform @ cup_rotation)
    
    # Approach arrow (line from outside pointing toward contact)
    arrow_start = grasp.position - grasp.approach * GRIPPER_ARROW_LENGTH
    arrow_end = grasp.position
    
    pts = np.array([arrow_start, arrow_end], dtype=np.float32).T
    vis[path]["arrow"].set_object(
        geom.Line(geom.PointsGeometry(pts), geom.LineBasicMaterial(color=0xffffff, linewidth=2))
    )


def visualize_model_with_grasps(
    vis: meshcat.Visualizer,
    model_name: str,
    mesh: o3d.geometry.TriangleMesh,
    grasps: list[GraspLabel],
    object_pose: np.ndarray,
    pose_name: str,
    base_position: np.ndarray,
    model_color: int = 0x88aacc,
) -> str | None:
    """
    Visualize a model in a specific pose with only the best graspable grasp.
    
    Returns:
        Name of selected grasp, or None if no valid grasp
    """
    path = f"models/{model_name}/{pose_name}"
    
    # Combined transform: base_position + object_pose
    full_pose = object_pose.copy()
    full_pose[:3, 3] += base_position
    
    # Get mesh points and transform them
    points = mesh_to_points(mesh, num_points=5000)
    points_h = np.hstack([points, np.ones((len(points), 1))])
    points_world = (full_pose @ points_h.T).T[:, :3]
    
    # Visualize mesh as point cloud
    vis[path]["mesh"].set_object(
        geom.Points(
            geom.PointsGeometry(points_world.T.astype(np.float32)),
            geom.PointsMaterial(size=0.003, color=model_color)
        )
    )
    
    # Select and visualize only the best graspable grasp
    best_grasp = select_best_grasp(grasps, full_pose)
    
    if best_grasp:
        visualize_grasp(vis, f"{path}/grasp", best_grasp, color=0x00ff00)
    
    # Add pose label (coordinate frame)
    frame_size = 0.03
    for axis, color, d in [("X", 0xff0000, [1,0,0]), ("Y", 0x00ff00, [0,1,0]), ("Z", 0x0000ff, [0,0,1])]:
        start = base_position
        direction = full_pose[:3, :3] @ np.array(d)
        end = start + direction * frame_size
        pts = np.array([start, end], dtype=np.float32).T
        vis[path]["frame"][axis].set_object(
            geom.Line(geom.PointsGeometry(pts), geom.LineBasicMaterial(color=color, linewidth=2))
        )
    
    return best_grasp.name if best_grasp else None


def visualize_all(
    models: dict[str, o3d.geometry.TriangleMesh],
    labels: dict[str, list[GraspLabel]],
) -> meshcat.Visualizer:
    """Main visualization function."""
    vis = meshcat.Visualizer().open()
    
    # World coordinate frame
    L = 0.15
    for axis, color, vec in [("X", 0xff0000, [L,0,0]), ("Y", 0x00ff00, [0,L,0]), ("Z", 0x0000ff, [0,0,L])]:
        pts = np.array([[0,0,0], vec], dtype=np.float32).T
        vis["world_frame"][axis].set_object(
            geom.Line(geom.PointsGeometry(pts), geom.LineBasicMaterial(color=color, linewidth=4))
        )
    
    # Ground plane reference
    ground_size = 1.0
    ground_pts = np.array([
        [-ground_size/2, 0, -ground_size/2],
        [ground_size/2, 0, -ground_size/2],
        [ground_size/2, 0, ground_size/2],
        [-ground_size/2, 0, ground_size/2],
        [-ground_size/2, 0, -ground_size/2],
    ], dtype=np.float32).T
    vis["ground"].set_object(
        geom.Line(geom.PointsGeometry(ground_pts), geom.LineBasicMaterial(color=0x444444, linewidth=1))
    )
    
    # Model colors
    model_colors = [0x88aacc, 0xcc88aa, 0xaacc88, 0xccaa88]
    
    # Layout: models in rows, poses in columns
    model_spacing_z = 0.4  # Space between different models
    pose_spacing_x = 0.25   # Space between different poses
    
    model_names = list(models.keys())
    
    for model_idx, model_name in enumerate(model_names):
        mesh = models[model_name]
        grasps = labels.get(model_name, [])
        
        if not grasps:
            print(f"[{model_name}] No grasp labels found, skipping visualization")
            continue
        
        poses = generate_poses(model_name)
        model_color = model_colors[model_idx % len(model_colors)]
        
        print(f"\n[{model_name}] {len(grasps)} grasps, {len(poses)} poses")
        
        for pose_idx, (pose_name, pose_transform) in enumerate(poses):
            # Position in grid
            x_pos = (pose_idx - len(poses) / 2) * pose_spacing_x
            z_pos = model_idx * model_spacing_z
            base_position = np.array([x_pos, 0, z_pos])
            
            selected_grasp = visualize_model_with_grasps(
                vis=vis,
                model_name=model_name,
                mesh=mesh,
                grasps=grasps,
                object_pose=pose_transform,
                pose_name=pose_name,
                base_position=base_position,
                model_color=model_color,
            )
            
            if selected_grasp:
                print(f"  [{pose_name}] grasp #{selected_grasp}")
            else:
                print(f"  [{pose_name}] NO VALID GRASP (not facing up)")
    
    print("\n" + "=" * 60)
    print(f"[Meshcat] URL: {vis.url()}")
    print("=" * 60)
    print("\nLegend:")
    print("  • Point clouds = Model meshes in different poses")
    print("  • Green cylinders = Suction cup at grasp point")
    print("  • White lines = Approach direction (gripper comes from above)")
    print("  • RGB axes = Object coordinate frame")
    print("\nOnly showing grasps where gripper can approach from above.")
    print("Poses (left to right):")
    print("  upright → rotated_45 → rotated_90 → lying_x → lying_z → upside_down → tilted_30")
    
    return vis


def main():
    print("=" * 60)
    print("GRASP VISUALIZER")
    print("=" * 60)
    
    # Load models
    model_paths = get_all_models(MODEL_DIR)
    print(f"\nFound {len(model_paths)} models in {MODEL_DIR}:")
    
    models = {}
    for name, path in model_paths.items():
        print(f"  Loading {name}...")
        models[name] = load_mesh(path)
    
    # Load labels
    labels = load_grasp_labels(LABELS_FILE)
    print(f"\nLoaded grasp labels from {LABELS_FILE}:")
    for name, grasps in labels.items():
        print(f"  {name}: {len(grasps)} grasps")
    
    # Visualize
    vis = visualize_all(models, labels)
    
    # Keep running
    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()

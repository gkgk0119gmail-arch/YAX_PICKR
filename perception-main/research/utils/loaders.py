"""
Model and data loading utilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


# =============================================================================
# GRASP CONFIGURATION
# =============================================================================
@dataclass
class GraspCandidate:
    """A candidate grasp pose in object frame."""
    name: str                    # Human-readable name (e.g., "top", "side_x")
    position: np.ndarray         # Contact point in object frame
    approach: np.ndarray         # Approach direction (unit vector, points INTO object)
    quality: float = 1.0         # Base quality score (higher = preferred)
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.approach = np.asarray(self.approach, dtype=np.float64)
        self.approach = self.approach / np.linalg.norm(self.approach)


@dataclass 
class GraspConfig:
    """Grasp configuration for a specific object model."""
    model_name: str
    candidates: list[GraspCandidate] = field(default_factory=list)


# Default grasp labels file
DEFAULT_GRASP_LABELS_FILE = Path("/home/jeong/perception/app/grasp_labels.json")


def load_grasp_labels(labels_file: Path = DEFAULT_GRASP_LABELS_FILE) -> dict[str, GraspConfig]:
    """
    Load grasp configurations from JSON labels file.
    
    Args:
        labels_file: Path to grasp_labels.json
        
    Returns:
        Dictionary mapping model_name -> GraspConfig
    """
    if not labels_file.exists():
        print(f"[Grasp] Warning: No labels file at {labels_file}")
        return {}
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    configs = {}
    for model_name, grasp_list in data.items():
        candidates = [
            GraspCandidate(
                name=g["name"],
                position=g["position"],
                approach=g["approach"],
                quality=g.get("quality", 1.0),
            )
            for g in grasp_list
        ]
        configs[model_name] = GraspConfig(
            model_name=model_name,
            candidates=candidates,
        )
        print(f"[Grasp] Loaded {len(candidates)} grasps for '{model_name}'")
    
    return configs


# =============================================================================
# MODEL LOADING
# =============================================================================
DEFAULT_MODEL_NUM_SAMPLES = 100000


def load_model(mesh_path: str, num_samples: int = DEFAULT_MODEL_NUM_SAMPLES) -> o3d.geometry.PointCloud:
    """
    Load model from mesh file (STL, OBJ, etc.) and sample points.
    
    Args:
        mesh_path: Path to mesh file
        num_samples: Number of points to sample from mesh surface
        
    Returns:
        Point cloud sampled from mesh
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"Could not load mesh from {mesh_path}")
    
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    
    print(f"   Mesh: {mesh_path}")
    print(f"   Sampled {len(pcd.points)} points")
    
    return pcd


def load_obj_as_pointcloud(filepath: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Load OBJ/mesh file as downsampled point cloud with normals and colors.
    
    Args:
        filepath: Path to mesh file
        voxel_size: Voxel size for downsampling
        
    Returns:
        Downsampled point cloud with normals and colors (if texture available)
    """
    mesh = None
    vertex_colors = None
    
    # Try loading with Open3D (may fail on non-standard MTL files)
    try:
        mesh = o3d.io.read_triangle_mesh(str(filepath), enable_post_processing=False)
    except Exception:
        pass
    
    # If Open3D failed or got no vertices, try manual OBJ parsing
    if mesh is None or not mesh.has_vertices():
        mesh, vertex_colors = _load_obj_manual(filepath)
    
    if not mesh.has_vertices():
        raise ValueError(f"OBJ file has no vertices: {filepath}")
    
    # Apply vertex colors to mesh if we got them from manual parsing
    if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Sample points from mesh surface
    if mesh.has_triangles() and len(mesh.triangles) > 0:
        n_points = max(10000, len(mesh.vertices) * 5)
        # sample_points_uniformly preserves vertex colors if mesh has them
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
    
    # Downsample (voxel_down_sample averages colors)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 4, max_nn=30
    ))
    
    return pcd


def _parse_mtl_for_texture(mtl_path: Path) -> Optional[Path]:
    """Parse MTL file to find diffuse texture path (map_Kd)."""
    if not mtl_path.exists():
        return None
    
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('map_Kd '):
                tex_rel_path = line[7:].strip()
                tex_path = mtl_path.parent / tex_rel_path
                if tex_path.exists():
                    return tex_path
    return None


def _load_texture_image(tex_path: Path) -> Optional[np.ndarray]:
    """Load texture image as RGB numpy array."""
    try:
        import cv2
        img = cv2.imread(str(tex_path))
        if img is not None:
            # Convert BGR to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def _load_obj_manual(filepath: Path) -> tuple[o3d.geometry.TriangleMesh, Optional[np.ndarray]]:
    """
    Manually parse OBJ file to extract vertices, faces, and vertex colors from texture.
    This bypasses ASSIMP and handles non-standard MTL files.
    
    Args:
        filepath: Path to OBJ file
        
    Returns:
        Tuple of (Open3D TriangleMesh, vertex_colors or None)
    """
    vertices = []
    tex_coords = []
    faces = []
    face_tex_indices = []
    mtl_file = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'mtllib' and len(parts) >= 2:
                # Material library reference
                mtl_file = filepath.parent / parts[1]
            
            elif parts[0] == 'v' and len(parts) >= 4:
                # Vertex: v x y z
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                except ValueError:
                    continue
            
            elif parts[0] == 'vt' and len(parts) >= 3:
                # Texture coordinate: vt u v
                try:
                    u, v = float(parts[1]), float(parts[2])
                    tex_coords.append([u, v])
                except ValueError:
                    continue
            
            elif parts[0] == 'f' and len(parts) >= 4:
                # Face: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                try:
                    face_v = []
                    face_vt = []
                    for p in parts[1:4]:  # Only take first 3 (triangles)
                        indices = p.split('/')
                        v_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                        face_v.append(v_idx)
                        if len(indices) > 1 and indices[1]:
                            vt_idx = int(indices[1]) - 1
                            face_vt.append(vt_idx)
                    faces.append(face_v)
                    if len(face_vt) == 3:
                        face_tex_indices.append(face_vt)
                except (ValueError, IndexError):
                    continue
    
    mesh = o3d.geometry.TriangleMesh()
    vertex_colors = None
    
    if vertices:
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
    
    if faces:
        mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
        mesh.compute_vertex_normals()
    
    # Try to load texture and compute vertex colors
    if mtl_file and tex_coords and face_tex_indices:
        tex_path = _parse_mtl_for_texture(mtl_file)
        if tex_path:
            tex_img = _load_texture_image(tex_path)
            if tex_img is not None:
                vertex_colors = _compute_vertex_colors_from_texture(
                    vertices, tex_coords, faces, face_tex_indices, tex_img
                )
    
    return mesh, vertex_colors


def _compute_vertex_colors_from_texture(
    vertices: list,
    tex_coords: list,
    faces: list,
    face_tex_indices: list,
    tex_img: np.ndarray,
) -> np.ndarray:
    """
    Compute per-vertex colors by sampling texture at UV coordinates.
    
    Since vertices can have different UVs in different faces, we average
    the colors from all faces that use each vertex.
    """
    n_verts = len(vertices)
    h, w = tex_img.shape[:2]
    
    # Accumulate colors for each vertex
    color_sum = np.zeros((n_verts, 3), dtype=np.float64)
    color_count = np.zeros(n_verts, dtype=np.int32)
    
    tex_coords_arr = np.array(tex_coords, dtype=np.float64)
    
    for face_v, face_vt in zip(faces, face_tex_indices):
        for v_idx, vt_idx in zip(face_v, face_vt):
            if vt_idx < len(tex_coords_arr):
                u, v = tex_coords_arr[vt_idx]
                # UV to pixel coords (v is flipped in most textures)
                px = int(np.clip(u * (w - 1), 0, w - 1))
                py = int(np.clip((1 - v) * (h - 1), 0, h - 1))
                color = tex_img[py, px]
                color_sum[v_idx] += color
                color_count[v_idx] += 1
    
    # Average and normalize to [0, 1]
    valid = color_count > 0
    vertex_colors = np.zeros((n_verts, 3), dtype=np.float64)
    vertex_colors[valid] = color_sum[valid] / color_count[valid, np.newaxis]
    vertex_colors /= 255.0
    
    return vertex_colors


def load_all_models(model_dir: Path, voxel_size: float) -> dict[str, o3d.geometry.PointCloud]:
    """
    Load all 3D models from directory.
    
    Searches for models in two ways:
    1. OBJ files directly in model_dir (model_name = filename stem)
    2. Subdirectories containing 3DModel.obj (model_name = subdirectory name)
       Example: /data/can/3DModel.obj -> model_name = "can"
    
    Args:
        model_dir: Directory containing .obj files or subdirectories with models
        voxel_size: Voxel size for downsampling
        
    Returns:
        Dictionary mapping model_name -> point cloud
    """
    models = {}
    
    # Method 1: OBJ files directly in model_dir
    obj_files = list(model_dir.glob("*.obj"))
    
    # Method 2: Subdirectories with 3DModel.obj
    subdir_models = []
    for subdir in model_dir.iterdir():
        if subdir.is_dir():
            model_file = subdir / "3DModel.obj"
            if model_file.exists():
                subdir_models.append((subdir.name, model_file))
    
    total_models = len(obj_files) + len(subdir_models)
    if total_models == 0:
        print(f"[Models] No models found in {model_dir}")
        print(f"[Models] Searched for: *.obj files and subdirs/3DModel.obj")
        return models
    
    print(f"[Models] Found {total_models} models in {model_dir}")
    if obj_files:
        print(f"  - {len(obj_files)} OBJ files in root")
    if subdir_models:
        print(f"  - {len(subdir_models)} subdirectory models: {[n for n, _ in subdir_models]}")
    
    # Load OBJ files from root
    for obj_path in obj_files:
        try:
            pcd = load_obj_as_pointcloud(obj_path, voxel_size)
            model_name = obj_path.stem
            models[model_name] = pcd
            
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            print(f"  [{model_name}] {len(pcd.points):,} pts, "
                  f"size: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm")
        except Exception as e:
            print(f"  [Warning] Failed to load {obj_path.name}: {e}")
    
    # Load models from subdirectories
    for model_name, model_path in subdir_models:
        try:
            pcd = load_obj_as_pointcloud(model_path, voxel_size)
            models[model_name] = pcd
            
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            print(f"  [{model_name}] {len(pcd.points):,} pts, "
                  f"size: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm "
                  f"(from {model_path.parent.name}/)")
        except Exception as e:
            print(f"  [Warning] Failed to load {model_name}/3DModel.obj: {e}")
    
    return models


def load_scene(pcd_path: str) -> o3d.geometry.PointCloud:
    """
    Load scene point cloud from PCD/PLY file.
    
    Args:
        pcd_path: Path to point cloud file
        
    Returns:
        Loaded point cloud
    """
    print(f"\n{'='*60}")
    print("LOADING SCENE")
    print(f"{'='*60}")
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"   Scene: {pcd_path}")
    print(f"   Loaded {len(pcd.points)} points")
    
    return pcd

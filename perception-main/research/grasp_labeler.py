"""
Interactive grasp point labeler for suction gripper.

Usage:
    python grasp_labeler.py              # Label all models
    python grasp_labeler.py --model can  # Label specific model

Instructions:
    1. Shift+Click on mesh vertices to select grasp points
    2. Press 'Q' or close window when done selecting
    3. Enter grasp names and quality scores in terminal
    4. Repeat for each model

Output saved to: grasp_labels.json

"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import open3d as o3d


# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR = Path("/home/jeong/perception/data")
OUTPUT_FILE = Path("/home/jeong/perception/app/grasp_labels.json")


@dataclass
class GraspLabel:
    """A labeled grasp point."""
    name: str
    position: list[float]  # [x, y, z] in object frame
    approach: list[float]  # Unit vector pointing INTO object
    quality: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "position": self.position,
            "approach": self.approach,
            "quality": self.quality,
        }


def load_mesh(filepath: Path) -> o3d.geometry.TriangleMesh:
    """Load a mesh and compute normals."""
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print(f"  Loaded: {filepath.name}")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.triangles):,}")
    return mesh


def get_all_models(model_dir: Path) -> dict[str, Path]:
    """Find all OBJ files in model directory."""
    models = {}
    for obj_file in model_dir.glob("*.obj"):
        model_name = obj_file.stem
        models[model_name] = obj_file
    return models


def pick_points_on_mesh(mesh: o3d.geometry.TriangleMesh, model_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Interactive point picking on mesh.
    
    Returns:
        List of (position, normal) tuples for picked points
    """
    print(f"\n{'='*60}")
    print(f"LABELING: {model_name}")
    print(f"{'='*60}")
    print("Instructions:")
    print("  • Shift + Left Click  → Select a grasp point")
    print("  • Shift + Right Click → Deselect last point")
    print("  • Q or close window   → Finish selection")
    print("  • Mouse drag          → Rotate view")
    print("  • Scroll              → Zoom")
    print(f"{'='*60}\n")
    
    # Convert mesh to point cloud for picking
    # (Open3D picking works better with point clouds)
    pcd = mesh.sample_points_uniformly(number_of_points=50000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    
    # Visualizer with picking
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=f"Grasp Labeler - {model_name}", width=1280, height=960)
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.15])
    opt.point_size = 2.0
    
    # Run picker
    vis.run()
    
    # Get picked points
    picked_indices = vis.get_picked_points()
    vis.destroy_window()
    
    if not picked_indices:
        print("No points selected.")
        return []
    
    # Extract positions and normals
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    results = []
    for picked_point in picked_indices:
        # PickedPoint has .index and .coord attributes
        idx = picked_point.index
        pos = points[idx]
        normal = normals[idx]
        # Approach is opposite of surface normal (into the object)
        approach = -normal / np.linalg.norm(normal)
        results.append((pos.copy(), approach.copy()))
        print(f"  Point {len(results)}: pos={pos.round(4)}, approach={approach.round(4)}")
    
    return results


def label_grasps_for_model(mesh: o3d.geometry.TriangleMesh, model_name: str) -> list[GraspLabel]:
    """Interactive grasp labeling for a single model."""
    picked_points = pick_points_on_mesh(mesh, model_name)
    
    if not picked_points:
        return []
    
    labels = []
    print(f"\n✓ Auto-labeling {len(picked_points)} selected points:")
    
    for i, (pos, approach) in enumerate(picked_points):
        # Auto-label with index and quality=1.0
        name = str(i + 1)
        quality = 1.0
        
        label = GraspLabel(
            name=name,
            position=pos.tolist(),
            approach=approach.tolist(),
            quality=quality,
        )
        labels.append(label)
        print(f"  {name}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    
    return labels


def save_labels(labels: dict[str, list[GraspLabel]], output_file: Path):
    """Save all labels to JSON file."""
    data = {}
    for model_name, grasp_labels in labels.items():
        data[model_name] = [g.to_dict() for g in grasp_labels]
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Saved labels to: {output_file}")


def load_existing_labels(output_file: Path) -> dict[str, list[GraspLabel]]:
    """Load existing labels if they exist."""
    if not output_file.exists():
        return {}
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    labels = {}
    for model_name, grasp_list in data.items():
        labels[model_name] = [
            GraspLabel(
                name=g["name"],
                position=g["position"],
                approach=g["approach"],
                quality=g.get("quality", 1.0),
            )
            for g in grasp_list
        ]
    
    return labels


def visualize_labeled_grasps(model_name: str, mesh: o3d.geometry.TriangleMesh, grasps: list[GraspLabel]):
    """Visualize a model with its labeled grasp points."""
    print(f"\nVisualizing {len(grasps)} grasps for {model_name}...")
    
    geometries = []
    
    # Add mesh (as wireframe for visibility)
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.paint_uniform_color([0.7, 0.7, 0.8])
    geometries.append(mesh_copy)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord_frame)
    
    # Add grasp indicators
    colors = [
        [1.0, 0.3, 0.3],  # red
        [0.3, 1.0, 0.3],  # green
        [0.3, 0.3, 1.0],  # blue
        [1.0, 1.0, 0.3],  # yellow
        [1.0, 0.3, 1.0],  # magenta
        [0.3, 1.0, 1.0],  # cyan
    ]
    
    for i, grasp in enumerate(grasps):
        pos = np.array(grasp.position)
        approach = np.array(grasp.approach)
        color = colors[i % len(colors)]
        
        # Grasp point sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
        
        # Approach arrow (cylinder + cone)
        arrow_length = 0.04
        arrow_start = pos
        arrow_end = pos - approach * arrow_length  # Arrow points toward object
        
        # Create arrow as line set for simplicity
        line_points = [arrow_start.tolist(), arrow_end.tolist()]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        line_set.colors = o3d.utility.Vector3dVector([color])
        geometries.append(line_set)
        
        print(f"  {grasp.name}: pos={np.round(pos, 3)}, quality={grasp.quality}")
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Grasp Labels - {model_name}", width=1280, height=960)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.15])
    
    vis.run()
    vis.destroy_window()


def generate_code_snippet(labels: dict[str, list[GraspLabel]]):
    """Generate Python code for grasp configs."""
    print("\n" + "=" * 60)
    print("GENERATED CODE (copy to 5.py)")
    print("=" * 60)
    
    for model_name, grasps in labels.items():
        print(f'\n    # === {model_name.upper()} ===')
        print(f'    configs["{model_name}"] = GraspConfig(')
        print(f'        model_name="{model_name}",')
        print(f'        candidates=[')
        
        for g in grasps:
            pos_str = f"[{g.position[0]:.6f}, {g.position[1]:.6f}, {g.position[2]:.6f}]"
            app_str = f"[{g.approach[0]:.4f}, {g.approach[1]:.4f}, {g.approach[2]:.4f}]"
            print(f'            GraspCandidate(')
            print(f'                name="{g.name}",')
            print(f'                position={pos_str},')
            print(f'                approach={app_str},')
            print(f'                quality={g.quality},')
            print(f'            ),')
        
        print(f'        ]')
        print(f'    )')


def main():
    parser = argparse.ArgumentParser(description="Label suction grasp points on 3D models")
    parser.add_argument("--model", type=str, help="Label specific model (e.g., 'can')")
    parser.add_argument("--view", action="store_true", help="View existing labels instead of labeling")
    parser.add_argument("--code", action="store_true", help="Generate code snippet from labels")
    args = parser.parse_args()
    
    # Find all models
    models = get_all_models(MODEL_DIR)
    print(f"Found {len(models)} model(s) in {MODEL_DIR}:")
    for name, path in models.items():
        print(f"  • {name}: {path.name}")
    
    if not models:
        print("No OBJ files found!")
        return
    
    # Load existing labels
    all_labels = load_existing_labels(OUTPUT_FILE)
    
    # Filter models if specified
    if args.model:
        if args.model not in models:
            print(f"Model '{args.model}' not found! Available: {list(models.keys())}")
            return
        models = {args.model: models[args.model]}
    
    # Generate code snippet
    if args.code:
        if all_labels:
            generate_code_snippet(all_labels)
        else:
            print("No labels found. Label some grasps first!")
        return
    
    # View mode
    if args.view:
        for model_name, model_path in models.items():
            if model_name not in all_labels:
                print(f"No labels for {model_name}")
                continue
            mesh = load_mesh(model_path)
            visualize_labeled_grasps(model_name, mesh, all_labels[model_name])
        return
    
    # Labeling mode
    for model_name, model_path in models.items():
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*60}")
        
        # Check for existing labels
        if model_name in all_labels:
            existing = all_labels[model_name]
            print(f"Found {len(existing)} existing labels:")
            for g in existing:
                print(f"  • {g.name}: quality={g.quality}")
            
            choice = input("\n(A)dd more, (R)eplace all, or (S)kip? [S]: ").strip().lower()
            if choice == 's' or not choice:
                print("Skipping...")
                continue
            elif choice == 'r':
                all_labels[model_name] = []
        
        # Load and label
        mesh = load_mesh(model_path)
        new_labels = label_grasps_for_model(mesh, model_name)
        
        if new_labels:
            if model_name in all_labels:
                all_labels[model_name].extend(new_labels)
            else:
                all_labels[model_name] = new_labels
            
            # Save after each model
            save_labels(all_labels, OUTPUT_FILE)
            
            # Show visualization
            view_choice = input("\nVisualize labeled grasps? [Y/n]: ").strip().lower()
            if view_choice != 'n':
                visualize_labeled_grasps(model_name, mesh, all_labels[model_name])
    
    # Final summary
    print("\n" + "=" * 60)
    print("LABELING COMPLETE")
    print("=" * 60)
    
    for model_name, grasps in all_labels.items():
        print(f"\n{model_name}: {len(grasps)} grasps")
        for g in grasps:
            print(f"  • {g.name} (quality={g.quality})")
    
    # Offer to generate code
    if all_labels:
        gen_code = input("\nGenerate Python code snippet? [Y/n]: ").strip().lower()
        if gen_code != 'n':
            generate_code_snippet(all_labels)


if __name__ == "__main__":
    main()

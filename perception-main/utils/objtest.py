#!/usr/bin/env python3
"""Simple script to read an OBJ file and get height measurements."""

import sys
from pathlib import Path


def parse_obj_vertices(filepath: str) -> list[tuple[float, float, float]]:
    """Parse vertex positions from an OBJ file."""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
    return vertices


def get_measurements(vertices: list[tuple[float, float, float]]) -> dict:
    """Calculate bounding box measurements from vertices."""
    if not vertices:
        return {}
    
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    width = max(xs) - min(xs)   # X axis
    height = max(ys) - min(ys)  # Y axis (typically vertical)
    depth = max(zs) - min(zs)   # Z axis
    
    return {
        'width': width,
        'height': height,
        'depth': depth,
        'bounds': {
            'x': (min(xs), max(xs)),
            'y': (min(ys), max(ys)),
            'z': (min(zs), max(zs)),
        }
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python objtest.py <path_to_obj_file>")
        sys.exit(1)
    
    obj_path = sys.argv[1]
    
    if not Path(obj_path).exists():
        print(f"Error: File not found: {obj_path}")
        sys.exit(1)
    
    print(f"Reading OBJ file: {obj_path}")
    
    vertices = parse_obj_vertices(obj_path)
    print(f"Found {len(vertices)} vertices")
    
    if not vertices:
        print("No vertices found in file!")
        sys.exit(1)
    
    measurements = get_measurements(vertices)
    
    print("\n--- Measurements ---")
    print(f"Width  (X): {measurements['width']:.4f}")
    print(f"Height (Y): {measurements['height']:.4f}")
    print(f"Depth  (Z): {measurements['depth']:.4f}")
    
    print("\n--- Bounding Box ---")
    bounds = measurements['bounds']
    print(f"X: {bounds['x'][0]:.4f} to {bounds['x'][1]:.4f}")
    print(f"Y: {bounds['y'][0]:.4f} to {bounds['y'][1]:.4f}")
    print(f"Z: {bounds['z'][0]:.4f} to {bounds['z'][1]:.4f}")
    
    # Identify the longest dimension (top to bottom)
    dims = [
        ('Width (X)', measurements['width']),
        ('Height (Y)', measurements['height']),
        ('Depth (Z)', measurements['depth']),
    ]
    longest = max(dims, key=lambda x: x[1])
    print(f"\n--- Longest dimension ---")
    print(f"{longest[0]}: {longest[1]:.4f}")


if __name__ == "__main__":
    main()

import numpy as np
import trimesh
from pathlib import Path

def main():
    project_root = Path('engine')
    processed_dir = project_root / 'processed'
    depth_path = processed_dir / 'depth.npy'
    glb_path = processed_dir / 'terrain.glb'

    print("==================================================")
    print("1. DEPTH.NPY STATS")
    print("==================================================")
    
    if depth_path.exists():
        depth = np.load(depth_path)
        print(f"Shape: {depth.shape}")
        print(f"Dtype: {depth.dtype}")
        print(f"Min: {depth.min()}")
        print(f"Max: {depth.max()}")
        print(f"Mean: {depth.mean()}")
        print(f"Std: {depth.std()}")
        print(f"Percentiles:")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  {p}%: {np.percentile(depth, p)}")
    else:
        print("depth.npy not found")

    print("\n==================================================")
    print("2. TERRAIN.GLB MESH STATS")
    print("==================================================")
    
    if glb_path.exists():
        scene = trimesh.load(glb_path, process=False)
        
        # GLB loads as a Scene if it has multiple nodes, or Trimesh if single
        if isinstance(scene, trimesh.Scene):
            print(f"Loaded as Scene.")
            for node_name in scene.graph.nodes:
                transform, _ = scene.graph.get(node_name)
                print(f"Transform for {node_name}:\n{transform}")
            
            # Combine to a single mesh for bounding box check
            mesh = scene.dump(concatenate=True)
        else:
            print(f"Loaded as Trimesh.")
            mesh = scene

        print(f"Vertex count: {len(mesh.vertices)}")
        print(f"Face count: {len(mesh.faces)}")
        bounds_min = mesh.bounds[0]
        bounds_max = mesh.bounds[1]
        extents = mesh.extents
        print(f"Bounds Min: {bounds_min}")
        print(f"Bounds Max: {bounds_max}")
        print(f"Extents (X/Y/Z): {extents}")
        
        print("\nSample Vertices:")
        print(f"Vertex 0: {mesh.vertices[0]}")
        print(f"Vertex 1000: {mesh.vertices[1000]}")
        print(f"Vertex -1: {mesh.vertices[-1]}")
    else:
        print("terrain.glb not found")

if __name__ == '__main__':
    main()

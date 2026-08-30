"""
DepthWizard V1.5 - Terrain Generator
Outputs a continuous height-field mesh.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import trimesh

class TerrainGenerator:
    def __init__(self, project_root=None, height_path=None, output_path=None,
                 height_scale=10.0, mesh_stride=2, world_depth=100.0):
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
        self.processed_dir = self.project_root / "engine" / "processed"
        self.height_path = Path(height_path).resolve() if height_path else self.processed_dir / "refined_height.npy"
        self.output_path = Path(output_path).resolve() if output_path else self.processed_dir / "terrain.glb"
        self.height_scale = float(height_scale)
        self.mesh_stride = max(1, int(mesh_stride))
        self.world_depth = float(world_depth)

    def generate(self) -> Path:
        print("[TerrainGenerator] Loading height field...")
        if not self.height_path.exists():
            raise FileNotFoundError(f"Height map not found: {self.height_path}")
            
        height_map = np.load(self.height_path).astype(np.float32)
        depth_height, depth_width = height_map.shape
        world_width = self.world_depth * (depth_width / depth_height)
        
        print("[TerrainGenerator] Building continuous 3D scene...")
        
        rows = np.arange(0, depth_height, self.mesh_stride)
        cols = np.arange(0, depth_width, self.mesh_stride)
        if rows[-1] != depth_height - 1: rows = np.append(rows, depth_height - 1)
        if cols[-1] != depth_width - 1: cols = np.append(cols, depth_width - 1)

        sampled_height = height_map[np.ix_(rows, cols)]
        gh, gw = sampled_height.shape
        
        xx, zz = np.meshgrid(cols.astype(np.float32) / max(depth_width-1, 1), rows.astype(np.float32) / max(depth_height-1, 1))
        
        # X and Z
        base_verts = np.column_stack((
            (xx.reshape(-1) - 0.5) * world_width, 
            sampled_height.reshape(-1) * self.height_scale, 
            (zz.reshape(-1) - 0.5) * self.world_depth
        ))
        
        # UVs
        base_uvs = np.column_stack((xx.reshape(-1), 1.0 - zz.reshape(-1)))

        rr, cc = np.meshgrid(np.arange(gh-1), np.arange(gw-1), indexing="ij")
        tl = rr*gw + cc
        tr = tl + 1
        bl = (rr+1)*gw + cc
        br = bl + 1

        all_base_faces = np.vstack((
            np.column_stack((tl.reshape(-1), bl.reshape(-1), tr.reshape(-1))),
            np.column_stack((tr.reshape(-1), bl.reshape(-1), br.reshape(-1)))
        )).astype(np.int64)
        
        base_mesh = trimesh.Trimesh(vertices=base_verts, faces=all_base_faces, process=False)
        base_mesh.visual = trimesh.visual.TextureVisuals(uv=base_uvs)
        
        scene = trimesh.Scene({'base_terrain': base_mesh})
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        print("[TerrainGenerator] Exporting continuous GLB...")
        scene.export(self.output_path, file_type="glb")
        
        print(f"[TerrainGenerator] Vertices: {len(base_mesh.vertices):,}")
        print(f"[TerrainGenerator] Faces: {len(base_mesh.faces):,}")
        print(f"[TerrainGenerator] Saved: {self.output_path}")
        
        return self.output_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--height-scale", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--world-depth", type=float, default=100.0)
    args = parser.parse_args()
    TerrainGenerator(
        height_path=args.height, 
        output_path=args.output, 
        height_scale=args.height_scale, 
        mesh_stride=args.stride, 
        world_depth=args.world_depth
    ).generate()

if __name__ == "__main__":
    main()
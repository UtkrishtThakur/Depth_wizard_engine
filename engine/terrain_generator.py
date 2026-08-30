"""
DepthWizard V1.5 - Terrain Generator

Input:
    engine/processed/refined_height.npy
    engine/processed/structure_regions.json

Output:
    engine/processed/terrain.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import cv2

class TerrainGenerator:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        height_path: Optional[Path] = None,
        regions_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        height_scale: float = 10.0,
        mesh_stride: int = 2,
        world_depth: float = 100.0,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.processed_dir = self.project_root / "engine" / "processed"

        self.height_path = (
            Path(height_path).resolve()
            if height_path is not None
            else self.processed_dir / "refined_height.npy"
        )

        self.regions_path = (
            Path(regions_path).resolve()
            if regions_path is not None
            else self.processed_dir / "structure_regions.json"
        )

        self.output_path = (
            Path(output_path).resolve()
            if output_path is not None
            else self.processed_dir / "terrain.glb"
        )

        self.height_scale = float(height_scale)
        self.mesh_stride = max(1, int(mesh_stride))
        self.world_depth = float(world_depth)

    def build_scene(self, height_map: np.ndarray, regions_data: list[dict]) -> trimesh.Scene:
        depth_height, depth_width = height_map.shape
        aspect = depth_width / depth_height
        world_width = self.world_depth * aspect

        def img_to_world(c, r):
            x = (c / max(depth_width - 1, 1)) - 0.5
            z = (r / max(depth_height - 1, 1)) - 0.5
            return x * world_width, z * self.world_depth

        def img_to_uv(c, r):
            u = c / max(depth_width - 1, 1)
            v = 1.0 - (r / max(depth_height - 1, 1))
            return u, v

        structure_mask = np.zeros((depth_height, depth_width), dtype=np.uint8)
        for reg in regions_data:
            poly_pts = np.array(reg["polygon"], dtype=np.int32)
            cv2.fillPoly(structure_mask, [poly_pts], 1)

        # -------------------------------------------------------------
        # Base Terrain
        # -------------------------------------------------------------
        rows = np.arange(0, depth_height, self.mesh_stride, dtype=np.int32)
        cols = np.arange(0, depth_width, self.mesh_stride, dtype=np.int32)
        
        if rows[-1] != depth_height - 1:
            rows = np.append(rows, depth_height - 1)
        if cols[-1] != depth_width - 1:
            cols = np.append(cols, depth_width - 1)

        sampled_height = height_map[np.ix_(rows, cols)]
        
        grid_height, grid_width = sampled_height.shape
        x_norm = cols.astype(np.float32) / max(depth_width - 1, 1)
        v_norm = rows.astype(np.float32) / max(depth_height - 1, 1)

        xx, zz = np.meshgrid(x_norm, v_norm)
        xx = (xx - 0.5) * world_width
        zz = (zz - 0.5) * self.world_depth
        yy = sampled_height * self.height_scale

        base_verts = np.column_stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))).astype(np.float32)
        
        uu, vv_uv2 = np.meshgrid(x_norm, 1.0 - v_norm)
        base_uvs = np.column_stack((uu.reshape(-1), vv_uv2.reshape(-1))).astype(np.float32)

        row_indices = np.arange(grid_height - 1, dtype=np.int32)
        col_indices = np.arange(grid_width - 1, dtype=np.int32)
        rr, cc = np.meshgrid(row_indices, col_indices, indexing="ij")

        top_left = rr * grid_width + cc
        top_right = top_left + 1
        bottom_left = (rr + 1) * grid_width + cc
        bottom_right = bottom_left + 1

        faces_a = np.column_stack((top_left.reshape(-1), bottom_left.reshape(-1), top_right.reshape(-1)))
        faces_b = np.column_stack((top_right.reshape(-1), bottom_left.reshape(-1), bottom_right.reshape(-1)))
        
        all_base_faces = np.vstack((faces_a, faces_b)).astype(np.int64)
        
        sampled_mask = structure_mask[np.ix_(rows, cols)].reshape(-1)
        v0, v1, v2 = all_base_faces[:, 0], all_base_faces[:, 1], all_base_faces[:, 2]
        face_mask_vals = sampled_mask[v0] + sampled_mask[v1] + sampled_mask[v2]
        valid_base_faces = all_base_faces[face_mask_vals < 3]

        scene_dict = {}
        
        if len(valid_base_faces) > 0:
            base_mesh = trimesh.Trimesh(vertices=base_verts, faces=valid_base_faces, process=False)
            base_mesh.visual = trimesh.visual.TextureVisuals(uv=base_uvs)
            scene_dict['base_terrain'] = base_mesh

        # -------------------------------------------------------------
        # Structural Regions (Roofs & Walls)
        # -------------------------------------------------------------
        roof_verts_list = []
        roof_faces_list = []
        roof_uvs_list = []
        
        wall_verts_list = []
        wall_faces_list = []
        
        roof_vert_offset = 0
        wall_vert_offset = 0
        
        import scipy.spatial
        from matplotlib.path import Path as MplPath
        
        for reg in regions_data:
            poly_pts = np.array(reg["polygon"], dtype=np.float32) # [N, 2] (c, r)
            if len(poly_pts) < 3:
                continue
                
            a, b, c_plane = reg["plane"]["a"], reg["plane"]["b"], reg["plane"]["c"]
            base_h = reg["local_base_height"] * self.height_scale
            
            path = MplPath(poly_pts)
            
            min_c, min_r = np.floor(poly_pts.min(axis=0)).astype(int)
            max_c, max_r = np.ceil(poly_pts.max(axis=0)).astype(int)
            
            c_grid = np.arange(min_c, max_c + 1, self.mesh_stride)
            r_grid = np.arange(min_r, max_r + 1, self.mesh_stride)
            if len(c_grid) > 0 and len(r_grid) > 0:
                CC, RR = np.meshgrid(c_grid, r_grid)
                pts = np.column_stack((CC.reshape(-1), RR.reshape(-1)))
                inside = path.contains_points(pts)
                pts_inside = pts[inside]
                all_pts = np.vstack((poly_pts, pts_inside))
            else:
                all_pts = poly_pts

            all_pts = np.unique(all_pts, axis=0)
            
            if len(all_pts) < 3:
                continue
                
            try:
                tri = scipy.spatial.Delaunay(all_pts)
            except:
                continue
                
            centroids = all_pts[tri.simplices].mean(axis=1)
            tri_inside = path.contains_points(centroids)
            simplices = tri.simplices[tri_inside]
            
            if len(simplices) == 0:
                continue
            
            V_roof = []
            UV_roof = []
            for (col, row) in all_pts:
                x, z = img_to_world(col, row)
                y_raw = a * col + b * row + c_plane
                y = y_raw * self.height_scale
                V_roof.append([x, y, z])
                UV_roof.append(img_to_uv(col, row))
                
            V_roof = np.array(V_roof, dtype=np.float32)
            UV_roof = np.array(UV_roof, dtype=np.float32)
            
            roof_verts_list.append(V_roof)
            roof_faces_list.append(simplices + roof_vert_offset)
            roof_uvs_list.append(UV_roof)
            roof_vert_offset += len(V_roof)
            
            V_wall = []
            F_wall = []
            
            for i in range(len(poly_pts)):
                p1 = poly_pts[i]
                p2 = poly_pts[(i+1) % len(poly_pts)]
                
                c1, r1 = p1
                c2, r2 = p2
                
                x1, z1 = img_to_world(c1, r1)
                x2, z2 = img_to_world(c2, r2)
                
                y1_top = (a * c1 + b * r1 + c_plane) * self.height_scale
                y2_top = (a * c2 + b * r2 + c_plane) * self.height_scale
                
                if y1_top - base_h < 0.001 and y2_top - base_h < 0.001:
                    continue
                
                idx = len(V_wall)
                V_wall.extend([
                    [x1, y1_top, z1],
                    [x2, y2_top, z2],
                    [x2, base_h, z2],
                    [x1, base_h, z1]
                ])
                
                F_wall.append([idx, idx+3, idx+1])
                F_wall.append([idx+1, idx+3, idx+2])
                
            if len(V_wall) > 0:
                wall_verts_list.append(np.array(V_wall, dtype=np.float32))
                wall_faces_list.append(np.array(F_wall, dtype=np.int64) + wall_vert_offset)
                wall_vert_offset += len(V_wall)

        if len(roof_verts_list) > 0:
            all_roof_verts = np.vstack(roof_verts_list)
            all_roof_faces = np.vstack(roof_faces_list)
            all_roof_uvs = np.vstack(roof_uvs_list)
            
            roof_mesh = trimesh.Trimesh(vertices=all_roof_verts, faces=all_roof_faces, process=False)
            roof_mesh.visual = trimesh.visual.TextureVisuals(uv=all_roof_uvs)
            scene_dict['roofs'] = roof_mesh
            
        num_wall_faces = 0
        if len(wall_verts_list) > 0:
            all_wall_verts = np.vstack(wall_verts_list)
            all_wall_faces = np.vstack(wall_faces_list)
            
            wall_mesh = trimesh.Trimesh(vertices=all_wall_verts, faces=all_wall_faces, process=False)
            wall_color = np.array([50, 50, 50, 255], dtype=np.uint8)
            wall_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(wall_color, (len(all_wall_verts), 1)))
            scene_dict['walls'] = wall_mesh
            num_wall_faces = len(all_wall_faces)

        return trimesh.Scene(scene_dict), num_wall_faces

    def generate(self) -> Path:
        print("[TerrainGenerator] Loading height field and structure regions...")
        
        if not self.height_path.exists():
            raise FileNotFoundError(f"Height map not found: {self.height_path}")
        height_map = np.load(self.height_path).astype(np.float32)
        
        regions_data = []
        if self.regions_path.exists():
            with open(self.regions_path, "r") as f:
                regions_data = json.load(f)
                
        print(f"[TerrainGenerator] Height map: {height_map.shape[1]}x{height_map.shape[0]}")
        print(f"[TerrainGenerator] Structural Regions: {len(regions_data)}")
        
        print("[TerrainGenerator] Building 3D scene...")
        scene, num_wall_faces = self.build_scene(height_map, regions_data)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for name, mesh in scene.geometry.items():
            if np.isnan(mesh.vertices).any() or np.isinf(mesh.vertices).any():
                raise ValueError(f"Mesh '{name}' contains NaN/Inf vertices")

        print("[TerrainGenerator] Exporting GLB...")
        scene.export(self.output_path, file_type="glb")

        total_verts = sum(len(m.vertices) for m in scene.geometry.values())
        total_faces = sum(len(m.faces) for m in scene.geometry.values())
        
        print(f"[TerrainGenerator] Total Vertices: {total_verts:,}")
        print(f"[TerrainGenerator] Total Faces: {total_faces:,}")
        print(f"[TerrainGenerator] Wall Faces: {num_wall_faces:,}")
        print(f"[TerrainGenerator] Saved: {self.output_path}")

        return self.output_path

def main() -> None:
    parser = argparse.ArgumentParser(description="DepthWizard V1.5 - Generate 3D terrain")

    parser.add_argument("--height", type=Path, default=None)
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--height-scale", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--world-depth", type=float, default=100.0)

    args = parser.parse_args()

    generator = TerrainGenerator(
        height_path=args.height,
        regions_path=args.regions,
        output_path=args.output,
        height_scale=args.height_scale,
        mesh_stride=args.stride,
        world_depth=args.world_depth,
    )
    generator.generate()

if __name__ == "__main__":
    main()
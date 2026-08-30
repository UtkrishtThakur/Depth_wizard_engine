"""
DepthWizard V1.5 - Wall Texture Generator

Input:
    engine/processed/terrain.glb
    engine/processed/enhanced_texture.png
    
Output:
    engine/processed/terrain.glb (updated walls with baked shading)
    engine/processed/texture_observation_mask.png

Generates plausible appearance and physics-based shading for vertical structures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image
import cv2

class WallTextureGenerator:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        terrain_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
        mask_path: Optional[Path] = None,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.processed_dir = self.project_root / "engine" / "processed"

        self.terrain_path = (
            Path(terrain_path).resolve()
            if terrain_path is not None
            else self.processed_dir / "terrain.glb"
        )

        self.image_path = (
            Path(image_path).resolve()
            if image_path is not None
            else self.processed_dir / "enhanced_texture.png"
        )

        self.mask_path = (
            Path(mask_path).resolve()
            if mask_path is not None
            else self.processed_dir / "texture_observation_mask.png"
        )

    def generate(self) -> None:
        print("[WallTextureGenerator] Loading terrain geometry...")
        if not self.terrain_path.exists():
            raise FileNotFoundError(f"Terrain GLB not found: {self.terrain_path}")
            
        scene = trimesh.load(self.terrain_path, force="scene")
        
        print("[WallTextureGenerator] Loading enhanced texture for color sampling...")
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
            
        img = cv2.imread(str(self.image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        
        # 1. Physics-based directional light
        light_dir = np.array([0.5, 0.8, 0.3])
        light_dir = light_dir / np.linalg.norm(light_dir)
        ambient = 0.3
        diffuse_strength = 0.7

        # 2. Process walls
        if 'walls' in scene.geometry:
            wall_mesh = scene.geometry['walls']
            vertices = wall_mesh.vertices
            faces = wall_mesh.faces
            
            # Recompute face normals manually for flat shading
            normals = wall_mesh.face_normals
            
            vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
            vertex_colors[:, 3] = 255
            
            # We assume walls were generated as quads (2 triangles)
            # Find top vertices (highest Y) to sample color
            for i, face in enumerate(faces):
                v_pts = vertices[face]
                normal = normals[i]
                
                # Directional shading
                n_dot_l = max(0.0, np.dot(normal, light_dir))
                shading = ambient + diffuse_strength * n_dot_l
                
                # Top vertex to find UV
                top_v = v_pts[np.argmax(v_pts[:, 1])]
                
                # Reverse world to UV projection
                # xx = (u - 0.5) * world_width => u = xx / world_width + 0.5
                # zz = (0.5 - v) * world_depth => v = 0.5 - zz / world_depth
                world_depth = 100.0
                aspect = W / H
                world_width = world_depth * aspect
                
                u = (top_v[0] / world_width) + 0.5
                v = 0.5 + (top_v[2] / world_depth)
                
                px = int(np.clip(u * W, 0, W - 1))
                py = int(np.clip(v * H, 0, H - 1))
                
                # Sample local color and add noise
                base_color = img[py, px].astype(np.float32)
                
                # Apply shading
                shaded_color = np.clip(base_color * shading, 0, 255)
                
                # Assign to vertices of this face
                for vid in face:
                    vertex_colors[vid, :3] = shaded_color.astype(np.uint8)
            
            wall_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
            print("[WallTextureGenerator] Applied physics-based shading to walls.")
            
        # Overwrite terrain.glb
        scene.export(self.terrain_path, file_type="glb")
        
        # 3. Observation Mask
        print("[WallTextureGenerator] Generating observation mask...")
        mask = np.ones((H, W), dtype=np.uint8) * 255
        # The entire enhanced texture is observed except the walls, but the walls 
        # aren't mapped to the texture image, they use vertex colors.
        # So the texture image itself is 100% observed pixels.
        Image.fromarray(mask).save(self.mask_path)
        print(f"[WallTextureGenerator] Saved observation mask to {self.mask_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DepthWizard V1.5 - Wall Texture Generator")
    parser.add_argument("--terrain", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    generator = WallTextureGenerator(
        terrain_path=args.terrain,
        image_path=args.image,
    )
    generator.generate()

if __name__ == "__main__":
    main()

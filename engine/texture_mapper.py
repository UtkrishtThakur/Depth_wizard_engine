"""
DepthWizard V1.5 - Texture Mapper

Input:
    engine/processed/terrain.glb (contains surface and wall meshes)
    input/<source image>

Output:
    engine/processed/textured_terrain.glb

Applies the original RGB imagery onto the surface geometry, leaving walls intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


class TextureMapper:
    """Map original RGB image onto the 3D terrain geometry."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        terrain_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.input_dir = self.project_root / "input"
        self.processed_dir = self.project_root / "engine" / "processed"

        self.terrain_path = (
            Path(terrain_path).resolve()
            if terrain_path is not None
            else self.processed_dir / "terrain.glb"
        )

        self.image_path = (
            Path(image_path).resolve()
            if image_path is not None
            else None
        )

        self.output_path = (
            Path(output_path).resolve()
            if output_path is not None
            else self.processed_dir / "textured_terrain.glb"
        )

    def find_image(self) -> Path:
        """Find the first supported RGB image in input/."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        images = sorted(
            p
            for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not images:
            raise FileNotFoundError(f"No supported image found in {self.input_dir}")

        return images[0]

    def load_scene(self) -> trimesh.Scene:
        """Load the untextured terrain scene (may contain multiple meshes)."""
        if not self.terrain_path.exists():
            raise FileNotFoundError(f"Terrain scene not found: {self.terrain_path}")

        scene = trimesh.load(self.terrain_path, force="scene")
        if not scene.geometry:
            raise ValueError("No geometry found in GLB")

        return scene.copy()

    def load_image(self) -> Image.Image:
        """Load the source image as RGB."""
        if self.image_path:
            image_path = self.image_path
        else:
            enhanced_path = self.processed_dir / "enhanced_texture.png"
            if enhanced_path.exists():
                image_path = enhanced_path
            else:
                image_path = self.find_image()

        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    def export(self) -> Path:
        """Run pipeline to produce textured GLB."""
        print(f"[TextureMapper] Terrain: {self.terrain_path}")
        scene = self.load_scene()
        
        total_vertices = sum(len(m.vertices) for m in scene.geometry.values())
        total_faces = sum(len(m.faces) for m in scene.geometry.values())
        print(f"[TextureMapper] Total vertices: {total_vertices:,}")
        print(f"[TextureMapper] Total faces: {total_faces:,}")

        image = self.load_image()
        print(f"[TextureMapper] Texture image size: {image.width}x{image.height}")

        # Material for the surface
        surface_material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=0.9,
        )

        # Apply texture only to meshes that have UVs (the surface mesh)
        textured_count = 0
        for name, mesh in scene.geometry.items():
            if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=mesh.visual.uv,
                    material=surface_material,
                    image=image,
                )
                textured_count += 1
                print(f"[TextureMapper] Applied texture to mesh: {name} ({len(mesh.faces):,} faces)")
            else:
                print(f"[TextureMapper] Skipped texture for mesh: {name} ({len(mesh.faces):,} faces) - No UVs")

        if textured_count == 0:
            print("[TextureMapper] WARNING: No meshes with UVs found to texture!")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        scene.export(self.output_path, file_type="glb")
        
        print(f"[TextureMapper] Saved: {self.output_path}")

        return self.output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="DepthWizard V1.5 - Texture Mapper")
    parser.add_argument("--terrain", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    mapper = TextureMapper(
        terrain_path=args.terrain,
        image_path=args.image,
        output_path=args.output,
    )
    mapper.export()


if __name__ == "__main__":
    main()

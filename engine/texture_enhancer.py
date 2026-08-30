"""
DepthWizard V1 - Texture Enhancer

Input:
    input/<source image>
    
Output:
    engine/processed/enhanced_texture.png

Upscales and enhances the original RGB texture for higher quality 3D projection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


class TextureEnhancer:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        image_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        scale: int = 2,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.input_dir = self.project_root / "input"
        self.processed_dir = self.project_root / "engine" / "processed"

        self.image_path = (
            Path(image_path).resolve()
            if image_path is not None
            else None
        )

        self.output_path = (
            Path(output_path).resolve()
            if output_path is not None
            else self.processed_dir / "enhanced_texture.png"
        )

        self.scale = max(1, int(scale))

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

    def process(self) -> Path:
        image_path = self.image_path or self.find_image()
        print(f"[TextureEnhancer] Source: {image_path}")
        
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Load image with OpenCV
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # OpenCV loads as BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        print(f"[TextureEnhancer] Original: {orig_w}x{orig_h}")
        print(f"[TextureEnhancer] Scale: {self.scale}x")

        # 1. Upscale (Lanczos)
        new_w = orig_w * self.scale
        new_h = orig_h * self.scale
        img_up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Convert to LAB for luminance processing
        lab = cv2.cvtColor(img_up, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 2. Local contrast enhancement (CLAHE on L channel)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge back
        lab_enhanced = cv2.merge((cl, a, b))
        img_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # 3. Mild Denoising
        # Using bilateral filter to preserve edges while smoothing flat areas
        img_denoised = cv2.bilateralFilter(img_contrast, d=5, sigmaColor=25, sigmaSpace=25)

        # 4. Conservative Sharpening (Unsharp Masking)
        gaussian = cv2.GaussianBlur(img_denoised, (0, 0), 2.0)
        img_sharpened = cv2.addWeighted(img_denoised, 1.5, gaussian, -0.5, 0)

        # Save result
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # PIL for saving handles RGB natively
        out_img = Image.fromarray(img_sharpened)
        out_img.save(self.output_path)
        
        print(f"[TextureEnhancer] Enhanced: {new_w}x{new_h}")
        print(f"[TextureEnhancer] Saved: {self.output_path}")

        return self.output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="DepthWizard V1 - Texture Enhancer")

    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to source RGB image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save enhanced_texture.png",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Texture upscale factor (1, 2, 4)",
    )

    args = parser.parse_args()

    enhancer = TextureEnhancer(
        image_path=args.image,
        output_path=args.output,
        scale=args.scale,
    )
    enhancer.process()


if __name__ == "__main__":
    main()

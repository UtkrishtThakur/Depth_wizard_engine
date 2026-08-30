"""
DepthWizard V1 - Depth to Height Transformation

Input:
    engine/processed/depth.npy
    
Output:
    engine/processed/height.npy
    engine/processed/height.png
    engine/processed/depth_direct.png

Transforms raw disparity-like depth into a stable relative height field.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image

# Configuration Constants
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 0.08
BILATERAL_SIGMA_SPACE = 7
GAMMA = 1.0


class DepthToHeight:
    """Converts a raw relative depth map to a stable relative height field."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        depth_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        height_mode: str = "direct",
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.processed_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else self.project_root / "engine" / "processed"
        )

        self.depth_path = (
            Path(depth_path).resolve()
            if depth_path is not None
            else self.processed_dir / "depth.npy"
        )

        self.height_mode = height_mode

    def process(self) -> Path:
        """Run the transformation pipeline."""
        
        print("[DepthToHeight] Loading depth...")
        if not self.depth_path.exists():
            raise FileNotFoundError(f"Depth map not found: {self.depth_path}")

        depth = np.load(self.depth_path).astype(np.float32)

        print(f"[DepthToHeight] Raw range: {float(depth.min()):.6f} -> {float(depth.max()):.6f}")

        # -----------------------------------------------------------
        # Robust Normalization
        # -----------------------------------------------------------
        low = np.percentile(depth, 2)
        high = np.percentile(depth, 98)
        print(f"[DepthToHeight] P02: {low:.6f}")
        print(f"[DepthToHeight] P98: {high:.6f}")

        depth_clipped = np.clip(depth, low, high)
        normalized = (depth_clipped - low) / max(high - low, 1e-6)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Polarity
        if self.height_mode == "inverse":
            print("[DepthToHeight] Applying inverse depth mapping...")
            normalized = 1.0 - normalized
        else:
            print("[DepthToHeight] Applying direct depth mapping...")
            pass

        # Save direct visualization
        direct_img = (normalized * 255.0).astype(np.uint8)
        direct_path = self.processed_dir / "depth_direct.png"
        Image.fromarray(direct_img).save(direct_path)

        # -----------------------------------------------------------
        # Edge-Aware Smoothing
        # -----------------------------------------------------------
        print("[DepthToHeight] Applying bilateral smoothing...")
        # cv2.bilateralFilter works best on float32 arrays
        smoothed = cv2.bilateralFilter(
            normalized,
            d=BILATERAL_D,
            sigmaColor=BILATERAL_SIGMA_COLOR,
            sigmaSpace=BILATERAL_SIGMA_SPACE,
        )

        # -----------------------------------------------------------
        # Gamma Contrast
        # -----------------------------------------------------------
        print(f"[DepthToHeight] Applying gamma ({GAMMA})...")
        height = np.power(np.clip(smoothed, 0.0, 1.0), GAMMA)

        # -----------------------------------------------------------
        # Save Outputs
        # -----------------------------------------------------------
        print(f"[DepthToHeight] Final height range: {float(height.min()):.6f} -> {float(height.max()):.6f}")
        print(f"[DepthToHeight] Final height mean: {float(height.mean()):.6f}")
        print(f"[DepthToHeight] Final height std: {float(height.std()):.6f}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        npy_path = self.processed_dir / "height.npy"
        png_path = self.processed_dir / "height.png"

        np.save(npy_path, height.astype(np.float32))
        print(f"[DepthToHeight] Saved height.npy")

        height_img = (height * 255.0).astype(np.uint8)
        Image.fromarray(height_img).save(png_path)
        print(f"[DepthToHeight] Saved height.png")

        return npy_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DepthWizard V1 - Depth to Height Transformation"
    )

    parser.add_argument(
        "--depth",
        type=Path,
        default=None,
        help="Path to raw depth.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--height-mode",
        type=str,
        default="direct",
        choices=["direct", "inverse"],
        help="Polarity mapping. 'direct' means higher depth value = higher elevation.",
    )

    args = parser.parse_args()

    transformer = DepthToHeight(
        depth_path=args.depth,
        output_dir=args.output_dir,
        height_mode=args.height_mode,
    )

    transformer.process()


if __name__ == "__main__":
    main()

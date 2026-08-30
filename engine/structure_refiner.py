"""
DepthWizard V1.5 - Structure Refiner
Applies continuous edge-aware structural height enhancement without breaking topology.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image

class StructureRefiner:
    def __init__(
        self, project_root=None, height_path=None, image_path=None, output_dir=None,
        small_scale=2.0, medium_scale=8.0, large_scale=32.0,
        structure_gain=1.3, edge_strength=0.5
    ):
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
        self.processed_dir = Path(output_dir).resolve() if output_dir else self.project_root / "engine" / "processed"
        self.height_path = Path(height_path).resolve() if height_path else self.processed_dir / "height.npy"
        self.image_path = Path(image_path).resolve() if image_path else None
        
        self.small_scale = float(small_scale)
        self.medium_scale = float(medium_scale)
        self.large_scale = float(large_scale)
        self.structure_gain = float(structure_gain)
        self.edge_strength = float(edge_strength)

    def process(self) -> Path:
        print("[StructureRefiner] Loading height field...")
        H = np.load(self.height_path).astype(np.float32)
        
        # 1. Base terrain (edge-aware smoothing)
        print("[StructureRefiner] Computing multi-scale decomposition...")
        # Using bilateral filter for edge-aware medium scale
        H_medium = cv2.bilateralFilter(H, d=int(self.medium_scale)*2+1, sigmaColor=0.1, sigmaSpace=self.medium_scale)
        H_large = cv2.GaussianBlur(H, (0, 0), sigmaX=self.large_scale, sigmaY=self.large_scale)
        H_small = cv2.GaussianBlur(H, (0, 0), sigmaX=self.small_scale, sigmaY=self.small_scale)
        
        support_terrain = H_medium
        structural_residual = H - support_terrain

        # 2. Structural Confidence
        print("[StructureRefiner] Computing structure confidence...")
        gy, gx = np.gradient(H_small)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        laplacian = cv2.Laplacian(H_small, cv2.CV_32F)
        curv_mag = np.abs(laplacian)
        
        persistence = np.abs(H_medium - H_large) + np.abs(H_small - H_medium)
        
        def norm(x): 
            p = np.percentile(x, 98)
            return np.clip(x / max(p, 1e-6), 0, 1)
            
        geo_confidence = (norm(grad_mag) + norm(curv_mag) + norm(persistence)) / 3.0
        
        rgb_grad_mag = np.zeros_like(grad_mag)
        if self.image_path and self.image_path.exists() and self.edge_strength > 0:
            img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (H.shape[1], H.shape[0]))
                img_float = img.astype(np.float32) / 255.0
                ig_y, ig_x = np.gradient(img_float)
                rgb_grad_mag = norm(np.sqrt(ig_x**2 + ig_y**2))

        confidence = geo_confidence + self.edge_strength * rgb_grad_mag
        if confidence.max() > 0:
            confidence = confidence / confidence.max()
        confidence = cv2.GaussianBlur(confidence, (0, 0), sigmaX=self.small_scale, sigmaY=self.small_scale)
        confidence = np.clip(confidence, 0, 1)

        # 3. Apply Local Relief Amplification
        print("[StructureRefiner] Enhancing continuous structure relief...")
        # enhanced_residual = residual * (1 + gain * confidence)
        enhanced_residual = structural_residual * (1.0 + self.structure_gain * confidence)
        
        H_new = support_terrain + enhanced_residual
        
        # Ensure it stays within [0, 1] range to avoid clipping spikes
        low = np.percentile(H_new, 1)
        high = np.percentile(H_new, 99)
        if high > low:
            H_new = (H_new - low) / (high - low)
        final_H = np.clip(H_new, 0.0, 1.0)
        
        print(f"[StructureRefiner] Before: Mean={H.mean():.6f}, P95={np.percentile(H, 95):.6f}, P99={np.percentile(H, 99):.6f}")
        print(f"[StructureRefiner] After: Mean={final_H.mean():.6f}, P95={np.percentile(final_H, 95):.6f}, P99={np.percentile(final_H, 99):.6f}")
        
        gy_ref, gx_ref = np.gradient(final_H)
        slope_ref = np.sqrt(gx_ref**2 + gy_ref**2)
        slope_p95 = np.percentile(slope_ref, 95)
        structure_edges = (slope_ref > (slope_p95 * 0.5)).astype(np.float32)
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        refined_npy = self.processed_dir / "refined_height.npy"
        np.save(refined_npy, final_H.astype(np.float32))
        np.save(self.processed_dir / "structure_edges.npy", structure_edges.astype(np.bool_))
        np.save(self.processed_dir / "structural_residual.npy", structural_residual.astype(np.float32))
        
        Image.fromarray((final_H * 255.0).astype(np.uint8)).save(self.processed_dir / "refined_height.png")
        Image.fromarray((support_terrain * 255.0).astype(np.uint8)).save(self.processed_dir / "support_terrain.png")
        Image.fromarray((confidence * 255.0).astype(np.uint8)).save(self.processed_dir / "structure_confidence.png")
        Image.fromarray((structure_edges * 255.0).astype(np.uint8)).save(self.processed_dir / "structure_edges.png")
        
        print(f"[StructureRefiner] Saved: {refined_npy.name}")
        return refined_npy

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--structure-gain", type=float, default=1.3)
    args = parser.parse_args()
    StructureRefiner(height_path=args.height, image_path=args.image, structure_gain=args.structure_gain).process()

if __name__ == "__main__":
    main()

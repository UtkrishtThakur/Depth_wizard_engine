"""
DepthWizard V1.5 - Structure Refiner

Input:
    engine/processed/height.npy
    input/<source image> (optional, for edge alignment)
    
Output:
    engine/processed/refined_height.npy
    engine/processed/structure_confidence.png
    engine/processed/structure_edges.png
    engine/processed/structure_edges.npy
    engine/processed/structure_regions.json

Applies generalized, multi-scale structure-aware elevation refinement.
Extracts piecewise-smooth structural regions and explicit geometric boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

class StructureRefiner:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        height_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        small_scale: float = 2.0,
        medium_scale: float = 8.0,
        large_scale: float = 32.0,
        small_gain: float = 0.2,     
        medium_gain: float = 1.0,    
        large_gain: float = 1.0,     
        local_gain: float = 1.5,     
        edge_strength: float = 0.5,
        regularize: bool = True,
        planar_fitting: bool = True,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.input_dir = self.project_root / "input"
        self.processed_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else self.project_root / "engine" / "processed"
        )

        self.height_path = (
            Path(height_path).resolve()
            if height_path is not None
            else self.processed_dir / "height.npy"
        )

        self.image_path = (
            Path(image_path).resolve() if image_path is not None else None
        )

        self.small_scale = float(small_scale)
        self.medium_scale = float(medium_scale)
        self.large_scale = float(large_scale)
        
        self.small_gain = float(small_gain)
        self.medium_gain = float(medium_gain)
        self.large_gain = float(large_gain)
        self.local_gain = float(local_gain)
        self.edge_strength = float(edge_strength)
        
        self.regularize = regularize
        self.planar_fitting = planar_fitting

    def find_image(self) -> Path:
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

    def _fit_plane(self, Y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, list[float]]:
        z_coords, x_coords = np.where(mask)
        if len(z_coords) < 3:
            return Y, float("inf"), [0.0, 0.0, 0.0]
            
        y_vals = Y[z_coords, x_coords]
        
        # A = [X, Z, 1]
        A = np.column_stack((x_coords, z_coords, np.ones_like(x_coords)))
        
        # Fit
        res = np.linalg.lstsq(A, y_vals, rcond=None)
        coeffs = res[0]
        
        # Reconstruct
        y_fit = coeffs[0] * x_coords + coeffs[1] * z_coords + coeffs[2]
        
        # Compute mean absolute error
        mae = np.mean(np.abs(y_fit - y_vals))
        
        # Create output plane
        Y_plane = Y.copy()
        Y_plane[z_coords, x_coords] = y_fit
        
        return Y_plane, float(mae), coeffs.tolist()

    def process(self) -> Path:
        print("[StructureRefiner] Loading height field...")
        if not self.height_path.exists():
            raise FileNotFoundError(f"Height map not found: {self.height_path}")

        H = np.load(self.height_path).astype(np.float32)
        print(f"[StructureRefiner] Raw height range: {H.min():.6f} -> {H.max():.6f}")
        print(f"[StructureRefiner] Raw height mean: {H.mean():.6f}, std: {H.std():.6f}")
        
        raw_p01 = np.percentile(H, 1)
        raw_p99 = np.percentile(H, 99)
        gy_raw, gx_raw = np.gradient(H)
        raw_slope = np.sqrt(gx_raw**2 + gy_raw**2)
        print(f"[StructureRefiner] Raw P01: {raw_p01:.6f}, P99: {raw_p99:.6f}")
        print(f"[StructureRefiner] Raw Mean Slope: {raw_slope.mean():.6f}, Max Slope: {raw_slope.max():.6f}")

        print("[StructureRefiner] Computing multi-scale decomposition...")
        H_large = cv2.GaussianBlur(H, (0, 0), sigmaX=self.large_scale, sigmaY=self.large_scale)
        H_medium = cv2.GaussianBlur(H, (0, 0), sigmaX=self.medium_scale, sigmaY=self.medium_scale)
        H_small = cv2.GaussianBlur(H, (0, 0), sigmaX=self.small_scale, sigmaY=self.small_scale)

        detail_large = H_large
        detail_medium = H_medium - H_large
        detail_small = H - H_medium

        print("[StructureRefiner] Computing structure confidence...")
        gy, gx = np.gradient(H_small)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        mean_sq = cv2.GaussianBlur(H**2, (0, 0), sigmaX=self.medium_scale, sigmaY=self.medium_scale)
        sq_mean = H_medium**2
        variance = np.clip(mean_sq - sq_mean, 0, None)
        std_dev = np.sqrt(variance)
        
        laplacian = cv2.Laplacian(H_small, cv2.CV_32F)
        curv_mag = np.abs(laplacian)
        
        persistence = np.abs(H_medium - H_large) + np.abs(H_small - H_medium)
        
        grad_p98 = np.percentile(grad_mag, 98)
        grad_norm = np.clip(grad_mag / max(grad_p98, 1e-6), 0, 1)
        
        std_p98 = np.percentile(std_dev, 98)
        std_norm = np.clip(std_dev / max(std_p98, 1e-6), 0, 1)
        
        curv_p98 = np.percentile(curv_mag, 98)
        curv_norm = np.clip(curv_mag / max(curv_p98, 1e-6), 0, 1)
        
        pers_p98 = np.percentile(persistence, 98)
        pers_norm = np.clip(persistence / max(pers_p98, 1e-6), 0, 1)
        
        geo_confidence = (grad_norm + std_norm + curv_norm + pers_norm) / 4.0
        
        rgb_grad_mag = np.zeros_like(grad_mag)
        image_path = self.image_path or self.find_image()
        if image_path.exists() and self.edge_strength > 0:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (H.shape[1], H.shape[0]))
                img_float = img.astype(np.float32) / 255.0
                ig_y, ig_x = np.gradient(img_float)
                rgb_grad_mag = np.sqrt(ig_x**2 + ig_y**2)
                
                rgb_p98 = np.percentile(rgb_grad_mag, 98)
                if rgb_p98 > 0:
                    rgb_grad_mag = np.clip(rgb_grad_mag / rgb_p98, 0, 1)

        confidence = geo_confidence + self.edge_strength * rgb_grad_mag
        if confidence.max() > 0:
            confidence = confidence / confidence.max()
            
        confidence = cv2.GaussianBlur(confidence, (0, 0), sigmaX=self.small_scale, sigmaY=self.small_scale)
        confidence = np.clip(confidence, 0, 1)

        print("[StructureRefiner] Assembling refined height...")
        
        effective_small_gain = self.small_gain * confidence
        
        H_refined = (
            detail_large * self.large_gain + 
            detail_medium * self.medium_gain + 
            detail_small * effective_small_gain
        )
        
        local_structure = H_refined - H_medium
        H_refined = H_refined + local_structure * (self.local_gain - 1.0) * confidence

        low = np.percentile(H_refined, 2)
        high = np.percentile(H_refined, 98)
        if high > low:
            H_refined = (H_refined - low) / (high - low)
        H_refined = np.clip(H_refined, 0.0, 1.0)
        
        print("[StructureRefiner] Extracting structural boundaries...")
        gy_ref, gx_ref = np.gradient(H_refined)
        slope_ref = np.sqrt(gx_ref**2 + gy_ref**2)
        slope_p95 = np.percentile(slope_ref, 95)
        structure_edges = (slope_ref > (slope_p95 * 0.5)).astype(np.float32)
        
        num_planar_regions = 0
        regions_json_data = []
        
        if self.planar_fitting:
            print("[StructureRefiner] Performing conditional planar fitting & extracting regions...")
            
            # Combine edge constraint with minimum confidence requirement
            region_mask = ((confidence > 0.3) & (structure_edges < 0.5)).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
            
            min_area = 200
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > min_area:
                    mask = (labels == label)
                    
                    Y_plane, mae, coeffs = self._fit_plane(H_refined, mask)
                    
                    # Local base height calculation
                    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
                    ring = (dilated - mask.astype(np.uint8)) > 0
                    ring_heights = H_refined[ring]
                    
                    if len(ring_heights) < 10:
                        continue
                        
                    local_base_height = float(np.percentile(ring_heights, 10))
                    
                    if mae < 0.02: 
                        H_refined[mask] = Y_plane[mask]
                        num_planar_regions += 1
                        
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours) > 0:
                            c = max(contours, key=cv2.contourArea)
                            epsilon = 0.005 * cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, epsilon, True)
                            
                            polygon = approx.reshape(-1, 2).tolist()
                            if len(polygon) >= 3:
                                regions_json_data.append({
                                    "polygon": polygon,
                                    "local_base_height": local_base_height,
                                    "plane": {
                                        "a": coeffs[0],
                                        "b": coeffs[1],
                                        "c": coeffs[2]
                                    }
                                })

        if self.regularize:
            print("[StructureRefiner] Applying edge-aware surface regularization...")
            H_refined = cv2.bilateralFilter(H_refined, d=11, sigmaColor=0.08, sigmaSpace=15)
            H_refined = np.clip(H_refined, 0.0, 1.0)

        gy_fin, gx_fin = np.gradient(H_refined)
        slope_fin = np.sqrt(gx_fin**2 + gy_fin**2)
        slope_p95_fin = np.percentile(slope_fin, 95)
        final_edges = (slope_fin > max(0.01, slope_p95_fin * 0.5)).astype(np.float32)

        change = np.abs(H_refined - H)
        ref_p01 = np.percentile(H_refined, 1)
        ref_p99 = np.percentile(H_refined, 99)
        
        print(f"[StructureRefiner] Refined height range: {H_refined.min():.6f} -> {H_refined.max():.6f}")
        print(f"[StructureRefiner] Refined height mean: {H_refined.mean():.6f}, std: {H_refined.std():.6f}")
        print(f"[StructureRefiner] Refined P01: {ref_p01:.6f}, P99: {ref_p99:.6f}")
        print(f"[StructureRefiner] Refined Mean Slope: {slope_fin.mean():.6f}, Max Slope: {slope_fin.max():.6f}")
        print(f"[StructureRefiner] Mean absolute change: {change.mean():.6f}")
        print(f"[StructureRefiner] Max absolute change: {change.max():.6f}")
        print(f"[StructureRefiner] Planar regions extracted: {len(regions_json_data)}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        refined_npy_path = self.processed_dir / "refined_height.npy"
        refined_png_path = self.processed_dir / "refined_height.png"
        conf_png_path = self.processed_dir / "structure_confidence.png"
        edges_npy_path = self.processed_dir / "structure_edges.npy"
        edges_png_path = self.processed_dir / "structure_edges.png"
        regions_json_path = self.processed_dir / "structure_regions.json"

        np.save(refined_npy_path, H_refined.astype(np.float32))
        np.save(edges_npy_path, final_edges.astype(np.bool_))
        
        Image.fromarray((H_refined * 255.0).astype(np.uint8)).save(refined_png_path)
        Image.fromarray((confidence * 255.0).astype(np.uint8)).save(conf_png_path)
        Image.fromarray((final_edges * 255.0).astype(np.uint8)).save(edges_png_path)
        
        with open(regions_json_path, 'w') as f:
            json.dump(regions_json_data, f)
        
        print(f"[StructureRefiner] Saved: {refined_npy_path.name}")
        print(f"[StructureRefiner] Saved: {edges_npy_path.name}")
        print(f"[StructureRefiner] Saved: {regions_json_path.name}")

        return refined_npy_path


def main() -> None:
    parser = argparse.ArgumentParser(description="DepthWizard V1.5 - Structure Refiner")

    parser.add_argument("--height", type=Path, default=None, help="Path to height.npy")
    parser.add_argument("--image", type=Path, default=None, help="Path to RGB image")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--small-scale", type=float, default=2.0)
    parser.add_argument("--medium-scale", type=float, default=8.0)
    parser.add_argument("--large-scale", type=float, default=32.0)
    parser.add_argument("--small-gain", type=float, default=0.2)
    parser.add_argument("--medium-gain", type=float, default=1.0)
    parser.add_argument("--large-gain", type=float, default=1.0)
    parser.add_argument("--local-gain", type=float, default=1.5)
    parser.add_argument("--edge-strength", type=float, default=0.5)
    parser.add_argument("--no-regularize", action="store_true", help="Disable bilateral smoothing")
    parser.add_argument("--no-planar-fitting", action="store_true", help="Disable planar fitting")

    args = parser.parse_args()

    refiner = StructureRefiner(
        height_path=args.height,
        image_path=args.image,
        output_dir=args.output_dir,
        small_scale=args.small_scale,
        medium_scale=args.medium_scale,
        large_scale=args.large_scale,
        small_gain=args.small_gain,
        medium_gain=args.medium_gain,
        large_gain=args.large_gain,
        local_gain=args.local_gain,
        edge_strength=args.edge_strength,
        regularize=not args.no_regularize,
        planar_fitting=not args.no_planar_fitting,
    )
    refiner.process()


if __name__ == "__main__":
    main()

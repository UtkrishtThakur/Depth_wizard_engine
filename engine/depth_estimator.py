"""
DepthWizard V1 - Depth Estimator

Pipeline:

input image
    ↓
OpenCV RGB conversion
    ↓
Depth Anything V2 preprocessing
    ↓
PyTorch tensor
    ↓
Depth Anything V2 Small
    ↓
relative depth map
    ↓
depth.npy + depth.png

This module does NOT:
- convert depth to metric elevation
- generate DSMs
- generate meshes
- perform 3D rendering

Those are later stages.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from models.depth_anything_v2.dpt import DepthAnythingV2
from models.depth_anything_v2.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


class DepthEstimator:
    """Runs Depth Anything V2 Small inference."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "auto",
        input_size: int = 518,
    ) -> None:
        self.project_root = Path(__file__).resolve().parent.parent

        self.model_path = (
            Path(model_path).resolve()
            if model_path is not None
            else self.project_root / "models" / "depth_anything_v2_vits.pth"
        )

        self.input_dir = self.project_root / "input"
        self.output_dir = self.project_root / "engine" / "processed"

        self.input_size = input_size
        self.device = self._resolve_device(device)

        self.model: Optional[DepthAnythingV2] = None

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve the requested device, with automatic fallback."""

        if device != "auto":
            requested = torch.device(device)

            if requested.type == "cuda" and not torch.cuda.is_available():
                print(
                    "[DepthEstimator] CUDA requested but unavailable. "
                    "Falling back to CPU."
                )
                return torch.device("cpu")

            return requested

        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Input discovery
    # ------------------------------------------------------------------

    def find_input_image(self) -> Path:
        """Find the first supported image in input/."""

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}"
            )

        images = sorted(
            path
            for path in self.input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not images:
            raise FileNotFoundError(
                f"No supported image found in {self.input_dir}. "
                f"Supported extensions: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if len(images) > 1:
            print(
                f"[DepthEstimator] Found {len(images)} input images. "
                f"Using: {images[0].name}"
            )

        return images[0]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Construct and load the official Depth Anything V2 Small model."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Depth model checkpoint not found: {self.model_path}"
            )

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }
        }

        print("[DepthEstimator] Loading Depth Anything V2 Small...")
        print(f"[DepthEstimator] Checkpoint: {self.model_path}")
        print(f"[DepthEstimator] Device: {self.device}")

        config = model_configs["vits"]

        self.model = DepthAnythingV2(**config)

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
        )

        # Checkpoints can be either:
        # 1. the raw state_dict
        # 2. wrapped inside a "model" key
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        print("[DepthEstimator] Model loaded successfully.")

    # ------------------------------------------------------------------
    # Model preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Apply the official Depth Anything V2 preprocessing.

        Returns:
            tensor: [1, 3, H, W] float32 tensor
            original_size: (height, width)
        """

        if image_bgr is None:
            raise ValueError("OpenCV failed to decode the image.")

        original_height, original_width = image_bgr.shape[:2]

        transform = Compose(
            [
                Resize(
                    width=self.input_size,
                    height=self.input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                PrepareForNet(),
            ]
        )

        # Official V2 path expects BGR from cv2 and converts to RGB.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32) / 255.0

        transformed = transform({"image": image_rgb})
        image = transformed["image"]

        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        return tensor, (original_height, original_width)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Run monocular depth inference.

        Returns:
            H x W float32 NumPy array containing relative depth.
        """

        if self.model is None:
            self.load_model()

        source_path = (
            Path(image_path).resolve()
            if image_path is not None
            else self.find_input_image()
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Input image does not exist: {source_path}"
            )

        print(f"[DepthEstimator] Source: {source_path.name}")

        image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError(
                f"Could not decode image: {source_path}"
            )

        original_height, original_width = image_bgr.shape[:2]

        tensor, _ = self._preprocess(image_bgr)

        print(
            "[DepthEstimator] Tensor shape: "
            f"{tuple(tensor.shape)}"
        )

        print(
            "[DepthEstimator] Running inference..."
        )

        start_time = time.perf_counter()

        # Execute the network directly.
        depth = self.model(tensor)

        # Restore the prediction to the original image resolution.
        depth = torch.nn.functional.interpolate(
            depth[:, None],
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

        depth = depth.detach().float().cpu().numpy()

        elapsed = time.perf_counter() - start_time

        depth = np.asarray(depth, dtype=np.float32)

        if depth.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D depth map, got shape {depth.shape}"
            )

        if not np.isfinite(depth).all():
            raise RuntimeError(
                "Depth prediction contains NaN or infinite values."
            )

        print(
            f"[DepthEstimator] Inference complete in {elapsed:.2f}s"
        )

        print(
            f"[DepthEstimator] Depth shape: {depth.shape[0]}x{depth.shape[1]}"
        )

        print(
            f"[DepthEstimator] Depth dtype: {depth.dtype}"
        )

        print(
            f"[DepthEstimator] Depth range: "
            f"{depth.min():.6f} -> {depth.max():.6f}"
        )

        return depth

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_outputs(
        self,
        depth: np.ndarray,
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """Save numerical depth and a grayscale visualization."""

        if depth.ndim != 2:
            raise ValueError(
                f"Expected 2D depth array, got {depth.shape}"
            )

        output_path = (
            Path(output_dir).resolve()
            if output_dir is not None
            else self.output_dir
        )

        output_path.mkdir(parents=True, exist_ok=True)

        npy_path = output_path / "depth.npy"
        png_path = output_path / "depth.png"

        # Preserve real floating-point depth.
        np.save(npy_path, depth.astype(np.float32))

        # Visualization only.
        depth_min = float(depth.min())
        depth_max = float(depth.max())

        if depth_max > depth_min:
            depth_normalized = (
                (depth - depth_min)
                / (depth_max - depth_min)
                * 255.0
            )
        else:
            depth_normalized = np.zeros_like(depth)

        depth_visual = np.clip(
            depth_normalized,
            0,
            255,
        ).astype(np.uint8)

        if not cv2.imwrite(
            str(png_path),
            depth_visual,
        ):
            raise RuntimeError(
                f"Failed to save depth visualization: {png_path}"
            )

        print(f"[DepthEstimator] Saved: {npy_path}")
        print(f"[DepthEstimator] Saved: {png_path}")

        return npy_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DepthWizard V1 - Depth Anything V2 Small inference"
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional explicit image path.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional Depth Anything V2 checkpoint path.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, cuda:0, or mps.",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Depth Anything V2 input size.",
    )

    args = parser.parse_args()

    estimator = DepthEstimator(
        model_path=args.model,
        device=args.device,
        input_size=args.input_size,
    )

    depth = estimator.predict(
        image_path=args.image,
    )

    estimator.save_outputs(depth)

    print("[DepthEstimator] Done.")


if __name__ == "__main__":
    main()
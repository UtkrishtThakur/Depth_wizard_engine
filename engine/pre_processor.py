"""
DepthWizard V1 - Image Preprocessor

Responsibilities:
1. Find an input image inside ../input/
2. Open it and force RGB format
3. Resize it while preserving aspect ratio
4. Make the resulting dimensions divisible by 32
5. Convert it into a PyTorch tensor in [1, 3, H, W] with values in [0, 1]
6. Save a processed-image preview for inspection

This file intentionally does NOT run the depth model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class PreprocessedImage:
    """Result returned by the preprocessing pipeline."""

    tensor: torch.Tensor
    image: Image.Image
    source_path: Path
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


class PreProcessor:
    """
    Loads an image from the DepthWizard input directory and prepares it
    for the depth-estimation stage.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        target_longest_side: int = 768,
        size_multiple: int = 32,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.input_dir = self.project_root / "input"
        self.output_dir = self.project_root / "engine" / "processed"

        self.target_longest_side = target_longest_side
        self.size_multiple = size_multiple

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_input_image(self) -> Path:
        """Return the first supported image found in input/."""

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}"
            )

        images = sorted(
            path
            for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not images:
            raise FileNotFoundError(
                f"No supported image found in {self.input_dir}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if len(images) > 1:
            print(
                f"[PreProcessor] Found {len(images)} images. "
                f"Using: {images[0].name}"
            )

        return images[0]

    def _calculate_size(self, width: int, height: int) -> tuple[int, int]:
        """
        Resize while preserving aspect ratio.

        The longest side is limited to target_longest_side, then both
        dimensions are rounded to a multiple of size_multiple.
        """

        longest_side = max(width, height)

        if longest_side <= self.target_longest_side:
            scale = 1.0
        else:
            scale = self.target_longest_side / longest_side

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        new_width = max(
            self.size_multiple,
            round(new_width / self.size_multiple) * self.size_multiple,
        )
        new_height = max(
            self.size_multiple,
            round(new_height / self.size_multiple) * self.size_multiple,
        )

        return new_width, new_height

    def preprocess(self, image_path: Optional[Path] = None) -> PreprocessedImage:
        """
        Load and preprocess an image.

        Returns:
            PreprocessedImage containing:
            - PIL RGB image
            - float32 tensor shaped [1, 3, H, W]
            - source and size metadata
        """

        source_path = (
            Path(image_path).resolve()
            if image_path is not None
            else self.find_input_image()
        )

        if not source_path.exists():
            raise FileNotFoundError(f"Image not found: {source_path}")

        with Image.open(source_path) as loaded:
            image = loaded.convert("RGB")

        original_size = image.size
        new_size = self._calculate_size(*original_size)

        if original_size != new_size:
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert RGB pixels to float32 [0, 1].
        image_np = np.asarray(image, dtype=np.float32) / 255.0

        # HWC -> CHW, then add batch dimension.
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.contiguous()

        return PreprocessedImage(
            tensor=tensor,
            image=image,
            source_path=source_path,
            original_size=original_size,
            processed_size=image.size,
        )

    def save_preview(self, result: PreprocessedImage) -> Path:
        """Save the processed RGB image for visual inspection."""

        output_path = self.output_dir / "preprocessed.png"
        result.image.save(output_path)
        return output_path


def main() -> None:
    processor = PreProcessor()

    print(f"[PreProcessor] Project: {processor.project_root}")
    print(f"[PreProcessor] Input:   {processor.input_dir}")

    result = processor.preprocess()
    preview_path = processor.save_preview(result)

    print(f"[PreProcessor] Source:   {result.source_path.name}")
    print(f"[PreProcessor] Original: {result.original_size[0]}x{result.original_size[1]}")
    print(f"[PreProcessor] Processed: {result.processed_size[0]}x{result.processed_size[1]}")
    print(f"[PreProcessor] Tensor:    {tuple(result.tensor.shape)}")
    print(f"[PreProcessor] Range:     {result.tensor.min().item():.3f} -> {result.tensor.max().item():.3f}")
    print(f"[PreProcessor] Preview:   {preview_path}")


if __name__ == "__main__":
    main()
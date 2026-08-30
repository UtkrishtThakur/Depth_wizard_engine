"""
DepthWizard V1.5 - Main Engine Pipeline

Runs the complete image -> depth -> height -> structure refinement -> terrain -> texture pipeline.

Usage:

    python engine/main.py

Optional:

    python engine/main.py --image input/my_image.jpg
    python engine/main.py --height-scale 10
    python engine/main.py --stride 2
    python engine/main.py --texture-scale 2
    python engine/main.py --device cpu

Pipeline:

Input RGB
    ↓
PreProcessor
    ↓
DepthEstimator
    ↓
depth.npy
    ↓
DepthToHeight
    ↓
height.npy
    ↓
StructureRefiner
    ↓
refined_height.npy
    ↓
TerrainGenerator
    ↓
terrain.glb
    ↓
TextureEnhancer
    ↓
enhanced_texture.png
    ↓
TextureMapper
    ↓
textured_terrain.glb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from engine.pre_processor import PreProcessor
    from engine.depth_estimator import DepthEstimator
    from engine.depth_to_height import DepthToHeight
    from engine.structure_refiner import StructureRefiner
    from engine.terrain_generator import TerrainGenerator
    from engine.texture_enhancer import TextureEnhancer
    from engine.texture_mapper import TextureMapper
except ImportError:
    from pre_processor import PreProcessor
    from depth_estimator import DepthEstimator
    from depth_to_height import DepthToHeight
    from structure_refiner import StructureRefiner
    from terrain_generator import TerrainGenerator
    from texture_enhancer import TextureEnhancer
    from texture_mapper import TextureMapper


class DepthWizardPipeline:
    """Orchestrates the complete DepthWizard V1.5 processing pipeline."""

    def __init__(
        self,
        project_root: Path | None = None,
        image_path: Path | None = None,
        device: str = "auto",
        input_size: int = 518,
        height_scale: float = 10.0,
        mesh_stride: int = 2,
        height_mode: str = "direct",
        texture_scale: int = 2,
        small_scale: float = 2.0,
        medium_scale: float = 8.0,
        large_scale: float = 32.0,
        small_gain: float = 0.5,
        medium_gain: float = 1.1,
        large_gain: float = 1.0,
        local_gain: float = 1.2,
        edge_strength: float = 0.5,
        regularize: bool = True,
        planar_fitting: bool = True,
    ) -> None:

        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.image_path = (
            Path(image_path).resolve()
            if image_path is not None
            else None
        )

        self.device = device
        self.input_size = input_size
        self.height_scale = height_scale
        self.mesh_stride = mesh_stride
        self.height_mode = height_mode
        self.texture_scale = texture_scale
        
        self.small_scale = small_scale
        self.medium_scale = medium_scale
        self.large_scale = large_scale
        self.small_gain = small_gain
        self.medium_gain = medium_gain
        self.large_gain = large_gain
        self.local_gain = local_gain
        self.edge_strength = edge_strength
        self.regularize = regularize
        self.planar_fitting = planar_fitting

        self.input_dir = self.project_root / "input"
        self.processed_dir = self.project_root / "engine" / "processed"

        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run every stage sequentially."""

        pipeline_start = time.perf_counter()

        print()
        print("=" * 60)
        print("DepthWizard V1.5")
        print("Single-View RGB → 3D Terrain")
        print("=" * 60)
        print()

        # --------------------------------------------------------------
        # 1. Preprocessing
        # --------------------------------------------------------------

        print("[1/7] PREPROCESSING")
        print("-" * 60)

        processor = PreProcessor(
            project_root=self.project_root,
        )

        preprocess_result = processor.preprocess(
            image_path=self.image_path,
        )

        processor.save_preview(
            preprocess_result,
        )

        source_image = preprocess_result.source_path

        print(
            f"[Pipeline] Source image: {source_image.name}"
        )
        if not (self.processed_dir / "preprocessed.png").exists():
            raise RuntimeError("PreProcessor failed to output preprocessed.png")

        print()

        # --------------------------------------------------------------
        # 2. Depth estimation
        # --------------------------------------------------------------

        print("[2/7] DEPTH ESTIMATION")
        print("-" * 60)

        estimator = DepthEstimator(
            model_path=(
                self.project_root
                / "models"
                / "depth_anything_v2_vits.pth"
            ),
            device=self.device,
            input_size=self.input_size,
        )

        depth = estimator.predict(
            image_path=source_image,
        )

        depth_npy, _ = estimator.save_outputs(
            depth,
            output_dir=self.processed_dir,
        )

        if not depth_npy.exists():
            raise RuntimeError(f"DepthEstimator failed to output {depth_npy}")

        print()

        # --------------------------------------------------------------
        # 3. Depth → relative height
        # --------------------------------------------------------------

        print("[3/7] DEPTH → HEIGHT")
        print("-" * 60)

        height_converter = DepthToHeight(
            project_root=self.project_root,
            depth_path=self.processed_dir / "depth.npy",
            output_dir=self.processed_dir,
            height_mode=self.height_mode,
        )

        height_npy = height_converter.process()

        if not height_npy.exists():
            raise RuntimeError(f"DepthToHeight failed to output {height_npy}")

        print()
        
        # --------------------------------------------------------------
        # 4. Structure Refinement
        # --------------------------------------------------------------

        print("[4/7] STRUCTURE REFINEMENT")
        print("-" * 60)

        refiner = StructureRefiner(
            project_root=self.project_root,
            height_path=self.processed_dir / "height.npy",
            image_path=source_image,
            output_dir=self.processed_dir,
            small_scale=self.small_scale,
            medium_scale=self.medium_scale,
            large_scale=self.large_scale,
            small_gain=self.small_gain,
            medium_gain=self.medium_gain,
            large_gain=self.large_gain,
            local_gain=self.local_gain,
            edge_strength=self.edge_strength,
            regularize=self.regularize,
            planar_fitting=self.planar_fitting,
        )

        refined_height_npy = refiner.process()

        if not refined_height_npy.exists():
            raise RuntimeError(f"StructureRefiner failed to output {refined_height_npy}")

        print()

        # --------------------------------------------------------------
        # 5. Terrain generation
        # --------------------------------------------------------------

        print("[5/7] TERRAIN GENERATION")
        print("-" * 60)

        terrain_generator = TerrainGenerator(
            project_root=self.project_root,
            height_path=self.processed_dir / "refined_height.npy",
            output_path=self.processed_dir / "terrain.glb",
            height_scale=self.height_scale,
            mesh_stride=self.mesh_stride,
        )

        terrain_glb = terrain_generator.generate()

        if not terrain_glb.exists():
            raise RuntimeError(f"TerrainGenerator failed to output {terrain_glb}")

        print()

        # --------------------------------------------------------------
        # 6. Texture enhancement
        # --------------------------------------------------------------

        print("[6/7] TEXTURE ENHANCEMENT")
        print("-" * 60)

        texture_enhancer = TextureEnhancer(
            project_root=self.project_root,
            image_path=source_image,
            output_path=self.processed_dir / "enhanced_texture.png",
            scale=self.texture_scale,
        )

        enhanced_png = texture_enhancer.process()

        if not enhanced_png.exists():
            raise RuntimeError(f"TextureEnhancer failed to output {enhanced_png}")

        print()

        # --------------------------------------------------------------
        # 7. Texture mapping
        # --------------------------------------------------------------

        print("[7/7] TEXTURE MAPPING")
        print("-" * 60)

        texture_mapper = TextureMapper(
            project_root=self.project_root,
            terrain_path=self.processed_dir / "terrain.glb",
            image_path=self.processed_dir / "enhanced_texture.png",
            output_path=self.processed_dir / "textured_terrain.glb",
        )

        textured_glb = texture_mapper.export()

        if not textured_glb.exists():
            raise RuntimeError(f"TextureMapper failed to output {textured_glb}")

        print()

        # --------------------------------------------------------------
        # Complete
        # --------------------------------------------------------------

        elapsed = time.perf_counter() - pipeline_start

        print("=" * 60)
        print("DEPTHWIZARD PIPELINE COMPLETE")
        print("=" * 60)

        print()
        print(f"Input:")
        print(f"  {source_image}")

        print()
        print("Outputs:")

        outputs = [
            self.processed_dir / "preprocessed.png",
            self.processed_dir / "depth.npy",
            self.processed_dir / "depth.png",
            self.processed_dir / "height.npy",
            self.processed_dir / "height.png",
            self.processed_dir / "refined_height.npy",
            self.processed_dir / "refined_height.png",
            self.processed_dir / "structure_confidence.png",
            self.processed_dir / "structure_edges.npy",
            self.processed_dir / "structure_edges.png",
            self.processed_dir / "structure_regions.json",
            self.processed_dir / "terrain.glb",
            self.processed_dir / "enhanced_texture.png",
            self.processed_dir / "textured_terrain.glb",
        ]

        for output in outputs:
            if output.exists():
                print(f"  ✓ {output}")
            else:
                print(f"  ✗ MISSING: {output}")

        print()
        print(
            f"Total pipeline time: {elapsed:.2f}s"
        )
        print()


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "DepthWizard V1.5 - "
            "Single-view RGB to textured 3D terrain"
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Optional input image. "
            "If omitted, the first supported image in input/ is used."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, or mps.",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Depth Anything V2 model input size.",
    )

    parser.add_argument(
        "--height-scale",
        type=float,
        default=10.0,
        help="Relative vertical terrain scale.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Mesh sampling stride.",
    )

    parser.add_argument(
        "--height-mode",
        choices=("direct", "inverse"),
        default="direct",
        help="Depth-to-height direction.",
    )

    parser.add_argument(
        "--texture-scale",
        type=int,
        choices=(1, 2, 4),
        default=2,
        help="RGB texture upscaling factor.",
    )
    
    parser.add_argument("--small-scale", type=float, default=2.0)
    parser.add_argument("--medium-scale", type=float, default=8.0)
    parser.add_argument("--large-scale", type=float, default=32.0)
    parser.add_argument("--small-gain", type=float, default=0.5)
    parser.add_argument("--medium-gain", type=float, default=1.1)
    parser.add_argument("--large-gain", type=float, default=1.0)
    parser.add_argument("--local-gain", type=float, default=1.2)
    parser.add_argument("--edge-strength", type=float, default=0.5)
    parser.add_argument("--no-regularize", action="store_true", help="Disable bilateral smoothing")
    parser.add_argument("--no-planar-fitting", action="store_true", help="Disable planar fitting")

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    pipeline = DepthWizardPipeline(
        image_path=args.image,
        device=args.device,
        input_size=args.input_size,
        height_scale=args.height_scale,
        mesh_stride=args.stride,
        height_mode=args.height_mode,
        texture_scale=args.texture_scale,
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

    try:
        pipeline.run()
    except Exception as e:
        print(f"\n[DepthWizard] ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

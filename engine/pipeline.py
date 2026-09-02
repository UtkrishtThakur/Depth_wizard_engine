"""
DepthWizard Engine - Pipeline Runner

Provides a callable run_pipeline() function for programmatic use by the API
worker, while preserving the existing CLI interface in engine/main.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from engine.pre_processor import PreProcessor
from engine.depth_estimator import DepthEstimator
from engine.depth_to_height import DepthToHeight
from engine.structure_refiner import StructureRefiner
from engine.terrain_generator import TerrainGenerator
from engine.texture_enhancer import TextureEnhancer
from engine.texture_mapper import TextureMapper


# Stage constants matching the API schema
STAGE_PREPROCESSING = "PREPROCESSING"
STAGE_DEPTH_ESTIMATION = "DEPTH_ESTIMATION"
STAGE_DEPTH_TO_HEIGHT = "DEPTH_TO_HEIGHT"
STAGE_STRUCTURE_REFINEMENT = "STRUCTURE_REFINEMENT"
STAGE_TERRAIN_GENERATION = "TERRAIN_GENERATION"
STAGE_TEXTURE_ENHANCEMENT = "TEXTURE_ENHANCEMENT"
STAGE_WALL_TEXTURE_GENERATION = "WALL_TEXTURE_GENERATION"
STAGE_TEXTURE_MAPPING = "TEXTURE_MAPPING"
STAGE_FINALIZING = "FINALIZING"
STAGE_COMPLETED = "COMPLETED"

ALL_STAGES = [
    STAGE_PREPROCESSING,
    STAGE_DEPTH_ESTIMATION,
    STAGE_DEPTH_TO_HEIGHT,
    STAGE_STRUCTURE_REFINEMENT,
    STAGE_TERRAIN_GENERATION,
    STAGE_TEXTURE_ENHANCEMENT,
    STAGE_WALL_TEXTURE_GENERATION,
    STAGE_TEXTURE_MAPPING,
    STAGE_FINALIZING,
    STAGE_COMPLETED,
]


@dataclass
class PipelineArtifact:
    """Describes a single output file produced by the pipeline."""
    artifact_type: str
    filename: str
    path: Path
    mime_type: str
    size_bytes: int


@dataclass
class PipelineResult:
    """Complete result of a pipeline run."""
    success: bool
    elapsed_seconds: float
    artifacts: list[PipelineArtifact] = field(default_factory=list)
    error: Optional[str] = None


# Type alias for the progress callback
ProgressCallback = Callable[[str, str, int], None]


def _noop_callback(stage: str, message: str, progress: int) -> None:
    """Default callback: prints to stdout like the original CLI."""
    print(f"[{stage}] {message}")


_ARTIFACT_MANIFEST = [
    ("PREPROCESSED_IMAGE", "preprocessed.png", "image/png"),
    ("DEPTH_NPY", "depth.npy", "application/octet-stream"),
    ("DEPTH_PNG", "depth.png", "image/png"),
    ("HEIGHT_NPY", "height.npy", "application/octet-stream"),
    ("HEIGHT_PNG", "height.png", "image/png"),
    ("REFINED_HEIGHT_NPY", "refined_height.npy", "application/octet-stream"),
    ("REFINED_HEIGHT_PNG", "refined_height.png", "image/png"),
    ("STRUCTURE_CONFIDENCE", "structure_confidence.png", "image/png"),
    ("STRUCTURE_EDGES", "structure_edges.npy", "application/octet-stream"),
    ("TERRAIN_GLB", "terrain.glb", "model/gltf-binary"),
    ("ENHANCED_TEXTURE", "enhanced_texture.png", "image/png"),
    ("TEXTURED_TERRAIN_GLB", "textured_terrain.glb", "model/gltf-binary"),
]


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    project_root: Path | None = None,
    device: str = "cpu",
    input_size: int = 518,
    height_scale: float = 10.0,
    mesh_stride: int = 2,
    height_mode: str = "direct",
    texture_scale: int = 2,
    small_scale: float = 2.0,
    medium_scale: float = 8.0,
    large_scale: float = 32.0,
    structure_gain: float = 1.3,
    edge_strength: float = 0.5,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """
    Execute the full DepthWizard pipeline.

    This is the single entry point used by both the CLI and the API worker.

    Args:
        input_path: Path to the source RGB image.
        output_dir: Directory where all outputs will be written.
        project_root: Project root (defaults to parent of engine/).
        device: Torch device string.
        progress_callback: Optional callback(stage, message, progress%).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    if progress_callback is None:
        progress_callback = _noop_callback

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path).resolve()
    pipeline_start = time.perf_counter()

    try:
        # ── 1. Preprocessing ─────────────────────────────────────────
        progress_callback(STAGE_PREPROCESSING, "Starting preprocessing", 0)

        processor = PreProcessor(project_root=project_root)
        preprocess_result = processor.preprocess(image_path=input_path)
        # Save preview to the process-specific output directory
        preview_path = output_dir / "preprocessed.png"
        preprocess_result.image.save(preview_path)

        source_image = preprocess_result.source_path

        progress_callback(STAGE_PREPROCESSING, f"Preprocessed {source_image.name}", 10)

        # ── 2. Depth estimation ──────────────────────────────────────
        progress_callback(STAGE_DEPTH_ESTIMATION, "Loading depth model", 12)

        estimator = DepthEstimator(
            model_path=project_root / "models" / "depth_anything_v2_vits.pth",
            device=device,
            input_size=input_size,
        )

        progress_callback(STAGE_DEPTH_ESTIMATION, "Running depth inference", 15)
        depth = estimator.predict(image_path=source_image)
        depth_npy, _ = estimator.save_outputs(depth, output_dir=output_dir)

        if not depth_npy.exists():
            raise RuntimeError("DepthEstimator failed to output depth.npy")

        progress_callback(STAGE_DEPTH_ESTIMATION, "Depth estimation complete", 25)

        # ── 3. Depth → Height ────────────────────────────────────────
        progress_callback(STAGE_DEPTH_TO_HEIGHT, "Converting depth to height", 27)

        height_converter = DepthToHeight(
            project_root=project_root,
            depth_path=output_dir / "depth.npy",
            output_dir=output_dir,
            height_mode=height_mode,
        )
        height_npy = height_converter.process()

        if not height_npy.exists():
            raise RuntimeError("DepthToHeight failed to output height.npy")

        progress_callback(STAGE_DEPTH_TO_HEIGHT, "Height conversion complete", 35)

        # ── 4. Structure refinement ──────────────────────────────────
        progress_callback(STAGE_STRUCTURE_REFINEMENT, "Refining structure", 37)

        refiner = StructureRefiner(
            project_root=project_root,
            height_path=output_dir / "height.npy",
            image_path=source_image,
            output_dir=output_dir,
            small_scale=small_scale,
            medium_scale=medium_scale,
            large_scale=large_scale,
            structure_gain=structure_gain,
            edge_strength=edge_strength,
        )
        refined_height_npy = refiner.process()

        if not refined_height_npy.exists():
            raise RuntimeError("StructureRefiner failed")

        progress_callback(STAGE_STRUCTURE_REFINEMENT, "Structure refinement complete", 50)

        # ── 5. Terrain generation ────────────────────────────────────
        progress_callback(STAGE_TERRAIN_GENERATION, "Generating terrain mesh", 52)

        terrain_gen = TerrainGenerator(
            project_root=project_root,
            height_path=output_dir / "refined_height.npy",
            output_path=output_dir / "terrain.glb",
            height_scale=height_scale,
            mesh_stride=mesh_stride,
            world_depth=100.0,
        )
        terrain_gen.generate()

        progress_callback(STAGE_TERRAIN_GENERATION, "Terrain generation complete", 65)

        # ── 6. Texture enhancement ───────────────────────────────────
        progress_callback(STAGE_TEXTURE_ENHANCEMENT, "Enhancing texture", 67)

        texture_enhancer = TextureEnhancer(
            project_root=project_root,
            image_path=source_image,
            output_path=output_dir / "enhanced_texture.png",
            scale=texture_scale,
        )
        enhanced_png = texture_enhancer.process()

        if not enhanced_png.exists():
            raise RuntimeError("TextureEnhancer failed")

        progress_callback(STAGE_TEXTURE_ENHANCEMENT, "Texture enhancement complete", 75)

        # ── 7. Wall texture generation (optional, best-effort) ───────
        progress_callback(STAGE_WALL_TEXTURE_GENERATION, "Generating wall textures", 77)
        try:
            from engine.wall_texture_generator import WallTextureGenerator
            wall_gen = WallTextureGenerator(
                project_root=project_root,
                terrain_path=output_dir / "terrain.glb",
                image_path=output_dir / "enhanced_texture.png",
                device=device,
                wall_mode="lama",
            )
            wall_gen.generate()
            progress_callback(STAGE_WALL_TEXTURE_GENERATION, "Wall texture generation complete", 85)
        except Exception as wall_err:
            progress_callback(STAGE_WALL_TEXTURE_GENERATION, f"Wall textures skipped: {wall_err}", 85)

        # ── 8. Texture mapping ───────────────────────────────────────
        progress_callback(STAGE_TEXTURE_MAPPING, "Mapping texture onto terrain", 87)

        texture_mapper = TextureMapper(
            project_root=project_root,
            terrain_path=output_dir / "terrain.glb",
            image_path=output_dir / "enhanced_texture.png",
            output_path=output_dir / "textured_terrain.glb",
        )
        textured_glb = texture_mapper.export()

        if not textured_glb.exists():
            raise RuntimeError("TextureMapper failed to produce textured_terrain.glb")

        progress_callback(STAGE_TEXTURE_MAPPING, "Texture mapping complete", 95)

        # ── 9. Finalize ──────────────────────────────────────────────
        progress_callback(STAGE_FINALIZING, "Collecting artifacts", 97)

        elapsed = time.perf_counter() - pipeline_start
        artifacts: list[PipelineArtifact] = []

        for atype, fname, mime in _ARTIFACT_MANIFEST:
            fpath = output_dir / fname
            if fpath.exists():
                artifacts.append(PipelineArtifact(
                    artifact_type=atype,
                    filename=fname,
                    path=fpath,
                    mime_type=mime,
                    size_bytes=fpath.stat().st_size,
                ))

        progress_callback(STAGE_COMPLETED, f"Pipeline complete in {elapsed:.2f}s", 100)

        return PipelineResult(success=True, elapsed_seconds=elapsed, artifacts=artifacts)

    except Exception as exc:
        elapsed = time.perf_counter() - pipeline_start
        return PipelineResult(success=False, elapsed_seconds=elapsed, error=str(exc))

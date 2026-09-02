"""
DepthWizard API - Generation Service

Bridges between the API/worker and the engine pipeline.
Calls engine.pipeline.run_pipeline() with proper callbacks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from engine.pipeline import run_pipeline, PipelineResult, ProgressCallback


def execute_pipeline(
    input_path: Path,
    output_dir: Path,
    project_root: Path,
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """
    Execute the DepthWizard engine pipeline.

    This is the only bridge between the API and the engine.
    The engine code is never duplicated here.
    """
    return run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        project_root=project_root,
        device=device,
        progress_callback=progress_callback,
    )

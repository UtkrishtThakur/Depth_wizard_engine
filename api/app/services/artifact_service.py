"""
DepthWizard API - Artifact Service

Resolves artifact file paths for serving.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import PROCESS_DIR
from app.db.models import Artifact, Process


def get_model_path(db: Session, process_id: UUID) -> Path | None:
    """
    Resolve the path to textured_terrain.glb for a specific process.

    Returns None if the process doesn't exist, isn't complete,
    or the file doesn't exist on disk.
    """
    process = db.get(Process, process_id)
    if process is None:
        return None

    # Resolve from the process-specific output directory
    output_dir = PROCESS_DIR / str(process_id) / "output"
    glb_path = output_dir / "textured_terrain.glb"

    if glb_path.exists() and glb_path.stat().st_size > 0:
        return glb_path

    return None


def get_artifact_path(db: Session, process_id: UUID, artifact_id: UUID) -> Path | None:
    """Resolve the filesystem path for a specific artifact."""
    artifact = db.get(Artifact, artifact_id)
    if artifact is None or artifact.process_id != process_id:
        return None

    fpath = Path(artifact.path)
    if fpath.exists():
        return fpath

    return None

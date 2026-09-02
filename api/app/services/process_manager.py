"""
DepthWizard API - Process Manager Service

Handles CRUD for processes and orchestrates the lifecycle.
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import PROCESS_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from app.db.models import Artifact, Process, ProcessLog, ProcessStatus, LogLevel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def validate_image(data: bytes, filename: str) -> str:
    """Validate uploaded image data. Returns sanitized filename or raises."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError(f"File too large ({len(data)} bytes, max {max_bytes})")

    # Quick magic-byte validation
    if data[:2] == b'\xff\xd8':
        pass  # JPEG
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        pass  # PNG
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        pass  # WebP
    elif data[:4] in (b'II\x2a\x00', b'MM\x00\x2a'):
        pass  # TIFF
    else:
        raise ValueError("File content does not match a supported image format")

    return filename


def create_process(
    db: Session,
    image_data: bytes,
    original_filename: str,
    requested_by: Optional[str] = None,
    delivery_target: Optional[str] = None,
) -> Process:
    """Create a new process: save input, insert DB row, return Process."""
    validated_name = validate_image(image_data, original_filename)

    process_id = uuid.uuid4()
    process_dir = PROCESS_DIR / str(process_id)
    input_dir = process_dir / "input"
    output_dir = process_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(validated_name).suffix.lower()
    stored_filename = f"source{ext}"
    input_path = input_dir / stored_filename
    input_path.write_bytes(image_data)

    process = Process(
        id=process_id,
        status=ProcessStatus.QUEUED,
        progress=0,
        input_filename=original_filename,
        input_path=str(input_path),
        requested_by=requested_by,
        delivery_target=delivery_target,
    )
    db.add(process)

    log_entry = ProcessLog(
        process_id=process_id,
        level=LogLevel.INFO,
        stage=None,
        message=f"Process created for {original_filename}",
        progress=0,
    )
    db.add(log_entry)

    db.commit()
    db.refresh(process)
    return process


def get_process(db: Session, process_id: uuid.UUID) -> Optional[Process]:
    """Retrieve a process by UUID."""
    return db.get(Process, process_id)


def get_process_artifacts(db: Session, process_id: uuid.UUID) -> list[Artifact]:
    """Get all artifacts for a process."""
    stmt = select(Artifact).where(Artifact.process_id == process_id).order_by(Artifact.created_at)
    return list(db.scalars(stmt).all())


def get_process_logs(db: Session, process_id: uuid.UUID) -> list[ProcessLog]:
    """Get all logs for a process."""
    stmt = select(ProcessLog).where(ProcessLog.process_id == process_id).order_by(ProcessLog.timestamp)
    return list(db.scalars(stmt).all())


def claim_next_queued(db: Session) -> Optional[Process]:
    """
    Atomically claim the next QUEUED process.

    Uses SELECT ... FOR UPDATE SKIP LOCKED to prevent two workers
    from claiming the same job.
    """
    stmt = (
        select(Process)
        .where(Process.status == ProcessStatus.QUEUED)
        .order_by(Process.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    process = db.scalars(stmt).first()
    if process is None:
        return None

    process.status = ProcessStatus.PROCESSING
    process.started_at = _utcnow()
    db.commit()
    db.refresh(process)
    return process


def mark_completed(db: Session, process: Process) -> None:
    """Mark a process as completed."""
    process.status = ProcessStatus.COMPLETED
    process.progress = 100
    process.current_stage = "COMPLETED"
    process.completed_at = _utcnow()
    db.commit()


def mark_failed(db: Session, process: Process, error_message: str) -> None:
    """Mark a process as failed."""
    process.status = ProcessStatus.FAILED
    process.error_message = error_message
    process.failed_at = _utcnow()
    db.commit()


def update_progress(
    db: Session,
    process: Process,
    stage: str,
    message: str,
    progress: int,
) -> None:
    """Update process progress and add a log entry."""
    process.current_stage = stage
    process.progress = progress

    log_entry = ProcessLog(
        process_id=process.id,
        level=LogLevel.INFO,
        stage=stage,
        message=message,
        progress=progress,
    )
    db.add(log_entry)
    db.commit()


def register_artifacts(db: Session, process: Process, artifacts: list[dict]) -> None:
    """Register pipeline output artifacts in the database."""
    for art in artifacts:
        db_artifact = Artifact(
            process_id=process.id,
            artifact_type=art["artifact_type"],
            filename=art["filename"],
            path=art["path"],
            mime_type=art["mime_type"],
            size_bytes=art["size_bytes"],
        )
        db.add(db_artifact)
    db.commit()

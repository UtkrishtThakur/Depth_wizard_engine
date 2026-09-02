"""
DepthWizard API - Background Worker

Polls for QUEUED processes and executes the engine pipeline.

Run as a standalone process:
    python -m app.services.worker
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from app.config import (
    PROCESS_DIR,
    PROJECT_ROOT,
    DEVICE,
    WORKER_POLL_INTERVAL,
)
from app.db.database import SessionLocal
from app.db.models import ProcessStatus, LogLevel, ProcessLog
from app.services import process_manager
from app.services.generation_service import execute_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WORKER] %(levelname)s %(message)s",
)
logger = logging.getLogger("worker")

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Received shutdown signal, finishing current job...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def process_one_job() -> bool:
    """
    Try to claim and process one queued job.

    Returns True if a job was processed, False if the queue was empty.
    """
    db = SessionLocal()
    try:
        process = process_manager.claim_next_queued(db)
        if process is None:
            return False

        process_id = process.id
        logger.info(f"Claimed process {process_id}")

        input_path = Path(process.input_path)
        output_dir = PROCESS_DIR / str(process_id) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        def progress_callback(stage: str, message: str, progress: int):
            """Called by the engine to report progress."""
            nonlocal db, process
            try:
                # Re-attach to session if needed
                if process not in db:
                    db.expire_all()
                    process = db.get(process.__class__, process_id)

                process_manager.update_progress(db, process, stage, message, progress)
                logger.info(f"[{process_id}] {stage}: {message} ({progress}%)")
            except Exception as e:
                logger.warning(f"Failed to persist progress: {e}")

        try:
            result = execute_pipeline(
                input_path=input_path,
                output_dir=output_dir,
                project_root=PROJECT_ROOT,
                device=DEVICE,
                progress_callback=progress_callback,
            )

            # Refresh the process from DB
            db.expire_all()
            process = db.get(process.__class__, process_id)

            if result.success:
                # Verify the final GLB exists
                glb_path = output_dir / "textured_terrain.glb"
                if not glb_path.exists() or glb_path.stat().st_size == 0:
                    process_manager.mark_failed(db, process, "Final GLB not produced")
                    logger.error(f"[{process_id}] GLB missing after pipeline")
                    return True

                # Register artifacts
                artifact_dicts = [
                    {
                        "artifact_type": a.artifact_type,
                        "filename": a.filename,
                        "path": str(a.path),
                        "mime_type": a.mime_type,
                        "size_bytes": a.size_bytes,
                    }
                    for a in result.artifacts
                ]
                process_manager.register_artifacts(db, process, artifact_dicts)
                process_manager.mark_completed(db, process)
                logger.info(f"[{process_id}] COMPLETED in {result.elapsed_seconds:.2f}s")
            else:
                process_manager.mark_failed(db, process, result.error or "Unknown error")
                logger.error(f"[{process_id}] FAILED: {result.error}")

        except Exception as e:
            logger.exception(f"[{process_id}] Pipeline exception")
            db.expire_all()
            process = db.get(process.__class__, process_id)
            if process:
                # Add error log
                error_log = ProcessLog(
                    process_id=process_id,
                    level=LogLevel.ERROR,
                    stage=process.current_stage,
                    message=str(e),
                )
                db.add(error_log)
                process_manager.mark_failed(db, process, str(e))

        return True

    finally:
        db.close()


def run_worker_loop():
    """Main worker loop: poll for jobs until shutdown."""
    logger.info("Worker started, polling for jobs...")

    while not _shutdown:
        try:
            had_job = process_one_job()
            if not had_job:
                time.sleep(WORKER_POLL_INTERVAL)
        except Exception:
            logger.exception("Worker loop error")
            time.sleep(WORKER_POLL_INTERVAL * 2)

    logger.info("Worker shutting down.")


if __name__ == "__main__":
    run_worker_loop()

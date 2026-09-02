"""
DepthWizard API - Process endpoints.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.process import (
    ArtifactResponse,
    ArtifactsListResponse,
    ProcessCreateResponse,
    ProcessResponse,
)
from app.schemas.logs import LogEntry, LogsResponse
from app.services import process_manager
from app.services.artifact_service import get_model_path
from app.services.logging_service import stream_process_logs

router = APIRouter()


@router.post("/process", status_code=202, response_model=ProcessCreateResponse)
async def create_process(
    image: UploadFile = File(...),
    requested_by: str | None = Form(None),
    delivery_target: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload an image and create a new generation process.

    Returns immediately with process UUID. The worker will pick up the job.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    data = await image.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        process = process_manager.create_process(
            db=db,
            image_data=data,
            original_filename=image.filename,
            requested_by=requested_by,
            delivery_target=delivery_target,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ProcessCreateResponse(
        success=True,
        process=ProcessResponse.model_validate(process),
    )


@router.get("/process/{process_id}", response_model=ProcessResponse)
def get_process(process_id: UUID, db: Session = Depends(get_db)):
    """Get the current status of a process."""
    process = process_manager.get_process(db, process_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Process not found")
    return ProcessResponse.model_validate(process)


@router.get("/process/{process_id}/logs", response_model=LogsResponse)
def get_process_logs(process_id: UUID, db: Session = Depends(get_db)):
    """Get all historical logs for a process."""
    process = process_manager.get_process(db, process_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Process not found")

    logs = process_manager.get_process_logs(db, process_id)
    return LogsResponse(
        process_id=process_id,
        logs=[LogEntry.model_validate(l) for l in logs],
    )


@router.get("/process/{process_id}/logs/stream")
async def stream_logs(process_id: UUID, db: Session = Depends(get_db)):
    """SSE stream of real-time process logs."""
    process = process_manager.get_process(db, process_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Process not found")

    return StreamingResponse(
        stream_process_logs(process_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/process/{process_id}/artifacts", response_model=ArtifactsListResponse)
def get_artifacts(process_id: UUID, db: Session = Depends(get_db)):
    """List all artifacts for a process."""
    process = process_manager.get_process(db, process_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Process not found")

    artifacts = process_manager.get_process_artifacts(db, process_id)
    artifact_responses = []
    for a in artifacts:
        resp = ArtifactResponse.model_validate(a)
        resp.url = f"/api/v1/process/{process_id}/artifacts/{a.id}"
        artifact_responses.append(resp)

    return ArtifactsListResponse(process_id=process_id, artifacts=artifact_responses)


@router.get("/process/{process_id}/model")
def get_model(process_id: UUID, db: Session = Depends(get_db)):
    """
    Serve the final textured_terrain.glb for a specific process.

    Resolves path from this process's UUID — never from a shared directory.
    """
    glb_path = get_model_path(db, process_id)
    if glb_path is None:
        raise HTTPException(
            status_code=404,
            detail="Model not available. Process may still be running or failed.",
        )

    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename="textured_terrain.glb",
    )

"""
DepthWizard API - Pydantic schemas for processes and artifacts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


# ── Process ──────────────────────────────────────────────────────────

class ProcessCreate(BaseModel):
    requested_by: Optional[str] = None
    delivery_target: Optional[str] = None


class ProcessResponse(BaseModel):
    id: UUID
    status: str
    progress: int
    current_stage: Optional[str] = None
    input_filename: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    requested_by: Optional[str] = None
    delivery_target: Optional[str] = None

    model_config = {"from_attributes": True}


class ProcessCreateResponse(BaseModel):
    success: bool
    process: ProcessResponse


# ── Artifact ─────────────────────────────────────────────────────────

class ArtifactResponse(BaseModel):
    id: UUID
    process_id: UUID
    artifact_type: str
    filename: str
    mime_type: str
    size_bytes: int
    created_at: datetime
    url: Optional[str] = None

    model_config = {"from_attributes": True}


class ArtifactsListResponse(BaseModel):
    process_id: UUID
    artifacts: list[ArtifactResponse]

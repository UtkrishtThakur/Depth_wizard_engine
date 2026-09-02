"""
DepthWizard API - Pydantic schemas for process logs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class LogEntry(BaseModel):
    id: UUID
    timestamp: datetime
    level: str
    stage: Optional[str] = None
    message: str
    progress: Optional[int] = None

    model_config = {"from_attributes": True}


class LogsResponse(BaseModel):
    process_id: UUID
    logs: list[LogEntry]

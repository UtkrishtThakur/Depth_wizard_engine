"""
DepthWizard API - SQLAlchemy ORM models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


# ── Enum values ──────────────────────────────────────────────────────

class ProcessStatus:
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProcessStage:
    PREPROCESSING = "PREPROCESSING"
    DEPTH_ESTIMATION = "DEPTH_ESTIMATION"
    DEPTH_TO_HEIGHT = "DEPTH_TO_HEIGHT"
    STRUCTURE_REFINEMENT = "STRUCTURE_REFINEMENT"
    TERRAIN_GENERATION = "TERRAIN_GENERATION"
    TEXTURE_ENHANCEMENT = "TEXTURE_ENHANCEMENT"
    WALL_TEXTURE_GENERATION = "WALL_TEXTURE_GENERATION"
    TEXTURE_MAPPING = "TEXTURE_MAPPING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"


class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


PROCESS_STATUS_ENUM = SAEnum(
    "QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED",
    name="process_status",
    create_constraint=True,
)

LOG_LEVEL_ENUM = SAEnum(
    "DEBUG", "INFO", "WARNING", "ERROR",
    name="log_level",
    create_constraint=True,
)


# ── Models ───────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Process(Base):
    __tablename__ = "processes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(PROCESS_STATUS_ENUM, nullable=False, default=ProcessStatus.QUEUED)
    progress = Column(Integer, nullable=False, default=0)
    current_stage = Column(String(64), nullable=True)

    input_filename = Column(String(256), nullable=False)
    input_path = Column(Text, nullable=False)

    requested_by = Column(String(256), nullable=True)
    delivery_target = Column(String(512), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)

    error_message = Column(Text, nullable=True)

    artifacts = relationship("Artifact", back_populates="process", cascade="all, delete-orphan")
    logs = relationship("ProcessLog", back_populates="process", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    process_id = Column(UUID(as_uuid=True), ForeignKey("processes.id", ondelete="CASCADE"), nullable=False)
    artifact_type = Column(String(64), nullable=False)
    filename = Column(String(256), nullable=False)
    path = Column(Text, nullable=False)
    mime_type = Column(String(128), nullable=False)
    size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    process = relationship("Process", back_populates="artifacts")


class ProcessLog(Base):
    __tablename__ = "process_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    process_id = Column(UUID(as_uuid=True), ForeignKey("processes.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    level = Column(LOG_LEVEL_ENUM, nullable=False, default=LogLevel.INFO)
    stage = Column(String(64), nullable=True)
    message = Column(Text, nullable=False)
    progress = Column(Integer, nullable=True)

    process = relationship("Process", back_populates="logs")

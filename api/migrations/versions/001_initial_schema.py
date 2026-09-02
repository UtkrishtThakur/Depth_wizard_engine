"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-09-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    process_status = sa.Enum(
        "QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED",
        name="process_status",
    )
    log_level = sa.Enum(
        "DEBUG", "INFO", "WARNING", "ERROR",
        name="log_level",
    )
    process_status.create(op.get_bind(), checkfirst=True)
    log_level.create(op.get_bind(), checkfirst=True)

    # processes
    op.create_table(
        "processes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("status", process_status, nullable=False, server_default="QUEUED"),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("current_stage", sa.String(64), nullable=True),
        sa.Column("input_filename", sa.String(256), nullable=False),
        sa.Column("input_path", sa.Text, nullable=False),
        sa.Column("requested_by", sa.String(256), nullable=True),
        sa.Column("delivery_target", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )

    # artifacts
    op.create_table(
        "artifacts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("process_id", UUID(as_uuid=True), sa.ForeignKey("processes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("artifact_type", sa.String(64), nullable=False),
        sa.Column("filename", sa.String(256), nullable=False),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_artifacts_process_id", "artifacts", ["process_id"])

    # process_logs
    op.create_table(
        "process_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("process_id", UUID(as_uuid=True), sa.ForeignKey("processes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("level", log_level, nullable=False, server_default="INFO"),
        sa.Column("stage", sa.String(64), nullable=True),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("progress", sa.Integer, nullable=True),
    )
    op.create_index("ix_process_logs_process_id", "process_logs", ["process_id"])


def downgrade() -> None:
    op.drop_table("process_logs")
    op.drop_table("artifacts")
    op.drop_table("processes")
    sa.Enum(name="process_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="log_level").drop(op.get_bind(), checkfirst=True)

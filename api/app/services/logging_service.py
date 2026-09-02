"""
DepthWizard API - Logging Service

Provides SSE streaming for real-time process logs.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import ProcessLog, Process, ProcessStatus


async def stream_process_logs(
    process_id: UUID,
    poll_interval: float = 1.0,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE-formatted log events for a process.

    Polls the database for new log entries and yields them.
    Terminates when the process reaches a terminal state.
    """
    last_seen_count = 0

    while True:
        db = SessionLocal()
        try:
            # Check process status
            process = db.get(Process, process_id)
            if process is None:
                yield _sse_event("error", {"message": "Process not found"})
                return

            # Fetch new logs
            stmt = (
                select(ProcessLog)
                .where(ProcessLog.process_id == process_id)
                .order_by(ProcessLog.timestamp)
            )
            all_logs = list(db.scalars(stmt).all())

            # Emit only new ones
            new_logs = all_logs[last_seen_count:]
            for log in new_logs:
                event_data = {
                    "stage": log.stage,
                    "message": log.message,
                    "progress": log.progress,
                    "level": log.level,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                }
                yield _sse_event("log", event_data)

            last_seen_count = len(all_logs)

            # Check terminal state
            if process.status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED, ProcessStatus.CANCELLED):
                event_data = {
                    "process_id": str(process_id),
                    "status": process.status,
                }
                if process.status == ProcessStatus.COMPLETED:
                    yield _sse_event("complete", event_data)
                elif process.status == ProcessStatus.FAILED:
                    event_data["error"] = process.error_message
                    yield _sse_event("error", event_data)
                else:
                    yield _sse_event("cancelled", event_data)
                return

        finally:
            db.close()

        await asyncio.sleep(poll_interval)


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event."""
    json_str = json.dumps(data)
    return f"event: {event_type}\ndata: {json_str}\n\n"

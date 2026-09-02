"""
DepthWizard API - FastAPI Application

Serves the HTTP API. Does NOT load ML models.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.db.database import engine, Base
from app.api.health import router as health_router
from app.api.processes import router as processes_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup (Alembic is primary, this is fallback)."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="DepthWizard API",
    description="Single-view RGB to 3D terrain generation API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(processes_router, prefix="/api/v1", tags=["processes"])


@app.get("/")
def root():
    return {"service": "DepthWizard API", "version": "1.0.0"}

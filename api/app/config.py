"""
DepthWizard API - Configuration

All settings are driven by environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from pathlib import Path


# ── Database ─────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://depthwizard:depthwizard@postgres:5432/depthwizard",
)

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", "/app"))
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "/app/data"))
PROCESS_DIR: Path = DATA_DIR / "processes"
MODEL_ROOT: Path = Path(os.getenv("MODEL_ROOT", "/app/models"))
DEPTH_MODEL_PATH: Path = Path(os.getenv(
    "DEPTH_MODEL_PATH",
    str(MODEL_ROOT / "depth_anything_v2_vits.pth"),
))
LAMA_MODEL_PATH: Path = Path(os.getenv(
    "LAMA_MODEL_PATH",
    str(MODEL_ROOT / "lama"),
))

# ── Worker ───────────────────────────────────────────────────────────
MAX_CONCURRENT_GENERATIONS: int = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1"))
WORKER_POLL_INTERVAL: float = float(os.getenv("WORKER_POLL_INTERVAL", "2.0"))

# ── CORS ─────────────────────────────────────────────────────────────
_raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS: list[str] = [
    origin.strip().strip("\"'").rstrip("/")
    for origin in _raw_cors.split(",")
    if origin.strip()
]

# ── Device ───────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cpu")

# ── Upload ───────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

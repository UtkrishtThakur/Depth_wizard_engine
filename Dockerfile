FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api

WORKDIR /app

# Runtime/system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy only the application/runtime content needed in production
COPY api /app/api
COPY engine /app/engine
COPY models /app/models

# Runtime data directory
RUN mkdir -p /app/data

EXPOSE 8000

# Production API default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
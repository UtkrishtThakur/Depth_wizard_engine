FROM python:3.11-slim

# Optimize Python execution and prevent cache files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ensure /app and /app/engine are on PYTHONPATH so local modules can be imported
ENV PYTHONPATH=/app:/app/engine

WORKDIR /app

# Only copy requirements for cache-efficient dependency installation layer
COPY requirements.txt .

# Install dependencies (CPU-optimized versions)
RUN pip install --no-cache-dir -r requirements.txt

# Keep the development container alive without executing a specific application
CMD ["sleep", "infinity"]

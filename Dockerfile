# Multi-stage production container for Plant Disease Classification MLOps
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY configs/ configs/
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
COPY plant_disease_resnet50.pth .

# Expose API and Streamlit UI ports
EXPOSE 8000 8501

# Healthcheck targeting the FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default launch command: FastAPI production server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

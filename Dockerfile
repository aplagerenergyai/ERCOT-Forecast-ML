# ============================================================================
# ERCOT ML Pipeline - Production Container
# 
# Multi-purpose container for:
# - Feature engineering (build_features.py)
# - Model training (train_lgbm.py, train_xgb.py, train_deep.py)
# - Real-time inference (score.py via Uvicorn)
# - Batch inference
# 
# Supports: Azure ML, Azure Container Apps, GitHub Actions, Docker Desktop
# ============================================================================

FROM python:3.10-slim as base

# Metadata
LABEL maintainer="ERCOT ML Team"
LABEL description="ERCOT DART Spread Prediction - Training & Inference"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including ODBC drivers for SQL Server
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    curl \
    ca-certificates \
    gnupg2 \
    unixodbc \
    unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy application code
COPY build_features.py .
COPY dataloader.py .
COPY metrics.py .
COPY train_lgbm.py .
COPY train_xgb.py .
COPY train_deep.py .
COPY score.py .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/outputs

# Set default environment variables
ENV MODEL_TYPE=lgbm \
    MODEL_PATH=/app/models/lgbm \
    DATA_PATH=/app/data \
    LOG_LEVEL=INFO \
    LOG_DIR=/app/logs \
    PORT=5001

# Expose port for inference
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Default entrypoint (can be overridden)
CMD ["uvicorn", "score:app", "--host", "0.0.0.0", "--port", "5001"]


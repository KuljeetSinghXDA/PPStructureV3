# syntax=docker/dockerfile:1.7
FROM --platform=linux/arm64 python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Default CPU thread tuning; can be overridden in .env
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

# System deps for OpenCV/headless rendering and common C libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin exact framework/tooling versions for PP-StructureV3
# paddlepaddle 3.2.0 (CPU, manylinux aarch64) and paddleocr[doc-parser] 3.2.0
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY app /app/app

# Model cache (persisted by a named volume in compose)
ENV PADDLE_MODEL_HOME=/root/.paddlex

# Default runtime env (overridden by .env via compose)
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8080

EXPOSE 8080

# Uvicorn entrypoint
CMD ["bash", "-lc", "uvicorn app.main:app --host ${APP_HOST} --port ${APP_PORT}"]

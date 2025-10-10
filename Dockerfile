# syntax=docker/dockerfile:1.7
FROM --platform=linux/arm64 python:3.12-slim-bookworm

# Threading and runtime defaults (see .env to override)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    ENABLE_MKLDNN=True

# Base system deps for build + runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build patchelf \
    libopenblas-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Upgrade pip toolchain early
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# ---------- Build PaddlePaddle (CPU) from source ----------
# Follow official Linux source build: clone -> cmake (CPU) -> make -> install wheel
# Reference: compile with -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
RUN git clone https://github.com/PaddlePaddle/Paddle.git && \
    cd Paddle && \
    git checkout v3.2.0 && \
    mkdir -p build && cd build && \
    cmake .. -DPY_VERSION=3.12 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" && \
    python -m pip install --no-cache-dir python/dist/*.whl

# ---------- Install PaddleOCR (PP-StructureV3) from source ----------
RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git && \
    cd PaddleOCR && \
    git checkout v3.2.0 && \
    python -m pip install --no-cache-dir -e .[doc-parser]

# App dependencies (server only; core OCR deps come via PaddleOCR extras)
COPY requirements.txt /opt/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements.txt

# App code
WORKDIR /app
COPY app /app/app

# Model cache (persisted by volume)
ENV PADDLE_MODEL_HOME=/root/.paddlex

# Default host/port; Dokploy maps domain and container port
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8080

EXPOSE 8080

CMD ["bash", "-lc", "uvicorn app.main:app --host ${APP_HOST} --port ${APP_PORT}"]

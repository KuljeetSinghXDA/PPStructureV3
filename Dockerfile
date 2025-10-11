# ===== Stage 1: Build PaddlePaddle (CPU-only, ARM64) =====
FROM python:3.11-slim AS paddle-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build patchelf pkg-config \
    protobuf-compiler libprotobuf-dev \
    python3-dev libopenblas-dev liblapack-dev gfortran \
    ca-certificates wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /paddle
RUN git clone --depth 1 https://github.com/PaddlePaddle/Paddle.git . && \
    git submodule update --init --recursive

# Python build deps (per Paddle’s Linux source build guidance)
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir -r /paddle/python/requirements.txt && \
    python -m pip install --no-cache-dir "protobuf==3.20.2"

WORKDIR /paddle/build
# CPU

# ===== Stage 1: Build PaddlePaddle (CPU-only, ARM64) =====
FROM python:3.11-slim AS paddle-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build patchelf \
    protobuf-compiler libprotobuf-dev \
    libopenblas-dev liblapack-dev gfortran \
    ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*  # Toolchain + OpenBLAS/LAPACK for CPU builds [web:121][web:122]

# Get Paddle source (use a release tag if desired)
WORKDIR /paddle
RUN git clone --depth 1 https://github.com/PaddlePaddle/Paddle.git . && git submodule update --init --recursive  # Source per official docs [web:121][web:122]

# Configure CMake: CPU only, no MKLDNN/oneDNN, build Python wheel
# Notes:
# - ON_INFER=ON builds inference libs with the wheel; WITH_TESTING=OFF speeds up and reduces memory
# - PY_VERSION must match the Python in this container (3.11 here)
WORKDIR /paddle/build
RUN cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_GPU=OFF -DWITH_IPU=OFF -DWITH_XPU=OFF -DWITH_TENSORRT=OFF \
  -DWITH_MKL=OFF -DWITH_SYSTEM_BLAS=ON -DWITH_OPENBLAS=ON \
  -DWITH_MKLDNN=OFF -DWITH_ONEDNN=OFF \
  -DWITH_PYTHON=ON -DPY_VERSION=3.11 \
  -DWITH_TESTING=OFF -DON_INFER=ON  # CPU build flags from official guidance and known stable switches [web:122][web:47][web:126]

# Build (adjust -j to avoid OOM on small VMs)
ARG BUILD_JOBS=2
RUN ninja -j${BUILD_JOBS} && ls -lah /paddle/build/python/dist  # Parallel build; wheel lands under python/dist [web:122]

# Export wheel
RUN mkdir -p /wheel && cp -v /paddle/build/python/dist/*whl /wheel/  # Wheel output per docs [web:122]

# ===== Stage 2: Runtime with compiled Paddle + PaddleOCR API =====
FROM python:3.11-slim

# Minimal libs used by CV backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 && \
    rm -rf /var/lib/apt/lists/*  # Common runtime libs for OCR stacks [web:55]

# Global stability flags: disable MKLDNN and cap thread counts
ENV FLAGS_use_mkldnn=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # Disable MKLDNN and cap threads at runtime [web:47][web:93]

# Install compiled Paddle and server deps
COPY --from=paddle-builder /wheel /tmp/wheel
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir /tmp/wheel/*.whl && \
    python -m pip install --no-cache-dir "paddleocr[doc-parser]" fastapi uvicorn[standard] python-multipart  # PaddleOCR + API stack [web:55]

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]  # Single worker to avoid multi-process native conflicts [web:121]

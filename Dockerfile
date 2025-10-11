# ===== Stage 1: Build PaddlePaddle (CPU-only, ARM64) =====
FROM python:3.11-slim AS paddle-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build patchelf pkg-config \
    protobuf-compiler libprotobuf-dev \
    python3-dev libopenblas-dev liblapack-dev gfortran ca-certificates wget unzip && \
    rm -rf /var/lib/apt/lists/*  # Toolchain + Python headers + BLAS/LAPACK + protobuf [web:121][web:122]

WORKDIR /paddle
RUN git clone --depth 1 https://github.com/PaddlePaddle/Paddle.git . && git submodule update --init --recursive  # Source + third-party deps [web:121][web:122]

# Python-side build requirements (as per docs) and a compatible protobuf on Py side
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir -r /paddle/python/requirements.txt && \
    python -m pip install --no-cache-dir "protobuf==3.20.2"  # Version used in docs for stable builds [web:121]

WORKDIR /paddle/build
# CPU-only, ARM-friendly settings: no MKL/MKLDNN/oneDNN, OpenBLAS enabled, inference-on, testing-off
RUN cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_GPU=OFF -DWITH_IPU=OFF -DWITH_XPU=OFF -DWITH_TENSORRT=OFF \
  -DWITH_MKL=OFF -DWITH_SYSTEM_BLAS=ON -DWITH_OPENBLAS=ON \
  -DWITH_MKLDNN=OFF -DWITH_ONEDNN=OFF \
  -DWITH_ARM=ON -DWITH_AVX=OFF \
  -DWITH_PYTHON=ON -DPY_VERSION=3.11 \
  -DWITH_TESTING=OFF -DON_INFER=ON  # CPU build per official guidance [web:122][web:155]

# Build (keep parallelism low on small A1 shapes)
ARG BUILD_JOBS=2
RUN ninja -j${BUILD_JOBS} && ls -lah /paddle/build/python/dist  # Wheel under python/dist [web:122]

RUN mkdir -p /wheel && cp -v /paddle/build/python/dist/*whl /wheel/  # Export wheel [web:122]

# ===== Stage 2: Runtime with compiled Paddle + PaddleOCR API =====
FROM python:3.11-slim

# Runtime libs for OCR backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 && \
    rm -rf /var/lib/apt/lists/*  # Common runtime libs [web:55]

# Global stability flags: fully disable MKLDNN and cap threads
ENV FLAGS_use_mkldnn=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # Stability on ARM64 [web:47][web:93][web:97]

COPY --from=paddle-builder /wheel /tmp/wheel
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir /tmp/wheel/*.whl && \
    python -m pip install --no-cache-dir "paddleocr[doc-parser]" fastapi uvicorn[standard] python-multipart  # PP-StructureV3 + API [web:55]

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]  # Single worker during stabilization [web:121]

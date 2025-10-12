# ===== Stage 1: Build PaddlePaddle (CPU-only, ARM64, Inference-Only) =====
FROM python:3.11-slim AS paddle-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build patchelf pkg-config \
    protobuf-compiler libprotobuf-dev \
    python3-dev libopenblas-dev liblapack-dev gfortran \
    ca-certificates wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /paddle
# Pin Paddle to 3.2 stable
RUN git clone --depth 1 --branch release/3.2 https://github.com/PaddlePaddle/Paddle.git . && \
    git submodule update --init --recursive

# Build deps
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir -r /paddle/python/requirements.txt

# Clean build dir and configure hardened CPU/ARM inference build
RUN rm -rf /paddle/build && mkdir /paddle/build
WORKDIR /paddle/build
RUN cmake .. -G Ninja \
-DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DON_INFER=ON \
  -DWITH_STATIC_LIB=OFF \
  -DWITH_PIR=OFF -DPADDLE_WITH_PIR=OFF -DPIR_FULL=OFF \
  -DWITH_SPARSE_TENSOR=OFF \
  -DWITH_GPU=OFF -DWITH_XPU=OFF -DWITH_TENSORRT=OFF \
  -DWITH_MKL=OFF -DWITH_SYSTEM_BLAS=ON -DWITH_OPENBLAS=ON \
  -DWITH_MKLDNN=OFF -DWITH_ONEDNN=OFF \
  -DWITH_ARM=ON -DWITH_AVX=OFF -DWITH_XBYAK=OFF \
  -DWITH_DISTRIBUTE=OFF -DWITH_BRPC=OFF -DWITH_PSCORE=OFF -DWITH_GLOO=OFF \
  -DWITH_CINN=OFF -DWITH_CUDNN_FRONTEND=OFF -DWITH_CUTLASS=OFF -DWITH_FLASH_ATTENTION=OFF \
  -DWITH_NCCL=OFF -DWITH_RCCL=OFF \
  -DWITH_CUSTOM_DEVICE=OFF -DWITH_HETERPS=OFF -DWITH_PSLIB=OFF \
  -DWITH_ARM_BRPC=OFF -DWITH_XPU_BKCL=OFF \
  -DWITH_INFERENCE_API_TEST=OFF -DWITH_TESTING=OFF \
  -DWITH_SHARED_PHI=OFF -DWITH_CRYPTO=OFF \
  -DWITH_LITE=OFF \
  -DWITH_STRIP=ON -DWITH_UNITY_BUILD=OFF \
  -DWITH_PYTHON=ON -DPY_VERSION=3.11

ARG BUILD_JOBS=1
RUN ninja -j${BUILD_JOBS} && ls -lah /paddle/build/python/dist
RUN mkdir -p /wheel && cp -v /paddle/build/python/dist/*.whl /wheel/

# ===== Stage 2: Runtime =====
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=paddle-builder /wheel /tmp/wheel
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir /tmp/wheel/*.whl && \
    python -m pip install --no-cache-dir "paddleocr==3.2" fastapi uvicorn[standard] python-multipart

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

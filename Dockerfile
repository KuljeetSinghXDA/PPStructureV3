# Syntax: docker/dockerfile:1
# Multi-stage build to compile PaddlePaddle 3.2.0 for aarch64 CPU, then run PaddleOCR 3.2.0 service.

# ---- Build stage: compile PaddlePaddle CPU for aarch64 ----
FROM debian:bookworm AS paddle-build
ARG PYVER=3.12
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl wget gnupg build-essential cmake ninja-build \
    python${PYVER} python${PYVER}-dev python3-pip python-is-python3 \
    patchelf pkg-config swig protobuf-compiler libprotobuf-dev \
    libopenblas-dev liblapack-dev libssl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip wheel setuptools
# Get Paddle source at 3.2.0
RUN git clone --depth 1 --branch v3.2.0 https://github.com/PaddlePaddle/Paddle.git /src/Paddle
WORKDIR /src/Paddle
# Configure CPU-only with OpenBLAS, Python wheel build, inference enabled
RUN mkdir -p build && cd build && \
    cmake .. \
      -G Ninja \
      -DWITH_GPU=OFF \
      -DWITH_XPU=OFF \
      -DWITH_IPU=OFF \
      -DWITH_MLU=OFF \
      -DWITH_ASCEND=OFF \
      -DWITH_ROCM=OFF \
      -DWITH_MKL=OFF \
      -DWITH_OPENBLAS=ON \
      -DWITH_DISTRIBUTE=OFF \
      -DWITH_TESTING=OFF \
      -DWITH_PYTHON=ON \
      -DON_INFER=ON \
      -DPY_VERSION=${PYVER} \
      -DCMAKE_BUILD_TYPE=Release && \
    ninja -j$(nproc)
# Build Python wheel
RUN cd build && ninja paddle_python && \
    ls -l python/dist && \
    cp python/dist/*.whl /tmp/paddlepaddle.whl

# ---- Runtime stage ----
FROM python:3.12-slim-bookworm AS runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
# System libs for OpenCV, fonts, and PDF rasterization
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    fonts-dejavu fonts-noto-cjk poppler-utils \
    && rm -rf /var/lib/apt/lists/*
# Copy compiled Paddle wheel and install exact pins
COPY --from=paddle-build /tmp/paddlepaddle.whl /tmp/paddlepaddle.whl
RUN python -m pip install /tmp/paddlepaddle.whl && rm -f /tmp/paddlepaddle.whl
# Exact pins for PaddleOCR and server stack
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt
# App code
WORKDIR /app
COPY app /app/app
# Ensure model cache directory exists
RUN mkdir -p /root/.paddlex/official_models
# Non-root runtime
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app /root/.paddlex
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ===== Stage 1: Build PaddlePaddle 3.2 (CPU) =====
FROM python:3.11-slim AS paddle-builder
ARG PY_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build patchelf pkg-config \
    protobuf-compiler libprotobuf-dev \
    python3-dev libopenblas-dev liblapack-dev gfortran \
    ca-certificates wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /paddle
RUN git clone --depth 1 --branch release/3.2 https://github.com/PaddlePaddle/Paddle.git . && \
    git submodule update --init --recursive

# Per compile docs: install python deps and protobuf 3.20.x before CMake
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir -r /paddle/python/requirements.txt && \
    python -m pip install --no-cache-dir "protobuf==3.20.2"

RUN rm -rf /paddle/build && mkdir /paddle/build
WORKDIR /paddle/build

# CPU build per official instructions (PY_VERSION must match Python)
RUN cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_GPU=OFF \
  -DWITH_PYTHON=ON -DPY_VERSION=${PY_VERSION}

RUN ninja -j2 && ls -lah /paddle/build/python/dist
RUN mkdir -p /wheel && cp -v /paddle/build/python/dist/*.whl /wheel/

# ===== Stage 2: Runtime with compiled Paddle + PaddleOCR =====
FROM python:3.11-slim

# No defaults here; all runtime values are provided via env at deploy time
WORKDIR /app
COPY --from=paddle-builder /wheel /tmp/wheel

# Install Paddle 3.2 wheel and PaddleOCR (official quick start)
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir /tmp/wheel/*.whl && \
    python -m pip install --no-cache-dir "paddleocr[all]"

# App code
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

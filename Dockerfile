# syntax=docker/dockerfile:1.7
# Target arm64/aarch64 CPU
FROM --platform=linux/arm64/v8 python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps commonly required by Paddle/OpenCV on CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Pip + PaddlePaddle CPU for aarch64 + PaddleOCR 3.2.0 + server deps
# The aarch64 wheel index is required for PaddlePaddle on arm64 CPU.

RUN python -m pip install --no-cache-dir "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

    pip install --no-cache-dir \
      "paddleocr[all]" \
      fastapi \
      "uvicorn[standard]" \
      opencv-python-headless==4.10.* \
      numpy \
      pyyaml \
      shapely \
      pillow \
      beautifulsoup4 \
      lxml \
      markdownify

WORKDIR /app
COPY ppstructurev3_server.py /app/ppstructurev3_server.py

# Reasonable CPU thread defaults
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_MAX_THREADS=4

EXPOSE 8000

# Start the API
CMD ["uvicorn", "ppstructurev3_server:app", "--host", "0.0.0.0", "--port", "8000"]

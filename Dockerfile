# syntax=docker/dockerfile:1

FROM python:3.13-slim

# System dependencies for PaddleOCR runtime (OpenCV headless, PDF utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tzdata \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps:
# - PaddlePaddle from the requested index, pinned to 3.3.0
# - PaddleOCR 3.3.0 with doc-parser extras for PP-StructureV3
# - FastAPI stack
RUN python -m pip install --upgrade pip && \
    pip install "paddlepaddle==3.3.0" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ && \
    pip install "paddleocr[doc-parser]==3.3.0" && \
    pip install "fastapi==0.114.*" "uvicorn[standard]==0.30.*" "python-multipart==0.0.9" "pillow==10.*"

# Conservative CPU threading hints
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Copy application
COPY app.py /app/app.py

EXPOSE 8000

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

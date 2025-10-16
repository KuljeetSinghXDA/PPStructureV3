# syntax=docker/dockerfile:1

FROM python:3.13-slim

# System dependencies for PaddleOCR runtime (OpenCV headless, font/render libs, PDF utils)
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

# Python dependencies
# - PaddlePaddle from the requested index, pinned to 3.3.0
# - PaddleOCR with doc-parser extras to enable PP-StructureV3 and its submodules
# - FastAPI stack and helpers
RUN python -m pip install --upgrade pip && \
    pip install "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ && \
    pip install "paddleocr[all]==3.3.0" && \
    pip install "fastapi" "uvicorn[standard]" "python-multipart" 

# Copy application
COPY app.py /app/app.py

EXPOSE 8000

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

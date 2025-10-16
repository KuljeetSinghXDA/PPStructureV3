# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \

# System deps required by wheels like opencv-headless/PDF/image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 1) Install PaddlePaddle CPU wheel (3.0.0) from official index
#    (PaddleOCR 3.2.0 is compatible; this index hosts manylinux wheels)
#    See official install guide. 
#    https://www.paddleocr.ai/main/en/version3.x/installation.html
RUN python -m pip install --upgrade pip \
 && python -m pip install "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 2) Install PaddleOCR 3.2.0 with doc-parser extras and API server deps
RUN python -m pip install "paddleocr[all]" fastapi==0.115.5 uvicorn[standard]==0.32.0

# Copy app
WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8000

# Default command: one worker is fine; increase if CPU-bound concurrency needed
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

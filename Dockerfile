FROM python:3.13-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies required by wheels like opencv-headless, PDF, image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 1) Install PaddlePaddle CPU wheel (latest stable) from official index
#    https://www.paddlepaddle.org.cn/packages/stable/cpu/
RUN python -m pip install --upgrade pip \
 && python -m pip install "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 2) Install PaddleOCR with all extras + FastAPI and Uvicorn
RUN python -m pip install "paddleocr[all]" fastapi==0.115.5 uvicorn[standard]==0.32.0

# Copy app
WORKDIR /app
COPY app.py /app/app.py

# Expose FastAPI port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

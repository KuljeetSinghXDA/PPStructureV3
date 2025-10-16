FROM python:3.13-slim

# Prevents Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies required by opencv-headless, PDF, and image libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PaddlePaddle CPU wheel (latest, ARM64 or x86 auto-detected)
# Official wheels are hosted here: https://www.paddlepaddle.org.cn/packages/stable/cpu/
RUN python -m pip install "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install PaddleOCR (with all extras), FastAPI, Uvicorn, and python-multipart
RUN python -m pip install \
    "paddleocr[all]" \
    fastapi \
    uvicorn[standard] \
    python-multipart

# Copy application
WORKDIR /app
COPY app.py /app/app.py

# Expose FastAPI port
EXPOSE 8000

# Default command to start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

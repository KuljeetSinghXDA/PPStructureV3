# Use the official Python 3.13 slim image for arm64
# Note: You must build this on an arm64 machine or use buildx with platform=linux/arm64
FROM --platform=linux/arm64 python:3.13-slim

# Prevents Python from writing pyc files to disk and buffering output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Set a fixed directory for PaddlePaddle models to cache
    XDG_CACHE_HOME=/app/.cache

# Install system dependencies required by PaddlePaddle and Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libglib2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy the application file
COPY app.py /app/

# Install PaddlePaddle and PaddleOCR
# We use the nightly CPU index for PaddlePaddle as it's more likely to have recent manylinux wheels.
# We pin PaddleOCR to 3.3.0 as requested.
# Note: There is no guarantee a compatible arm64 wheel for PaddlePaddle exists.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ && \
    pip install "paddleocr==3.3.0"

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]

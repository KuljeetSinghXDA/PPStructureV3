FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ARM64
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle from nightly build (CPU version for ARM64)
RUN pip install --no-cache-dir "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# Install PaddleOCR 3.3.0 with doc-parser dependencies
RUN pip install --no-cache-dir "paddleocr[all]==3.3.0"

# Install FastAPI and additional dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles
# Copy application file
COPY app.py .

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

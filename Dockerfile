FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PaddlePaddle and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle and PaddleOCR
RUN pip install --no-cache-dir "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
RUN pip install --no-cache-dir "paddleocr[all]==3.3.0"

# Install additional dependencies for FastAPI and web server
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    python-magic \
    pydantic

# Copy application code
COPY app.py /app/
COPY models_config.py /app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

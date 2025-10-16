FROM --platform=linux/arm64 python:3.13-slim

# Install system dependencies for OpenCV, Pillow, and PDF handling
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle for ARM64 CPU using the provided nightly index
RUN pip install --no-cache-dir "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# Install PaddleOCR with all features (doc-parser, ie, trans, etc.) for PP-StructureV3
RUN pip install --no-cache-dir "paddleocr[all]==3.3.0"

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn

# Copy the application file
COPY app.py /app.py

# Expose port
EXPOSE 80

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

FROM python:3.13-slim

# Install system dependencies for PDF handling (poppler for pdf2image), OpenCV (libGL for headless), and general utils
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PaddlePaddle 3.3.0 from nightly CPU index (supports ARM64 via compatible wheels)
RUN pip install --no-cache-dir "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# Install PaddleOCR 3.3.0 with all extras for full PP-StructureV3 features
RUN pip install --no-cache-dir "paddleocr[all]==3.3.0"

# Install FastAPI dependencies for file uploads and server
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy the application code
COPY app.py .

# Expose port and run the server
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

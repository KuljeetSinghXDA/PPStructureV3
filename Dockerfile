FROM python:3.13-slim

# Install system dependencies for PaddleOCR CPU inference
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle from nightly CPU index for ARM64 compatibility
RUN pip install --no-cache-dir "paddlepaddle" -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# Install PaddleOCR 3.3.0 with all features (including PP-StructureV3)
RUN pip install --no-cache-dir "paddleocr[all]==3.3.0"

# Install FastAPI and Uvicorn for the API server
RUN pip install --no-cache-dir fastapi uvicorn pyyaml

# Copy the application file
COPY app.py /app.py

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

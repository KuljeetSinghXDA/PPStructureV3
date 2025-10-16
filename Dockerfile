# Use Python 3.13 slim base image (CPU)
FROM python:3.13-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle (CPU), PaddleX with OCR extras, and FastAPI server requirements
RUN pip install --no-cache-dir \
    "paddlepaddle==3.2.0" \
    "paddlex[ocr]==3.2.0" \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

# Copy the application code
COPY app.py /app.py

# Expose the API port (optional, default to 8000)
EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

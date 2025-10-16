# Use Python 3.13-slim as base image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PADDLE_PDX_MODEL_SOURCE=HuggingFace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    paddlepaddle==3.0.0 \
    paddleocr==3.2.0 \
    "fastapi[standard]" \
    uvicorn \
    python-multipart \
    pydantic \
    pillow \
    numpy \
    opencv-python-headless \
    pdf2docx \
    pypdfium2

# Pre-download models (optional - can be done at runtime)
# This ensures models are cached in the container
RUN python -c "from paddleocr import PPStructureV3; PPStructureV3(layout_model_name='PP-DocLayout-L', text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='en_PP-OCRv5_mobile_rec', table_recognition_model_name='SLANet_plus', use_doc_preprocessor=False, use_table_recognition=True, use_formula_recognition=True, use_seal_recognition=True, use_chart_recognition=True).predict('', input_type='img', do_visualize=False)" 2>/dev/null || true

# Copy application files
COPY pp_structure_api.py /app/

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "pp_structure_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

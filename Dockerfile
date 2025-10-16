FROM python:3.13-slim

# Noninteractive apt to avoid debconf warnings
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    DEBCONF_NOWARNINGS=yes

# Install system dependencies (GL/GUI libs for CV backends, wget for model downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies:
# - Upgrade pip
# - PaddlePaddle CPU (Armv8/AArch64) from official source
# - PaddleOCR (with all optional extras), FastAPI, Uvicorn, python-multipart for file uploads
# - PyMuPDF for PDF support
RUN python -m pip install --no-cache-dir -U pip --root-user-action=ignore && \
    python -m pip install --no-cache-dir paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ --root-user-action=ignore && \
    python -m pip install --no-cache-dir "paddleocr[all]" fastapi uvicorn[standard] python-multipart PyMuPDF --root-user-action=ignore

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# app/Dockerfile
FROM python:3.13-slim

# Noninteractive apt to avoid debconf warnings
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    DEBCONF_NOWARNINGS=yes \
    # Prefer HF as model source per PaddleOCR 3.0.2+ notes (can be changed to BOS)
    PADDLE_PDX_MODEL_SOURCE=huggingface \
    # Avoid OpenMP oversubscription for CPU inference
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Common GUI/GL libs needed by cv backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

# Latest pip + Paddle CPU wheel (aarch64/py3.13 available on PyPI as of 2025-09-08)
# Install PaddleOCR with doc-parser extras, FastAPI stack, and PyMuPDF (PDF page-range support)
RUN python -m pip install --no-cache-dir -U pip \
 && python -m pip install --no-cache-dir \
    "paddlepaddle" \
    "paddleocr[doc-parser]" \
    fastapi uvicorn[standard] python-multipart \
    "pymupdf" \
    "orjson"

WORKDIR /app
COPY app /app/app

EXPOSE 8000
# Single worker is fine (CPU-only + predictable memory). Increase if needed.
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

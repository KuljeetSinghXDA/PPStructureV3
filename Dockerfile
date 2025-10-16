# app/Dockerfile
FROM python:3.12-slim

# Noninteractive apt
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    DEBCONF_NOWARNINGS=yes

# Common GUI/GL libs, plus poppler-utils for PDF rasterization (pdf2image/pymupdf compatible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 wget poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Speed and compatibility notes:
# - PaddlePaddle CPU stable wheel (arm64 supported via official index)
# - PaddleOCR with all extras to cover PP-StructureV3, chart, formula, etc.
# - FastAPI + Uvicorn, multipart for uploads, and PDF helpers
RUN python -m pip install --no-cache-dir -U pip --root-user-action=ignore \
 && python -m pip install --no-cache-dir "paddlepaddle" \
      -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ --root-user-action=ignore \
 && python -m pip install --no-cache-dir "paddleocr[all]" fastapi uvicorn[standard] python-multipart \
      "pymupdf" "pdf2image" --root-user-action=ignore

# Performance-related environment hints for CPU
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_MAX_THREADS=4

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

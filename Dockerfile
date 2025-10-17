# syntax=docker/dockerfile:1.7
# Build:  docker buildx build --platform linux/arm64 -t ppstructv3:cpu .
# Run:    docker run --rm -p 8080:8080 -e OMP_NUM_THREADS=1 ppstructv3:cpu

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Prefer HF for model downloads (PaddleOCR 3.x supports switching sources)
    PADDLE_PDX_MODEL_SOURCE=huggingface

# Minimal runtime libs for Paddle/OpenCV/PyMuPDF on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install PaddlePaddle from the requested nightly CPU index (no version pin).
#    If you need a particular version later, adjust at build time.
RUN python -m pip install "paddlepaddle" \
    -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 2) Install PaddleOCR 3.3 codebase and API/runtime deps (no extra pins).
#    Using the 3.3 release branch per your request.
#    Extras: doc-parser pulls parser dependencies used by PP-StructureV3.
RUN python -m pip install \
      "paddleocr[all]==3.3.0" \
      fastapi uvicorn[standard] python-multipart

# App
COPY app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

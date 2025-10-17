# syntax=docker/dockerfile:1.7

# Build (arm64):
#   docker buildx build --platform linux/arm64 -t ppstructv3:cpu .
# Run:
#   docker run --rm -p 8080:8080 -e OMP_NUM_THREADS=4 ppstructv3:cpu

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Minimal native libs needed at runtime (OpenMP, OpenCV GUI shims, CA store)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) PaddlePaddle from your requested nightly CPU index (no pin)
RUN python -m pip install "paddlepaddle" \
    -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 2) PaddleOCR from release/3.3 branch + only the dependency group needed by PP-StructureV3
#    (you can switch to [all] if you want every extra)
RUN python -m pip install \
    paddleocr[all] \
    fastapi uvicorn[standard] \
    python-multipart

COPY app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

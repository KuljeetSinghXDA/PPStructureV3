FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Use Noninteractive only for this RUN; include apt-utils to reduce debconf warnings
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      apt-utils \
      libgomp1 \
      libglib2.0-0 \
      libgl1 \
      ca-certificates \
      curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install PaddlePaddle (CPU) from the official index
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ paddlepaddle

# 2) Install PaddleX with OCR extras required by PP-StructureV3
RUN python -m pip install --no-cache-dir "paddlex[ocr]"

# 3) Install PaddleOCR and API/server deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# App code
COPY app ./app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 75"]

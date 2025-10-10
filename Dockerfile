FROM python:3.12.6-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential pkg-config git curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    poppler-utils fonts-dejavu fonts-noto-cjk locales \
 && rm -rf /var/lib/apt/lists/*

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen

RUN pip install --no-cache-dir \
    paddlepaddle==3.2.0 \
    "paddleocr[doc-parser]==3.2.0" \
    fastapi==0.115.2 \
    uvicorn[standard]==0.30.6 \
    python-multipart==0.0.9 \
    requests==2.32.3

WORKDIR /app
COPY app /app/app

CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-2}"]

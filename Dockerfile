# ---- Builder ----
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and use a virtualenv so we can copy only what we need
RUN python -m pip install --upgrade pip && python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PaddlePaddle first from the CPU index, then the rest
RUN python -m pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ paddlepaddle
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---- Runtime ----
FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    OMP_NUM_THREADS=4 \
    PADDLE_PDX_MODEL_SOURCE=BOS \
    PATH="/opt/venv/bin:$PATH"

# Only the minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libgl1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in the venv from builder with all Python deps
COPY --from=builder /opt/venv /opt/venv

# App code
COPY app ./app

# No EXPOSE needed; Compose 'expose' handles container-to-container access
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 75"]

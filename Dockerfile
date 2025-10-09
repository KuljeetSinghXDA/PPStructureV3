FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libgl1 ca-certificates curl ccache && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) PaddlePaddle CPU (multi-arch aarch64 supported on your host)
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ paddlepaddle

# 2) Core deps from requirements (installs paddleocr which pulls paddlex base)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# 3) Ensure PP-StructureV3 extras are installed for the exact paddlex version
RUN python - <<'PY'\n\
import paddlex, subprocess, sys\n\
v = paddlex.__version__\n\
subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'paddlex[ocr]=={v}'])\n\
print('Installed paddlex[ocr]==', v)\n\
PY

# 4) App code
COPY app ./app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 75"]

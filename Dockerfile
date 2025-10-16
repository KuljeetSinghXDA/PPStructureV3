FROM python:3.11-slim

# Noninteractive apt
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    DEBCONF_NOWARNINGS=yes

# Common GUI/GL + OpenMP runtime needed by OpenCV/Paddle CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 libgomp1 wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Runtime perf hints (tunable at runtime via envs)
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

# Latest pip + core deps:
# - paddlepaddle (CPU) for aarch64 (PyPI provides wheels)
# - paddleocr[all] for doc parser, KIE, translation etc. (includes PP-Structure V3)
# - API stack + helpers
RUN python -m pip install --no-cache-dir -U pip setuptools wheel --root-user-action=ignore \
 && python -m pip install --no-cache-dir paddlepaddle --root-user-action=ignore \
 && python -m pip install --no-cache-dir "paddleocr[all]" fastapi "uvicorn[standard]" python-multipart \
        pypdfium2 "beautifulsoup4>=4.12" "orjson>=3.9" --root-user-action=ignore

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

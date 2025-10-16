FROM python:3.13-slim

# Noninteractive apt to avoid debconf warnings
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    DEBCONF_NOWARNINGS=yes

# Latest GL/GUI libs commonly required by CV backends used by PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps:
# - paddlepaddle (CPU; ARMv8/AArch64 supported) from the official CPU index
# - paddleocr[all] full feature set (structure, formula, chart, etc.)
# - fastapi, uvicorn, python-multipart for API
# - pypdfium2 for PDF rendering (no poppler system dependency)
RUN python -m pip install --no-cache-dir -U pip --root-user-action=ignore \
 && python -m pip install --no-cache-dir paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ --root-user-action=ignore \
 && python -m pip install --no-cache-dir "paddleocr[all]" fastapi uvicorn[standard] python-multipart pypdfium2 --root-user-action=ignore

# Threading defaults (can be overridden via environment)
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

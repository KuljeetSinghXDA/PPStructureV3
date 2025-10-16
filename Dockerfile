FROM python:3.13-slim


ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    DEBCONF_NOWARNINGS=yes \
    PIP_NO_CACHE_DIR=1

# Minimal system libs for common CV backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Core Python deps: Paddle (CPU), PaddleOCR, FastAPI stack, and PyMuPDF (import fitz)
# Note: PyMuPDF provides wheels for Py3.9–3.13; no external deps when wheel is available
RUN python -m pip install -U pip --root-user-action=ignore && \
    python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ --root-user-action=ignore && \
    python -m pip install "paddleocr[all]" fastapi uvicorn[standard] python-multipart --root-user-action=ignore && \
    python -m pip install "pymupdf" --root-user-action=ignore

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

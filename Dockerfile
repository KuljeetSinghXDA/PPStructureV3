# ===== Stage 2: Runtime =====
FROM python:3.11-slim

# Install runtime libs + temporary build tools for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    libopenblas0 libgfortran5 libgomp1 \
    build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

COPY --from=paddle-builder /wheel /tmp/wheel

# Safe install sequence preserving custom Paddle wheel
RUN python -m pip install --no-cache-dir -U pip && \
    # 1) Install custom ARM64 Paddle wheel first
    python -m pip install --no-cache-dir /tmp/wheel/*.whl && \
    # 2) Install PaddleOCR/PaddleX without dependencies to prevent Paddle replacement
    python -m pip install --no-cache-dir --no-deps "paddleocr==3.2.*" && \
    python -m pip install --no-cache-dir --no-deps "paddlex==3.2.*" && \
    # 3) Manually satisfy paddlex[ocr] extras completely (per official requirements)
    python -m pip install --no-cache-dir \
        opencv-python-headless opencv-contrib-python \
        pillow pyyaml shapely scikit-image imgaug \
        pyclipper lmdb tqdm numpy visualdl rapidfuzz cython \
        lanms-neo attrdict easydict \
        reportlab pypdf pdfminer.six PyMuPDF pypdfium2>=4 \
        ftfy imagesize lxml openpyxl premailer \
        scikit-learn tokenizers==0.19.1 \
        onnx onnxruntime matplotlib requests typing_extensions && \
    # 4) Install FastAPI stack
    python -m pip install --no-cache-dir fastapi uvicorn[standard] python-multipart && \
    # 5) Verify installation integrity
    python -c "import paddle; print('✓ Paddle version:', paddle.__version__); print('✓ Paddle path:', paddle.__file__)" && \
    python -c "import paddleocr; print('✓ PaddleOCR imported successfully')" && \
    python -c "import paddlex; print('✓ PaddleX imported successfully')" && \
    # 6) Clean up build tools to reduce image size
    apt-get purge -y build-essential g++ && \
    apt-get autoremove -y && \
    rm -rf /tmp/wheel /var/lib/apt/lists/*

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

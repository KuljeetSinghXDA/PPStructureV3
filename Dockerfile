FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies + build tools (needed for some native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    libopenblas0 libgfortran5 libgomp1 \
    build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

# Install from PyPI (has ARM64 wheels for most packages)
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir \
        paddlepaddle==3.2.0 \
        "paddlex[ocr]==3.2.*" && \
    python -m pip install --no-cache-dir \
        fastapi uvicorn[standard] python-multipart && \
    apt-get purge -y build-essential g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

# Use official PaddlePaddle CPU runtime image (multi-arch, includes ARM64 builds)
FROM paddlepaddle/paddle:latest

# Noninteractive apt to avoid debconf warnings if we install anything later
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    DEBCONF_NOWARNINGS=yes

# Install API/server deps and OCR pipeline; Paddle is already present in base
RUN python -m pip install --no-cache-dir -U pip --root-user-action=ignore \
 && python -m pip install --no-cache-dir "paddleocr[doc-parser]" fastapi uvicorn[standard] python-multipart --root-user-action=ignore

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","1"]

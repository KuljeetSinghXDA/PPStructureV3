FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 wget && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U pip \
 && python -m pip install --no-cache-dir paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
 && python -m pip install --no-cache-dir "paddleocr[doc-parser]" fastapi uvicorn[standard]

WORKDIR /app
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000","--workers","2"]

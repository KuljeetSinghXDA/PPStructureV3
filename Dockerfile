FROM python:3.12.6-slim-bookworm

# System deps for OpenCV and fonts (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Exact Python deps
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip==24.2
RUN pip install \
    paddlepaddle==3.2.0 \
    "paddleocr[doc-parser]==3.2.0" \
    fastapi==0.115.0 \
    uvicorn==0.30.6

# App files
WORKDIR /app
COPY app /app/app
COPY scripts /app/scripts

# Pre-fetch flagship models at build time to bake them into the image
# Uses env defaults for model names and toggles; safe no-op if unset
ENV USE_DOC_ORIENTATION_CLASSIFY=false \
    USE_DOC_UNWARPING=false \
    USE_TEXTLINE_ORIENTATION=false \
    USE_REGION_DETECTION=true \
    USE_TABLE_RECOGNITION=true \
    USE_FORMULA_RECOGNITION=false \
    USE_CHART_RECOGNITION=false \
    USE_SEAL_RECOGNITION=false \
    LAYOUT_DETECTION_MODEL_NAME=PP-DocLayout_plus-L \
    REGION_DETECTION_MODEL_NAME=PP-DocBlockLayout \
    TEXT_DETECTION_MODEL_NAME=PP-OCRv5_server_det \
    TEXT_RECOGNITION_MODEL_NAME=PP-OCRv5_server_rec \
    TABLE_ORIENTATION_CLASSIFY_MODEL_NAME=PP-LCNet_x1_0_table_cls \
    WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME=SLANeXt_wired \
    WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME=SLANeXt_wireless \
    WIRED_TABLE_CELLS_DETECTION_MODEL_NAME=RT-DETR-L_wired_table_cell_det \
    WIRELESS_TABLE_CELLS_DETECTION_MODEL_NAME=RT-DETR-L_wireless_table_cell_det \
    ENABLE_MKLDNN=true \
    CPU_THREADS=4 \
    PADDLE_PDX_MODEL_SOURCE=HUGGINGFACE

RUN python -u /app/scripts/prefetch_models.py

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

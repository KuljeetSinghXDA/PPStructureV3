# Build on ARM64 host or use: docker build --platform=linux/arm64/v8
FROM --platform=linux/arm64/v8 python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4

# System deps for PDFs and image backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# PaddlePaddle nightly CPU (ARM64) and PaddleOCR 3.3.0 doc-parser group
RUN python -m pip install --upgrade pip && \
    python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    python -m pip install "paddleocr[doc-parser]==3.2.0" fastapi "uvicorn[standard]" python-multipart pymupdf

# FastAPI app: PP-StructureV3 native outputs, chart disabled, accuracy tuning applied
RUN cat > /app.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import shutil
import os
import json

from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 API (ARM64, native, charts off)", version="3.3.0")

# Initialize PP-StructureV3 with requested models, chart disabled, and accuracy tuning
pipeline = PPStructureV3(
    layout_detection_model_name="PP-DocLayout-L",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    # Disable chart recognition to avoid PP-Chart2Table path in 3.3.0
    use_chart_recognition=False,
    # Keep other subpipelines at defaults (tables/formulas/region on)
    cpu_threads=4,                   # Ampere A1 4 OCPU
    # Accuracy-oriented tuning for small fonts / dense tables
    text_det_limit_side_len=1920,
    text_det_limit_type="max",
    text_det_thresh=0.20,
    text_det_box_thresh=0.30,
    text_det_unclip_ratio=2.5
)

def predict_collect(path: Path) -> Dict[str, Any]:
    # Predict supports images and multi-page PDFs
    results = pipeline.predict(
        str(path),
        use_e2e_wireless_table_rec_model=True,
        use_ocr_results_with_table_cells=True
    )
    pages = []
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        json_files, md_files = [], []
        # Save native per-page artifacts then read back
        for res in results:
            res.save_to_json(save_path=str(out_dir))
            res.save_to_markdown(save_path=str(out_dir))
        for p in sorted(out_dir.glob("*.json")):
            json_files.append(p)
        for p in sorted(out_dir.glob("*.md")):
            md_files.append(p)
        for i in range(max(len(json_files), len(md_files))):
            page_json = {}
            page_md = ""
            if i < len(json_files):
                try:
                    page_json = json.loads(json_files[i].read_text(encoding="utf-8"))
                except Exception:
                    page_json = {}
            if i < len(md_files):
                page_md = md_files[i].read_text(encoding="utf-8")
            pages.append({"page_index": i, "json": page_json, "markdown": page_md})
    return {"pages": pages}

@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    outputs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for uf in files:
            suffix = Path(uf.filename or "").suffix or ".bin"
            target = tmpdir / (Path(uf.filename or f"upload{len(outputs)}{suffix}").name)
            with target.open("wb") as w:
                shutil.copyfileobj(uf.file, w)
            file_res = predict_collect(target)
            outputs.append({"filename": uf.filename, **file_res})
    return JSONResponse({"files": outputs})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

WORKDIR /
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

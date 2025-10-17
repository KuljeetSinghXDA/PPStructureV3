# Build on ARM64 host or use: docker build --platform=linux/arm64/v8
FROM --platform=linux/arm64/v8 python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4

# Add cmake and compiler toolchain before pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    cmake \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# Keep PaddleOCR 3.3.0 with doc-parser extras, ONNX stack, and API deps
RUN python -m pip install \
    "paddleocr[doc-parser]==3.3.0" \
    onnxruntime \
    paddle2onnx \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pymupdf

# Optional: mitigate libgomp TLS issues on some ARM64 distros
# ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Embed the FastAPI app that uses native PP-StructureV3 JSON/Markdown outputs
RUN cat > /app.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import shutil
import os

from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 API (ARM64, ONNX HPI)", version="3.3.0")

# Initialize pipeline with requested models and high-performance inference enabled
pipeline = PPStructureV3(
    layout_detection_model_name="PP-DocLayout-L",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    enable_hpi=True,                 # auto-convert and select ONNX Runtime on CPU when available
    cpu_threads=4,                   # match Ampere A1 4 OCPU
    # Accuracy-oriented tuning for dense medical reports
    text_det_limit_side_len=1920,
    text_det_limit_type="max",
    text_det_thresh=0.20,
    text_det_box_thresh=0.30,
    text_det_unclip_ratio=2.5
    # Keep other submodules at defaults
)

def run_and_collect(path: Path) -> Dict[str, Any]:
    # Predict supports images and multi-page PDFs; returns per-page results
    results = pipeline.predict(
        str(path),
        use_e2e_wireless_table_rec_model=True,
        use_ocr_results_with_table_cells=True
    )
    # Use native save_to_json/save_to_markdown for fidelity; then read back
    pages = []
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        json_files = []
        md_files = []
        for idx, res in enumerate(results):
            res.save_to_json(save_path=str(out_dir))
            res.save_to_markdown(save_path=str(out_dir))
        # Collect saved artifacts; rely on sorted order as page order
        for p in sorted(out_dir.glob("*.json")):
            json_files.append(p)
        for p in sorted(out_dir.glob("*.md")):
            md_files.append(p)
        # Pair JSON and MD by index
        for i in range(max(len(json_files), len(md_files))):
            page_json = {}
            page_md = ""
            if i < len(json_files):
                try:
                    import json
                    page_json = json.loads(json_files[i].read_text(encoding="utf-8"))
                except Exception:
                    page_json = {}
            if i < len(md_files):
                page_md = md_files[i].read_text(encoding="utf-8")
            pages.append({"json": page_json, "markdown": page_md})
    return {"pages": pages}

@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    outputs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for uf in files:
            suffix = os.path.splitext(uf.filename or "")[1]
            if not suffix:
                suffix = ".bin"
            # Accept images and PDFs (pipeline will handle multi-page PDF)
            target = tmpdir / (Path(uf.filename).name or f"upload{suffix}")
            with target.open("wb") as w:
                shutil.copyfileobj(uf.file, w)
            file_res = run_and_collect(target)
            outputs.append({"filename": uf.filename, **file_res})
    return JSONResponse({"files": outputs})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

WORKDIR /
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

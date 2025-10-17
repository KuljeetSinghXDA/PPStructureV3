FROM --platform=linux/arm64/v8 python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ && \
    python -m pip install "paddleocr[doc-parser]==3.3.0" fastapi "uvicorn[standard]" python-multipart pymupdf

RUN cat > /app.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pathlib import Path
import tempfile, shutil, os, json
from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 (ARM64, charts OFF)", version="3.3.0")

pipeline = PPStructureV3(
    layout_detection_model_name="PP-DocLayout-L",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_chart_recognition=False,         # hard OFF to bypass PP-Chart2Table path
    cpu_threads=4,
    text_det_limit_side_len=1920,
    text_det_limit_type="max",
    text_det_thresh=0.20,
    text_det_box_thresh=0.30,
    text_det_unclip_ratio=2.5
)

def predict_collect(p: Path) -> Dict[str, Any]:
    results = pipeline.predict(
        str(p),
        use_e2e_wireless_table_rec_model=True,
        use_ocr_results_with_table_cells=True
    )
    pages = []
    with tempfile.TemporaryDirectory() as outd:
        outd = Path(outd)
        jf, mf = [], []
        for r in results:
            r.save_to_json(save_path=str(outd))
            r.save_to_markdown(save_path=str(outd))
        for q in sorted(outd.glob("*.json")): jf.append(q)
        for q in sorted(outd.glob("*.md")): mf.append(q)
        for i in range(max(len(jf), len(mf))):
            pj, pm = {}, ""
            if i < len(jf):
                try: pj = json.loads(jf[i].read_text(encoding="utf-8"))
                except Exception: pj = {}
            if i < len(mf):
                pm = mf[i].read_text(encoding="utf-8")
            pages.append({"page_index": i, "json": pj, "markdown": pm})
    return {"pages": pages}

@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)):
    if not files: raise HTTPException(status_code=400, detail="No files provided.")
    outs = []
    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        for uf in files:
            suffix = Path(uf.filename or "").suffix or ".bin"
            tgt = tmpd / (Path(uf.filename or f"upload{len(outs)}{suffix}").name)
            with tgt.open("wb") as w: shutil.copyfileobj(uf.file, w)
            outs.append({"filename": uf.filename, **predict_collect(tgt)})
    return JSONResponse({"files": outs})
EOF

WORKDIR /
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

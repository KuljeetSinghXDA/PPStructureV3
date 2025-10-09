import os
import io
import json
import tempfile
import multiprocessing
import numpy as np
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Environment (via Dokploy UI or .env through Compose)
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))

# Image OCR models (existing endpoint)
TEXT_DET_MODEL = os.getenv("TEXT_DET_MODEL", "PP-OCRv5_server_det")
TEXT_REC_MODEL = os.getenv("TEXT_REC_MODEL", "PP-OCRv5_server_rec")

# PP-StructureV3 override for recognition model (improves English-only PDFs)
STRUCTURE_REC_MODEL = os.getenv("STRUCTURE_REC_MODEL", None)  # e.g., "en_PP-OCRv5_server_rec"

# Official model cache root
OFFICIAL_DIR = Path("/root/.paddlex/official_models")
DET_DIR = OFFICIAL_DIR / TEXT_DET_MODEL
REC_DIR = OFFICIAL_DIR / TEXT_REC_MODEL

app = FastAPI()

@app.on_event("startup")
def load_models():
    # Image OCR pipeline (existing)
    ocr_kwargs = dict(
        lang=OCR_LANG,
        ocr_version=OCR_VERSION,
        device="cpu",
        enable_hpi=False,
        cpu_threads=CPU_THREADS,
        text_detection_model_name=TEXT_DET_MODEL,
        text_recognition_model_name=TEXT_REC_MODEL,
    )
    if DET_DIR.exists():
        ocr_kwargs["det_model_dir"] = str(DET_DIR)
    if REC_DIR.exists():
        ocr_kwargs["rec_model_dir"] = str(REC_DIR)
    app.state.ocr = PaddleOCR(**ocr_kwargs)

    # Prepare PP-StructureV3 factory; instantiate per-request to avoid holding many large models in memory
    app.state.structure_rec_model = STRUCTURE_REC_MODEL

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "lang": OCR_LANG,
        "ocr_version": OCR_VERSION,
        "det_model": TEXT_DET_MODEL,
        "rec_model": TEXT_REC_MODEL,
        "structure_rec_model": STRUCTURE_REC_MODEL,
        "det_cached": DET_DIR.exists(),
        "rec_cached": REC_DIR.exists(),
    }

def _bytes_to_ndarray(b: bytes):
    with Image.open(io.BytesIO(b)) as im:
        return np.array(im.convert("RGB"))

@app.post("/ocr")
async def ocr_endpoint(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        content = await f.read()
        img = _bytes_to_ndarray(content)
        preds = app.state.ocr.predict(input=img)
        for res in preds:
            results.append({
                "filename": f.filename,
                "texts": res.json.get("rec_texts", []),
                "scores": [float(s) for s in res.json.get("rec_scores", [])],
                "boxes": res.json.get("rec_boxes", []),
            })
    return JSONResponse({"results": results})

@app.post("/parse_pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Write to a temp file for PP-StructureV3
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    # Build PP-StructureV3 with optional English recognizer override
    structure_kwargs = {}
    if app.state.structure_rec_model:
        structure_kwargs["text_recognition_model_name"] = app.state.structure_rec_model

    try:
        pipeline = PPStructureV3(**structure_kwargs)
        output = pipeline.predict(input=tmp_path)

        # Collect per-page JSON and build combined Markdown
        page_json = []
        md_list = []
        md_images_all = []

        for res in output:
            page_json.append(res.json)
            md_info = res.markdown
            md_list.append(md_info)
            md_images_all.append(md_info.get("markdown_images", {}))

        combined_markdown = pipeline.concatenate_markdown_pages(md_list)

        return JSONResponse({
            "filename": file.filename,
            "pages": page_json,
            "markdown": combined_markdown,
        })
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

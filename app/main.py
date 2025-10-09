import os
import io
import tempfile
import multiprocessing
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Environment (from .env or platform)
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))
TEXT_DET_MODEL = os.getenv("TEXT_DET_MODEL", "PP-OCRv5_server_det")
TEXT_REC_MODEL = os.getenv("TEXT_REC_MODEL", "PP-OCRv5_server_rec")

OFFICIAL_DIR = Path("/root/.paddlex/official_models")
DET_DIR = OFFICIAL_DIR / TEXT_DET_MODEL
REC_DIR = OFFICIAL_DIR / TEXT_REC_MODEL

app = FastAPI()

@app.on_event("startup")
def load_models():
    # Image OCR pipeline (PP-OCRv5)
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

    # Document parsing pipeline (PP-StructureV3) for PDFs/images -> JSON/Markdown
    app.state.struct = PPStructureV3(
        lang=OCR_LANG,
        device="cpu",
        cpu_threads=CPU_THREADS,
    )

    print(f"[startup] OCR det={TEXT_DET_MODEL} cached={DET_DIR.exists()} | "
          f"rec={TEXT_REC_MODEL} cached={REC_DIR.exists()} | "
          f"lang={OCR_LANG} version={OCR_VERSION} cpu_threads={CPU_THREADS}")

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "lang": OCR_LANG,
        "ocr_version": OCR_VERSION,
        "det_model": TEXT_DET_MODEL,
        "rec_model": TEXT_REC_MODEL,
        "pp_structure": True,
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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")
    # Save PDF to a temporary file for the pipeline
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name
        tmp.write(await file.read())

    try:
        results = app.state.struct.predict(input=pdf_path)
        pages = []
        md_pages = []
        for idx, res in enumerate(results, start=1):
            pages.append({
                "page": idx,
                "json": res.json,
                "markdown": res.markdown,
            })
            md_pages.append(res.markdown or "")

        combined_markdown = "\n\n".join(md_pages)
        return JSONResponse({
            "filename": file.filename,
            "pages": pages,
            "markdown": combined_markdown
        })
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass

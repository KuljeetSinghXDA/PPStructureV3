import os
import io
import multiprocessing
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from paddleocr import PaddleOCR
from PIL import Image

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Environment (set in Dokploy UI)
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))
TEXT_DET_MODEL = os.getenv("TEXT_DET_MODEL", "PP-OCRv5_server_det")
TEXT_REC_MODEL = os.getenv("TEXT_REC_MODEL", "PP-OCRv5_server_rec")

# Resolve model folders where official models are cached by PaddleX/PaddleOCR
OFFICIAL_DIR = Path("/root/.paddlex/official_models")
DET_DIR = OFFICIAL_DIR / TEXT_DET_MODEL
REC_DIR = OFFICIAL_DIR / TEXT_REC_MODEL

app = FastAPI()

@app.on_event("startup")
def load_models():
    # Strategy:
    # - Always pass model names so PaddleOCR will auto-download if folders are missing.
    # - If folders exist, also pass det_model_dir/rec_model_dir to hard-bind to server-grade models.
    kwargs = dict(
        lang=OCR_LANG,
        ocr_version=OCR_VERSION,
        device="cpu",
        enable_hpi=False,
        cpu_threads=CPU_THREADS,
        text_detection_model_name=TEXT_DET_MODEL,
        text_recognition_model_name=TEXT_REC_MODEL,
    )
    if DET_DIR.exists():
        kwargs["det_model_dir"] = str(DET_DIR)
    if REC_DIR.exists():
        kwargs["rec_model_dir"] = str(REC_DIR)

    app.state.ocr = PaddleOCR(**kwargs)

    print(f"[startup] Using det={TEXT_DET_MODEL} dir_exists={DET_DIR.exists()} | "
          f"rec={TEXT_REC_MODEL} dir_exists={REC_DIR.exists()} | "
          f"lang={OCR_LANG} ocr_version={OCR_VERSION} cpu_threads={CPU_THREADS}")

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "lang": OCR_LANG,
        "ocr_version": OCR_VERSION,
        "det_model": TEXT_DET_MODEL,
        "rec_model": TEXT_REC_MODEL,
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

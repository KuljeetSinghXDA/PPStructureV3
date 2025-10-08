import os
import io
import multiprocessing
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from paddleocr import PaddleOCR
from PIL import Image

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Read environment variables set in Dokploy UI
OCR_LANG = os.getenv("OCR_LANG", "en")
DET_MODEL = os.getenv("TEXT_DET_MODEL", "PP-OCRv5_server_det")
REC_MODEL = os.getenv("TEXT_REC_MODEL", "PP-OCRv5_server_rec")
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))

app = FastAPI()

@app.on_event("startup")
def load_models():
    # Correctly initialize PaddleOCR using model directory arguments
    # to override the default language-based model selection.
    app.state.ocr = PaddleOCR(
        lang=OCR_LANG,
        det_model_dir=f'/root/.paddlex/official_models/{DET_MODEL}',
        rec_model_dir=f'/root/.paddlex/official_models/{REC_MODEL}',
        use_angle_cls=False,
        device="cpu",
        cpu_threads=CPU_THREADS
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok", "lang": OCR_LANG, "det_model": DET_MODEL, "rec_model": REC_MODEL}

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


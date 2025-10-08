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

OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
USE_DOC_ORI = os.getenv("USE_DOC_ORI", "false").lower() == "true"
USE_UNWARP = os.getenv("USE_UNWARP", "false").lower() == "true"
USE_TEXTLINE_ORI = os.getenv("USE_TEXTLINE_ORI", "false").lower() == "true"
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))

TEXT_DET_MODEL = os.getenv("TEXT_DET_MODEL") or "PP-OCRv5_server_det"
TEXT_REC_MODEL = os.getenv("TEXT_REC_MODEL") or "PP-OCRv5_server_rec"

# Valid PaddleX-style config for the OCR pipeline
PDX_CONFIG = {
    "pipeline_name": "OCR",
    "SubModules": {
        "TextDetection": {
            "model_name": TEXT_DET_MODEL,
            "model_dir": None
        },
        "TextRecognition": {
            "model_name": TEXT_REC_MODEL,
            "model_dir": None
        }
    }
}

app = FastAPI()

@app.on_event("startup")
def load_models():
    app.state.ocr = PaddleOCR(
        lang=OCR_LANG,
        ocr_version=OCR_VERSION,
        use_doc_orientation_classify=USE_DOC_ORI,
        use_doc_unwarping=USE_UNWARP,
        use_textline_orientation=USE_TEXTLINE_ORI,
        device="cpu",
        enable_hpi=False,
        cpu_threads=CPU_THREADS,
        paddlex_config=PDX_CONFIG,
    )
    print(f"Loaded models: det={TEXT_DET_MODEL}, rec={TEXT_REC_MODEL}, lang={OCR_LANG}, version={OCR_VERSION}")

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "lang": OCR_LANG,
        "version": OCR_VERSION,
        "det_model": TEXT_DET_MODEL,
        "rec_model": TEXT_REC_MODEL
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
                "scores": [float(s) for s in preds[0].json.get("rec_scores", [])] if preds else [],
                "boxes": res.json.get("rec_boxes", []),
            })
    return JSONResponse({"results": results})

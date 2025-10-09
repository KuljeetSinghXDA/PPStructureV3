import os
import io
import tempfile
import multiprocessing
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Environment (from .env via Compose env_file)
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))
TEXT_DET_MODEL = os.getenv("TEXT_DET_MODEL", "PP-OCRv5_server_det")
TEXT_REC_MODEL = os.getenv("TEXT_REC_MODEL", "PP-OCRv5_server_rec")

# PP-StructureV3 options (add to .env if needed)
USE_TABLE_REC = os.getenv("USE_TABLE_RECOGNITION", "true").lower() == "true"
USE_FORMULA_REC = os.getenv("USE_FORMULA_RECOGNITION", "false").lower() == "true"
USE_CHART_REC = os.getenv("USE_CHART_RECOGNITION", "false").lower() == "true"
USE_SEAL_REC = os.getenv("USE_SEAL_RECOGNITION", "false").lower() == "true"

# Resolve model folders where official models are cached by PaddleX/PaddleOCR
OFFICIAL_DIR = Path("/root/.paddlex/official_models")
DET_DIR = OFFICIAL_DIR / TEXT_DET_MODEL
REC_DIR = OFFICIAL_DIR / TEXT_REC_MODEL

app = FastAPI()

@app.on_event("startup")
def load_models():
    # General OCR pipeline (PP-OCRv5)
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

    print(f"[startup] OCR: det={TEXT_DET_MODEL} dir_exists={DET_DIR.exists()} | "
          f"rec={TEXT_REC_MODEL} dir_exists={REC_DIR.exists()} | "
          f"lang={OCR_LANG} ocr_version={OCR_VERSION} cpu_threads={CPU_THREADS}")

    # PP-StructureV3 pipeline for PDF/document parsing
    app.state.structure = PPStructureV3(
        lang=OCR_LANG,
        device="cpu",
        cpu_threads=CPU_THREADS,
        use_table_recognition=USE_TABLE_REC,
        use_formula_recognition=USE_FORMULA_REC,
        use_chart_recognition=USE_CHART_REC,
        use_seal_recognition=USE_SEAL_REC,
    )

    print(f"[startup] PP-StructureV3: table={USE_TABLE_REC} formula={USE_FORMULA_REC} "
          f"chart={USE_CHART_REC} seal={USE_SEAL_REC}")

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
        "structure_enabled": True,
    }

def _bytes_to_ndarray(b: bytes):
    with Image.open(io.BytesIO(b)) as im:
        return np.array(im.convert("RGB"))

@app.post("/ocr")
async def ocr_endpoint(files: List[UploadFile] = File(...)):
    """Image-based OCR using PP-OCRv5 pipeline."""
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
async def parse_pdf_endpoint(file: UploadFile = File(...)):
    """PDF/document parsing using PP-StructureV3."""
    # Save uploaded file to a temporary location for PP-StructureV3 to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run PP-StructureV3 on the PDF
        results = app.state.structure.predict(input=tmp_path)
        
        # Collect per-page structured outputs
        pages = []
        for idx, res in enumerate(results, start=1):
            pages.append({
                "page": idx,
                "json": res.json,
                "markdown": res.markdown,
            })
        
        return JSONResponse({
            "filename": file.filename,
            "pages": pages,
        })
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

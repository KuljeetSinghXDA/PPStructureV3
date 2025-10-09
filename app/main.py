import os
import io
import tempfile
import multiprocessing
import numpy as np
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image

def _cpu_threads() -> int:
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# OCR language and pipeline
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")

# Threading
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))

# Server-grade det/rec model names (auto-download and cache under /root/.paddlex)
TEXT_DET_MODEL = os.getenv("TEXT_DET_MODEL", "PP-OCRv5_server_det")
TEXT_REC_MODEL = os.getenv("TEXT_REC_MODEL", "PP-OCRv5_server_rec")

# PP-StructureV3 feature toggles (lab reports focus)
USE_REGION_DET = os.getenv("USE_REGION_DETECTION", "true").lower() == "true"
USE_TABLE_REC = os.getenv("USE_TABLE_RECOGNITION", "true").lower() == "true"
USE_FORMULA_REC = os.getenv("USE_FORMULA_RECOGNITION", "false").lower() == "true"
USE_CHART_REC = os.getenv("USE_CHART_RECOGNITION", "false").lower() == "true"
USE_SEAL_REC = os.getenv("USE_SEAL_RECOGNITION", "false").lower() == "true"
USE_DOC_ORI = os.getenv("USE_DOC_ORI", "true").lower() == "true"
USE_DOC_UNWARP = os.getenv("USE_DOC_UNWARPING", "true").lower() == "true"
USE_TEXTLINE_ORI = os.getenv("USE_TEXTLINE_ORIENTATION", "false").lower() == "true"
CHART_MODEL = os.getenv("CHART_RECOGNITION_MODEL_NAME", None)  # e.g., "PP-Chart2Table"

# Resolve model folders (if already cached, bind them explicitly)
OFFICIAL_DIR = Path("/root/.paddlex/official_models")
DET_DIR = OFFICIAL_DIR / TEXT_DET_MODEL
REC_DIR = OFFICIAL_DIR / TEXT_REC_MODEL

app = FastAPI()

@app.on_event("startup")
def load_models():
    # General OCR (images) using PP-OCRv5
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

    # PP-StructureV3 (PDFs and complex docs)
    struct_kwargs = dict(
        lang=OCR_LANG,
        device="cpu",
        cpu_threads=CPU_THREADS,
        use_region_detection=USE_REGION_DET,
        use_doc_orientation_classify=USE_DOC_ORI,
        use_doc_unwarping=USE_DOC_UNWARP,
        use_textline_orientation=USE_TEXTLINE_ORI,
        use_table_recognition=USE_TABLE_REC,
        use_formula_recognition=USE_FORMULA_REC,
        use_chart_recognition=USE_CHART_REC if CHART_MODEL is None else True,
        use_seal_recognition=USE_SEAL_REC,
    )
    if CHART_MODEL is not None:
        struct_kwargs["chart_recognition_model_name"] = CHART_MODEL

    app.state.structure = PPStructureV3(**struct_kwargs)

    print(f"[startup] OCR: det={TEXT_DET_MODEL}({DET_DIR.exists()}) rec={TEXT_REC_MODEL}({REC_DIR.exists()}) "
          f"| lang={OCR_LANG} ver={OCR_VERSION} cpu_threads={CPU_THREADS}")
    print(f"[startup] StructureV3: region={USE_REGION_DET} table={USE_TABLE_REC} "
          f"formula={USE_FORMULA_REC} chart={USE_CHART_REC} seal={USE_SEAL_REC} "
          f"doc_ori={USE_DOC_ORI} unwarp={USE_DOC_UNWARP} tl_ori={USE_TEXTLINE_ORI} chart_model={CHART_MODEL}")

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
        "structure_flags": {
            "region": USE_REGION_DET,
            "table": USE_TABLE_REC,
            "formula": USE_FORMULA_REC,
            "chart": USE_CHART_REC or (CHART_MODEL is not None),
            "seal": USE_SEAL_REC,
            "doc_orientation": USE_DOC_ORI,
            "doc_unwarping": USE_DOC_UNWARP,
            "textline_orientation": USE_TEXTLINE_ORI,
        },
    }

def _bytes_to_ndarray(b: bytes):
    with Image.open(io.BytesIO(b)) as im:
        return np.array(im.convert("RGB"))

@app.post("/ocr")
async def ocr_endpoint(files: List[UploadFile] = File(...)):
    """Image-based OCR (PP-OCRv5)."""
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
    """PDF/document parsing (PP-StructureV3) -> per-page JSON and Markdown."""
    # Save uploaded PDF to a temp file for the pipeline to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = app.state.structure.predict(input=tmp_path)

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
        Path(tmp_path).unlink(missing_ok=True)

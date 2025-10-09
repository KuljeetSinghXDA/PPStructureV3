import os
import io
import json
import glob
import shutil
import tempfile
import multiprocessing
import numpy as np
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from PIL import Image

from paddleocr import PaddleOCR, PPStructureV3
from .guide import GUIDE_CONTENT

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Environment (from .env via Compose env_file)
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))
TEXT_DET_MODEL = os.getenv("TEXT_DET_MODEL", "PP-OCRv5_server_det")
TEXT_REC_MODEL = os.getenv("TEXT_REC_MODEL", "PP-OCRv5_server_rec")

# Structure defaults
STRUCT_DEFAULT_FORMAT = os.getenv("STRUCT_DEFAULT_FORMAT", "json").lower()
USE_DOC_ORI = os.getenv("USE_DOC_ORI", "false").lower() == "true"
USE_UNWARP = os.getenv("USE_UNWARP", "false").lower() == "true"
USE_TEXTLINE_ORI = os.getenv("USE_TEXTLINE_ORI", "false").lower() == "true"

# Official models cache
OFFICIAL_DIR = Path("/root/.paddlex/official_models")
DET_DIR = OFFICIAL_DIR / TEXT_DET_MODEL
REC_DIR = OFFICIAL_DIR / TEXT_REC_MODEL

app = FastAPI(
    title="PaddleOCR & PP-StructureV3 API",
    description="High-accuracy OCR and document parsing optimized for lab reports and medical documents",
    version="1.0.0"
)

def _bytes_to_ndarray(b: bytes):
    with Image.open(io.BytesIO(b)) as im:
        return np.array(im.convert("RGB"))

@app.on_event("startup")
def load_models():
    # OCR: force server det/rec; pass names to auto-download if missing; bind dirs if cached
    ocr_kwargs = dict(
        lang=OCR_LANG,
        ocr_version=OCR_VERSION,
        device="cpu",
        enable_hpi=False,
        cpu_threads=CPU_THREADS,
        text_detection_model_name=TEXT_DET_MODEL,
        text_recognition_model_name=TEXT_REC_MODEL,
        use_doc_orientation_classify=USE_DOC_ORI,
        use_doc_unwarping=USE_UNWARP,
        use_textline_orientation=USE_TEXTLINE_ORI,
    )
    if DET_DIR.exists():
        ocr_kwargs["det_model_dir"] = str(DET_DIR)
    if REC_DIR.exists():
        ocr_kwargs["rec_model_dir"] = str(REC_DIR)
    app.state.ocr = PaddleOCR(**ocr_kwargs)

    # PP-StructureV3: document parsing (layout, tables); optimized for English
    app.state.struct = PPStructureV3(
        lang=OCR_LANG,
        device="cpu",
        use_doc_orientation_classify=USE_DOC_ORI,
        use_doc_unwarping=USE_UNWARP,
        use_textline_orientation=USE_TEXTLINE_ORI,
        cpu_threads=CPU_THREADS
    )

    print(f"[startup] OCR det={TEXT_DET_MODEL}({DET_DIR.exists()}) rec={TEXT_REC_MODEL}({REC_DIR.exists()}) | "
          f"struct_lang={OCR_LANG} fmt={STRUCT_DEFAULT_FORMAT} cpu_threads={CPU_THREADS}")

@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint redirects to guide"""
    return HTMLResponse(content="""
    <html>
        <head><title>PaddleOCR API</title></head>
        <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto;">
            <h1>PaddleOCR & PP-StructureV3 API</h1>
            <p>High-accuracy OCR and document parsing for lab reports and medical documents.</p>
            <ul>
                <li><a href="/guide">📖 API Guide & Documentation</a></li>
                <li><a href="/docs">🔧 Interactive API Docs (Swagger)</a></li>
                <li><a href="/healthz">💚 Health Check</a></li>
            </ul>
            <h3>Quick Start</h3>
            <p><strong>Simple Text Extraction:</strong> POST to <code>/ocr</code></p>
            <p><strong>Document Parsing (Tables, Formulas):</strong> POST to <code>/structure</code></p>
        </body>
    </html>
    """)

@app.get("/guide", response_class=PlainTextResponse)
def guide():
    """Complete API guide with examples and use cases"""
    return GUIDE_CONTENT

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
        "struct_default_format": STRUCT_DEFAULT_FORMAT
    }

@app.post("/ocr")
async def ocr_endpoint(files: List[UploadFile] = File(...)):
    """
    Fast text recognition using PP-OCRv5 server models.
    Best for: Simple text extraction, receipts, labels, forms.
    """
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

@app.post("/structure")
async def structure_endpoint(
    files: List[UploadFile] = File(...),
    output_format: Literal["json", "markdown"] = Query(None, description="Output format: json or markdown")
):
    """
    Advanced document parsing with PP-StructureV3.
    Supports: Layout detection, table extraction, formula recognition, reading order.
    Best for: Lab reports with tables, invoices, scientific papers.
    """
    ofmt = (output_format or STRUCT_DEFAULT_FORMAT).lower()
    if ofmt not in ("json", "markdown"):
        ofmt = "json"

    tmpdir = tempfile.mkdtemp(prefix="ppstructv3_")
    payload = []

    try:
        for f in files:
            content = await f.read()
            img = _bytes_to_ndarray(content)
            outputs = app.state.struct.predict(input=img)

            if ofmt == "json":
                for res in outputs:
                    res.save_to_json(save_path=tmpdir)
                json_files = sorted(glob.glob(f"{tmpdir}/*.json"))
                json_docs = []
                for jf in json_files:
                    try:
                        with open(jf, "r", encoding="utf-8") as fh:
                            json_docs.append(json.load(fh))
                    except Exception:
                        pass
                payload.append({"filename": f.filename, "documents": json_docs})
            else:
                for res in outputs:
                    res.save_to_markdown(save_path=tmpdir)
                md_files = sorted(glob.glob(f"{tmpdir}/*.md"))
                md_docs = []
                for mf in md_files:
                    try:
                        with open(mf, "r", encoding="utf-8") as fh:
                            md_docs.append(fh.read())
                    except Exception:
                        pass
                payload.append({"filename": f.filename, "documents_markdown": md_docs})

        if ofmt == "json":
            return JSONResponse({"results": payload})
        return PlainTextResponse(
            "\n\n".join(
                f"# {item['filename']}\n\n" + "\n\n".join(item["documents_markdown"])
                for item in payload
            ),
            media_type="text/markdown"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

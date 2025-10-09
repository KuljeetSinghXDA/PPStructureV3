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
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from paddleocr import PaddleOCR, PPStructureV3

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Environment (from .env or Dokploy)
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

# Display base URL in help (domain first, fallback to local)
BASE_URL = os.getenv("BASE_URL", "https://ocr.aiengineops.com")

app = FastAPI()

def _bytes_to_ndarray(b: bytes):
    with Image.open(io.BytesIO(b)) as im:
        return np.array(im.convert("RGB"))

@app.on_event("startup")
def load_models():
    # OCR: lock to server det/rec if cached; also pass names to auto-download if missing
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

    # PP-StructureV3: optimized for English document parsing (layout/tables/reading order)
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

def _usage_text() -> str:
    return f"""
PaddleOCR 3.x + PP-StructureV3 API
Base URL: {BASE_URL}

Overview
- /healthz (GET): Service status and loaded model info. 
- /ocr (POST): General OCR for images using PP-OCRv5 server models (English), returns text lines, scores, and boxes. 
- /structure (POST): PP-StructureV3 document parsing (layout + tables + reading order) with output_format={{{json|markdown}}}, tuned for English lab reports and other documents. 

Notes
- Send one or more images via multipart/form-data with key "files". 
- Supported formats: JPEG, PNG, TIFF; for PDF, convert to page images client-side before sending. 
- For best results on lab reports: scan at 300 DPI+, good lighting, avoid skew, crop margins if possible. 

1) /healthz (GET)
- Check status and current configuration. 
Example:
  curl -s {BASE_URL}/healthz | jq .

2) /ocr (POST)
Purpose
- High-accuracy text extraction from images using PP-OCRv5 server-grade detection + recognition (English). 

Request
- Method: POST
- URL: {BASE_URL}/ocr
- Content-Type: multipart/form-data
- Fields:
  - files: One or more image files (repeat this field to send multiple images). 

Response (JSON)
- results: List of per-file results.
  - filename: Original filename.
  - texts: Recognized text lines.
  - scores: Confidence scores per line.
  - boxes: Quadrilateral boxes per line in image coordinates.

Examples
- cURL (single file):
  curl -X POST \\
    -F "files=@/path/lab_report_page1.jpg" \\
    {BASE_URL}/ocr

- cURL (multiple files):
  curl -X POST \\
    -F "files=@/path/lab_report_page1.jpg" \\
    -F "files=@/path/lab_report_page2.jpg" \\
    {BASE_URL}/ocr

- Python (requests):
  import requests
  files = [('files', open('lab_report_page1.jpg','rb')),
           ('files', open('lab_report_page2.jpg','rb'))]
  r = requests.post("{BASE_URL}/ocr", files=files)
  print(r.json())

- Node (fetch):
  import FormData from "form-data";
  import fs from "fs";
  const fd = new FormData();
  fd.append("files", fs.createReadStream("lab_report_page1.jpg"));
  fd.append("files", fs.createReadStream("lab_report_page2.jpg"));
  const res = await fetch("{BASE_URL}/ocr", {{ method: "POST", body: fd }});
  console.log(await res.json());

Typical lab report post-processing
- Join lines by reading order or filter lines by keywords like "Hemoglobin", "WBC", "Reference Range". 
- Use regex to capture numeric values and units (e.g., "13.5 g/dL", "7500 /µL") and map to structured fields. 

3) /structure (POST)
Purpose
- Full document parsing pipeline (PP-StructureV3) for layout segmentation, table recognition, reading order, and chart/table extraction. 

Request
- Method: POST
- URL: {BASE_URL}/structure?output_format={{json|markdown}}
- Default: output_format=json
- Content-Type: multipart/form-data
- Fields:
  - files: One or more image files (repeat this field for multiple images). 

Response
- JSON mode: "results" returns per-file document JSONs from PP-StructureV3 (layout blocks, tables, cells, order). 
- Markdown mode: plain text concatenation of per-file markdown outputs (.md) for easy downstream consumption. 

Examples (lab reports)
- cURL JSON (default):
  curl -X POST \\
    -F "files=@/path/lab_report_page1.jpg" \\
    -F "files=@/path/lab_report_page2.jpg" \\
    "{BASE_URL}/structure?output_format=json" | jq .

- cURL Markdown:
  curl -X POST \\
    -F "files=@/path/lab_report_page1.jpg" \\
    "{BASE_URL}/structure?output_format=markdown"

- Python (requests, JSON):
  import requests
  files = [('files', open('lab_report_page1.jpg','rb'))]
  r = requests.post("{BASE_URL}/structure?output_format=json", files=files)
  print(r.json())

- Node (fetch, Markdown):
  import FormData from "form-data";
  import fs from "fs";
  const fd = new FormData();
  fd.append("files", fs.createReadStream("lab_report_page1.jpg"));
  const res = await fetch("{BASE_URL}/structure?output_format=markdown", {{ method: "POST", body: fd }});
  console.log(await res.text());

Interpreting outputs
- JSON: parse "layout" blocks, "tables" with cells, and reading order; map analyte names to values and reference ranges. 
- Markdown: ready-to-ingest textual structure for quick display or LLM summarization; still parseable with regex if needed. 

Beyond lab reports (other documents/images)
- Invoices/receipts: use /structure JSON to extract tables and totals; /ocr for line‑level text if layout is simple. 
- Forms: /structure identifies fields and structure; combine with /ocr to read free‑text answers. 
- Academic pages with charts/tables: PP-StructureV3 extracts tables and chart descriptions to JSON/MD. 
- ID cards/simple certificates: /ocr is typically sufficient; ensure face/ID regions are not occluded. 

Operational tips
- Batch multiple pages in one request by repeating -F files=... in multipart form. 
- If throughput is needed, parallelize requests client‑side; for stability, keep OMP_NUM_THREADS=1 and tune CPU_THREADS. 
- Preprocess: de-skew and crop; convert PDFs to images at ~300 DPI; use lossless PNG for critical numeric tables. 

This help is available at:
- GET {BASE_URL}/
- GET {BASE_URL}/help
- GET {BASE_URL}/ocr
- GET {BASE_URL}/structure
"""

@app.get("/")
def root_usage():
    return PlainTextResponse(_usage_text(), status_code=200)

@app.get("/help")
def help_usage():
    return PlainTextResponse(_usage_text(), status_code=200)

@app.get("/ocr")
def ocr_help():
    # GET help for OCR endpoint (POST is the actual inference)
    doc = _usage_text()
    # Extract OCR section for brevity if desired; here we return the full usage for completeness
    return PlainTextResponse(doc, status_code=200)

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

@app.get("/structure")
def structure_help():
    # GET help for Structure endpoint (POST is the actual inference)
    return PlainTextResponse(_usage_text(), status_code=200)

@app.post("/structure")
async def structure_endpoint(
    files: List[UploadFile] = File(...),
    output_format: Literal["json", "markdown"] = Query(None)
):
    ofmt = (output_format or STRUCT_DEFAULT_FORMAT).lower()
    if ofmt not in ("json", "markdown"):
        ofmt = "json"

    # Use a temp dir to capture pipeline file outputs
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
        return PlainTextResponse("\n\n".join(
            f"# {item['filename']}\n\n" + "\n\n".join(item.get("documents_markdown", []))
            for item in payload
        ))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

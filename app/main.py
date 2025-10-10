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

app = FastAPI()

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

    # PP-StructureV3: document parsing (layout, tables) for English
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
        for idx, f in enumerate(files):
            content = await f.read()
            filename = (f.filename or "").lower()
            content_type = getattr(f, "content_type", "") or ""
            is_pdf = filename.endswith(".pdf") or (content_type == "application/pdf")

            # Use a per-file output subdir to avoid mixing results across multiple inputs
            out_dir = os.path.join(tmpdir, f"out_{idx}")
            os.makedirs(out_dir, exist_ok=True)

            if is_pdf:
                # Save PDF to a temp file and let PP-StructureV3 handle pagination internally
                tf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                try:
                    tf.write(content)
                    tf.flush()
                    tmp_pdf_path = tf.name
                finally:
                    tf.close()

                try:
                    outputs = app.state.struct.predict(input=tmp_pdf_path)
                finally:
                    # Cleanup the temporary PDF regardless of inference outcome
                    try:
                        os.remove(tmp_pdf_path)
                    except Exception:
                        pass
            else:
                # Image input: convert to RGB ndarray
                img = _bytes_to_ndarray(content)
                outputs = app.state.struct.predict(input=img)

            # Save per-file results via helper methods
            if ofmt == "json":
                for res in outputs:
                    res.save_to_json(save_path=out_dir)
                json_files = sorted(glob.glob(f"{out_dir}/*.json"))
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
                    res.save_to_markdown(save_path=out_dir)
                md_files = sorted(glob.glob(f"{out_dir}/*.md"))
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
            "\n\n".join(f"# {item['filename']}\n\n" + "\n\n".join(item["documents_markdown"]) for item in payload),
            media_type="text/markdown"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

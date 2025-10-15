# ppstructurev3_server.py
import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# ---- PaddleOCR import (ensure paddleocr>=3.0 / 3.2.0 is installed) ----
from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values) =================
DEVICE = "cpu"
CPU_THREADS = 4

# Optional accuracy boosters (toggle at pipeline init)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model overrides (you can change these to point to other official models)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (None means pipeline defaults)
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# Optional: where to persist downloaded model files / cache (default uses user home)
MODEL_CACHE_DIR = os.environ.get("PADDLE_MODEL_CACHE", str(Path.home() / ".paddlex"))

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize pipeline once at app start; parameters follow docs
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=True,
        enable_hpi=False,
        cpu_threads=CPU_THREADS,
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
        paddlex_model_cache_dir=MODEL_CACHE_DIR if "paddlex_model_cache_dir" in PPStructureV3.__init__.__code__.co_varnames else None,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------- Helper utilities -----------------
def _secure_extension(filename: str) -> str:
    return Path(filename).suffix.lower()

async def _save_upload_to_tempfile(upload: UploadFile, tmp_dir: Path) -> Path:
    """
    Save UploadFile into a temp path and return the Path.
    """
    dest = tmp_dir / Path(upload.filename).name
    with dest.open("wb") as f:
        # stream in chunks to avoid memory blowup
        while True:
            chunk = await upload.read(2 ** 16)
            if not chunk:
                break
            f.write(chunk)
    return dest

def _enforce_limits(file_path: Path):
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)")

def _read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

# ----------------- /parse endpoint -----------------
@app.post("/parse", response_class=JSONResponse)
async def parse_file(
    file: UploadFile = File(...),
    # optional overrides at request time (safe subset)
    use_doc_orientation_classify: bool = Query(USE_DOC_ORIENTATION_CLASSIFY),
    use_doc_unwarping: bool = Query(USE_DOC_UNWARPING),
    use_textline_orientation: bool = Query(USE_TEXTLINE_ORIENTATION),
    # allow toggling table/chart/formula processing per-request
    use_table_recognition: bool = Query(USE_TABLE_RECOGNITION),
    use_formula_recognition: bool = Query(USE_FORMULA_RECOGNITION),
    use_chart_recognition: bool = Query(USE_CHART_RECOGNITION),
    # quick model name overrides (optional)
    text_recognition_model_name: Optional[str] = Query(None),
):
    # basic validation & save upload
    tmp_dir = Path(tempfile.mkdtemp(prefix="ppstrv3_"))
    try:
        saved_path = await _save_upload_to_tempfile(file, tmp_dir)
        _enforce_limits(saved_path)

        # Acquire semaphore to limit concurrency
        acquired = app.state.predict_sem.acquire(timeout=60)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy — try again later")

        try:
            # Build per-request predict kwargs (only the documented ones)
            predict_kwargs: Dict[str, Any] = {
                # pipeline doc: pass input as path to file (pdf or image). PDF pages will be processed individually.
                "input": str(saved_path),
                "use_doc_orientation_classify": bool(use_doc_orientation_classify),
                "use_doc_unwarping": bool(use_doc_unwarping),
                "use_textline_orientation": bool(use_textline_orientation),
                "use_table_recognition": bool(use_table_recognition),
                "use_formula_recognition": bool(use_formula_recognition),
                "use_chart_recognition": bool(use_chart_recognition),
            }
            if text_recognition_model_name:
                predict_kwargs["text_recognition_model_name"] = text_recognition_model_name

            # Run predict in threadpool because pipeline is CPU-bound / blocking.
            pipeline = app.state.pipeline
            raw_results = await run_in_threadpool(pipeline.predict, **predict_kwargs)

            # raw_results is an iterable/list of "result" objects (one per page/image).
            # The docs show using `res.save_to_json(save_path=...)` and `res.markdown`. We'll persist per-page JSON into tmp_dir/output_json
            output_json_dir = tmp_dir / "output_json"
            output_json_dir.mkdir(exist_ok=True)

            pages = []
            markdown_pages = []
            for idx, res in enumerate(raw_results):
                # res.save_to_json(save_path=...) is in docs — use it to get the pipeline's official JSON shape.
                try:
                    # this creates files like <inputname>_page_<n>.json in the path
                    res.save_to_json(save_path=str(output_json_dir))
                except Exception as e:
                    # If save_to_json unexpectedly fails for some versions, fall back to best-effort serialization.
                    # Best-effort: attempt to access res.__dict__ or res.markdown
                    pass

            # collect all JSON files created in output_json_dir (sorted)
            json_files = sorted(output_json_dir.glob("*.json"))
            for jf in json_files:
                j = _read_json_file(jf)
                pages.append(j)

            # collect markdown text if provided by results (res.markdown)
            # The predict result objects expose 'markdown' as documented; best-effort collect
            for res in raw_results:
                try:
                    md = res.markdown if hasattr(res, "markdown") else None
                    if md and isinstance(md, dict):
                        markdown_pages.append(md)
                    else:
                        # fallback: try to extract text content if present
                        markdown_pages.append({"text": md or ""})
                except Exception:
                    markdown_pages.append({"text": ""})

            # Concatenate markdown text across pages manually (avoid relying on pipeline helper that may be missing)
            concatenated_markdown_text = "\n\n".join(p.get("text", "") for p in markdown_pages if p.get("text"))
            # Collect markdown images mapping (if present)
            concatenated_markdown_images = {}
            for p in markdown_pages:
                imgs = p.get("markdown_images", {}) if isinstance(p, dict) else {}
                if isinstance(imgs, dict):
                    concatenated_markdown_images.update(imgs)

            response_payload = {
                "input_filename": file.filename,
                "num_pages": len(pages),
                "pages": pages,
                "markdown_pages": markdown_pages,
                "markdown": {
                    "text": concatenated_markdown_text,
                    "images": list(concatenated_markdown_images.keys()),
                },
            }

            return JSONResponse(content=response_payload)

        finally:
            # always release semaphore & cleanup saved file
            app.state.predict_sem.release()
    finally:
        # NOTE: keep temp directory long enough for caller to fetch images if you plan to expose them.
        # For safety we clean up here; if you want to persist outputs, change this behavior.
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

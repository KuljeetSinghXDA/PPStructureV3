import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values) =================
DEVICE = "cpu"  # e.g., "cpu", "gpu:0", "gpu:0,1" for multi-GPU
CPU_THREADS = 4

# Optional accuracy boosters
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
# Note: Region detection and seal recognition are available but disabled by default

# Model overrides (names per PP-StructureV3 docs)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANeXt_wired"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANeXt_wireless"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (None = use defaults)
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
CHUNK_SIZE = 1024 * 1024  # 1MB

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
        # model names
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        # thresholds/limits
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        # feature toggles
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file extension: {ext}")

async def _save_upload_to_temp(upload: UploadFile, dst_path: Path) -> None:
    total = 0
    with open(dst_path, "wb") as out:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
            out.write(chunk)
    await upload.close()

def _load_json_files(dir_path: Path) -> List[dict]:
    items = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                items.append(json.load(f))
        except Exception:
            # Skip malformed JSON (unlikely)
            continue
    return items

@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    output: Literal["json", "markdown"] = Query("json", description="Return JSON per page or combined Markdown"),
):
    _validate_extension(file.filename)
    suffix = Path(file.filename).suffix.lower()

    work_dir = Path(tempfile.mkdtemp(prefix="ppstructv3_"))
    in_path = work_dir / f"input{suffix}"
    try:
        await _save_upload_to_temp(file, in_path)

        # enforce single-flight prediction if desired
        await run_in_threadpool(app.state.predict_sem.acquire)
        try:
            # Run prediction in threadpool (blocking call)
            results = await run_in_threadpool(app.state.pipeline.predict, str(in_path))

            if output == "markdown":
                # Combine markdown across pages (PDF or multipage)
                markdown_list = []
                for res in results:
                    # Each res has `.markdown` per docs
                    md_info = getattr(res, "markdown", None)
                    if md_info is None:
                        # Fallback: permit save_to_markdown then read, but normally .markdown exists
                        pass
                    else:
                        markdown_list.append(md_info)
                combined = await run_in_threadpool(app.state.pipeline.concatenate_markdown_pages, markdown_list)
                return PlainTextResponse(combined)

            # Default: JSON output
            # Persist each page's JSON to a tmp dir using the official API then load it back
            out_dir = work_dir / "json"
            out_dir.mkdir(parents=True, exist_ok=True)
            for res in results:
                # Each res supports save_to_json(save_path=...)
                res.save_to_json(save_path=str(out_dir))
            pages = _load_json_files(out_dir)
            return JSONResponse(content={"pages": pages})
        finally:
            app.state.predict_sem.release()
    except HTTPException:
        # Bubble up validation errors
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict failed: {type(e).__name__}: {e}")
    finally:
        # Clean workspace
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

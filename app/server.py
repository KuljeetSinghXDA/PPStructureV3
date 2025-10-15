import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values) =================
DEVICE = "cpu"
CPU_THREADS = 4

# Optional accuracy boosters
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model overrides
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters
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

# ================= Utilities =================
def _ext_ok(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS

def _ensure_size_and_save(upload: UploadFile, dst_path: Path, max_mb: int) -> None:
    max_bytes = max_mb * 1024 * 1024
    total = 0
    with dst_path.open("wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail=f"File exceeds {max_mb} MB limit")
            out.write(chunk)
    upload.file.seek(0)

def _result_to_dict(res) -> dict:
    # Prefer stable dictionary extraction without disk I/O
    if hasattr(res, "to_dict") and callable(getattr(res, "to_dict")):
        return res.to_dict()
    if hasattr(res, "res"):
        return res.res
    # Fallback: use string representation if available
    try:
        s = str(res)
        # Not guaranteed JSON, best-effort
        return {"repr": s}
    except Exception:
        return {}

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
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
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse", response_class=JSONResponse)
async def parse(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Query(default=None, description="Direct image/PDF URL supported by PP-StructureV3"),
    response_format: Literal["json"] = Query(default="json"),
):
    if not file and not url:
        raise HTTPException(status_code=422, detail="Provide either an uploaded file or a URL")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ppstructv3_"))
    save_path = None
    input_path: str

    try:
        if file:
            if not file.filename:
                raise HTTPException(status_code=422, detail="Uploaded file has no filename")
            if not _ext_ok(file.filename):
                raise HTTPException(status_code=422, detail=f"Unsupported file type: {Path(file.filename).suffix}")
            save_path = tmp_dir / Path(file.filename).name
            _ensure_size_and_save(file, save_path, MAX_FILE_SIZE_MB)
            input_path = str(save_path)
        else:
            # URL path goes directly to pipeline
            input_path = url  # type: ignore

        # Acquire semaphore in threadpool to avoid blocking event loop
        await run_in_threadpool(app.state.predict_sem.acquire)

        def _run_predict():
            return app.state.pipeline.predict(input=input_path)

        outputs = await run_in_threadpool(_run_predict)

        # Build response
        pages = []
        for res in outputs:
            pages.append(_result_to_dict(res))

        return {"pages": pages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        # Always release semaphore and clean up temp files
        try:
            if getattr(app.state, "predict_sem", None):
                app.state.predict_sem.release()
        except Exception:
            pass
        try:
            if save_path and save_path.exists():
                # remove temp dir recursively
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

import os
import uuid
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

# Runtime toggles
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

# Model overrides (names from PaddleOCR model zoo; can be left as None to use defaults)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (use None for defaults per docs)
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

# Output root
OUTPUT_ROOT = Path("outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

def _safe_write_upload(tmp_dir: Path, up: UploadFile) -> Path:
    # Read in-memory (50MB cap default)
    content = up.file.read()
    size_mb = _bytes_to_mb(len(content))
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large: {size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB")
    # Persist to disk
    in_path = tmp_dir / up.filename
    with open(in_path, "wb") as f:
        f.write(content)
    return in_path

def _collect_artifacts(out_dir: Path) -> dict:
    json_files = sorted([str(p) for p in out_dir.glob("*.json")])
    md_files = sorted([str(p) for p in out_dir.glob("*.md")])
    pages = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                pages.append(json.load(f))
        except Exception:
            # Keep going; still return manifest
            continue
    combined_markdown = ""
    for mf in md_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                combined_markdown += f.read().rstrip() + "\n\n"
        except Exception:
            continue
    return {
        "pages": pages,
        "json_files": json_files,
        "markdown_files": md_files,
        "markdown_combined": combined_markdown.strip()
    }

def _run_predict_and_save(pipeline: PPStructureV3, input_path: Path, save_dir: Path):
    # Predict using pipeline; save page-wise JSON and Markdown via result helpers
    outputs = pipeline.predict(str(input_path))
    for res in outputs:
        # Official helpers to persist results
        res.save_to_json(save_path=str(save_dir))
        res.save_to_markdown(save_path=str(save_dir))

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize PP-StructureV3 with documented flags/parameters
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
    # No explicit teardown required; FastAPI will clean app.state on shutdown

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    save_as_job: Optional[bool] = Query(True, description="Persist results under outputs/<job_id>"),
):
    if not _ext_ok(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type; allowed: {sorted(ALLOWED_EXTENSIONS)}")

    # Prevent concurrent runs beyond limit
    if not app.state.predict_sem.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Too many concurrent requests; please retry later")

    try:
        job_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory(prefix="ppsv3_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            input_path = _safe_write_upload(tmpdir, file)

            # Output directory (temp first, then move if requested)
            temp_out_dir = tmpdir / "out"
            temp_out_dir.mkdir(parents=True, exist_ok=True)

            # Run predict in threadpool to avoid blocking event loop
            await run_in_threadpool(
                _run_predict_and_save,
                app.state.pipeline,
                input_path,
                temp_out_dir
            )

            # Finalize output location
            if save_as_job:
                final_dir = OUTPUT_ROOT / job_id
                final_dir.mkdir(parents=True, exist_ok=True)
                for p in temp_out_dir.iterdir():
                    shutil.move(str(p), str(final_dir / p.name))
                out_dir = final_dir
            else:
                out_dir = temp_out_dir

            artifacts = _collect_artifacts(out_dir)

            return JSONResponse(
                content={
                    "job_id": job_id if save_as_job else None,
                    "input_filename": file.filename,
                    "output_dir": str(out_dir),
                    "artifacts": artifacts,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        app.state.predict_sem.release()

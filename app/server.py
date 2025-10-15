import os
import io
import zipfile
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.concurrency import run_in_threadpool

# =====================================================================
# PaddleOCR / PP-StructureV3
# Ensure: pip install "paddlepaddle>=3.2.0" and "paddleocr>=3.2.0"
# =====================================================================
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
# NOTE: General OCR and other sub-pipelines are enabled internally by default unless toggled off in the constructor.

# Model overrides (official names; use as needed)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (None -> pipeline defaults)
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None

# Execution backends/accelerations
ENABLE_HPI = False
ENABLE_MKLDNN = True

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= Utility helpers =================
def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

def _bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

async def _save_upload_to_temp(upload: UploadFile, dst_dir: Path) -> Tuple[Path, int]:
    """
    Stream the UploadFile to a temporary path on disk, enforcing MAX_FILE_SIZE_MB.
    Returns (path, size_bytes).
    """
    _validate_extension(upload.filename)
    suffix = Path(upload.filename).suffix.lower()
    dst_path = dst_dir / f"input{suffix}"

    total = 0
    chunk_size = 1024 * 1024  # 1MB
    with open(dst_path, "wb") as f:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE_MB * 1024 * 1024:
                # Stop early to avoid writing oversized files
                try:
                    upload.file.close()
                except Exception:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {_bytes_to_mb(total):.2f}MB > {MAX_FILE_SIZE_MB}MB",
                )
            f.write(chunk)
    await upload.close()
    return dst_path, total

def _cleanup_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _zip_directory(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                file_path = Path(root) / name
                arcname = file_path.relative_to(src_dir)
                zf.write(file_path, arcname.as_posix())

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Construct the PP-StructureV3 pipeline using official parameter names (aligned with CLI flags).
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,

        # Model selection
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,

        # Fine-grained thresholds/limits (None -> defaults)
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,

        # Sub-pipeline toggles
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

# ================= /parse endpoint =================
@app.post(
    "/parse",
    responses={
        200: {"content": {"application/json": {}, "application/zip": {}}},
        400: {"description": "Bad Request"},
        413: {"description": "Payload Too Large"},
        500: {"description": "Internal Server Error"},
    },
)
async def parse(
    file: UploadFile = File(..., description="PDF or image"),
    as_zip: bool = Query(
        False,
        description="Return a ZIP containing JSON pages, Markdown, and images. If false, return JSON body with Markdown text.",
    ),
    save_artifacts: bool = Query(
        True,
        description="Persist artifacts on disk (within a temp session dir). Required for ZIP responses."
    ),
):
    """
    Parse a PDF/image document into structured JSON pages and a combined Markdown file.
    - If as_zip=True: returns a single ZIP file containing all artifacts.
    - If as_zip=False: returns a JSON body with parse metadata and the Markdown text content.
    """
    # Session temp dir
    session_dir = Path(tempfile.mkdtemp(prefix="ppstructv3_"))
    input_dir = session_dir / "input"
    output_dir = session_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save upload to disk with size guard
        input_path, nbytes = await _save_upload_to_temp(file, input_dir)

        # Acquire semaphore to limit concurrent predictions
        sem = app.state.predict_sem
        await run_in_threadpool(sem.acquire)
        try:
            # Run prediction in a thread to avoid blocking the event loop
            def _do_predict():
                # We intentionally do not pass save_path here to keep control of filenames;
                # we'll explicitly save JSON and Markdown below.
                return app.state.pipeline.predict(input=str(input_path))

            results = await run_in_threadpool(_do_predict)

            # Persist page JSON files and collect Markdown page dicts
            markdown_pages = []
            json_paths: List[str] = []

            for idx, res in enumerate(results):
                # Save per-page JSON using official API
                # This writes files into output_dir with pipeline-defined naming
                res.save_to_json(save_path=str(output_dir))

                # Capture Markdown page info (contains markdown text and images)
                md_info = res.markdown  # dict with "markdown" and "markdown_images"
                markdown_pages.append(md_info)

            # Assemble full Markdown from page fragments using the official helper
            markdown_text = app.state.pipeline.concatenate_markdown_pages(markdown_pages)

            # Save Markdown file
            md_path = output_dir / f"{input_path.stem}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            # Save Markdown referenced images
            for md in markdown_pages:
                md_images = md.get("markdown_images", {}) or {}
                for rel_path, image in md_images.items():
                    img_path = output_dir / rel_path
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    # PIL Image object; save with inferred format
                    image.save(img_path)

            # Collect JSON file names for metadata (list all *.json in output_dir)
            for p in sorted(output_dir.rglob("*.json")):
                json_paths.append(p.name)

            if as_zip:
                # Package all artifacts into a single ZIP
                zip_path = session_dir / f"{input_path.stem}_ppstructurev3.zip"
                _zip_directory(output_dir, zip_path)
                # Clean up everything after sending
                bg = BackgroundTasks()
                bg.add_task(_cleanup_dir, session_dir)
                return FileResponse(
                    path=zip_path,
                    filename=zip_path.name,
                    media_type="application/zip",
                    background=bg,
                )

            # Otherwise, return a JSON body (and keep a short-lived temp folder)
            resp = {
                "filename": Path(file.filename).name,
                "size_mb": round(_bytes_to_mb(nbytes), 2),
                "pages": len(results),
                "artifacts_dir": "outputs",
                "json_files": json_paths,  # page JSONs saved by the pipeline
                "markdown_file": md_path.name,
                "markdown_text": markdown_text,
                "note": "For a single downloadable archive (JSONs, .md, images), call with ?as_zip=true",
            }

            # Clean up session dir unless the caller wants artifacts preserved (ZIP path uses BackgroundTasks)
            if not save_artifacts:
                _cleanup_dir(session_dir)

            return JSONResponse(resp)

        finally:
            # Always release the semaphore
            sem.release()

    except HTTPException:
        # Preserve HTTPException as-is
        # Clean up temp dir on error
        _cleanup_dir(session_dir)
        raise
    except Exception as e:
        _cleanup_dir(session_dir)
        raise HTTPException(status_code=500, detail=f"Parse failed: {e}") from e

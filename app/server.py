from __future__ import annotations

import io
import os
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

import orjson
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

from .ppengine import PPStructureEngine
from .pdf_utils import render_pdf_to_images
from .normalize import normalize_ppstructure_result
from .markdownify import pages_to_markdown

# ================= Core Configuration Defaults =================
DEVICE = os.getenv("PP_DEVICE", "cpu")
CPU_THREADS = int(os.getenv("PP_CPU_THREADS", "4"))
ENABLE_HPI = os.getenv("PP_ENABLE_HPI", "0") == "1"  # GPU-oriented; off on CPU
ENABLE_MKLDNN = os.getenv("PP_ENABLE_MKLDNN", "1") == "1"

# Optional accuracy boosters (off by default)
USE_DOC_ORIENTATION_CLASSIFY = os.getenv("PP_USE_DOC_ORIENTATION", "0") == "1"
USE_DOC_UNWARPING = os.getenv("PP_USE_DOC_UNWARPING", "0") == "1"
USE_TEXTLINE_ORIENTATION = os.getenv("PP_USE_TEXTLINE_ORIENTATION", "0") == "1"

# Subpipeline toggles
USE_TABLE_RECOGNITION = os.getenv("PP_USE_TABLE", "1") == "1"
USE_FORMULA_RECOGNITION = os.getenv("PP_USE_FORMULA", "0") == "1"
USE_CHART_RECOGNITION = os.getenv("PP_USE_CHART", "0") == "1"

# Model overrides (None => use PaddleOCR 3.x defaults, e.g., PP-OCRv5_server)
LAYOUT_DETECTION_MODEL_NAME = os.getenv("PP_LAYOUT_MODEL_NAME")  # e.g., "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = os.getenv("PP_TEXT_DET_MODEL_NAME")  # e.g., "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = os.getenv("PP_TEXT_REC_MODEL_NAME")  # e.g., "en_PP-OCRv5_server_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = os.getenv("PP_WIRED_TABLE_MODEL_NAME")  # e.g., "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = os.getenv("PP_WIRELESS_TABLE_MODEL_NAME")  # e.g., "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = os.getenv("PP_TABLE_CLS_MODEL_NAME")  # e.g., "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = os.getenv("PP_FORMULA_MODEL_NAME")  # e.g., "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = os.getenv("PP_CHART_MODEL_NAME")  # e.g., "PP-Chart2Table"

# Thresholds and batch sizes (None => library defaults)
LAYOUT_THRESHOLD = os.getenv("PP_LAYOUT_THRESHOLD")
TEXT_DET_THRESH = os.getenv("PP_TEXT_DET_THRESH")
TEXT_DET_BOX_THRESH = os.getenv("PP_TEXT_DET_BOX_THRESH")
TEXT_DET_UNCLIP_RATIO = os.getenv("PP_TEXT_DET_UNCLIP_RATIO")
TEXT_DET_LIMIT_SIDE_LEN = os.getenv("PP_TEXT_DET_LIMIT_SIDE_LEN")
TEXT_DET_LIMIT_TYPE = os.getenv("PP_TEXT_DET_LIMIT_TYPE")
TEXT_REC_SCORE_THRESH = os.getenv("PP_TEXT_REC_SCORE_THRESH")
TEXT_RECOGNITION_BATCH_SIZE = os.getenv("PP_TEXT_REC_BATCH_SIZE")

# Convert numeric envs
def _env_float(v): return float(v) if v is not None else None
def _env_int(v): return int(v) if v is not None else None

LAYOUT_THRESHOLD = _env_float(LAYOUT_THRESHOLD)
TEXT_DET_THRESH = _env_float(TEXT_DET_THRESH)
TEXT_DET_BOX_THRESH = _env_float(TEXT_DET_BOX_THRESH)
TEXT_DET_UNCLIP_RATIO = _env_float(TEXT_DET_UNCLIP_RATIO)
TEXT_DET_LIMIT_SIDE_LEN = _env_int(TEXT_DET_LIMIT_SIDE_LEN)
TEXT_REC_SCORE_THRESH = _env_float(TEXT_REC_SCORE_THRESH)
TEXT_RECOGNITION_BATCH_SIZE = _env_int(TEXT_RECOGNITION_BATCH_SIZE)

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = int(os.getenv("PP_MAX_FILE_SIZE_MB", "50"))
MAX_PARALLEL_PREDICT = int(os.getenv("PP_MAX_PARALLEL_PREDICT", "1"))

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a default shared engine optimized for CPU
    app.state.engine = PPStructureEngine(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
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
        show_log=False,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PP-Structure V3 /parse API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    info = {
        "status": "ok",
        "device": DEVICE,
        "enable_mkldnn": ENABLE_MKLDNN,
        "max_parallel": MAX_PARALLEL_PREDICT,
    }
    return info


def _validate_ext(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")


def _spill_upload_to_temp(upload: UploadFile, dst_path: str, max_mb: int) -> int:
    """Stream copy upload to disk with size limit; return bytes written."""
    total = 0
    chunk = 1024 * 1024
    with open(dst_path, "wb") as f:
        while True:
            data = upload.file.read(chunk)
            if not data:
                break
            total += len(data)
            if total > max_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File too large (>{max_mb} MB)")
            f.write(data)
    return total


def _maybe_override_engine(
    base_engine: PPStructureEngine,
    # toggles:
    use_doc_orientation_classify: Optional[bool] = None,
    use_doc_unwarping: Optional[bool] = None,
    use_textline_orientation: Optional[bool] = None,
    use_table_recognition: Optional[bool] = None,
    use_formula_recognition: Optional[bool] = None,
    use_chart_recognition: Optional[bool] = None,
    # models:
    layout_detection_model_name: Optional[str] = None,
    text_detection_model_name: Optional[str] = None,
    text_recognition_model_name: Optional[str] = None,
    wired_table_structure_recognition_model_name: Optional[str] = None,
    wireless_table_structure_recognition_model_name: Optional[str] = None,
    table_classification_model_name: Optional[str] = None,
    formula_recognition_model_name: Optional[str] = None,
    chart_recognition_model_name: Optional[str] = None,
    # thresholds/batch:
    layout_threshold: Optional[float] = None,
    text_det_thresh: Optional[float] = None,
    text_det_box_thresh: Optional[float] = None,
    text_det_unclip_ratio: Optional[float] = None,
    text_det_limit_side_len: Optional[int] = None,
    text_det_limit_type: Optional[str] = None,
    text_rec_score_thresh: Optional[float] = None,
    text_recognition_batch_size: Optional[int] = None,
) -> PPStructureEngine:
    """
    Reuse the base engine unless request overrides differ; in that case, create a temporary engine.
    """
    req_overrides = any([
        use_doc_orientation_classify is not None,
        use_doc_unwarping is not None,
        use_textline_orientation is not None,
        use_table_recognition is not None,
        use_formula_recognition is not None,
        use_chart_recognition is not None,
        layout_detection_model_name,
        text_detection_model_name,
        text_recognition_model_name,
        wired_table_structure_recognition_model_name,
        wireless_table_structure_recognition_model_name,
        table_classification_model_name,
        formula_recognition_model_name,
        chart_recognition_model_name,
        layout_threshold is not None,
        text_det_thresh is not None,
        text_det_box_thresh is not None,
        text_det_unclip_ratio is not None,
        text_det_limit_side_len is not None,
        text_det_limit_type is not None,
        text_rec_score_thresh is not None,
        text_recognition_batch_size is not None,
    ])
    if not req_overrides:
        return base_engine

    # Build a new engine with overrides merged with base environment defaults
    return PPStructureEngine(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
        use_doc_orientation_classify=use_doc_orientation_classify if use_doc_orientation_classify is not None else USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=use_doc_unwarping if use_doc_unwarping is not None else USE_DOC_UNWARPING,
        use_textline_orientation=use_textline_orientation if use_textline_orientation is not None else USE_TEXTLINE_ORIENTATION,
        use_table_recognition=use_table_recognition if use_table_recognition is not None else USE_TABLE_RECOGNITION,
        use_formula_recognition=use_formula_recognition if use_formula_recognition is not None else USE_FORMULA_RECOGNITION,
        use_chart_recognition=use_chart_recognition if use_chart_recognition is not None else USE_CHART_RECOGNITION,
        layout_detection_model_name=layout_detection_model_name or LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=text_detection_model_name or TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=text_recognition_model_name or TEXT_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=wired_table_structure_recognition_model_name or None,
        wireless_table_structure_recognition_model_name=wireless_table_structure_recognition_model_name or None,
        table_classification_model_name=table_classification_model_name or TABLE_CLASSIFICATION_MODEL_NAME,
        formula_recognition_model_name=formula_recognition_model_name or FORMULA_RECOGNITION_MODEL_NAME,
        chart_recognition_model_name=chart_recognition_model_name or CHART_RECOGNITION_MODEL_NAME,
        layout_threshold=layout_threshold if layout_threshold is not None else LAYOUT_THRESHOLD,
        text_det_thresh=text_det_thresh if text_det_thresh is not None else TEXT_DET_THRESH,
        text_det_box_thresh=text_det_box_thresh if text_det_box_thresh is not None else TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=text_det_unclip_ratio if text_det_unclip_ratio is not None else TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=text_det_limit_side_len if text_det_limit_side_len is not None else TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=text_det_limit_type if text_det_limit_type is not None else TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=text_rec_score_thresh if text_rec_score_thresh is not None else TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=text_recognition_batch_size if text_recognition_batch_size is not None else TEXT_RECOGNITION_BATCH_SIZE,
        show_log=False,
    )


@app.post(
    "/parse",
    summary="Parse a document or image with PP-Structure V3 and return JSON and/or Markdown.",
)
async def parse(
    file: UploadFile = File(..., description="PDF or image"),
    output_format: Literal["json", "markdown", "both"] = Query("json", description="Select output format"),
    table_format: Literal["markdown", "html"] = Query("markdown", description="Table rendering within Markdown"),
    include_page_headings: bool = Query(True, description="Include '## Page N' headings in Markdown output"),

    # PDF controls
    pdf_dpi: int = Query(180, ge=72, le=600, description="DPI for PDF rasterization"),
    page_range: str = Query("all", description="Pages to parse, e.g., 'all', '1-3,6'"),
    max_pages: Optional[int] = Query(None, ge=1, description="Truncate to at most this many pages"),

    # Optional feature toggles (per-request overrides)
    use_doc_orientation_classify: Optional[bool] = Query(None),
    use_doc_unwarping: Optional[bool] = Query(None),
    use_textline_orientation: Optional[bool] = Query(None),

    use_table_recognition: Optional[bool] = Query(None),
    use_formula_recognition: Optional[bool] = Query(None),
    use_chart_recognition: Optional[bool] = Query(None),

    # Model overrides (names depend on PaddleOCR)
    layout_detection_model_name: Optional[str] = Query(None),
    text_detection_model_name: Optional[str] = Query(None),
    text_recognition_model_name: Optional[str] = Query(None),
    wired_table_structure_recognition_model_name: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(None),
    table_classification_model_name: Optional[str] = Query(None),
    formula_recognition_model_name: Optional[str] = Query(None),
    chart_recognition_model_name: Optional[str] = Query(None),

    # Thresholds and batch sizes
    layout_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_unclip_ratio: Optional[float] = Query(None, ge=0.0, le=10.0),
    text_det_limit_side_len: Optional[int] = Query(None, ge=16, le=4096),
    text_det_limit_type: Optional[str] = Query(None),
    text_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_recognition_batch_size: Optional[int] = Query(None, ge=1, le=256),

    # Output controls
    return_raw: bool = Query(True, description="Include raw Paddle output in JSON"),
):
    _validate_ext(file.filename or "")
    with tempfile.TemporaryDirectory(prefix="ppstruct_") as td:
        dst = os.path.join(td, Path(file.filename).name)
        # Save upload with size check
        await run_in_threadpool(_spill_upload_to_temp, file, dst, MAX_FILE_SIZE_MB)

        # Build engine (reuse or override from base)
        engine = _maybe_override_engine(
            app.state.engine,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            layout_detection_model_name=layout_detection_model_name,
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,
            wired_table_structure_recognition_model_name=wired_table_structure_recognition_model_name,
            wireless_table_structure_recognition_model_name=wireless_table_structure_recognition_model_name,
            table_classification_model_name=table_classification_model_name,
            formula_recognition_model_name=formula_recognition_model_name,
            chart_recognition_model_name=chart_recognition_model_name,
            layout_threshold=layout_threshold,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_rec_score_thresh=text_rec_score_thresh,
            text_recognition_batch_size=text_recognition_batch_size,
        )

        ext = Path(dst).suffix.lower()
        pages: List[Dict[str, Any]] = []

        # Acquire global semaphore to limit parallel parsing
        with app.state.predict_sem:
            if ext == ".pdf":
                pil_pages = await run_in_threadpool(
                    render_pdf_to_images, dst, pdf_dpi, page_range, max_pages
                )
                if not pil_pages:
                    raise HTTPException(status_code=400, detail="No pages selected/rendered from PDF.")
                for pil_img, pno, w, h in pil_pages:
                    raw_items = await run_in_threadpool(engine.infer_image, pil_img)
                    normalized = normalize_ppstructure_result(raw_items, pno, w, h)
                    if not return_raw:
                        normalized.pop("raw", None)
                    pages.append(normalized)
            else:
                # Single image path; infer as one page
                from PIL import Image
                pil_img = Image.open(dst).convert("RGB")
                w, h = pil_img.size
                raw_items = await run_in_threadpool(engine.infer_image, pil_img)
                normalized = normalize_ppstructure_result(raw_items, 0, w, h)
                if not return_raw:
                    normalized.pop("raw", None)
                pages.append(normalized)

        # Assemble response according to requested format
        if output_format == "json":
            payload = {
                "pages": pages,
                "meta": {
                    "device": DEVICE,
                    "enable_mkldnn": ENABLE_MKLDNN,
                    "pdf_dpi": pdf_dpi,
                    "table_format": table_format,
                },
            }
            return JSONResponse(content=orjson.loads(orjson.dumps(payload)))
        elif output_format == "markdown":
            md = await run_in_threadpool(pages_to_markdown, pages, table_format, include_page_headings)
            return PlainTextResponse(md, media_type="text/markdown")
        else:  # both
            md = await run_in_threadpool(pages_to_markdown, pages, table_format, include_page_headings)
            payload = {
                "pages": pages,
                "markdown": md,
                "meta": {
                    "device": DEVICE,
                    "enable_mkldnn": ENABLE_MKLDNN,
                    "pdf_dpi": pdf_dpi,
                    "table_format": table_format,
                },
            }
            return JSONResponse(content=orjson.loads(orjson.dumps(payload)))

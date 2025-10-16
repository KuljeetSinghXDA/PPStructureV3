import os
import io
import json
import shutil
import tempfile
import threading
import inspect
from pathlib import Path
from typing import List, Optional, Literal, Union, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# PaddleOCR pipeline
from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Defaults) =================
# These are server defaults; they can be overridden per request.
DEVICE = "cpu"  # CPU-only as requested
CPU_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))

# Inference optimizations
ENABLE_HPI = False       # High performance inference toggle (module-level, CPU-safe) [paddlepaddle.github.io]
ENABLE_MKLDNN = True     # oneDNN/MKL-DNN acceleration on CPU

# Optional pre-/post-processing modules (accuracy boosters) [paddlepaddle.github.io]
USE_DOC_ORIENTATION_CLASSIFY = True
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = True
USE_CHART_RECOGNITION = True

# Model configuration (latest PP-StructureV3 family) [paddlepaddle.github.io]
# Layout detection
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
# OCR models (PP-OCRv5 family; server variants favor accuracy vs mobile for speed)
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_server_rec"
# Table structure (wired/wireless)
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
# Formula recognition
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
# Chart recognition
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (None = pipeline defaults)
# Text detection module max_side_limit (a.k.a limit_side_len) is important on long pages [paddlepaddle.github.io]
LAYOUT_THRESHOLD: Optional[float] = None
TEXT_DET_THRESH: Optional[float] = None
TEXT_DET_BOX_THRESH: Optional[float] = None
TEXT_DET_UNCLIP_RATIO: Optional[float] = None
TEXT_DET_LIMIT_SIDE_LEN: Optional[int] = None  # aka max_side_limit in docs
TEXT_DET_LIMIT_TYPE: Optional[str] = None      # 'min' or 'max' based on detector
TEXT_REC_SCORE_THRESH: Optional[float] = None
TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= Utility helpers =================

def _ext_ok(name: str) -> bool:
    return Path(name.lower()).suffix in ALLOWED_EXTENSIONS

def _size_ok(upload: UploadFile) -> bool:
    # Starlette UploadFile doesn't have size; we enforce in-memory read cap per request
    return True

def _parse_page_range(page_range: Optional[str]) -> Optional[List[int]]:
    if not page_range:
        return None
    pages = set()
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip()
            b = b.strip()
            if not a.isdigit() or not b.isdigit():
                raise HTTPException(status_code=400, detail=f"Invalid page range segment: {part}")
            start = int(a)
            end = int(b)
            if start <= 0 or end <= 0 or end < start:
                raise HTTPException(status_code=400, detail=f"Invalid page range segment: {part}")
            for p in range(start, end + 1):
                pages.add(p)
        else:
            if not part.isdigit():
                raise HTTPException(status_code=400, detail=f"Invalid page index: {part}")
            p = int(part)
            if p <= 0:
                raise HTTPException(status_code=400, detail=f"Invalid page index: {part}")
            pages.add(p)
    return sorted(pages)

def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        return {}
    accepted = {}
    for name, param in sig.parameters.items():
        if name in kwargs:
            accepted[name] = kwargs[name]
    return accepted

def _make_pipeline(
    device: str = DEVICE,
    enable_mkldnn: bool = ENABLE_MKLDNN,
    enable_hpi: bool = ENABLE_HPI,
    cpu_threads: int = CPU_THREADS,
    layout_detection_model_name: str = LAYOUT_DETECTION_MODEL_NAME,
    text_detection_model_name: str = TEXT_DETECTION_MODEL_NAME,
    text_recognition_model_name: str = TEXT_RECOGNITION_MODEL_NAME,
    wired_table_structure_recognition_model_name: str = WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
    wireless_table_structure_recognition_model_name: str = WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
    table_classification_model_name: str = TABLE_CLASSIFICATION_MODEL_NAME,
    formula_recognition_model_name: str = FORMULA_RECOGNITION_MODEL_NAME,
    chart_recognition_model_name: str = CHART_RECOGNITION_MODEL_NAME,
    layout_threshold: Optional[float] = LAYOUT_THRESHOLD,
    text_det_thresh: Optional[float] = TEXT_DET_THRESH,
    text_det_box_thresh: Optional[float] = TEXT_DET_BOX_THRESH,
    text_det_unclip_ratio: Optional[float] = TEXT_DET_UNCLIP_RATIO,
    text_det_limit_side_len: Optional[int] = TEXT_DET_LIMIT_SIDE_LEN,
    text_det_limit_type: Optional[str] = TEXT_DET_LIMIT_TYPE,
    text_rec_score_thresh: Optional[float] = TEXT_REC_SCORE_THRESH,
    text_recognition_batch_size: Optional[int] = TEXT_RECOGNITION_BATCH_SIZE,
    use_doc_orientation_classify: bool = USE_DOC_ORIENTATION_CLASSIFY,
    use_doc_unwarping: bool = USE_DOC_UNWARPING,
    use_textline_orientation: bool = USE_TEXTLINE_ORIENTATION,
    use_table_recognition: bool = USE_TABLE_RECOGNITION,
    use_formula_recognition: bool = USE_FORMULA_RECOGNITION,
    use_chart_recognition: bool = USE_CHART_RECOGNITION,
) -> PPStructureV3:
    # Construct PP-StructureV3 with provided parameters (filter to match installed version)
    ctor_kwargs = dict(
        device=device,
        enable_mkldnn=enable_mkldnn,
        enable_hpi=enable_hpi,
        cpu_threads=cpu_threads,
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
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        use_table_recognition=use_table_recognition,
        use_formula_recognition=use_formula_recognition,
        use_chart_recognition=use_chart_recognition,
    )
    # Be resilient to version drift by filtering constructor kwargs
    ctor_filtered = _filter_kwargs_for_callable(PPStructureV3, ctor_kwargs)
    return PPStructureV3(**ctor_filtered)

def _needs_new_pipeline(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> bool:
    for k, v in overrides.items():
        if v is None:
            continue
        if k in defaults and defaults[k] != v:
            return True
    return False

def _collect_defaults() -> Dict[str, Any]:
    return dict(
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

# ================= App & Lifespan =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm a default pipeline that enables all features (CPU only)
    app.state.pipeline_defaults = _collect_defaults()
    app.state.pipeline = _make_pipeline(**app.state.pipeline_defaults)
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield
    # Cleanup (models cache kept by PaddleOCR inside site-packages; temp dirs handled per request)

app = FastAPI(title="PP-StructureV3 /parse API (CPU/ARM64)", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= /parse Endpoint =================

@app.post(
    "/parse",
    responses={
        200: {"content": {"application/json": {}, "text/markdown": {}}},
        400: {"description": "Bad Request"},
        500: {"description": "Server Error"},
    },
)
async def parse_document(
    file: UploadFile = File(..., description="PDF or single page image: .pdf, .jpg, .jpeg, .png, .bmp"),
    output_format: Literal["json", "markdown"] = Query("json", description="Return JSON or Markdown."),
    merge_pages: Optional[bool] = Query(None, description="If markdown, merge pages into one Markdown string."),
    page_range: Optional[str] = Query(None, description="Pages to parse, e.g. '1-3,5'. 1-indexed. Omit for all pages."),
    # Optional per-request overrides for modules and models
    use_table_recognition: Optional[bool] = Query(None),
    use_formula_recognition: Optional[bool] = Query(None),
    use_chart_recognition: Optional[bool] = Query(None),
    use_doc_orientation_classify: Optional[bool] = Query(None),
    use_doc_unwarping: Optional[bool] = Query(None),
    use_textline_orientation: Optional[bool] = Query(None),
    layout_detection_model_name: Optional[str] = Query(None),
    text_detection_model_name: Optional[str] = Query(None),
    text_recognition_model_name: Optional[str] = Query(None),
    wired_table_structure_recognition_model_name: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(None),
    table_classification_model_name: Optional[str] = Query(None),
    formula_recognition_model_name: Optional[str] = Query(None),
    chart_recognition_model_name: Optional[str] = Query(None),
    # Thresholds and batching
    layout_threshold: Optional[float] = Query(None),
    text_det_thresh: Optional[float] = Query(None),
    text_det_box_thresh: Optional[float] = Query(None),
    text_det_unclip_ratio: Optional[float] = Query(None),
    text_det_limit_side_len: Optional[int] = Query(None, description="Text detection max_side_limit (resize cap)."),
    text_det_limit_type: Optional[str] = Query(None, regex="^(min|max)$"),
    text_rec_score_thresh: Optional[float] = Query(None),
    text_recognition_batch_size: Optional[int] = Query(None),
    # Performance knobs
    cpu_threads: Optional[int] = Query(None, ge=1, le=64),
    enable_mkldnn: Optional[bool] = Query(None),
    enable_hpi: Optional[bool] = Query(None, description="High Performance Inference (CPU-safe)."),
):
    # Validate file
    if not _ext_ok(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    # Limit memory footprint and write to a temp file
    tmpdir = Path(tempfile.mkdtemp(prefix="ppstructv3_"))
    try:
        in_path = tmpdir / file.filename
        raw = await file.read()
        if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_FILE_SIZE_MB} MB.")
        in_path.write_bytes(raw)

        pages = _parse_page_range(page_range)
        # Default merge_pages behavior: True for markdown, False for json
        if merge_pages is None:
            merge_pages_eff = (output_format == "markdown")
        else:
            merge_pages_eff = merge_pages

        # Prepare overrides dict for pipeline constructor
        overrides = dict(
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
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
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_hpi=enable_hpi,
            device="cpu",
        )
        # Decide whether we need to spin up a temporary pipeline
        defaults = app.state.pipeline_defaults
        need_new = _needs_new_pipeline(defaults, overrides)

        # Build predict kwargs (filter dynamically to match installed version)
        predict_kwargs = dict(
            output_format=output_format,   # json | markdown (supported per PP-StructureV3 docs)
            page_numbers=pages,            # Some versions accept 'page_numbers' or 'pages'; filter will handle
            merge_pages=merge_pages_eff,
        )

        def _run():
            pipeline = app.state.pipeline
            # Create per-request pipeline if overrides differ from defaults (ensures all features are supported)
            if need_new:
                # Merge defaults with provided overrides (None means keep default)
                merged = {**defaults}
                for k, v in overrides.items():
                    if v is not None:
                        merged[k] = v
                pipeline_local = _make_pipeline(**merged)
                pipe = pipeline_local
            else:
                pipe = pipeline

            # Filter predict kwargs to match the installed pipeline signature
            call = getattr(pipe, "predict", pipe)
            pk = _filter_kwargs_for_callable(call, predict_kwargs)

            # Run prediction on a file path (PDF or image). The pipeline accepts path inputs.
            return call(in_path.as_posix(), **pk)

        # Serialize predictions with concurrency guard
        with app.state.predict_sem:
            result = await run_in_threadpool(_run)

        # Result normalization: handle flexible returns
        if output_format == "markdown":
            md_text: Optional[str] = None
            # Common return shapes:
            # - str markdown
            # - dict with 'markdown'
            # - list[dict] with 'markdown' per page
            if isinstance(result, str):
                md_text = result
            elif isinstance(result, dict) and "markdown" in result:
                md_text = result["markdown"]
            elif isinstance(result, list) and all(isinstance(x, dict) for x in result):
                # Concatenate per-page markdown with page separators if merge_pages; else return multi-page text as-is
                if merge_pages_eff:
                    parts = []
                    for i, page in enumerate(result, start=1):
                        if "markdown" in page:
                            parts.append(f"# Page {i}\n\n{page['markdown']}")
                    md_text = "\n\n".join(parts) if parts else ""
                else:
                    # Not merged: still emit a concatenation but without headers (consumer can split on '---' if needed)
                    parts = [page.get("markdown", "") for page in result]
                    md_text = "\n\n".join(parts)
            else:
                # Fallback: stringify
                md_text = str(result)
            return PlainTextResponse(content=md_text, media_type="text/markdown")

        # JSON branch
        # If the pipeline returned non-JSON-serializable items, attempt to convert or wrap.
        def _jsonify(obj):
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                # Fallback: convert to string
                return json.loads(json.dumps(obj, default=str))

        payload = _jsonify(result)
        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}") from e
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

import os
import io
import tempfile
import threading
import json
import shutil
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

# Optional external deps for PDF/images conversion and HTML->MD
# We keep them optional and degrade gracefully if missing.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pdf2image import convert_from_path as pdf2image_convert_from_path
except Exception:
    pdf2image_convert_from_path = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values) =================
DEVICE = "cpu"
CPU_THREADS = 4

# Optional accuracy boosters (Doc preprocessing)
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
LAYOUT_THRESHOLD: Optional[float] = None
TEXT_DET_THRESH: Optional[float] = None
TEXT_DET_BOX_THRESH: Optional[float] = None
TEXT_DET_UNCLIP_RATIO: Optional[float] = None
TEXT_DET_LIMIT_SIDE_LEN: Optional[int] = None
TEXT_DET_LIMIT_TYPE: Optional[str] = None
TEXT_REC_SCORE_THRESH: Optional[float] = None
TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None

# Additional optional batch sizes (will be filtered by introspection)
LAYOUT_BATCH_SIZE: Optional[int] = None
TEXT_DETECTION_BATCH_SIZE: Optional[int] = None
TABLE_STRUCTURE_RECOGNITION_BATCH_SIZE: Optional[int] = None
FORMULA_RECOGNITION_BATCH_SIZE: Optional[int] = None
CHART_RECOGNITION_BATCH_SIZE: Optional[int] = None

# Logging / verbosity (will only be passed if supported)
SHOW_LOG: Optional[bool] = True

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= Helpers: Introspection-safe pipeline init =================

def get_init_signature_params() -> Dict[str, inspect.Parameter]:
    sig = inspect.signature(PPStructureV3.__init__)
    return sig.parameters

def filter_kwargs_for_signature(kwargs: Dict[str, Any], sig_params: Dict[str, inspect.Parameter]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if (v is not None and k in sig_params)}

def build_base_pipeline_kwargs() -> Dict[str, Any]:
    """
    Build a superset of kwargs. We'll filter by the installed PPStructureV3 signature.
    """
    return {
        # Runtime
        "device": DEVICE,
        "enable_mkldnn": ENABLE_MKLDNN,
        "enable_hpi": ENABLE_HPI,
        "cpu_threads": CPU_THREADS,
        "show_log": SHOW_LOG,

        # Models
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,
        "wired_table_structure_recognition_model_name": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "wireless_table_structure_recognition_model_name": WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "table_classification_model_name": TABLE_CLASSIFICATION_MODEL_NAME,
        "formula_recognition_model_name": FORMULA_RECOGNITION_MODEL_NAME,
        "chart_recognition_model_name": CHART_RECOGNITION_MODEL_NAME,

        # Thresholds / limits
        "layout_threshold": LAYOUT_THRESHOLD,
        "text_det_thresh": TEXT_DET_THRESH,
        "text_det_box_thresh": TEXT_DET_BOX_THRESH,
        "text_det_unclip_ratio": TEXT_DET_UNCLIP_RATIO,
        "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
        "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
        "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,

        # Batch sizes
        "text_recognition_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
        "layout_batch_size": LAYOUT_BATCH_SIZE,
        "text_detection_batch_size": TEXT_DETECTION_BATCH_SIZE,
        "table_structure_recognition_batch_size": TABLE_STRUCTURE_RECOGNITION_BATCH_SIZE,
        "formula_recognition_batch_size": FORMULA_RECOGNITION_BATCH_SIZE,
        "chart_recognition_batch_size": CHART_RECOGNITION_BATCH_SIZE,

        # Feature toggles
        "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": USE_DOC_UNWARPING,
        "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        "use_table_recognition": USE_TABLE_RECOGNITION,
        "use_formula_recognition": USE_FORMULA_RECOGNITION,
        "use_chart_recognition": USE_CHART_RECOGNITION,
    }

def effective_pipeline_cache_key(kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """
    A deterministic key from kwargs to cache pipelines by effective config.
    Only includes items actually passed to __init__ for stability.
    """
    # Stable tuple of sorted items
    items = sorted(kwargs.items(), key=lambda x: x[0])
    return tuple(items)

class PipelineCache:
    def __init__(self, max_size: int = 4):
        self._cache: Dict[Tuple[Tuple[str, Any], ...], PPStructureV3] = {}
        self._order: List[Tuple[Tuple[str, Any], ...]] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def get_or_create(self, kwargs: Dict[str, Any]) -> PPStructureV3:
        key = effective_pipeline_cache_key(kwargs)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            # evict if needed
            if len(self._order) >= self._max_size:
                old_key = self._order.pop(0)
                try:
                    old_engine = self._cache.pop(old_key)
                    # best-effort cleanup if needed later
                    del old_engine
                except KeyError:
                    pass
            engine = PPStructureV3(**kwargs)
            self._cache[key] = engine
            self._order.append(key)
            return engine

# ================= PDF conversion helpers =================

def convert_pdf_to_images(
    pdf_path: str,
    dpi: int = 200,
    page_indices: Optional[List[int]] = None
) -> List[str]:
    """
    Convert a PDF to a list of image file paths (PNG).
    Prefers PyMuPDF; falls back to pdf2image if available; else raises.
    page_indices: 1-based indices to render (if None -> all pages).
    """
    out_paths: List[str] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="ppsv3_pdf_"))
    try:
        if fitz is not None:
            doc = fitz.open(pdf_path)
            try:
                total = doc.page_count
                indices = page_indices or list(range(1, total + 1))
                for pnum in indices:
                    if pnum < 1 or pnum > total:
                        continue
                    page = doc.load_page(pnum - 1)
                    # DPI -> scale
                    zoom = dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_path = str(tmp_dir / f"page_{pnum:04d}.png")
                    pix.save(img_path)
                    out_paths.append(img_path)
            finally:
                doc.close()
        elif pdf2image_convert_from_path is not None:
            # Note: pdf2image uses 1-based page numbers via first_page/last_page
            images = pdf2image_convert_from_path(pdf_path, dpi=dpi)
            if page_indices:
                # Re-map based on requested indices
                for idx in page_indices:
                    if 1 <= idx <= len(images):
                        img = images[idx - 1]
                        img_path = str(tmp_dir / f"page_{idx:04d}.png")
                        img.save(img_path)
                        out_paths.append(img_path)
            else:
                for i, img in enumerate(images, start=1):
                    img_path = str(tmp_dir / f"page_{i:04d}.png")
                    img.save(img_path)
                    out_paths.append(img_path)
        else:
            raise RuntimeError("Neither PyMuPDF nor pdf2image is available to render PDFs.")
        return out_paths
    except Exception:
        # Clean partials on failure
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

# ================= Output normalization =================

def _to_list(x: Any) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def html_table_to_markdown(html: str) -> Optional[str]:
    if not html:
        return None
    if BeautifulSoup is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None
    # Very lightweight HTML table to Markdown conversion
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            text = cell.get_text(strip=True).replace("|", "\\|")
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return None
    # Insert header separator if header exists
    if table.find("th") and len(rows) >= 1:
        col_count = rows[0].count("|") - 1
        sep = "|" + " --- |" * col_count
        rows.insert(1, sep)
    return "\n".join(rows)

def normalize_ppsv3_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single PP-StructureV3 detection item into a stable shape.
    Preserve raw payload under 'raw'.
    """
    itype = item.get("type")
    bbox = item.get("bbox") or item.get("box") or item.get("poly") or item.get("bbox_score")
    res = item.get("res")
    score = item.get("score") or (res.get("score") if isinstance(res, dict) else None)

    norm: Dict[str, Any] = {
        "type": itype,
        "bbox": bbox,
        "score": score,
        "raw": item
    }

    # Heuristics by type
    if itype in ("text", "paragraph", "title", "section"):
        text_val = res.get("text") if isinstance(res, dict) else (res if isinstance(res, str) else None)
        norm["text"] = text_val

    elif itype in ("table", "table_wired", "table_wireless"):
        html = None
        cells = None
        if isinstance(res, dict):
            html = res.get("html") or res.get("res_html") or res.get("structure_html")
            cells = res.get("cells") or res.get("cell_bbox") or res.get("cell_boxes")
        norm["table"] = {
            "html": html,
            "markdown": html_table_to_markdown(html) if html else None,
            "cells": cells
        }

    elif itype in ("formula", "equation"):
        latex = None
        if isinstance(res, dict):
            latex = res.get("latex") or res.get("text")
        elif isinstance(res, str):
            latex = res
        norm["formula"] = {"latex": latex}

    elif itype in ("chart", "figure"):
        # Some chart parsers emit table-like HTML, others JSON tables
        html = res.get("html") if isinstance(res, dict) else None
        table = res.get("table") if isinstance(res, dict) else None
        norm["chart"] = {
            "html": html,
            "markdown": html_table_to_markdown(html) if html else None,
            "table": table
        }

    return norm

def normalize_ppsv3_results(res: Any) -> Dict[str, Any]:
    """
    Normalize possible return types:
    - Single image: list[dict]
    - PDF pages: list[list[dict]] or dict with per-page
    We produce a consistent {pages: [ {items: ...} ]} structure.
    """
    normalized: Dict[str, Any] = {"pages": []}

    def normalize_page(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        nitems = [normalize_ppsv3_item(it) for it in items if isinstance(it, dict)]
        return {"items": nitems}

    if isinstance(res, list):
        # Could be list[dict] (single page) or list[list[dict]] (multi-page)
        if res and isinstance(res[0], list):
            for page_items in res:
                normalized["pages"].append(normalize_page([it for it in page_items if isinstance(it, dict)]))
        else:
            normalized["pages"].append(normalize_page([it for it in res if isinstance(it, dict)]))
    elif isinstance(res, dict):
        # Unknown return pattern, wrap it raw
        normalized["raw"] = res
    else:
        normalized["raw"] = res
    return normalized

# ================= App & Lifespan =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build default pipeline
    sig_params = get_init_signature_params()
    base_kwargs = build_base_pipeline_kwargs()
    safe_kwargs = filter_kwargs_for_signature(base_kwargs, sig_params)
    pipeline_cache = PipelineCache(max_size=4)
    default_pipeline = pipeline_cache.get_or_create(safe_kwargs)

    app.state.pipeline_signature = sig_params
    app.state.pipeline_base_kwargs = base_kwargs
    app.state.pipeline_cache = pipeline_cache
    app.state.default_pipeline_kwargs = safe_kwargs
    app.state.pipeline = default_pipeline
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= Utility: overrides and profiles =================

def apply_profile_overrides(profile: Optional[Literal["default", "pp", "full"]], overrides: Dict[str, Any]) -> None:
    """
    Apply high-level profiles to overrides.
    - default: no change
    - pp: enable doc preprocessing
    - full: pp + chart parsing
    """
    if profile == "pp":
        overrides.setdefault("use_doc_orientation_classify", True)
        overrides.setdefault("use_textline_orientation", True)
        overrides.setdefault("use_doc_unwarping", True)
    elif profile == "full":
        overrides.setdefault("use_doc_orientation_classify", True)
        overrides.setdefault("use_textline_orientation", True)
        overrides.setdefault("use_doc_unwarping", True)
        overrides.setdefault("use_chart_recognition", True)

def build_effective_overrides(
    base: Dict[str, Any],
    query: Dict[str, Any],
    sig_params: Dict[str, inspect.Parameter],
) -> Dict[str, Any]:
    """
    Create a new kwargs dict based on base + query overrides,
    filtered by the PPStructureV3 __init__ signature.
    """
    merged = dict(base)
    # overlay supported query keys (only set if not None)
    for k, v in query.items():
        if v is not None:
            merged[k] = v
    # filter
    return filter_kwargs_for_signature(merged, sig_params)

# ================= I/O validation =================

def ensure_allowed_ext_and_size(upload: UploadFile, max_mb: int) -> None:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    # Starlette UploadFile does not always give size; read header if provided
    content_length = upload.headers.get("content-length")
    if content_length:
        try:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_mb:
                raise HTTPException(status_code=413, detail=f"File too large (> {max_mb} MB)")
        except Exception:
            pass  # ignore header parse errors

def save_upload_to_temp(upload: UploadFile) -> str:
    tmp_dir = Path(tempfile.mkdtemp(prefix="ppsv3_in_"))
    dst_path = tmp_dir / (Path(upload.filename or "input").name)
    with open(dst_path, "wb") as f:
        # stream copy
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return str(dst_path)

def parse_pages_param(pages: Optional[str]) -> Optional[List[int]]:
    if not pages:
        return None
    try:
        out = []
        for part in pages.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                out.extend(list(range(min(a,b), max(a,b)+1)))
            else:
                out.append(int(part))
        # unique, sorted
        return sorted(set(out))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'pages' parameter. Use comma list or ranges, e.g. 1,3-5.")

# ================= /parse endpoint =================

@app.post("/parse")
async def parse(
    file: UploadFile = File(...),

    # High-level profile (maps to PP-StructureV3-default/pp/full)
    profile: Optional[Literal["default", "pp", "full"]] = Query(None, description="Quick config: default | pp (doc preprocessing) | full (pp + chart)"),

    # Feature toggles (request-level overrides)
    use_table_recognition: Optional[bool] = Query(None),
    use_formula_recognition: Optional[bool] = Query(None),
    use_chart_recognition: Optional[bool] = Query(None),
    use_doc_orientation_classify: Optional[bool] = Query(None),
    use_textline_orientation: Optional[bool] = Query(None),
    use_doc_unwarping: Optional[bool] = Query(None),

    # Model name overrides
    layout_detection_model_name: Optional[str] = Query(None),
    text_detection_model_name: Optional[str] = Query(None),
    text_recognition_model_name: Optional[str] = Query(None),
    wired_table_structure_recognition_model_name: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(None),
    table_classification_model_name: Optional[str] = Query(None),
    formula_recognition_model_name: Optional[str] = Query(None),
    chart_recognition_model_name: Optional[str] = Query(None),

    # Thresholds
    layout_threshold: Optional[float] = Query(None),
    text_det_thresh: Optional[float] = Query(None),
    text_det_box_thresh: Optional[float] = Query(None),
    text_det_unclip_ratio: Optional[float] = Query(None),
    text_det_limit_side_len: Optional[int] = Query(None),
    text_det_limit_type: Optional[str] = Query(None),
    text_rec_score_thresh: Optional[float] = Query(None),

    # Batching
    text_recognition_batch_size: Optional[int] = Query(None),
    layout_batch_size: Optional[int] = Query(None),
    text_detection_batch_size: Optional[int] = Query(None),
    table_structure_recognition_batch_size: Optional[int] = Query(None),
    formula_recognition_batch_size: Optional[int] = Query(None),
    chart_recognition_batch_size: Optional[int] = Query(None),

    # Output controls
    return_raw: bool = Query(False, description="Include raw pipeline outputs"),
    return_html: bool = Query(False, description="Try to include HTML (e.g., table/ chart HTML)"),
    return_markdown: bool = Query(False, description="Try to include Markdown conversions of HTML tables/ charts"),
    include_images: bool = Query(False, description="Reserved: attach images if pipeline supports; currently not embedded"),

    # PDF controls
    pages: Optional[str] = Query(None, description="Comma list of 1-based page numbers and/or ranges (e.g., 1,3-5)"),
    page_start: Optional[int] = Query(None, ge=1, description="1-based inclusive page start (applied if 'pages' not set)"),
    page_end: Optional[int] = Query(None, ge=1, description="1-based inclusive page end (applied if 'pages' not set)"),
    pdf_dpi: int = Query(200, ge=72, le=600, description="PDF render DPI if conversion required"),
):
    ensure_allowed_ext_and_size(file, MAX_FILE_SIZE_MB)
    src_path = save_upload_to_temp(file)
    ext = Path(src_path).suffix.lower()

    # Build request-level overrides and apply profile
    query_overrides: Dict[str, Any] = {
        # toggles
        "use_table_recognition": use_table_recognition,
        "use_formula_recognition": use_formula_recognition,
        "use_chart_recognition": use_chart_recognition,
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_textline_orientation": use_textline_orientation,
        "use_doc_unwarping": use_doc_unwarping,

        # models
        "layout_detection_model_name": layout_detection_model_name,
        "text_detection_model_name": text_detection_model_name,
        "text_recognition_model_name": text_recognition_model_name,
        "wired_table_structure_recognition_model_name": wired_table_structure_recognition_model_name,
        "wireless_table_structure_recognition_model_name": wireless_table_structure_recognition_model_name,
        "table_classification_model_name": table_classification_model_name,
        "formula_recognition_model_name": formula_recognition_model_name,
        "chart_recognition_model_name": chart_recognition_model_name,

        # thresholds
        "layout_threshold": layout_threshold,
        "text_det_thresh": text_det_thresh,
        "text_det_box_thresh": text_det_box_thresh,
        "text_det_unclip_ratio": text_det_unclip_ratio,
        "text_det_limit_side_len": text_det_limit_side_len,
        "text_det_limit_type": text_det_limit_type,
        "text_rec_score_thresh": text_rec_score_thresh,

        # batching
        "text_recognition_batch_size": text_recognition_batch_size,
        "layout_batch_size": layout_batch_size,
        "text_detection_batch_size": text_detection_batch_size,
        "table_structure_recognition_batch_size": table_structure_recognition_batch_size,
        "formula_recognition_batch_size": formula_recognition_batch_size,
        "chart_recognition_batch_size": chart_recognition_batch_size,
    }
    # Apply profile shorthands
    apply_profile_overrides(profile, query_overrides)

    # Build effective kwargs safely against installed version
    base_kwargs = app.state.pipeline_base_kwargs
    sig_params = app.state.pipeline_signature
    effective_kwargs = build_effective_overrides(base_kwargs, query_overrides, sig_params)

    # Acquire or reuse pipeline instance
    pipeline: PPStructureV3 = app.state.pipeline_cache.get_or_create(effective_kwargs)

    # Drive inference with concurrency limit
    async def run_infer_on_paths(paths: List[str]) -> Any:
        results = []
        for p in paths:
            res = await run_in_threadpool(pipeline, p)  # PPStructureV3 is callable
            results.append(res)
        return results

    # Compute page indices for PDF if any
    page_list: Optional[List[int]] = None
    if pages:
        page_list = parse_pages_param(pages)
    elif page_start is not None or page_end is not None:
        # Will resolve after loading PDF length; here we just pass the range
        if page_start is None or page_end is None or page_end < page_start:
            raise HTTPException(status_code=400, detail="Invalid page range. Provide 'page_start' and 'page_end' where end >= start.")

    acquired = app.state.predict_sem.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=429, detail="Too many concurrent requests. Please try again later.")
    try:
        if ext == ".pdf":
            # Render pages
            if page_list is not None:
                render_indices = page_list
            else:
                # If range is set, use it; else None -> all pages
                # We'll derive count within convert function if using fitz or pdf2image separately
                render_indices = None
                if page_start is not None and page_end is not None:
                    # Build list: we don't know total, but conversion function will clip invalid pages
                    render_indices = list(range(page_start, page_end + 1))

            try:
                img_paths = convert_pdf_to_images(src_path, dpi=pdf_dpi, page_indices=render_indices)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to render PDF pages: {e}")
            if not img_paths:
                raise HTTPException(status_code=400, detail="No pages to process after applying page selection.")
            raw_results = await run_infer_on_paths(img_paths)
            # raw_results is list[res] where each res is typically list[dict] (page items)
            normalized_pages = []
            for res in raw_results:
                norm = normalize_ppsv3_results(res)
                if norm.get("pages"):
                    normalized_pages.extend(norm["pages"])
                else:
                    # If single page result returned as list[dict]
                    items = res if isinstance(res, list) else []
                    normalized_pages.append({"items": [normalize_ppsv3_item(x) for x in items if isinstance(x, dict)]})

            out = {
                "file": Path(src_path).name,
                "type": "pdf",
                "config": effective_kwargs,
                "pages": normalized_pages
            }
            if return_raw:
                out["raw"] = raw_results
            return JSONResponse(out)
        else:
            # Single image
            raw_result = await run_in_threadpool(pipeline, src_path)
            norm = normalize_ppsv3_results(raw_result)
            out = {
                "file": Path(src_path).name,
                "type": "image",
                "config": effective_kwargs,
                "result": norm
            }
            if return_raw:
                out["raw"] = raw_result
            return JSONResponse(out)
    finally:
        try:
            app.state.predict_sem.release()
        except Exception:
            pass
        # Cleanup temp dir
        try:
            tmp_root = str(Path(src_path).parent)
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

# ================= Optional: inspect supported init args =================

@app.get("/ppstructurev3/init-args")
def ppstructurev3_init_args():
    """
    Introspect and return the supported __init__ keyword arguments for the installed PPStructureV3.
    Helpful for debugging override mismatches.
    """
    sig_params = app.state.pipeline_signature
    return {
        "supported_args": sorted(list(sig_params.keys())),
        "defaults": {
            k: (None if v.default is inspect._empty else v.default)
            for k, v in sig_params.items()
        }
    }

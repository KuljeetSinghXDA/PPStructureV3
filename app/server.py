import os
import io
import base64
import json
import shutil
import tempfile
import threading
import platform
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple, List

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from paddleocr import PPStructureV3

# ================= ARM64-aware defaults =================
DEVICE = "cpu"
CPU_THREADS = 4

# Subpipelines
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False
USE_REGION_DETECTION = True

# Models
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
WIRED_TABLE_CELLS_DET_MODEL_NAME = None
WIRELESS_TABLE_CELLS_DET_MODEL_NAME = None
TABLE_ORIENTATION_CLASSIFY_MODEL_NAME = None
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
SEAL_TEXT_DETECTION_MODEL_NAME = None
SEAL_TEXT_RECOGNITION_MODEL_NAME = None
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Optional model dirs
LAYOUT_DETECTION_MODEL_DIR = None
REGION_DETECTION_MODEL_DIR = None
TEXT_DETECTION_MODEL_DIR = None
TEXT_RECOGNITION_MODEL_DIR = None
TABLE_CLASSIFICATION_MODEL_DIR = None
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = None
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = None
WIRED_TABLE_CELLS_DET_MODEL_DIR = None
WIRELESS_TABLE_CELLS_DET_MODEL_DIR = None
TABLE_ORIENTATION_CLASSIFY_MODEL_DIR = None
FORMULA_RECOGNITION_MODEL_DIR = None
DOC_ORIENTATION_CLASSIFY_MODEL_DIR = None
DOC_UNWARPING_MODEL_DIR = None
TEXTLINE_ORIENTATION_MODEL_DIR = None
SEAL_TEXT_DETECTION_MODEL_DIR = None
SEAL_TEXT_RECOGNITION_MODEL_DIR = None
CHART_RECOGNITION_MODEL_DIR = None

# Thresholds / sizes / batches
LAYOUT_THRESHOLD = None
LAYOUT_NMS = None
LAYOUT_UNCLIP_RATIO = None
LAYOUT_MERGE_BBOXES_MODE = None

TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None

SEAL_DET_LIMIT_SIDE_LEN = None
SEAL_DET_LIMIT_TYPE = None
SEAL_DET_THRESH = None
SEAL_DET_BOX_THRESH = None
SEAL_DET_UNCLIP_RATIO = None
SEAL_REC_SCORE_THRESH = None

TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None
TEXTLINE_ORIENTATION_BATCH_SIZE = None
FORMULA_RECOGNITION_BATCH_SIZE = None
CHART_RECOGNITION_BATCH_SIZE = None
SEAL_TEXT_RECOGNITION_BATCH_SIZE = None

# Backend knobs (ARM64-friendly)
ENABLE_HPI = False
_ENABLE_MKLDNN_DEFAULT = platform.machine().lower() in ("x86_64", "amd64")
ENABLE_MKLDNN = bool(int(os.getenv("ENABLE_MKLDNN", "1" if _ENABLE_MKLDNN_DEFAULT else "0")))
USE_TENSORRT = False
PRECISION = "fp32"
MKLDNN_CACHE_CAPACITY = 10
PADDLEX_CONFIG = None

# I/O and service
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1
PIPELINE_CACHE_SIZE = 2

# ================= Helpers =================
def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_too_big(upload: UploadFile) -> bool:
    size_hdr = upload.headers.get("content-length")
    if size_hdr and size_hdr.isdigit():
        return int(size_hdr) > MAX_FILE_SIZE_MB * 1024 * 1024
    return False

def _build_config_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in params.items()))

def _make_pipeline(**kwargs) -> PPStructureV3:
    return PPStructureV3(**kwargs)

def _pil_to_base64(pil_img: Image.Image) -> str:
    """Convert PIL Image to base64 data URI"""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _embed_images_in_markdown(md_text: str, images_map: Dict[str, Image.Image]) -> str:
    """Replace image paths with base64 data URIs in both Markdown and HTML syntax"""
    for path, pil_img in images_map.items():
        data_uri = _pil_to_base64(pil_img)
        # Replace HTML img tags
        md_text = re.sub(
            rf'<img\s+[^>]*src\s*=\s*["\']?{re.escape(path)}["\']?[^>]*>',
            f'<img src="{data_uri}">',
            md_text,
            flags=re.IGNORECASE
        )
        # Replace Markdown image syntax
        md_text = md_text.replace(f"]({path})", f"]({data_uri})")
        md_text = md_text.replace(f'src="{path}"', f'src="{data_uri}"')
        md_text = md_text.replace(f"src='{path}'", f"src='{data_uri}'")
    return md_text

def _collect_results(pipeline: PPStructureV3, outputs, inline_images: bool):
    """
    Extract JSON and Markdown from pipeline results using documented APIs.
    
    res.markdown is a dict with keys:
    - "text" (or "markdown_texts"): the actual markdown content
    - "markdown_images" (or "images"): dict mapping paths to PIL Images
    """
    page_json = []
    markdown_list = []
    all_images = {}
    
    for res in outputs:
        # Extract JSON using documented attribute
        page_json.append(getattr(res, "json", {}))
        
        # Extract markdown dict
        md = getattr(res, "markdown", None)
        if md and isinstance(md, dict):
            markdown_list.append(md)
            # Collect images from this page (support both key names)
            imgs = md.get("markdown_images") or md.get("images") or {}
            if imgs and isinstance(imgs, dict):
                all_images.update(imgs)
        else:
            # Fallback for empty/missing markdown
            markdown_list.append({"text": "", "markdown_images": {}})
    
    # Merge markdown text across pages using pipeline method
    merged_md = ""
    if markdown_list:
        try:
            # Primary method: pipeline.concatenate_markdown_pages
            merged_md = pipeline.concatenate_markdown_pages(markdown_list)
        except Exception:
            # Fallback for older builds
            try:
                paddlex = getattr(pipeline, "paddlex_pipeline", None)
                if paddlex and hasattr(paddlex, "concatenate_markdown_pages"):
                    merged_md = paddlex.concatenate_markdown_pages(markdown_list)
            except Exception:
                pass
            
            # Final fallback: manual concatenation
            if not merged_md:
                texts = []
                for md_dict in markdown_list:
                    if isinstance(md_dict, dict):
                        # Try multiple key names for compatibility
                        text = md_dict.get("text") or md_dict.get("markdown_texts") or ""
                        if text:
                            texts.append(text)
                merged_md = "\n\n---\n\n".join(texts) if texts else ""
    
    # Optionally inline images as base64 data URIs
    if inline_images and merged_md and all_images:
        merged_md = _embed_images_in_markdown(merged_md, all_images)
    
    return page_json, merged_md

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = _make_pipeline(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        use_tensorrt=USE_TENSORRT,
        precision=PRECISION,
        mkldnn_cache_capacity=MKLDNN_CACHE_CAPACITY,
        cpu_threads=CPU_THREADS,
        paddlex_config=PADDLEX_CONFIG,
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        layout_detection_model_dir=LAYOUT_DETECTION_MODEL_DIR,
        region_detection_model_dir=REGION_DETECTION_MODEL_DIR,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_detection_model_dir=TEXT_DETECTION_MODEL_DIR,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        text_recognition_model_dir=TEXT_RECOGNITION_MODEL_DIR,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        table_classification_model_dir=TABLE_CLASSIFICATION_MODEL_DIR,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_dir=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_dir=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        wired_table_cells_detection_model_name=WIRED_TABLE_CELLS_DET_MODEL_NAME,
        wired_table_cells_detection_model_dir=WIRED_TABLE_CELLS_DET_MODEL_DIR,
        wireless_table_cells_detection_model_name=WIRELESS_TABLE_CELLS_DET_MODEL_NAME,
        wireless_table_cells_detection_model_dir=WIRELESS_TABLE_CELLS_DET_MODEL_DIR,
        table_orientation_classify_model_name=TABLE_ORIENTATION_CLASSIFY_MODEL_NAME,
        table_orientation_classify_model_dir=TABLE_ORIENTATION_CLASSIFY_MODEL_DIR,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        formula_recognition_model_dir=FORMULA_RECOGNITION_MODEL_DIR,
        doc_orientation_classify_model_dir=DOC_ORIENTATION_CLASSIFY_MODEL_DIR,
        doc_unwarping_model_dir=DOC_UNWARPING_MODEL_DIR,
        textline_orientation_model_dir=TEXTLINE_ORIENTATION_MODEL_DIR,
        seal_text_detection_model_name=SEAL_TEXT_DETECTION_MODEL_NAME,
        seal_text_detection_model_dir=SEAL_TEXT_DETECTION_MODEL_DIR,
        seal_text_recognition_model_name=SEAL_TEXT_RECOGNITION_MODEL_NAME,
        seal_text_recognition_model_dir=SEAL_TEXT_RECOGNITION_MODEL_DIR,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        chart_recognition_model_dir=CHART_RECOGNITION_MODEL_DIR,
        layout_threshold=LAYOUT_THRESHOLD,
        layout_nms=LAYOUT_NMS,
        layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
        layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        seal_det_limit_side_len=SEAL_DET_LIMIT_SIDE_LEN,
        seal_det_limit_type=SEAL_DET_LIMIT_TYPE,
        seal_det_thresh=SEAL_DET_THRESH,
        seal_det_box_thresh=SEAL_DET_BOX_THRESH,
        seal_det_unclip_ratio=SEAL_DET_UNCLIP_RATIO,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        textline_orientation_batch_size=TEXTLINE_ORIENTATION_BATCH_SIZE,
        formula_recognition_batch_size=FORMULA_RECOGNITION_BATCH_SIZE,
        chart_recognition_batch_size=CHART_RECOGNITION_BATCH_SIZE,
        seal_text_recognition_batch_size=SEAL_TEXT_RECOGNITION_BATCH_SIZE,
        seal_rec_score_thresh=SEAL_REC_SCORE_THRESH,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
        use_seal_recognition=USE_SEAL_RECOGNITION,
        use_region_detection=USE_REGION_DETECTION,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    app.state.pipeline_cache = OrderedDict()
    yield

app = FastAPI(
    title="PPStructureV3 /parse API", 
    version="1.8.0",
    description="Production-ready PP-StructureV3 endpoint with full ARM64 support and proper Markdown generation",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.8.0", "architecture": platform.machine()}

def _get_or_create_pipeline(app: FastAPI, effective: Dict[str, Any]) -> PPStructureV3:
    if not effective:
        return app.state.pipeline
    cache: OrderedDict = app.state.pipeline_cache
    eff_key = _build_config_key(effective)
    if eff_key in cache:
        pipe = cache.pop(eff_key)
        cache[eff_key] = pipe
        return pipe
    while len(cache) >= PIPELINE_CACHE_SIZE:
        cache.popitem(last=False)
    base_defaults = dict(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        use_tensorrt=USE_TENSORRT,
        precision=PRECISION,
        mkldnn_cache_capacity=MKLDNN_CACHE_CAPACITY,
        cpu_threads=CPU_THREADS,
        paddlex_config=PADDLEX_CONFIG,
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
        use_seal_recognition=USE_SEAL_RECOGNITION,
        use_region_detection=USE_REGION_DETECTION,
    )
    base_defaults.update(effective)
    final_params = {k: v for k, v in base_defaults.items() if v is not None}
    pipe = _make_pipeline(**final_params)
    cache[eff_key] = pipe
    return pipe

def _predict_table_kwargs(
    use_ocr_results_with_table_cells: Optional[bool],
    use_e2e_wired_table_rec_model: Optional[bool],
    use_e2e_wireless_table_rec_model: Optional[bool],
    use_wired_table_cells_trans_to_html: Optional[bool],
    use_wireless_table_cells_trans_to_html: Optional[bool],
    use_table_orientation_classify: Optional[bool],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if use_ocr_results_with_table_cells is not None:
        kwargs["use_ocr_results_with_table_cells"] = use_ocr_results_with_table_cells
    if use_e2e_wired_table_rec_model is not None:
        kwargs["use_e2e_wired_table_rec_model"] = use_e2e_wired_table_rec_model
    if use_e2e_wireless_table_rec_model is not None:
        kwargs["use_e2e_wireless_table_rec_model"] = use_e2e_wireless_table_rec_model
    if use_wired_table_cells_trans_to_html is not None:
        kwargs["use_wired_table_cells_trans_to_html"] = use_wired_table_cells_trans_to_html
    if use_wireless_table_cells_trans_to_html is not None:
        kwargs["use_wireless_table_cells_trans_to_html"] = use_wireless_table_cells_trans_to_html
    if use_table_orientation_classify is not None:
        kwargs["use_table_orientation_classify"] = use_table_orientation_classify
    return kwargs

@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="Image or PDF file to parse"),
    output_format: Literal["json", "markdown", "both"] = Query(
        "json",
        description="Output format: json (structured), markdown (text), or both"
    ),
    markdown_images: Literal["none", "inline"] = Query(
        "none",
        description="Image handling: none (paths) or inline (base64 data URIs)"
    ),
    
    # Backend config
    device: Optional[str] = Query(None, description="Inference device: cpu, gpu, gpu:0, etc."),
    enable_mkldnn: Optional[bool] = Query(None, description="Enable MKL-DNN CPU acceleration"),
    enable_hpi: Optional[bool] = Query(None, description="Enable high-performance inference (x86 only)"),
    use_tensorrt: Optional[bool] = Query(None, description="Enable TensorRT GPU acceleration"),
    precision: Optional[str] = Query(None, description="Precision: fp32 or fp16"),
    mkldnn_cache_capacity: Optional[int] = Query(None, description="MKL-DNN cache capacity"),
    cpu_threads: Optional[int] = Query(None, description="CPU inference threads"),
    paddlex_config: Optional[str] = Query(None, description="PaddleX config file path"),
    
    # Subpipeline toggles
    use_doc_orientation_classify: Optional[bool] = Query(None, description="Enable document orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(None, description="Enable document unwarping"),
    use_textline_orientation: Optional[bool] = Query(None, description="Enable text line orientation classification"),
    use_table_recognition: Optional[bool] = Query(None, description="Enable table recognition"),
    use_formula_recognition: Optional[bool] = Query(None, description="Enable formula recognition (LaTeX output)"),
    use_chart_recognition: Optional[bool] = Query(None, description="Enable chart parsing to tables"),
    use_seal_recognition: Optional[bool] = Query(None, description="Enable seal/stamp text recognition"),
    use_region_detection: Optional[bool] = Query(None, description="Enable document region detection"),
    
    # Table behavior (predict-time only)
    use_ocr_results_with_table_cells: Optional[bool] = Query(None, description="Use OCR for table cell text"),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None, description="End-to-end wired table recognition"),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None, description="End-to-end wireless table recognition"),
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None, description="Convert wired table cells to HTML"),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None, description="Convert wireless table cells to HTML"),
    use_table_orientation_classify: Optional[bool] = Query(None, description="Classify table orientation"),
    
    # Model names
    layout_detection_model_name: Optional[str] = Query(None, description="Layout detection model name"),
    layout_detection_model_dir: Optional[str] = Query(None, description="Layout detection model directory"),
    region_detection_model_name: Optional[str] = Query(None, description="Region detection model name"),
    region_detection_model_dir: Optional[str] = Query(None, description="Region detection model directory"),
    text_detection_model_name: Optional[str] = Query(None, description="Text detection model name"),
    text_detection_model_dir: Optional[str] = Query(None, description="Text detection model directory"),
    text_recognition_model_name: Optional[str] = Query(None, description="Text recognition model name"),
    text_recognition_model_dir: Optional[str] = Query(None, description="Text recognition model directory"),
    table_classification_model_name: Optional[str] = Query(None, description="Table classification model name"),
    table_classification_model_dir: Optional[str] = Query(None, description="Table classification model directory"),
    wired_table_structure_recognition_model_name: Optional[str] = Query(None),
    wired_table_structure_recognition_model_dir: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_dir: Optional[str] = Query(None),
    wired_table_cells_detection_model_name: Optional[str] = Query(None),
    wired_table_cells_detection_model_dir: Optional[str] = Query(None),
    wireless_table_cells_detection_model_name: Optional[str] = Query(None),
    wireless_table_cells_detection_model_dir: Optional[str] = Query(None),
    table_orientation_classify_model_name: Optional[str] = Query(None),
    table_orientation_classify_model_dir: Optional[str] = Query(None),
    formula_recognition_model_name: Optional[str] = Query(None, description="Formula recognition model name"),
    formula_recognition_model_dir: Optional[str] = Query(None, description="Formula recognition model directory"),
    chart_recognition_model_name: Optional[str] = Query(None, description="Chart recognition model name"),
    chart_recognition_model_dir: Optional[str] = Query(None, description="Chart recognition model directory"),
    doc_orientation_classify_model_name: Optional[str] = Query(None),
    doc_orientation_classify_model_dir: Optional[str] = Query(None),
    doc_unwarping_model_name: Optional[str] = Query(None),
    doc_unwarping_model_dir: Optional[str] = Query(None),
    textline_orientation_model_name: Optional[str] = Query(None),
    textline_orientation_model_dir: Optional[str] = Query(None),
    seal_text_detection_model_name: Optional[str] = Query(None),
    seal_text_detection_model_dir: Optional[str] = Query(None),
    seal_text_recognition_model_name: Optional[str] = Query(None),
    seal_text_recognition_model_dir: Optional[str] = Query(None),
    
    # Thresholds and batch sizes
    layout_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Layout detection confidence threshold"),
    layout_nms: Optional[bool] = Query(None, description="Enable NMS for layout detection"),
    layout_unclip_ratio: Optional[float] = Query(None, gt=0.0, description="Layout box expansion ratio"),
    layout_merge_bboxes_mode: Optional[str] = Query(None, description="Layout bbox merge mode"),
    text_det_limit_side_len: Optional[int] = Query(None, gt=0, description="Text detection max side length"),
    text_det_limit_type: Optional[str] = Query(None, description="Text detection limit type: min or max"),
    text_det_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Text detection pixel threshold"),
    text_det_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Text detection box threshold"),
    text_det_unclip_ratio: Optional[float] = Query(None, gt=0.0, description="Text detection unclip ratio"),
    text_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Text recognition score threshold"),
    text_recognition_batch_size: Optional[int] = Query(None, gt=0, description="Text recognition batch size"),
    textline_orientation_batch_size: Optional[int] = Query(None, gt=0),
    formula_recognition_batch_size: Optional[int] = Query(None, gt=0, description="Formula recognition batch size"),
    chart_recognition_batch_size: Optional[int] = Query(None, gt=0, description="Chart recognition batch size"),
    seal_text_recognition_batch_size: Optional[int] = Query(None, gt=0),
    seal_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    seal_det_limit_side_len: Optional[int] = Query(None, gt=0),
    seal_det_limit_type: Optional[str] = Query(None),
    seal_det_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    seal_det_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    seal_det_unclip_ratio: Optional[float] = Query(None, gt=0.0),
):
    """
    Parse document images or PDFs using PP-StructureV3 pipeline.
    
    **Supports all PP-StructureV3 features:**
    - Layout detection with 20+ categories (title, paragraph, table, formula, etc.)
    - High-accuracy OCR (80+ languages, handwriting, vertical text)
    - Table recognition with structure preservation
    - Formula recognition outputting LaTeX
    - Chart parsing to structured tables
    - Seal/stamp text recognition
    - Document preprocessing (orientation, unwarping)
    
    **Returns:**
    - JSON: Structured per-page results with bboxes, text, and metadata
    - Markdown: Human-readable format with headings, tables, and formulas
    - Both: Combined JSON + Markdown in single response
    
    **Markdown quality:**
    - ATX-style headings (# ## ###) preserve document hierarchy
    - Tables rendered with proper Markdown/HTML syntax
    - Reading order maintained across multi-column layouts
    - Formulas embedded as LaTeX blocks
    - Charts converted to structured tables
    - Optional base64 image inlining for self-contained output
    """
    
    # Validate upload
    if _file_too_big(file):
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    if not _ext_ok(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    # Save to temp disk location
    tmp_dir = tempfile.mkdtemp(prefix="ppsv3_")
    tmp_path = os.path.join(tmp_dir, file.filename)
    
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Build constructor parameter overrides
        effective = {}
        for k, v in dict(
            device=device, enable_mkldnn=enable_mkldnn, enable_hpi=enable_hpi,
            use_tensorrt=use_tensorrt, precision=precision, mkldnn_cache_capacity=mkldnn_cache_capacity,
            cpu_threads=cpu_threads, paddlex_config=paddlex_config,
            use_doc_orientation_classify=use_doc_orientation_classify, use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation, use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition, use_chart_recognition=use_chart_recognition,
            use_seal_recognition=use_seal_recognition, use_region_detection=use_region_detection,
            layout_detection_model_name=layout_detection_model_name,
            layout_detection_model_dir=layout_detection_model_dir,
            region_detection_model_name=region_detection_model_name,
            region_detection_model_dir=region_detection_model_dir,
            text_detection_model_name=text_detection_model_name,
            text_detection_model_dir=text_detection_model_dir,
            text_recognition_model_name=text_recognition_model_name,
            text_recognition_model_dir=text_recognition_model_dir,
            table_classification_model_name=table_classification_model_name,
            table_classification_model_dir=table_classification_model_dir,
            wired_table_structure_recognition_model_name=wired_table_structure_recognition_model_name,
            wired_table_structure_recognition_model_dir=wired_table_structure_recognition_model_dir,
            wireless_table_structure_recognition_model_name=wireless_table_structure_recognition_model_name,
            wireless_table_structure_recognition_model_dir=wireless_table_structure_recognition_model_dir,
            wired_table_cells_detection_model_name=wired_table_cells_detection_model_name,
            wired_table_cells_detection_model_dir=wired_table_cells_detection_model_dir,
            wireless_table_cells_detection_model_name=wireless_table_cells_detection_model_name,
            wireless_table_cells_detection_model_dir=wireless_table_cells_detection_model_dir,
            table_orientation_classify_model_name=table_orientation_classify_model_name,
            table_orientation_classify_model_dir=table_orientation_classify_model_dir,
            formula_recognition_model_name=formula_recognition_model_name,
            formula_recognition_model_dir=formula_recognition_model_dir,
            chart_recognition_model_name=chart_recognition_model_name,
            chart_recognition_model_dir=chart_recognition_model_dir,
            doc_orientation_classify_model_name=doc_orientation_classify_model_name,
            doc_orientation_classify_model_dir=doc_orientation_classify_model_dir,
            doc_unwarping_model_name=doc_unwarping_model_name,
            doc_unwarping_model_dir=doc_unwarping_model_dir,
            textline_orientation_model_name=textline_orientation_model_name,
            textline_orientation_model_dir=textline_orientation_model_dir,
            seal_text_detection_model_name=seal_text_detection_model_name,
            seal_text_detection_model_dir=seal_text_detection_model_dir,
            seal_text_recognition_model_name=seal_text_recognition_model_name,
            seal_text_recognition_model_dir=seal_text_recognition_model_dir,
            layout_threshold=layout_threshold, layout_nms=layout_nms, layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode, text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type, text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh, text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh, text_recognition_batch_size=text_recognition_batch_size,
            textline_orientation_batch_size=textline_orientation_batch_size,
            formula_recognition_batch_size=formula_recognition_batch_size,
            chart_recognition_batch_size=chart_recognition_batch_size,
            seal_text_recognition_batch_size=seal_text_recognition_batch_size,
            seal_rec_score_thresh=seal_rec_score_thresh,
            seal_det_limit_side_len=seal_det_limit_side_len, seal_det_limit_type=seal_det_limit_type,
            seal_det_thresh=seal_det_thresh, seal_det_box_thresh=seal_det_box_thresh,
            seal_det_unclip_ratio=seal_det_unclip_ratio,
        ).items():
            if v is not None:
                effective[k] = v
        
        # Get or create pipeline instance
        pipeline = _get_or_create_pipeline(app, effective)
        
        # Build predict-time table kwargs
        predict_kwargs = _predict_table_kwargs(
            use_ocr_results_with_table_cells,
            use_e2e_wired_table_rec_model,
            use_e2e_wireless_table_rec_model,
            use_wired_table_cells_trans_to_html,
            use_wireless_table_cells_trans_to_html,
            use_table_orientation_classify,
        )
        
        # Run inference with concurrency control
        acquired = app.state.predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy, please retry")
        
        try:
            outputs = await run_in_threadpool(lambda: pipeline.predict(input=tmp_path, **predict_kwargs))
        finally:
            app.state.predict_sem.release()
        
        # Extract results with proper markdown handling
        inline_flag = markdown_images == "inline"
        page_json, merged_md = _collect_results(pipeline, outputs, inline_flag)
        
        # Return in requested format
        if output_format == "json":
            return JSONResponse({
                "results": page_json,
                "pages": len(page_json),
                "file": file.filename
            })
        elif output_format == "markdown":
            return PlainTextResponse(merged_md or "", media_type="text/markdown")
        else:  # both
            return JSONResponse({
                "results": page_json,
                "markdown": merged_md or "",
                "pages": len(page_json),
                "file": file.filename
            })
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline inference failed: {type(e).__name__}: {str(e)}"
        )
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

import os
import json
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

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
USE_SEAL_RECOGNITION = False
USE_REGION_DETECTION = True

# Advanced table behavior toggles (additions)
USE_OCR_RESULTS_WITH_TABLE_CELLS = True
USE_E2E_WIRED_TABLE_REC_MODEL = False
USE_E2E_WIRELESS_TABLE_REC_MODEL = False
USE_WIRED_TABLE_CELLS_TRANS_TO_HTML = False
USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML = False
USE_TABLE_ORIENTATION_CLASSIFY = True

# Model overrides
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

# Optional model directories (local checkpoints)
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

# Detection/recognition parameters
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

# Inference/backend knobs
ENABLE_HPI = False
ENABLE_MKLDNN = True
USE_TENSORRT = False
PRECISION = "fp32"
MKLDNN_CACHE_CAPACITY = 10
PADDLEX_CONFIG = None  # optional unified config

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1
PIPELINE_CACHE_SIZE = 2  # simple LRU cache for alt configs

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

def _collect_result_json_and_markdown(pipeline: PPStructureV3, outputs):
    # Use documented attributes to extract per-page results and optional page-level markdown
    page_json = []
    markdown_list = []
    for res in outputs:
        page_json.append(getattr(res, "json", {}))
        md = getattr(res, "markdown", None)
        markdown_list.append(md if isinstance(md, dict) else {})
    merged_md = ""
    if markdown_list and any(markdown_list):
        try:
            merged_md = pipeline.concatenate_markdown_pages(markdown_list)
        except Exception:
            # Temporary fallback for older builds or environments
            try:
                merged_md = getattr(getattr(pipeline, "paddlex_pipeline", None), "concatenate_markdown_pages")(markdown_list)  # type: ignore
            except Exception:
                texts = []
                for item in markdown_list:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item.get("text", ""))
                merged_md = "\n\n".join(texts) if texts else ""
    return page_json, merged_md

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = _make_pipeline(
        # Backend
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        use_tensorrt=USE_TENSORRT,
        precision=PRECISION,
        mkldnn_cache_capacity=MKLDNN_CACHE_CAPACITY,
        cpu_threads=CPU_THREADS,
        paddlex_config=PADDLEX_CONFIG,

        # Models and dirs
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

        # Layout params
        layout_threshold=LAYOUT_THRESHOLD,
        layout_nms=LAYOUT_NMS,
        layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
        layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,

        # Text det params
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,

        # Seal det params
        seal_det_limit_side_len=SEAL_DET_LIMIT_SIDE_LEN,
        seal_det_limit_type=SEAL_DET_LIMIT_TYPE,
        seal_det_thresh=SEAL_DET_THRESH,
        seal_det_box_thresh=SEAL_DET_BOX_THRESH,
        seal_det_unclip_ratio=SEAL_DET_UNCLIP_RATIO,

        # Rec/Batch params
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        textline_orientation_batch_size=TEXTLINE_ORIENTATION_BATCH_SIZE,
        formula_recognition_batch_size=FORMULA_RECOGNITION_BATCH_SIZE,
        chart_recognition_batch_size=CHART_RECOGNITION_BATCH_SIZE,
        seal_text_recognition_batch_size=SEAL_TEXT_RECOGNITION_BATCH_SIZE,
        seal_rec_score_thresh=SEAL_REC_SCORE_THRESH,

        # Toggles
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
        use_seal_recognition=USE_SEAL_RECOGNITION,
        use_region_detection=USE_REGION_DETECTION,

        # Advanced table behavior toggles
        use_ocr_results_with_table_cells=USE_OCR_RESULTS_WITH_TABLE_CELLS,
        use_e2e_wired_table_rec_model=USE_E2E_WIRED_TABLE_REC_MODEL,
        use_e2e_wireless_table_rec_model=USE_E2E_WIRELESS_TABLE_REC_MODEL,
        use_wired_table_cells_trans_to_html=USE_WIRED_TABLE_CELLS_TRANS_TO_HTML,
        use_wireless_table_cells_trans_to_html=USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML,
        use_table_orientation_classify=USE_TABLE_ORIENTATION_CLASSIFY,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    app.state.pipeline_cache = OrderedDict()
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.4.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

def _effective_params_from_query(
    # Device/backend
    device: Optional[str], enable_mkldnn: Optional[bool], enable_hpi: Optional[bool],
    use_tensorrt: Optional[bool], precision: Optional[str], mkldnn_cache_capacity: Optional[int],
    cpu_threads: Optional[int], paddlex_config: Optional[str],

    # Toggles
    use_doc_orientation_classify: Optional[bool], use_doc_unwarping: Optional[bool],
    use_textline_orientation: Optional[bool], use_table_recognition: Optional[bool],
    use_formula_recognition: Optional[bool], use_chart_recognition: Optional[bool],
    use_seal_recognition: Optional[bool], use_region_detection: Optional[bool],

    # Advanced table behavior toggles
    use_ocr_results_with_table_cells: Optional[bool],
    use_e2e_wired_table_rec_model: Optional[bool],
    use_e2e_wireless_table_rec_model: Optional[bool],
    use_wired_table_cells_trans_to_html: Optional[bool],
    use_wireless_table_cells_trans_to_html: Optional[bool],
    use_table_orientation_classify: Optional[bool],

    # Models and dirs
    layout_detection_model_name: Optional[str], layout_detection_model_dir: Optional[str],
    region_detection_model_name: Optional[str], region_detection_model_dir: Optional[str],
    text_detection_model_name: Optional[str], text_detection_model_dir: Optional[str],
    text_recognition_model_name: Optional[str], text_recognition_model_dir: Optional[str],
    table_classification_model_name: Optional[str], table_classification_model_dir: Optional[str],
    wired_table_structure_recognition_model_name: Optional[str], wired_table_structure_recognition_model_dir: Optional[str],
    wireless_table_structure_recognition_model_name: Optional[str], wireless_table_structure_recognition_model_dir: Optional[str],
    wired_table_cells_detection_model_name: Optional[str], wired_table_cells_detection_model_dir: Optional[str],
    wireless_table_cells_detection_model_name: Optional[str], wireless_table_cells_detection_model_dir: Optional[str],
    table_orientation_classify_model_name: Optional[str], table_orientation_classify_model_dir: Optional[str],
    formula_recognition_model_name: Optional[str], formula_recognition_model_dir: Optional[str],
    chart_recognition_model_name: Optional[str], chart_recognition_model_dir: Optional[str],
    doc_orientation_classify_model_name: Optional[str], doc_orientation_classify_model_dir: Optional[str],
    doc_unwarping_model_name: Optional[str], doc_unwarping_model_dir: Optional[str],
    textline_orientation_model_name: Optional[str], textline_orientation_model_dir: Optional[str],
    seal_text_detection_model_name: Optional[str], seal_text_detection_model_dir: Optional[str],
    seal_text_recognition_model_name: Optional[str], seal_text_recognition_model_dir: Optional[str],

    # Parameters
    layout_threshold: Optional[float], layout_nms: Optional[bool],
    layout_unclip_ratio: Optional[float], layout_merge_bboxes_mode: Optional[str],
    text_det_limit_side_len: Optional[int], text_det_limit_type: Optional[str],
    text_det_thresh: Optional[float], text_det_box_thresh: Optional[float],
    text_det_unclip_ratio: Optional[float], text_rec_score_thresh: Optional[float],
    text_recognition_batch_size: Optional[int], textline_orientation_batch_size: Optional[int],
    formula_recognition_batch_size: Optional[int], chart_recognition_batch_size: Optional[int],
    seal_text_recognition_batch_size: Optional[int], seal_rec_score_thresh: Optional[float],
    seal_det_limit_side_len: Optional[int], seal_det_limit_type: Optional[str],
    seal_det_thresh: Optional[float], seal_det_box_thresh: Optional[float], seal_det_unclip_ratio: Optional[float],
) -> Dict[str, Any]:
    p = {}

    # Backend
    if device is not None: p["device"] = device
    if enable_mkldnn is not None: p["enable_mkldnn"] = enable_mkldnn
    if enable_hpi is not None: p["enable_hpi"] = enable_hpi
    if use_tensorrt is not None: p["use_tensorrt"] = use_tensorrt
    if precision is not None: p["precision"] = precision
    if mkldnn_cache_capacity is not None: p["mkldnn_cache_capacity"] = mkldnn_cache_capacity
    if cpu_threads is not None: p["cpu_threads"] = cpu_threads
    if paddlex_config is not None: p["paddlex_config"] = paddlex_config

    # Toggles
    if use_doc_orientation_classify is not None: p["use_doc_orientation_classify"] = use_doc_orientation_classify
    if use_doc_unwarping is not None: p["use_doc_unwarping"] = use_doc_unwarping
    if use_textline_orientation is not None: p["use_textline_orientation"] = use_textline_orientation
    if use_table_recognition is not None: p["use_table_recognition"] = use_table_recognition
    if use_formula_recognition is not None: p["use_formula_recognition"] = use_formula_recognition
    if use_chart_recognition is not None: p["use_chart_recognition"] = use_chart_recognition
    if use_seal_recognition is not None: p["use_seal_recognition"] = use_seal_recognition
    if use_region_detection is not None: p["use_region_detection"] = use_region_detection

    # Advanced table behavior toggles
    if use_ocr_results_with_table_cells is not None: p["use_ocr_results_with_table_cells"] = use_ocr_results_with_table_cells
    if use_e2e_wired_table_rec_model is not None: p["use_e2e_wired_table_rec_model"] = use_e2e_wired_table_rec_model
    if use_e2e_wireless_table_rec_model is not None: p["use_e2e_wireless_table_rec_model"] = use_e2e_wireless_table_rec_model
    if use_wired_table_cells_trans_to_html is not None: p["use_wired_table_cells_trans_to_html"] = use_wired_table_cells_trans_to_html
    if use_wireless_table_cells_trans_to_html is not None: p["use_wireless_table_cells_trans_to_html"] = use_wireless_table_cells_trans_to_html
    if use_table_orientation_classify is not None: p["use_table_orientation_classify"] = use_table_orientation_classify

    # Model names and dirs
    kvs = [
        ("layout_detection_model_name", layout_detection_model_name),
        ("layout_detection_model_dir", layout_detection_model_dir),
        ("region_detection_model_name", region_detection_model_name),
        ("region_detection_model_dir", region_detection_model_dir),
        ("text_detection_model_name", text_detection_model_name),
        ("text_detection_model_dir", text_detection_model_dir),
        ("text_recognition_model_name", text_recognition_model_name),
        ("text_recognition_model_dir", text_recognition_model_dir),
        ("table_classification_model_name", table_classification_model_name),
        ("table_classification_model_dir", table_classification_model_dir),
        ("wired_table_structure_recognition_model_name", wired_table_structure_recognition_model_name),
        ("wired_table_structure_recognition_model_dir", wired_table_structure_recognition_model_dir),
        ("wireless_table_structure_recognition_model_name", wireless_table_structure_recognition_model_name),
        ("wireless_table_structure_recognition_model_dir", wireless_table_structure_recognition_model_dir),
        ("wired_table_cells_detection_model_name", wired_table_cells_detection_model_name),
        ("wired_table_cells_detection_model_dir", wired_table_cells_detection_model_dir),
        ("wireless_table_cells_detection_model_name", wireless_table_cells_detection_model_name),
        ("wireless_table_cells_detection_model_dir", wireless_table_cells_detection_model_dir),
        ("table_orientation_classify_model_name", table_orientation_classify_model_name),
        ("table_orientation_classify_model_dir", table_orientation_classify_model_dir),
        ("formula_recognition_model_name", formula_recognition_model_name),
        ("formula_recognition_model_dir", formula_recognition_model_dir),
        ("chart_recognition_model_name", chart_recognition_model_name),
        ("chart_recognition_model_dir", chart_recognition_model_dir),
        ("doc_orientation_classify_model_name", doc_orientation_classify_model_name),
        ("doc_orientation_classify_model_dir", doc_orientation_classify_model_dir),
        ("doc_unwarping_model_name", doc_unwarping_model_name),
        ("doc_unwarping_model_dir", doc_unwarping_model_dir),
        ("textline_orientation_model_name", textline_orientation_model_name),
        ("textline_orientation_model_dir", textline_orientation_model_dir),
        ("seal_text_detection_model_name", seal_text_detection_model_name),
        ("seal_text_detection_model_dir", seal_text_detection_model_dir),
        ("seal_text_recognition_model_name", seal_text_recognition_model_name),
        ("seal_text_recognition_model_dir", seal_text_recognition_model_dir),
    ]
    for k, v in kvs:
        if v is not None:
            p[k] = v

    # Thresholds / batch sizes
    kvp = [
        ("layout_threshold", layout_threshold),
        ("layout_nms", layout_nms),
        ("layout_unclip_ratio", layout_unclip_ratio),
        ("layout_merge_bboxes_mode", layout_merge_bboxes_mode),
        ("text_det_limit_side_len", text_det_limit_side_len),
        ("text_det_limit_type", text_det_limit_type),
        ("text_det_thresh", text_det_thresh),
        ("text_det_box_thresh", text_det_box_thresh),
        ("text_det_unclip_ratio", text_det_unclip_ratio),
        ("text_rec_score_thresh", text_rec_score_thresh),
        ("text_recognition_batch_size", text_recognition_batch_size),
        ("textline_orientation_batch_size", textline_orientation_batch_size),
        ("formula_recognition_batch_size", formula_recognition_batch_size),
        ("chart_recognition_batch_size", chart_recognition_batch_size),
        ("seal_text_recognition_batch_size", seal_text_recognition_batch_size),
        ("seal_rec_score_thresh", seal_rec_score_thresh),
        ("seal_det_limit_side_len", seal_det_limit_side_len),
        ("seal_det_limit_type", seal_det_limit_type),
        ("seal_det_thresh", seal_det_thresh),
        ("seal_det_box_thresh", seal_det_box_thresh),
        ("seal_det_unclip_ratio", seal_det_unclip_ratio),
    ]
    for k, v in kvp:
        if v is not None:
            p[k] = v

    return p

def _get_or_create_pipeline(app: FastAPI, effective: Dict[str, Any]) -> PPStructureV3:
    # If no overrides, reuse default
    if not effective:
        return app.state.pipeline

    cache: OrderedDict = app.state.pipeline_cache
    eff_key = _build_config_key(effective)
    if eff_key in cache:
        pipe = cache.pop(eff_key)
        cache[eff_key] = pipe
        return pipe

    # Evict if needed
    while len(cache) >= PIPELINE_CACHE_SIZE:
        cache.popitem(last=False)

    # Merge defaults with overrides
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
        use_ocr_results_with_table_cells=USE_OCR_RESULTS_WITH_TABLE_CELLS,
        use_e2e_wired_table_rec_model=USE_E2E_WIRED_TABLE_REC_MODEL,
        use_e2e_wireless_table_rec_model=USE_E2E_WIRELESS_TABLE_REC_MODEL,
        use_wired_table_cells_trans_to_html=USE_WIRED_TABLE_CELLS_TRANS_TO_HTML,
        use_wireless_table_cells_trans_to_html=USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML,
        use_table_orientation_classify=USE_TABLE_ORIENTATION_CLASSIFY,
    )
    base_defaults.update(effective)
    final_params = {k: v for k, v in base_defaults.items() if v is not None}
    pipe = _make_pipeline(**final_params)
    cache[eff_key] = pipe
    return pipe

@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="Image or PDF"),
    output_format: Literal["json", "markdown", "both"] = Query("json"),

    # Backend
    device: Optional[str] = Query(None),
    enable_mkldnn: Optional[bool] = Query(None),
    enable_hpi: Optional[bool] = Query(None),
    use_tensorrt: Optional[bool] = Query(None),
    precision: Optional[str] = Query(None),
    mkldnn_cache_capacity: Optional[int] = Query(None),
    cpu_threads: Optional[int] = Query(None),
    paddlex_config: Optional[str] = Query(None),

    # Toggles
    use_doc_orientation_classify: Optional[bool] = Query(None),
    use_doc_unwarping: Optional[bool] = Query(None),
    use_textline_orientation: Optional[bool] = Query(None),
    use_table_recognition: Optional[bool] = Query(None),
    use_formula_recognition: Optional[bool] = Query(None),
    use_chart_recognition: Optional[bool] = Query(None),
    use_seal_recognition: Optional[bool] = Query(None),
    use_region_detection: Optional[bool] = Query(None),

    # Advanced table behavior toggles
    use_ocr_results_with_table_cells: Optional[bool] = Query(None),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None),
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None),
    use_table_orientation_classify: Optional[bool] = Query(None),

    # Model names
    layout_detection_model_name: Optional[str] = Query(None),
    region_detection_model_name: Optional[str] = Query(None),
    text_detection_model_name: Optional[str] = Query(None),
    text_recognition_model_name: Optional[str] = Query(None),
    table_classification_model_name: Optional[str] = Query(None),
    wired_table_structure_recognition_model_name: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(None),
    wired_table_cells_detection_model_name: Optional[str] = Query(None),
    wireless_table_cells_detection_model_name: Optional[str] = Query(None),
    table_orientation_classify_model_name: Optional[str] = Query(None),
    formula_recognition_model_name: Optional[str] = Query(None),
    chart_recognition_model_name: Optional[str] = Query(None),
    doc_orientation_classify_model_name: Optional[str] = Query(None),
    doc_unwarping_model_name: Optional[str] = Query(None),
    textline_orientation_model_name: Optional[str] = Query(None),
    seal_text_detection_model_name: Optional[str] = Query(None),
    seal_text_recognition_model_name: Optional[str] = Query(None),

    # Model dirs
    layout_detection_model_dir: Optional[str] = Query(None),
    region_detection_model_dir: Optional[str] = Query(None),
    text_detection_model_dir: Optional[str] = Query(None),
    text_recognition_model_dir: Optional[str] = Query(None),
    table_classification_model_dir: Optional[str] = Query(None),
    wired_table_structure_recognition_model_dir: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_dir: Optional[str] = Query(None),
    wired_table_cells_detection_model_dir: Optional[str] = Query(None),
    wireless_table_cells_detection_model_dir: Optional[str] = Query(None),
    table_orientation_classify_model_dir: Optional[str] = Query(None),
    formula_recognition_model_dir: Optional[str] = Query(None),
    chart_recognition_model_dir: Optional[str] = Query(None),
    doc_orientation_classify_model_dir: Optional[str] = Query(None),
    doc_unwarping_model_dir: Optional[str] = Query(None),
    textline_orientation_model_dir: Optional[str] = Query(None),
    seal_text_detection_model_dir: Optional[str] = Query(None),
    seal_text_recognition_model_dir: Optional[str] = Query(None),

    # Parameters
    layout_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    layout_nms: Optional[bool] = Query(None),
    layout_unclip_ratio: Optional[float] = Query(None, gt=0.0),
    layout_merge_bboxes_mode: Optional[str] = Query(None),
    text_det_limit_side_len: Optional[int] = Query(None, gt=0),
    text_det_limit_type: Optional[str] = Query(None),
    text_det_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_unclip_ratio: Optional[float] = Query(None, gt=0.0),
    text_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_recognition_batch_size: Optional[int] = Query(None, gt=0),
    textline_orientation_batch_size: Optional[int] = Query(None, gt=0),
    formula_recognition_batch_size: Optional[int] = Query(None, gt=0),
    chart_recognition_batch_size: Optional[int] = Query(None, gt=0),
    seal_text_recognition_batch_size: Optional[int] = Query(None, gt=0),
    seal_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    seal_det_limit_side_len: Optional[int] = Query(None, gt=0),
    seal_det_limit_type: Optional[str] = Query(None),
    seal_det_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    seal_det_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    seal_det_unclip_ratio: Optional[float] = Query(None, gt=0.0),
):
    # Validate upload
    if _file_too_big(file):
        raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB)")
    if not _ext_ok(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type; allowed: {sorted(ALLOWED_EXTENSIONS)}")

    # Persist to disk
    tmp_dir = tempfile.mkdtemp(prefix="ppsv3_")
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Build effective overrides
        effective = _effective_params_from_query(
            device, enable_mkldnn, enable_hpi, use_tensorrt, precision, mkldnn_cache_capacity, cpu_threads, paddlex_config,
            use_doc_orientation_classify, use_doc_unwarping, use_textline_orientation,
            use_table_recognition, use_formula_recognition, use_chart_recognition, use_seal_recognition, use_region_detection,
            use_ocr_results_with_table_cells, use_e2e_wired_table_rec_model, use_e2e_wireless_table_rec_model,
            use_wired_table_cells_trans_to_html, use_wireless_table_cells_trans_to_html, use_table_orientation_classify,
            layout_detection_model_name, layout_detection_model_dir,
            region_detection_model_name, region_detection_model_dir,
            text_detection_model_name, text_detection_model_dir,
            text_recognition_model_name, text_recognition_model_dir,
            table_classification_model_name, table_classification_model_dir,
            wired_table_structure_recognition_model_name, wired_table_structure_recognition_model_dir,
            wireless_table_structure_recognition_model_name, wireless_table_structure_recognition_model_dir,
            wired_table_cells_detection_model_name, wired_table_cells_detection_model_dir,
            wireless_table_cells_detection_model_name, wireless_table_cells_detection_model_dir,
            table_orientation_classify_model_name, table_orientation_classify_model_dir,
            formula_recognition_model_name, formula_recognition_model_dir,
            chart_recognition_model_name, chart_recognition_model_dir,
            doc_orientation_classify_model_name, doc_orientation_classify_model_dir,
            doc_unwarping_model_name, doc_unwarping_model_dir,
            textline_orientation_model_name, textline_orientation_model_dir,
            seal_text_detection_model_name, seal_text_detection_model_dir,
            seal_text_recognition_model_name, seal_text_recognition_model_dir,
            layout_threshold, layout_nms, layout_unclip_ratio, layout_merge_bboxes_mode,
            text_det_limit_side_len, text_det_limit_type, text_det_thresh, text_det_box_thresh, text_det_unclip_ratio,
            text_rec_score_thresh, text_recognition_batch_size, textline_orientation_batch_size,
            formula_recognition_batch_size, chart_recognition_batch_size, seal_text_recognition_batch_size, seal_rec_score_thresh,
            seal_det_limit_side_len, seal_det_limit_type, seal_det_thresh, seal_det_box_thresh, seal_det_unclip_ratio,
        )

        # Choose pipeline
        pipeline = _get_or_create_pipeline(app, effective)

        # Concurrency guard
        acquired = app.state.predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy")
        try:
            outputs = await run_in_threadpool(lambda: pipeline.predict(input=tmp_path))
        finally:
            app.state.predict_sem.release()

        # Collect results
        page_json, merged_md = _collect_result_json_and_markdown(pipeline, outputs)

        if output_format == "json":
            return JSONResponse({"results": page_json, "pages": len(page_json)})
        elif output_format == "markdown":
            return PlainTextResponse(merged_md or "")
        else:
            return JSONResponse({"results": page_json, "markdown": merged_md or "", "pages": len(page_json)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {str(e)}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

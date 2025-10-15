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

# Inference/backend knobs
ENABLE_HPI = False
ENABLE_MKLDNN = True
USE_TENSORRT = False
PRECISION = "fp32"
MKLDNN_CACHE_CAPACITY = 10

# I/O and service limits
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

def _collect_result_json_and_markdown(pipeline: PPStructureV3, outputs):
    """Extract JSON and Markdown from pipeline results using documented APIs"""
    page_json = []
    markdown_list = []
    
    for res in outputs:
        # Use documented json attribute 
        if hasattr(res, 'json'):
            page_json.append(res.json)
        else:
            # Fallback for older versions
            with tempfile.TemporaryDirectory() as td:
                res.save_to_json(save_path=td)
                jfiles = sorted(Path(td).glob("*.json"))
                if jfiles:
                    page_json.append(json.loads(jfiles[-1].read_text(encoding="utf-8")))
                else:
                    page_json.append({})
        
        # Collect markdown metadata
        if hasattr(res, 'markdown'):
            markdown_list.append(res.markdown)
        else:
            markdown_list.append({})
    
    # Use documented concatenate_markdown_pages method
    merged_md = ""
    if markdown_list and any(markdown_list):
        try:
            merged_md = pipeline.concatenate_markdown_pages(markdown_list)
        except Exception:
            # Fallback to workaround if needed (for older versions with the bug)
            try:
                merged_md = pipeline.paddlex_pipeline.concatenate_markdown_pages(markdown_list)
            except Exception:
                # Final fallback
                texts = []
                for item in markdown_list:
                    if isinstance(item, dict) and "markdown_texts" in item:
                        texts.append(item.get("markdown_texts", ""))
                merged_md = "\n\n".join(texts) if texts else ""
    
    return page_json, merged_md

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        use_tensorrt=USE_TENSORRT,
        precision=PRECISION,
        mkldnn_cache_capacity=MKLDNN_CACHE_CAPACITY,
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
        use_seal_recognition=USE_SEAL_RECOGNITION,
        use_region_detection=USE_REGION_DETECTION,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    app.state.pipeline_cache = OrderedDict()
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.3.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

def _effective_params_from_query(
    # Device and backend
    device: Optional[str], enable_mkldnn: Optional[bool], enable_hpi: Optional[bool],
    use_tensorrt: Optional[bool], precision: Optional[str], mkldnn_cache_capacity: Optional[int],
    cpu_threads: Optional[int], paddlex_config: Optional[str],
    
    # Toggles
    use_doc_orientation_classify: Optional[bool], use_doc_unwarping: Optional[bool],
    use_textline_orientation: Optional[bool], use_table_recognition: Optional[bool],
    use_formula_recognition: Optional[bool], use_chart_recognition: Optional[bool],
    use_seal_recognition: Optional[bool], use_region_detection: Optional[bool],
    
    # Models
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
    
    # Build complete parameter dict with all documented parameters
    params = {}
    
    # Device/Backend parameters
    if device is not None: params["device"] = device
    if enable_mkldnn is not None: params["enable_mkldnn"] = enable_mkldnn
    if enable_hpi is not None: params["enable_hpi"] = enable_hpi
    if use_tensorrt is not None: params["use_tensorrt"] = use_tensorrt
    if precision is not None: params["precision"] = precision
    if mkldnn_cache_capacity is not None: params["mkldnn_cache_capacity"] = mkldnn_cache_capacity
    if cpu_threads is not None: params["cpu_threads"] = cpu_threads
    if paddlex_config is not None: params["paddlex_config"] = paddlex_config
    
    # Toggle parameters
    if use_doc_orientation_classify is not None: params["use_doc_orientation_classify"] = use_doc_orientation_classify
    if use_doc_unwarping is not None: params["use_doc_unwarping"] = use_doc_unwarping
    if use_textline_orientation is not None: params["use_textline_orientation"] = use_textline_orientation
    if use_table_recognition is not None: params["use_table_recognition"] = use_table_recognition
    if use_formula_recognition is not None: params["use_formula_recognition"] = use_formula_recognition
    if use_chart_recognition is not None: params["use_chart_recognition"] = use_chart_recognition
    if use_seal_recognition is not None: params["use_seal_recognition"] = use_seal_recognition
    if use_region_detection is not None: params["use_region_detection"] = use_region_detection
    
    # Model names and directories
    if layout_detection_model_name is not None: params["layout_detection_model_name"] = layout_detection_model_name
    if layout_detection_model_dir is not None: params["layout_detection_model_dir"] = layout_detection_model_dir
    if region_detection_model_name is not None: params["region_detection_model_name"] = region_detection_model_name
    if region_detection_model_dir is not None: params["region_detection_model_dir"] = region_detection_model_dir
    if text_detection_model_name is not None: params["text_detection_model_name"] = text_detection_model_name
    if text_detection_model_dir is not None: params["text_detection_model_dir"] = text_detection_model_dir
    if text_recognition_model_name is not None: params["text_recognition_model_name"] = text_recognition_model_name
    if text_recognition_model_dir is not None: params["text_recognition_model_dir"] = text_recognition_model_dir
    if table_classification_model_name is not None: params["table_classification_model_name"] = table_classification_model_name
    if table_classification_model_dir is not None: params["table_classification_model_dir"] = table_classification_model_dir
    if wired_table_structure_recognition_model_name is not None: params["wired_table_structure_recognition_model_name"] = wired_table_structure_recognition_model_name
    if wired_table_structure_recognition_model_dir is not None: params["wired_table_structure_recognition_model_dir"] = wired_table_structure_recognition_model_dir
    if wireless_table_structure_recognition_model_name is not None: params["wireless_table_structure_recognition_model_name"] = wireless_table_structure_recognition_model_name
    if wireless_table_structure_recognition_model_dir is not None: params["wireless_table_structure_recognition_model_dir"] = wireless_table_structure_recognition_model_dir
    if wired_table_cells_detection_model_name is not None: params["wired_table_cells_detection_model_name"] = wired_table_cells_detection_model_name
    if wired_table_cells_detection_model_dir is not None: params["wired_table_cells_detection_model_dir"] = wired_table_cells_detection_model_dir
    if wireless_table_cells_detection_model_name is not None: params["wireless_table_cells_detection_model_name"] = wireless_table_cells_detection_model_name
    if wireless_table_cells_detection_model_dir is not None: params["wireless_table_cells_detection_model_dir"] = wireless_table_cells_detection_model_dir
    if table_orientation_classify_model_name is not None: params["table_orientation_classify_model_name"] = table_orientation_classify_model_name
    if table_orientation_classify_model_dir is not None: params["table_orientation_classify_model_dir"] = table_orientation_classify_model_dir
    if formula_recognition_model_name is not None: params["formula_recognition_model_name"] = formula_recognition_model_name
    if formula_recognition_model_dir is not None: params["formula_recognition_model_dir"] = formula_recognition_model_dir
    if chart_recognition_model_name is not None: params["chart_recognition_model_name"] = chart_recognition_model_name
    if chart_recognition_model_dir is not None: params["chart_recognition_model_dir"] = chart_recognition_model_dir
    if doc_orientation_classify_model_name is not None: params["doc_orientation_classify_model_name"] = doc_orientation_classify_model_name
    if doc_orientation_classify_model_dir is not None: params["doc_orientation_classify_model_dir"] = doc_orientation_classify_model_dir
    if doc_unwarping_model_name is not None: params["doc_unwarping_model_name"] = doc_unwarping_model_name
    if doc_unwarping_model_dir is not None: params["doc_unwarping_model_dir"] = doc_unwarping_model_dir
    if textline_orientation_model_name is not None: params["textline_orientation_model_name"] = textline_orientation_model_name
    if textline_orientation_model_dir is not None: params["textline_orientation_model_dir"] = textline_orientation_model_dir
    if seal_text_detection_model_name is not None: params["seal_text_detection_model_name"] = seal_text_detection_model_name
    if seal_text_detection_model_dir is not None: params["seal_text_detection_model_dir"] = seal_text_detection_model_dir
    if seal_text_recognition_model_name is not None: params["seal_text_recognition_model_name"] = seal_text_recognition_model_name
    if seal_text_recognition_model_dir is not None: params["seal_text_recognition_model_dir"] = seal_text_recognition_model_dir
    
    # Threshold and batch parameters
    if layout_threshold is not None: params["layout_threshold"] = layout_threshold
    if layout_nms is not None: params["layout_nms"] = layout_nms
    if layout_unclip_ratio is not None: params["layout_unclip_ratio"] = layout_unclip_ratio
    if layout_merge_bboxes_mode is not None: params["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
    if text_det_limit_side_len is not None: params["text_det_limit_side_len"] = text_det_limit_side_len
    if text_det_limit_type is not None: params["text_det_limit_type"] = text_det_limit_type
    if text_det_thresh is not None: params["text_det_thresh"] = text_det_thresh
    if text_det_box_thresh is not None: params["text_det_box_thresh"] = text_det_box_thresh
    if text_det_unclip_ratio is not None: params["text_det_unclip_ratio"] = text_det_unclip_ratio
    if text_rec_score_thresh is not None: params["text_rec_score_thresh"] = text_rec_score_thresh
    if text_recognition_batch_size is not None: params["text_recognition_batch_size"] = text_recognition_batch_size
    if textline_orientation_batch_size is not None: params["textline_orientation_batch_size"] = textline_orientation_batch_size
    if formula_recognition_batch_size is not None: params["formula_recognition_batch_size"] = formula_recognition_batch_size
    if chart_recognition_batch_size is not None: params["chart_recognition_batch_size"] = chart_recognition_batch_size
    if seal_text_recognition_batch_size is not None: params["seal_text_recognition_batch_size"] = seal_text_recognition_batch_size
    if seal_rec_score_thresh is not None: params["seal_rec_score_thresh"] = seal_rec_score_thresh
    if seal_det_limit_side_len is not None: params["seal_det_limit_side_len"] = seal_det_limit_side_len
    if seal_det_limit_type is not None: params["seal_det_limit_type"] = seal_det_limit_type
    if seal_det_thresh is not None: params["seal_det_thresh"] = seal_det_thresh
    if seal_det_box_thresh is not None: params["seal_det_box_thresh"] = seal_det_box_thresh
    if seal_det_unclip_ratio is not None: params["seal_det_unclip_ratio"] = seal_det_unclip_ratio
    
    return params

def _get_or_create_pipeline(app: FastAPI, effective: Dict[str, Any]) -> PPStructureV3:
    # If no overrides, use default pipeline
    if not effective:
        return app.state.pipeline
    
    # Check cache
    cache: OrderedDict = app.state.pipeline_cache
    eff_key = _build_config_key(effective)
    
    if eff_key in cache:
        pipe = cache.pop(eff_key)
        cache[eff_key] = pipe
        return pipe
    
    # Evict if needed
    while len(cache) >= PIPELINE_CACHE_SIZE:
        cache.popitem(last=False)
    
    # Create new pipeline with overrides applied to defaults
    base_params = {
        "device": DEVICE,
        "enable_mkldnn": ENABLE_MKLDNN,
        "enable_hpi": ENABLE_HPI,
        "use_tensorrt": USE_TENSORRT,
        "precision": PRECISION,
        "mkldnn_cache_capacity": MKLDNN_CACHE_CAPACITY,
        "cpu_threads": CPU_THREADS,
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,
        "wired_table_structure_recognition_model_name": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "wireless_table_structure_recognition_model_name": WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "table_classification_model_name": TABLE_CLASSIFICATION_MODEL_NAME,
        "formula_recognition_model_name": FORMULA_RECOGNITION_MODEL_NAME,
        "chart_recognition_model_name": CHART_RECOGNITION_MODEL_NAME,
        "layout_threshold": LAYOUT_THRESHOLD,
        "text_det_thresh": TEXT_DET_THRESH,
        "text_det_box_thresh": TEXT_DET_BOX_THRESH,
        "text_det_unclip_ratio": TEXT_DET_UNCLIP_RATIO,
        "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
        "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
        "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
        "text_recognition_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
        "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": USE_DOC_UNWARPING,
        "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        "use_table_recognition": USE_TABLE_RECOGNITION,
        "use_formula_recognition": USE_FORMULA_RECOGNITION,
        "use_chart_recognition": USE_CHART_RECOGNITION,
        "use_seal_recognition": USE_SEAL_RECOGNITION,
        "use_region_detection": USE_REGION_DETECTION,
    }
    
    # Apply overrides
    base_params.update(effective)
    
    # Remove None values
    final_params = {k: v for k, v in base_params.items() if v is not None}
    
    pipe = _make_pipeline(**final_params)
    cache[eff_key] = pipe
    return pipe

@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="Image or PDF file"),
    output_format: Literal["json", "markdown", "both"] = Query("json", description="Output format"),
    
    # Device and backend
    device: Optional[str] = Query(None, description="Device: cpu, gpu, gpu:0, etc."),
    enable_mkldnn: Optional[bool] = Query(None, description="Enable MKL-DNN acceleration"),
    enable_hpi: Optional[bool] = Query(None, description="Enable HPI acceleration"),
    use_tensorrt: Optional[bool] = Query(None, description="Enable TensorRT acceleration"),
    precision: Optional[str] = Query(None, description="Precision: fp32, fp16"),
    mkldnn_cache_capacity: Optional[int] = Query(None, description="MKL-DNN cache capacity"),
    cpu_threads: Optional[int] = Query(None, description="CPU threads for inference"),
    paddlex_config: Optional[str] = Query(None, description="PaddleX config file path"),
    
    # Module toggles
    use_doc_orientation_classify: Optional[bool] = Query(None, description="Enable document orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(None, description="Enable document unwarping"),
    use_textline_orientation: Optional[bool] = Query(None, description="Enable textline orientation classification"),
    use_table_recognition: Optional[bool] = Query(None, description="Enable table recognition"),
    use_formula_recognition: Optional[bool] = Query(None, description="Enable formula recognition"),
    use_chart_recognition: Optional[bool] = Query(None, description="Enable chart recognition"),
    use_seal_recognition: Optional[bool] = Query(None, description="Enable seal text recognition"),
    use_region_detection: Optional[bool] = Query(None, description="Enable region detection"),
    
    # Model names
    layout_detection_model_name: Optional[str] = Query(None, description="Layout detection model"),
    region_detection_model_name: Optional[str] = Query(None, description="Region detection model"),
    text_detection_model_name: Optional[str] = Query(None, description="Text detection model"),
    text_recognition_model_name: Optional[str] = Query(None, description="Text recognition model"),
    table_classification_model_name: Optional[str] = Query(None, description="Table classification model"),
    wired_table_structure_recognition_model_name: Optional[str] = Query(None, description="Wired table structure model"),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(None, description="Wireless table structure model"),
    wired_table_cells_detection_model_name: Optional[str] = Query(None, description="Wired table cell detection model"),
    wireless_table_cells_detection_model_name: Optional[str] = Query(None, description="Wireless table cell detection model"),
    table_orientation_classify_model_name: Optional[str] = Query(None, description="Table orientation classification model"),
    formula_recognition_model_name: Optional[str] = Query(None, description="Formula recognition model"),
    chart_recognition_model_name: Optional[str] = Query(None, description="Chart recognition model"),
    doc_orientation_classify_model_name: Optional[str] = Query(None, description="Document orientation classification model"),
    doc_unwarping_model_name: Optional[str] = Query(None, description="Document unwarping model"),
    textline_orientation_model_name: Optional[str] = Query(None, description="Textline orientation model"),
    seal_text_detection_model_name: Optional[str] = Query(None, description="Seal text detection model"),
    seal_text_recognition_model_name: Optional[str] = Query(None, description="Seal text recognition model"),
    
    # Model directories
    layout_detection_model_dir: Optional[str] = Query(None, description="Layout detection model directory"),
    region_detection_model_dir: Optional[str] = Query(None, description="Region detection model directory"),
    text_detection_model_dir: Optional[str] = Query(None, description="Text detection model directory"),
    text_recognition_model_dir: Optional[str] = Query(None, description="Text recognition model directory"),
    table_classification_model_dir: Optional[str] = Query(None, description="Table classification model directory"),
    wired_table_structure_recognition_model_dir: Optional[str] = Query(None, description="Wired table structure model directory"),
    wireless_table_structure_recognition_model_dir: Optional[str] = Query(None, description="Wireless table structure model directory"),
    wired_table_cells_detection_model_dir: Optional[str] = Query(None, description="Wired table cell detection model directory"),
    wireless_table_cells_detection_model_dir: Optional[str] = Query(None, description="Wireless table cell detection model directory"),
    table_orientation_classify_model_dir: Optional[str] = Query(None, description="Table orientation classification model directory"),
    formula_recognition_model_dir: Optional[str] = Query(None, description="Formula recognition model directory"),
    chart_recognition_model_dir: Optional[str] = Query(None, description="Chart recognition model directory"),
    doc_orientation_classify_model_dir: Optional[str] = Query(None, description="Document orientation classification model directory"),
    doc_unwarping_model_dir: Optional[str] = Query(None, description="Document unwarping model directory"),
    textline_orientation_model_dir: Optional[str] = Query(None, description="Textline orientation model directory"),
    seal_text_detection_model_dir: Optional[str] = Query(None, description="Seal text detection model directory"),
    seal_text_recognition_model_dir: Optional[str] = Query(None, description="Seal text recognition model directory"),
    
    # Detection/Recognition parameters
    layout_threshold: Optional[float] = Query(None, description="Layout detection confidence threshold", ge=0.0, le=1.0),
    layout_nms: Optional[bool] = Query(None, description="Enable layout NMS"),
    layout_unclip_ratio: Optional[float] = Query(None, description="Layout detection unclip ratio", gt=0.0),
    layout_merge_bboxes_mode: Optional[str] = Query(None, description="Layout bbox merging mode"),
    text_det_limit_side_len: Optional[int] = Query(None, description="Text detection image side length limit", gt=0),
    text_det_limit_type: Optional[str] = Query(None, description="Text detection limit type: max or min"),
    text_det_thresh: Optional[float] = Query(None, description="Text detection threshold", ge=0.0, le=1.0),
    text_det_box_thresh: Optional[float] = Query(None, description="Text detection box threshold", ge=0.0, le=1.0),
    text_det_unclip_ratio: Optional[float] = Query(None, description="Text detection unclip ratio", gt=0.0),
    text_rec_score_thresh: Optional[float] = Query(None, description="Text recognition confidence threshold", ge=0.0, le=1.0),
    text_recognition_batch_size: Optional[int] = Query(None, description="Text recognition batch size", gt=0),
    textline_orientation_batch_size: Optional[int] = Query(None, description="Textline orientation batch size", gt=0),
    formula_recognition_batch_size: Optional[int] = Query(None, description="Formula recognition batch size", gt=0),
    chart_recognition_batch_size: Optional[int] = Query(None, description="Chart recognition batch size", gt=0),
    seal_text_recognition_batch_size: Optional[int] = Query(None, description="Seal text recognition batch size", gt=0),
    seal_rec_score_thresh: Optional[float] = Query(None, description="Seal recognition confidence threshold", ge=0.0, le=1.0),
    seal_det_limit_side_len: Optional[int] = Query(None, description="Seal detection image side length limit", gt=0),
    seal_det_limit_type: Optional[str] = Query(None, description="Seal detection limit type: max or min"),
    seal_det_thresh: Optional[float] = Query(None, description="Seal detection threshold", ge=0.0, le=1.0),
    seal_det_box_thresh: Optional[float] = Query(None, description="Seal detection box threshold", ge=0.0, le=1.0),
    seal_det_unclip_ratio: Optional[float] = Query(None, description="Seal detection unclip ratio", gt=0.0),
):
    """
    Parse document images or PDFs using PP-StructureV3 pipeline.
    
    Supports all PP-StructureV3 features:
    - Layout detection and analysis
    - OCR (text detection and recognition) 
    - Table recognition and structure analysis
    - Formula recognition (LaTeX output)
    - Chart recognition and parsing
    - Seal text recognition
    - Document preprocessing (orientation, unwarping, textline orientation)
    
    Returns JSON with structured results per page, and/or Markdown format for documents.
    """
    
    # Validate file
    if _file_too_big(file):
        raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB)")
    
    if not _ext_ok(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    # Save uploaded file
    tmp_dir = tempfile.mkdtemp(prefix="ppstructurev3_")
    tmp_path = os.path.join(tmp_dir, file.filename)
    
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Build parameter overrides
        effective = _effective_params_from_query(
            device, enable_mkldnn, enable_hpi, use_tensorrt, precision, mkldnn_cache_capacity, cpu_threads, paddlex_config,
            use_doc_orientation_classify, use_doc_unwarping, use_textline_orientation,
            use_table_recognition, use_formula_recognition, use_chart_recognition, use_seal_recognition, use_region_detection,
            layout_detection_model_name, layout_detection_model_dir, region_detection_model_name, region_detection_model_dir,
            text_detection_model_name, text_detection_model_dir, text_recognition_model_name, text_recognition_model_dir,
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
        
        # Get pipeline (default or custom)
        pipeline = _get_or_create_pipeline(app, effective)
        
        # Run inference with concurrency control
        acquired = app.state.predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy, try again later")
        
        try:
            outputs = await run_in_threadpool(lambda: pipeline.predict(input=tmp_path))
        finally:
            app.state.predict_sem.release()
        
        # Extract results using documented APIs
        page_json, merged_markdown = _collect_result_json_and_markdown(pipeline, outputs)
        
        # Return in requested format
        if output_format == "json":
            return JSONResponse({
                "results": page_json,
                "pages": len(page_json)
            })
        elif output_format == "markdown":
            return PlainTextResponse(merged_markdown or "")
        else:  # both
            return JSONResponse({
                "results": page_json,
                "markdown": merged_markdown or "",
                "pages": len(page_json)
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

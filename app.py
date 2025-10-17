import os
import io
import base64
import json
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple, List, Union

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from paddleocr import PPStructureV3


# ================= ARM64-aware defaults (tuned for medical lab reports) =================
DEVICE = "cpu"
CPU_THREADS = 4

# Subpipelines
USE_DOC_ORIENTATION_CLASSIFY = True      # robust to rotated/scanned pages
USE_DOC_UNWARPING = False                # off to save CPU; enable only for warped photos
USE_TEXTLINE_ORIENTATION = True          # improves rotated headers/side labels
USE_TABLE_RECOGNITION = True             # lab reports are table-heavy
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False
USE_REGION_DETECTION = False             # OFF by default for stability on ARM64 CPU

# Core models (balanced for CPU + English medical reports)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# Optional table models (leave empty to let pipeline choose)
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = ""
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = ""
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
WIRED_TABLE_CELLS_DET_MODEL_NAME = ""
WIRELESS_TABLE_CELLS_DET_MODEL_NAME = ""
TABLE_ORIENTATION_CLASSIFY_MODEL_NAME = ""

# Optional model dirs ("" means not provided; prevents forwarding None)
LAYOUT_DETECTION_MODEL_DIR = ""
REGION_DETECTION_MODEL_DIR = ""
TEXT_DETECTION_MODEL_DIR = ""
TEXT_RECOGNITION_MODEL_DIR = ""
TABLE_CLASSIFICATION_MODEL_DIR = ""
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = ""
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = ""
WIRED_TABLE_CELLS_DET_MODEL_DIR = ""
WIRELESS_TABLE_CELLS_DET_MODEL_DIR = ""
TABLE_ORIENTATION_CLASSIFY_MODEL_DIR = ""
FORMULA_RECOGNITION_MODEL_DIR = ""
DOC_ORIENTATION_CLASSIFY_MODEL_DIR = ""
DOC_UNWARPING_MODEL_DIR = ""
TEXTLINE_ORIENTATION_MODEL_DIR = ""
SEAL_TEXT_DETECTION_MODEL_DIR = ""
SEAL_TEXT_RECOGNITION_MODEL_DIR = ""
CHART_RECOGNITION_MODEL_DIR = ""

# Layout thresholds (favor defaults unless you need class-wise tuning)
LAYOUT_THRESHOLD = -1.0
LAYOUT_NMS = False
LAYOUT_UNCLIP_RATIO = -1.0
LAYOUT_MERGE_BBOXES_MODE = ""            # '', 'all', 'text_and_table'

# Text detection tuning for small fonts typical in lab reports
TEXT_DET_LIMIT_SIDE_LEN = 1536           # larger value helps tiny text on A4 scans
TEXT_DET_LIMIT_TYPE = ""                 # use pipeline default
TEXT_DET_DB_THRESH = 0.30
TEXT_DET_DB_BOX_THRESH = 0.40
TEXT_DET_DB_UNCLIP_RATIO = 2.0
TEXT_DET_DB_SCORE_MODE = ""              # 'fast'/'slow' if supported; keep default

# Recognition filter and batches
TEXT_REC_SCORE_THRESH = 0.60
TEXT_RECOGNITION_BATCH_SIZE = 8
TEXTLINE_ORIENTATION_BATCH_SIZE = 8
FORMULA_RECOGNITION_BATCH_SIZE = 0       # 0/"" means don't send
CHART_RECOGNITION_BATCH_SIZE = 0
SEAL_TEXT_RECOGNITION_BATCH_SIZE = 0

# Backend knobs (ARM64 optimized)
ENABLE_HPI = False                       # HPI typically unavailable on ARM64 CPU
ENABLE_MKLDNN = False                    # DNNL tends to benefit x86_64 more
MKLDNN_CACHE_CAPACITY = 5
USE_TENSORRT = False                     # N/A on CPU
PRECISION = "fp32"                       # retain accuracy for medical text
PADDLEX_CONFIG = ""                      # reserved; keep empty to avoid forwarding

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
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

def _page_json_with_fallback(res) -> Dict[str, Any]:
    j = getattr(res, "json", None)
    if isinstance(j, dict):
        return j
    with tempfile.TemporaryDirectory() as td:
        try:
            res.save_to_json(save_path=td)
            files = sorted(Path(td).glob("*.json"))
            if files:
                return json.loads(files[-1].read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _open_image_any(val: Union[Image.Image, bytes, str]) -> Optional[Image.Image]:
    try:
        if isinstance(val, Image.Image):
            return val
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val)).convert("RGBA")
        if isinstance(val, str) and os.path.exists(val):
            return Image.open(val).convert("RGBA")
    except (UnidentifiedImageError, OSError):
        return None
    return None

def _embed_images_in_markdown(md_text: str, images_map: Dict[str, Union[Image.Image, bytes, str]]) -> str:
    for path, val in images_map.items():
        pil_img = _open_image_any(val)
        if pil_img is None:
            continue
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        md_text = md_text.replace(f'src="{path}"', f'src="{data_uri}"')
        md_text = md_text.replace(f"({path})", f"({data_uri})")
    return md_text

def _collect_results(pipeline: PPStructureV3, outputs, inline_images: bool):
    page_json: List[Dict[str, Any]] = []
    markdown_list: List[Dict[str, Any]] = []
    markdown_images_list: List[Dict[str, Union[Image.Image, bytes, str]]] = []

    for res in outputs:
        page_json.append(_page_json_with_fallback(res))
        md_dict = getattr(res, "markdown", None)
        if md_dict and isinstance(md_dict, dict):
            markdown_list.append(md_dict)
            imgs = md_dict.get("markdown_images", {})
            markdown_images_list.append(imgs if isinstance(imgs, dict) else {})
        else:
            markdown_list.append({})
            markdown_images_list.append({})

    merged_md = ""
    if markdown_list:
        try:
            merged_md = pipeline.concatenate_markdown_pages(markdown_list)
        except AttributeError:
            # Some builds expose the helper under paddlex_pipeline
            try:
                paddlex = getattr(pipeline, "paddlex_pipeline", None)
                if paddlex and hasattr(paddlex, "concatenate_markdown_pages"):
                    merged_md = paddlex.concatenate_markdown_pages(markdown_list)
            except Exception:
                merged_md = ""
        except Exception:
            merged_md = ""

    if inline_images and merged_md:
        merged_images: Dict[str, Union[Image.Image, bytes, str]] = {}
        for imgs in markdown_images_list:
            merged_images.update(imgs or {})
        if merged_images:
            merged_md = _embed_images_in_markdown(merged_md, merged_images)

    return page_json, merged_md

def _warmup_pipeline(pipeline: PPStructureV3) -> None:
    # Warm up with a real PNG file path (more stable than ndarray on some stacks)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        Image.new("RGB", (64, 64), color="white").save(tf.name)
        path = tf.name
    try:
        pipeline.predict(input=path)
    except Exception:
        pass
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ================= Signature-aware filtering/mapping =================
def _provided(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)) and v in (-1, -1.0, 0):
        return False
    if isinstance(v, str) and v == "":
        return False
    return v is not None

def _filter_by_signature(params: Dict[str, Any], fn) -> Dict[str, Any]:
    import inspect
    allowed = set(inspect.signature(fn).parameters.keys())
    return {k: v for k, v in params.items() if (k in allowed and _provided(v))}

def _build_init_kwargs() -> Dict[str, Any]:
    """
    Build PPStructureV3.__init__ kwargs strictly from the installed signature.
    Includes version-aware name mapping for chart and text_det DB params.
    """
    # Prepare candidates with commonly used/official keys
    cand: Dict[str, Any] = {
        # Core device/lang/threads/backend
        "device": DEVICE,
        "lang": "en",
        "cpu_threads": CPU_THREADS,
        "enable_mkldnn": ENABLE_MKLDNN,
        "mkldnn_cache_capacity": MKLDNN_CACHE_CAPACITY,
        "enable_hpi": ENABLE_HPI,

        # Toggles
        "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": USE_DOC_UNWARPING,
        "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        "use_table_recognition": USE_TABLE_RECOGNITION,
        "use_formula_recognition": USE_FORMULA_RECOGNITION,
        "use_chart_recognition": USE_CHART_RECOGNITION,     # mapped below if variant differs
        "use_seal_recognition": USE_SEAL_RECOGNITION,
        "use_region_detection": USE_REGION_DETECTION,

        # Models
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,

        # Optional dirs
        "layout_detection_model_dir": LAYOUT_DETECTION_MODEL_DIR,
        "region_detection_model_dir": REGION_DETECTION_MODEL_DIR,
        "text_detection_model_dir": TEXT_DETECTION_MODEL_DIR,
        "text_recognition_model_dir": TEXT_RECOGNITION_MODEL_DIR,

        # Table models
        "table_classifier_model_name": TABLE_CLASSIFICATION_MODEL_NAME,
        "table_classifier_model_dir": TABLE_CLASSIFICATION_MODEL_DIR,
        "wired_table_structure_recognition_model_name": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "wireless_table_structure_recognition_model_name": WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "wired_table_cells_detection_model_name": WIRED_TABLE_CELLS_DET_MODEL_NAME,
        "wireless_table_cells_detection_model_name": WIRELESS_TABLE_CELLS_DET_MODEL_NAME,
        "wired_table_structure_recognition_model_dir": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        "wireless_table_structure_recognition_model_dir": WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        "wired_table_cells_detection_model_dir": WIRED_TABLE_CELLS_DET_MODEL_DIR,
        "wireless_table_cells_detection_model_dir": WIRELESS_TABLE_CELLS_DET_MODEL_DIR,
        "table_orientation_classify_model_name": TABLE_ORIENTATION_CLASSIFY_MODEL_NAME,
        "table_orientation_classify_model_dir": TABLE_ORIENTATION_CLASSIFY_MODEL_DIR,

        # Preprocess dirs
        "doc_orientation_classify_model_dir": DOC_ORIENTATION_CLASSIFY_MODEL_DIR,
        "doc_unwarping_model_dir": DOC_UNWARPING_MODEL_DIR,
        "textline_orientation_model_dir": TEXTLINE_ORIENTATION_MODEL_DIR,

        # Seal/Chart models (mapped below as needed)
        "seal_text_detection_model_name": "",
        "seal_text_recognition_model_name": "",
        "seal_text_detection_model_dir": SEAL_TEXT_DETECTION_MODEL_DIR,
        "seal_text_recognition_model_dir": SEAL_TEXT_RECOGNITION_MODEL_DIR,
        "chart_recognition_model_name": "",
        "chart_recognition_model_dir": CHART_RECOGNITION_MODEL_DIR,

        # Layout thresholds
        "layout_threshold": LAYOUT_THRESHOLD,
        "layout_nms": LAYOUT_NMS,
        "layout_unclip_ratio": LAYOUT_UNCLIP_RATIO,
        "layout_merge_bboxes_mode": LAYOUT_MERGE_BBOXES_MODE,

        # Text detection (DB params)
        "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
        "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
        "text_det_db_thresh": TEXT_DET_DB_THRESH,
        "text_det_db_box_thresh": TEXT_DET_DB_BOX_THRESH,
        "text_det_db_unclip_ratio": TEXT_DET_DB_UNCLIP_RATIO,
        "text_det_db_score_mode": TEXT_DET_DB_SCORE_MODE,

        # Recognition filters/batching
        "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
        "text_rec_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
        "textline_orientation_batch_size": TEXTLINE_ORIENTATION_BATCH_SIZE,
        "formula_recognition_batch_size": FORMULA_RECOGNITION_BATCH_SIZE,
        "chart_recognition_batch_size": CHART_RECOGNITION_BATCH_SIZE,
        "seal_text_recognition_batch_size": SEAL_TEXT_RECOGNITION_BATCH_SIZE,
    }

    # Version-aware mapping: some releases expose 'use_chart_parsing' and 'chart_parsing_model_*'
    # If those exist, map values and remove the *_recognition equivalents during filtering.
    try:
        import inspect
        init_params = set(inspect.signature(PPStructureV3.__init__).parameters.keys())

        if "use_chart_parsing" in init_params and "use_chart_recognition" not in init_params:
            cand["use_chart_parsing"] = cand.pop("use_chart_recognition")
        if "chart_parsing_model_name" in init_params and "chart_recognition_model_name" not in init_params:
            cand["chart_parsing_model_name"] = cand.pop("chart_recognition_model_name")
        if "chart_parsing_model_dir" in init_params and "chart_recognition_model_dir" not in init_params:
            cand["chart_parsing_model_dir"] = cand.pop("chart_recognition_model_dir")
    except Exception:
        pass

    # Filter by actual signature and drop empty/"not provided" sentinel values
    return _filter_by_signature({k: v for k, v in cand.items() if _provided(v)}, PPStructureV3.__init__)

def _build_predict_kwargs(
    save_path: Optional[str],
    page_num: Optional[int],
    table_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    # Merge table predict-time knobs (only when supported) and core predict args
    cand = dict(table_kwargs)
    if save_path:
        cand["save_path"] = save_path
    if page_num is not None and page_num > 0:
        cand["page_num"] = page_num
    return _filter_by_signature(cand, PPStructureV3.predict)


# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build init kwargs strictly by signature to avoid leaking unknown keys
    init_kwargs = _build_init_kwargs()
    app.state.pipeline = PPStructureV3(**init_kwargs)

    # Warmup triggers model downloads and initializes graphs
    _warmup_pipeline(app.state.pipeline)

    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    app.state.pipeline_cache = OrderedDict()
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="2.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _build_config_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    # Stable cache key for variant pipelines (only construction-affecting keys)
    return tuple(sorted((k, v) for k, v in params.items()))


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

    # Start from our tuned defaults, then merge user overrides
    base_defaults = _build_init_kwargs()
    base_defaults.update(_filter_by_signature({k: v for k, v in effective.items() if _provided(v)}, PPStructureV3.__init__))

    pipe = PPStructureV3(**base_defaults)
    try:
        _warmup_pipeline(pipe)
    except Exception:
        pass

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
    # Only include provided values; filtering against predict() happens later
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


# Default to "both" so you get JSON + concatenated Markdown out of the box
@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown", "both"] = Query("both"),
    markdown_images: Literal["none", "inline"] = Query("none", description="Inline images as base64"),

    # Backend/runtime overrides
    device: Optional[str] = Query(None),
    enable_mkldnn: Optional[bool] = Query(None),
    enable_hpi: Optional[bool] = Query(None),
    mkldnn_cache_capacity: Optional[int] = Query(None),
    cpu_threads: Optional[int] = Query(None),

    # Toggles
    use_doc_orientation_classify: Optional[bool] = Query(None),
    use_doc_unwarping: Optional[bool] = Query(None),
    use_textline_orientation: Optional[bool] = Query(None),
    use_table_recognition: Optional[bool] = Query(None),
    use_formula_recognition: Optional[bool] = Query(None),
    use_chart_recognition: Optional[bool] = Query(None),
    use_seal_recognition: Optional[bool] = Query(None),
    use_region_detection: Optional[bool] = Query(None),

    # Models
    layout_detection_model_name: Optional[str] = Query(None),
    text_detection_model_name: Optional[str] = Query(None),
    text_recognition_model_name: Optional[str] = Query(None),
    table_classification_model_name: Optional[str] = Query(None),
    wired_table_structure_recognition_model_name: Optional[str] = Query(None),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(None),
    wired_table_cells_detection_model_name: Optional[str] = Query(None),
    wireless_table_cells_detection_model_name: Optional[str] = Query(None),
    table_orientation_classify_model_name: Optional[str] = Query(None),

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

    # Layout thresholds
    layout_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    layout_nms: Optional[bool] = Query(None),
    layout_unclip_ratio: Optional[float] = Query(None, gt=0.0),
    layout_merge_bboxes_mode: Optional[str] = Query(None),

    # Text detection tuning (DB params)
    text_det_limit_side_len: Optional[int] = Query(None, gt=0),
    text_det_limit_type: Optional[str] = Query(None),
    text_det_db_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_db_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_db_unclip_ratio: Optional[float] = Query(None, gt=0.0),
    text_det_db_score_mode: Optional[str] = Query(None),

    # Recognition
    text_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_recognition_batch_size: Optional[int] = Query(None, gt=0),
    textline_orientation_batch_size: Optional[int] = Query(None, gt=0),

    # Predict-time table behavior
    use_ocr_results_with_table_cells: Optional[bool] = Query(None),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None),
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None),
    use_table_orientation_classify: Optional[bool] = Query(None),

    # Page control
    page_num: Optional[int] = Query(None, gt=0),
):
    if _file_too_big(file):
        raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB)")
    if not _ext_ok(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type; allowed: {sorted(ALLOWED_EXTENSIONS)}")

    tmp_dir = tempfile.mkdtemp(prefix="ppsv3_")
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Build constructor overrides (strictly by signature)
        effective: Dict[str, Any] = {}
        for k, v in dict(
            device=device, enable_mkldnn=enable_mkldnn, enable_hpi=enable_hpi,
            mkldnn_cache_capacity=mkldnn_cache_capacity, cpu_threads=cpu_threads,

            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_seal_recognition=use_seal_recognition,
            use_region_detection=use_region_detection,

            layout_detection_model_name=layout_detection_model_name,
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,
            table_classification_model_name=table_classification_model_name,
            wired_table_structure_recognition_model_name=wired_table_structure_recognition_model_name,
            wireless_table_structure_recognition_model_name=wireless_table_structure_recognition_model_name,
            wired_table_cells_detection_model_name=wired_table_cells_detection_model_name,
            wireless_table_cells_detection_model_name=wireless_table_cells_detection_model_name,
            table_orientation_classify_model_name=table_orientation_classify_model_name,

            layout_detection_model_dir=layout_detection_model_dir,
            region_detection_model_dir=region_detection_model_dir,
            text_detection_model_dir=text_detection_model_dir,
            text_recognition_model_dir=text_recognition_model_dir,
            table_classification_model_dir=table_classification_model_dir,
            wired_table_structure_recognition_model_dir=wired_table_structure_recognition_model_dir,
            wireless_table_structure_recognition_model_dir=wireless_table_structure_recognition_model_dir,
            wired_table_cells_detection_model_dir=wired_table_cells_detection_model_dir,
            wireless_table_cells_detection_model_dir=wireless_table_cells_detection_model_dir,
            table_orientation_classify_model_dir=table_orientation_classify_model_dir,

            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,

            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_db_thresh=text_det_db_thresh,
            text_det_db_box_thresh=text_det_db_box_thresh,
            text_det_db_unclip_ratio=text_det_db_unclip_ratio,
            text_det_db_score_mode=text_det_db_score_mode,

            text_rec_score_thresh=text_rec_score_thresh,
            text_rec_batch_size=text_recognition_batch_size,
            textline_orientation_batch_size=textline_orientation_batch_size,
        ).items():
            if v is not None:
                effective[k] = v

        # Filter overrides by signature and merge with tuned defaults
        pipeline = _get_or_create_pipeline(app, effective)

        table_kwargs = _predict_table_kwargs(
            use_ocr_results_with_table_cells,
            use_e2e_wired_table_rec_model,
            use_e2e_wireless_table_rec_model,
            use_wired_table_cells_trans_to_html,
            use_wireless_table_cells_trans_to_html,
            use_table_orientation_classify,
        )
        predict_kwargs = _build_predict_kwargs(save_path=None, page_num=page_num, table_kwargs=table_kwargs)

        acquired = app.state.predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy")
        try:
            outputs = await run_in_threadpool(lambda: pipeline.predict(input=tmp_path, **predict_kwargs))
        finally:
            app.state.predict_sem.release()

        inline_flag = markdown_images == "inline"
        page_json, merged_md = _collect_results(pipeline, outputs, inline_flag)

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

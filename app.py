import os
import io
import json
import shutil
import tempfile
import inspect

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from paddleocr import PPStructureV3
import numpy as np

# =========================
# Global defaults (uppercase constants, no typing annotations)
# Optimized for medical lab reports (English), ARM64 CPU.
# =========================

# Device and threads
DEVICE = "cpu"
CPU_THREADS = 4
LANG = "en"

# Subpipelines (explicit on/off)
USE_DOC_ORIENTATION_CLASSIFY = True     # scans may be rotated
USE_DOC_UNWARPING = False               # off to save CPU for flat scans
USE_TEXTLINE_ORIENTATION = True         # helps with rotated headers/side labels
USE_TABLE_RECOGNITION = True            # lab reports are table-heavy
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False
USE_REGION_DETECTION = True             # if supported by the current version

# Models
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# Table models (keep generic; library selects defaults if left empty)
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
WIRED_TABLE_CELLS_DET_MODEL_NAME = ""
WIRELESS_TABLE_CELLS_DET_MODEL_NAME = ""
TABLE_ORIENTATION_CLASSIFY_MODEL_NAME = ""

# Optional specialized models
FORMULA_RECOGNITION_MODEL_NAME = ""     # keep off unless needed
SEAL_TEXT_DETECTION_MODEL_NAME = ""
SEAL_TEXT_RECOGNITION_MODEL_NAME = ""
CHART_RECOGNITION_MODEL_NAME = ""

# Optional model dirs ("" means not provided)
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

# Thresholds / sizes / batches (favor recall; filter low-confidence rec)
LAYOUT_THRESHOLD = -1.0                 # keep native default
LAYOUT_NMS = False
LAYOUT_UNCLIP_RATIO = -1.0
LAYOUT_MERGE_BBOXES_MODE = ""           # '', 'all', 'text_and_table'

# Text detection tuning for small fonts in lab reports
TEXT_DET_THRESH = 0.30
TEXT_DET_BOX_THRESH = 0.40
TEXT_DET_UNCLIP_RATIO = 2.0
TEXT_DET_LIMIT_SIDE_LEN = 1536          # higher limit to capture fine text
TEXT_DET_LIMIT_TYPE = ""                # library default (e.g., 'max'/'min')
TEXT_DET_DB_SCORE_MODE = ""             # 'fast'/'slow' if supported

# Seals (kept off; values present for completeness)
SEAL_DET_LIMIT_SIDE_LEN = -1
SEAL_DET_LIMIT_TYPE = ""
SEAL_DET_THRESH = -1.0
SEAL_DET_BOX_THRESH = -1.0
SEAL_DET_UNCLIP_RATIO = -1.0
SEAL_REC_SCORE_THRESH = -1.0

# Recognition/batching
TEXT_REC_SCORE_THRESH = 0.60            # filter low-confidence text
TEXT_RECOGNITION_BATCH_SIZE = 8         # modest CPU batch for throughput
TEXTLINE_ORIENTATION_BATCH_SIZE = 8
FORMULA_RECOGNITION_BATCH_SIZE = 4
CHART_RECOGNITION_BATCH_SIZE = 2
SEAL_TEXT_RECOGNITION_BATCH_SIZE = 4

# Backend knobs (ARM64 optimized)
ENABLE_HPI = False                      # not supported on ARM64
ENABLE_MKLDNN = False                   # MKLDNN mainly benefits x86_64
MKLDNN_CACHE_CAPACITY = 5
USE_TENSORRT = False
PRECISION = "fp32"                      # preserve accuracy for medical text
PADDLEX_CONFIG = ""                     # reserved; not used

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1                # single-threaded request processing
PIPELINE_CACHE_SIZE = 2

# Output behavior
RETURN_JSON_DEFAULT = True
RETURN_MARKDOWN_DEFAULT = True
CONCATENATE_MARKDOWN_DEFAULT = True
DEFAULT_SAVE_DIR = ""                   # "" = do not persist to disk

# PDF/page control
PAGE_NUM = -1                           # -1 = all pages

# =========================
# Utilities
# =========================

def _provided(v):
    # Treat '', -1, -1.0 as "not provided"; booleans are always provided
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)) and (v == -1 or v == -1.0):
        return False
    if isinstance(v, str) and v == "":
        return False
    return v is not None

def _lower_snake(name):
    return name.lower()

def _build_init_kwargs(cfg):
    # Map our uppercase config into PPStructureV3.__init__ kwargs where supported
    sig = inspect.signature(PPStructureV3.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "args", "kwargs"}

    # Candidate map (keys already aligned to PPStructureV3 as much as possible)
    cand = {
        # Device/lang/CPU
        "device": cfg["DEVICE"],
        "lang": LANG,
        "cpu_threads": cfg["CPU_THREADS"],
        "enable_mkldnn": ENABLE_MKLDNN,
        "mkldnn_cache_capacity": MKLDNN_CACHE_CAPACITY,

        # Optional components
        "use_doc_orientation_classify": cfg["USE_DOC_ORIENTATION_CLASSIFY"],
        "use_doc_unwarping": cfg["USE_DOC_UNWARPING"],
        "use_textline_orientation": cfg["USE_TEXTLINE_ORIENTATION"],
        "use_table_recognition": cfg["USE_TABLE_RECOGNITION"],
        "use_formula_recognition": cfg["USE_FORMULA_RECOGNITION"],
        "use_chart_parsing": cfg["USE_CHART_RECOGNITION"],
        "use_seal_recognition": cfg["USE_SEAL_RECOGNITION"],
        "use_region_detection": cfg["USE_REGION_DETECTION"],

        # Layout
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "layout_detection_model_dir": LAYOUT_DETECTION_MODEL_DIR,
        "layout_threshold": LAYOUT_THRESHOLD,
        "layout_nms": LAYOUT_NMS,
        "layout_unclip_ratio": LAYOUT_UNCLIP_RATIO,
        "layout_merge_bboxes_mode": LAYOUT_MERGE_BBOXES_MODE,

        # Text detection (DB)
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_detection_model_dir": TEXT_DETECTION_MODEL_DIR,
        "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
        "text_det_db_thresh": TEXT_DET_THRESH,
        "text_det_db_box_thresh": TEXT_DET_BOX_THRESH,
        "text_det_db_unclip_ratio": TEXT_DET_UNCLIP_RATIO,
        "text_det_db_score_mode": TEXT_DET_DB_SCORE_MODE,
        "text_det_limit_type": TEXT_DET_LIMIT_TYPE,

        # Text recognition
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,
        "text_recognition_model_dir": TEXT_RECOGNITION_MODEL_DIR,
        "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
        "text_rec_batch_size": TEXT_RECOGNITION_BATCH_SIZE,

        # Textline orientation batch (if exposed)
        "textline_orientation_batch_size": TEXTLINE_ORIENTATION_BATCH_SIZE,

        # Table models (pass only if supported)
        "table_structure_model_name": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "table_structure_model_dir": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        "table_classifier_model_name": TABLE_CLASSIFICATION_MODEL_NAME,
        "table_classifier_model_dir": TABLE_CLASSIFICATION_MODEL_DIR,
        "table_cell_det_model_name": WIRED_TABLE_CELLS_DET_MODEL_NAME,
        "table_cell_det_model_dir": WIRED_TABLE_CELLS_DET_MODEL_DIR,
        "table_orientation_classify_model_name": TABLE_ORIENTATION_CLASSIFY_MODEL_NAME,
        "table_orientation_classify_model_dir": TABLE_ORIENTATION_CLASSIFY_MODEL_DIR,

        # Formula / Seal / Chart
        "formula_recognition_model_name": FORMULA_RECOGNITION_MODEL_NAME,
        "formula_recognition_model_dir": FORMULA_RECOGNITION_MODEL_DIR,
        "formula_recognition_batch_size": FORMULA_RECOGNITION_BATCH_SIZE,

        "seal_text_detection_model_name": SEAL_TEXT_DETECTION_MODEL_NAME,
        "seal_text_detection_model_dir": SEAL_TEXT_DETECTION_MODEL_DIR,
        "seal_text_recognition_model_name": SEAL_TEXT_RECOGNITION_MODEL_NAME,
        "seal_text_recognition_model_dir": SEAL_TEXT_RECOGNITION_MODEL_DIR,
        "seal_text_recognition_batch_size": SEAL_TEXT_RECOGNITION_BATCH_SIZE,
        "seal_det_limit_side_len": SEAL_DET_LIMIT_SIDE_LEN,
        "seal_det_limit_type": SEAL_DET_LIMIT_TYPE,
        "seal_det_db_thresh": SEAL_DET_THRESH,
        "seal_det_db_box_thresh": SEAL_DET_BOX_THRESH,
        "seal_det_db_unclip_ratio": SEAL_DET_UNCLIP_RATIO,
        "seal_rec_score_thresh": SEAL_REC_SCORE_THRESH,

        "chart_parsing_model_name": CHART_RECOGNITION_MODEL_NAME,
        "chart_parsing_model_dir": CHART_RECOGNITION_MODEL_DIR,
        "chart_recognition_batch_size": CHART_RECOGNITION_BATCH_SIZE,

        # Optional model dirs for preprocessors (if supported)
        "doc_orientation_classify_model_dir": DOC_ORIENTATION_CLASSIFY_MODEL_DIR,
        "doc_unwarping_model_dir": DOC_UNWARPING_MODEL_DIR,
        "textline_orientation_model_dir": TEXTLINE_ORIENTATION_MODEL_DIR,

        # Backend knobs (ignored if not supported)
        "precision": PRECISION,
        "use_tensorrt": USE_TENSORRT,
        "enable_hpi": ENABLE_HPI,  # custom; ignored by library if unsupported
    }

    init_kwargs = {k: v for k, v in cand.items() if (k in allowed and _provided(v))}
    return init_kwargs

def _build_predict_kwargs(save_dir_for_native):
    sig = inspect.signature(PPStructureV3.predict)
    allowed = set(sig.parameters.keys()) - {"self"}
    cand = {
        "page_num": PAGE_NUM,
        "save_path": save_dir_for_native,
    }
    return {k: v for k, v in cand.items() if (k in allowed and _provided(v))}

def _validate_file(name, content):
    ext = os.path.splitext(name or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {ext}")
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB), limit is {MAX_FILE_SIZE_MB} MB")

# =========================
# App and pipeline cache
# =========================

app = FastAPI(
    title="PP-StructureV3 (CPU, ARM64) Service",
    version="1.4.0",
    description="PP-StructureV3 FastAPI with full param exposure, JSON/Markdown outputs, and PDF Markdown concatenation. Optimized for medical lab reports."
)

from functools import lru_cache

@lru_cache(maxsize=PIPELINE_CACHE_SIZE)
def _get_pipeline_cached(cache_key, init_kwargs_json):
    init_kwargs = json.loads(init_kwargs_json)
    return PPStructureV3(**init_kwargs)

def _cache_key_from_cfg(cfg):
    # Only include fields that affect pipeline construction
    return (
        cfg["DEVICE"], LANG, cfg["CPU_THREADS"], ENABLE_MKLDNN, MKLDNN_CACHE_CAPACITY,
        cfg["USE_DOC_ORIENTATION_CLASSIFY"], cfg["USE_DOC_UNWARPING"], cfg["USE_TEXTLINE_ORIENTATION"],
        cfg["USE_TABLE_RECOGNITION"], cfg["USE_FORMULA_RECOGNITION"], cfg["USE_CHART_RECOGNITION"],
        cfg["USE_SEAL_RECOGNITION"], cfg["USE_REGION_DETECTION"],
        LAYOUT_DETECTION_MODEL_NAME, LAYOUT_DETECTION_MODEL_DIR, LAYOUT_THRESHOLD, LAYOUT_NMS,
        LAYOUT_UNCLIP_RATIO, LAYOUT_MERGE_BBOXES_MODE,
        TEXT_DETECTION_MODEL_NAME, TEXT_DETECTION_MODEL_DIR, TEXT_DET_LIMIT_SIDE_LEN,
        TEXT_DET_THRESH, TEXT_DET_BOX_THRESH, TEXT_DET_UNCLIP_RATIO, TEXT_DET_DB_SCORE_MODE, TEXT_DET_LIMIT_TYPE,
        TEXT_RECOGNITION_MODEL_NAME, TEXT_RECOGNITION_MODEL_DIR, TEXT_REC_SCORE_THRESH, TEXT_RECOGNITION_BATCH_SIZE,
        WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME, WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        TABLE_CLASSIFICATION_MODEL_NAME, TABLE_CLASSIFICATION_MODEL_DIR,
        WIRED_TABLE_CELLS_DET_MODEL_NAME, WIRED_TABLE_CELLS_DET_MODEL_DIR,
        FORMULA_RECOGNITION_MODEL_NAME, FORMULA_RECOGNITION_MODEL_DIR, FORMULA_RECOGNITION_BATCH_SIZE,
        CHART_RECOGNITION_MODEL_NAME, CHART_RECOGNITION_MODEL_DIR, CHART_RECOGNITION_BATCH_SIZE,
        SEAL_TEXT_DETECTION_MODEL_NAME, SEAL_TEXT_DETECTION_MODEL_DIR,
        SEAL_TEXT_RECOGNITION_MODEL_NAME, SEAL_TEXT_RECOGNITION_MODEL_DIR, SEAL_TEXT_RECOGNITION_BATCH_SIZE,
    )

def _get_pipeline(cfg):
    init_kwargs = _build_init_kwargs(cfg)
    return _get_pipeline_cached(_cache_key_from_cfg(cfg), json.dumps(init_kwargs))

def _warmup(pipeline):
    # Trigger model downloads and graph initialization
    tiny = (255 * np.ones((64, 64, 3), dtype=np.uint8))
    try:
        pipeline.predict(input=tiny)
    except Exception:
        try:
            from PIL import Image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                Image.fromarray(tiny).save(tf.name)
                pipeline.predict(input=tf.name)
        except Exception:
            pass
        finally:
            try:
                os.remove(tf.name)
            except Exception:
                pass

@app.on_event("startup")
def _startup():
    cfg = {
        "DEVICE": DEVICE,
        "CPU_THREADS": CPU_THREADS,
        "USE_DOC_ORIENTATION_CLASSIFY": USE_DOC_ORIENTATION_CLASSIFY,
        "USE_DOC_UNWARPING": USE_DOC_UNWARPING,
        "USE_TEXTLINE_ORIENTATION": USE_TEXTLINE_ORIENTATION,
        "USE_TABLE_RECOGNITION": USE_TABLE_RECOGNITION,
        "USE_FORMULA_RECOGNITION": USE_FORMULA_RECOGNITION,
        "USE_CHART_RECOGNITION": USE_CHART_RECOGNITION,
        "USE_SEAL_RECOGNITION": USE_SEAL_RECOGNITION,
        "USE_REGION_DETECTION": USE_REGION_DETECTION,
    }
    pipe = _get_pipeline(cfg)
    _warmup(pipe)

# =========================
# Routes
# =========================

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def index():
    return {
        "name": "PP-StructureV3 (FastAPI)",
        "optimized_for": "Medical lab reports (English)",
        "defaults": {
            "device": DEVICE,
            "cpu_threads": CPU_THREADS,
            "layout_model": LAYOUT_DETECTION_MODEL_NAME,
            "det_model": TEXT_DETECTION_MODEL_NAME,
            "rec_model": TEXT_RECOGNITION_MODEL_NAME,
            "use_table_recognition": USE_TABLE_RECOGNITION,
            "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
            "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY
        },
        "endpoints": {"POST /parse": "Multipart: files=..., options=<JSON string> (optional)"}
    }

@app.post("/parse")
async def parse(
    files: list[UploadFile] = File(..., description="One or more images or PDFs"),
    options: str = Form(default="")
):
    # Defaults for this request
    ret_json = RETURN_JSON_DEFAULT
    ret_md = RETURN_MARKDOWN_DEFAULT
    concat_md = CONCATENATE_MARKDOWN_DEFAULT
    save_dir = DEFAULT_SAVE_DIR

    # Build a working cfg for this request (uppercase keys)
    cfg = {
        "DEVICE": DEVICE,
        "CPU_THREADS": CPU_THREADS,
        "USE_DOC_ORIENTATION_CLASSIFY": USE_DOC_ORIENTATION_CLASSIFY,
        "USE_DOC_UNWARPING": USE_DOC_UNWARPING,
        "USE_TEXTLINE_ORIENTATION": USE_TEXTLINE_ORIENTATION,
        "USE_TABLE_RECOGNITION": USE_TABLE_RECOGNITION,
        "USE_FORMULA_RECOGNITION": USE_FORMULA_RECOGNITION,
        "USE_CHART_RECOGNITION": USE_CHART_RECOGNITION,
        "USE_SEAL_RECOGNITION": USE_SEAL_RECOGNITION,
        "USE_REGION_DETECTION": USE_REGION_DETECTION,
    }

    # Merge overrides if provided
    if options:
        try:
            ov = json.loads(options)
        except Exception as e:
            raise ValueError(f"Invalid options JSON: {e}")
        # Toggle outputs
        if "return_json" in ov: ret_json = bool(ov["return_json"])
        if "return_markdown" in ov: ret_md = bool(ov["return_markdown"])
        if "concatenate_markdown" in ov: concat_md = bool(ov["concatenate_markdown"])
        if "save_dir" in ov: save_dir = str(ov["save_dir"])
        # Device/threads overrides
        if "device" in ov: cfg["DEVICE"] = str(ov["device"])
        if "cpu_threads" in ov: cfg["CPU_THREADS"] = int(ov["cpu_threads"])
        # Module toggles
        for k in [
            "USE_DOC_ORIENTATION_CLASSIFY","USE_DOC_UNWARPING","USE_TEXTLINE_ORIENTATION",
            "USE_TABLE_RECOGNITION","USE_FORMULA_RECOGNITION","USE_CHART_RECOGNITION",
            "USE_SEAL_RECOGNITION","USE_REGION_DETECTION"
        ]:
            if k in ov: cfg[k] = bool(ov[k])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    pipeline = _get_pipeline(cfg)

    overall = {
        "engine": "PP-StructureV3",
        "device": cfg["DEVICE"],
        "results": []
    }

    with tempfile.TemporaryDirectory(prefix="ppstructv3_req_") as req_tmp:
        for uf in files:
            original_name = uf.filename or "upload"
            content = await uf.read()
            _validate_file(original_name, content)

            base, ext = os.path.splitext(original_name)
            work_dir = tempfile.mkdtemp(prefix="file_", dir=req_tmp)
            in_path = os.path.join(work_dir, f"input{ext or ''}")
            with open(in_path, "wb") as f:
                f.write(content)

            native_dir = os.path.join(work_dir, "native")
            os.makedirs(native_dir, exist_ok=True)

            predict_kwargs = _build_predict_kwargs(native_dir)
            preds = pipeline.predict(input=in_path, **predict_kwargs)

            page_json = []
            page_markdown = []
            page_md_objs = []

            for res in preds:
                if ret_json:
                    try: res.save_to_json(save_path=native_dir)
                    except Exception: pass
                if ret_md:
                    try: res.save_to_markdown(save_path=native_dir)
                    except Exception: pass
                md_obj = getattr(res, "markdown", None)
                if isinstance(md_obj, dict):
                    page_md_objs.append(md_obj)

            if ret_json:
                for name in sorted(os.listdir(native_dir)):
                    if name.lower().endswith(".json"):
                        fp = os.path.join(native_dir, name)
                        try:
                            with open(fp, "r", encoding="utf-8") as jf:
                                page_json.append(json.load(jf))
                        except Exception:
                            with open(fp, "r", encoding="utf-8", errors="ignore") as jf:
                                page_json.append({"raw_json": jf.read()})

            if ret_md:
                for name in sorted(os.listdir(native_dir)):
                    if name.lower().endswith(".md"):
                        with open(os.path.join(native_dir, name), "r", encoding="utf-8", errors="ignore") as mf:
                            page_markdown.append(mf.read())

            markdown_combined = ""
            if ret_md and concat_md:
                combined_from_native = False
                try:
                    if hasattr(pipeline, "concatenate_markdown_pages") and page_md_objs:
                        merged = pipeline.concatenate_markdown_pages(page_md_objs)
                        if isinstance(merged, dict) and "markdown_texts" in merged:
                            markdown_combined = merged["markdown_texts"]
                            combined_from_native = True
                        elif isinstance(merged, str):
                            markdown_combined = merged
                            combined_from_native = True
                except Exception:
                    combined_from_native = False

                if not combined_from_native and page_markdown:
                    sep = "\n\n<!-- Page Break -->\n\n"
                    markdown_combined = sep.join(page_markdown)

                if markdown_combined:
                    with open(os.path.join(native_dir, f"{base}_combined.md"), "w", encoding="utf-8") as cf:
                        cf.write(markdown_combined)

            persisted_dir = ""
            if save_dir:
                persisted_dir = os.path.join(save_dir, base)
                os.makedirs(persisted_dir, exist_ok=True)
                for name in os.listdir(native_dir):
                    shutil.copy2(os.path.join(native_dir, name), os.path.join(persisted_dir, name))

            overall["results"].append({
                "filename": original_name,
                "pages": len(preds),
                "json": page_json if ret_json else [],
                "markdown_pages": page_markdown if ret_md else [],
                "markdown_combined": markdown_combined if ret_md else "",
                "saved_to": persisted_dir
            })

    return JSONResponse(overall)

import os
import io
import json
import shutil
import tempfile
import inspect

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
import numpy as np

# =========================
# Stable defaults (ARM64 CPU, medical lab reports)
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
USE_REGION_DETECTION = False            # OFF by default; can be heavy/fragile on ARM64

# Models (core)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# Optional model dirs ("" means not provided)
LAYOUT_DETECTION_MODEL_DIR = ""
TEXT_DETECTION_MODEL_DIR = ""
TEXT_RECOGNITION_MODEL_DIR = ""

# Recognition filter/batching
TEXT_REC_SCORE_THRESH = 0.60            # filter low-confidence text
TEXT_RECOGNITION_BATCH_SIZE = 8         # modest CPU batch

# Detection sizing (tuned for small fonts; safe defaults)
TEXT_DET_LIMIT_SIDE_LEN = 1536
TEXT_DET_LIMIT_TYPE = ""                # library default (e.g., 'max')
TEXT_DET_DB_THRESH = -1.0               # use model default
TEXT_DET_DB_BOX_THRESH = -1.0
TEXT_DET_DB_UNCLIP_RATIO = -1.0
TEXT_DET_DB_SCORE_MODE = ""             # 'fast'/'slow' if supported

# Layout thresholds
LAYOUT_THRESHOLD = -1.0
LAYOUT_UNCLIP_RATIO = -1.0
LAYOUT_NMS = False
LAYOUT_MERGE_BBOXES_MODE = ""           # '', 'all', 'text_and_table'

# Backend knobs (ARM64)
ENABLE_HPI = False                      # not supported on typical ARM64 CPU builds
ENABLE_MKLDNN = False                   # DNNL generally benefits x86_64 more
MKLDNN_CACHE_CAPACITY = 5

# Service I/O
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1
PIPELINE_CACHE_SIZE = 2

# Output behavior
RETURN_JSON_DEFAULT = True
RETURN_MARKDOWN_DEFAULT = True
CONCATENATE_MARKDOWN_DEFAULT = True
DEFAULT_SAVE_DIR = ""

# PDF/page control
PAGE_NUM = -1                           # -1 = all pages


# =========================
# Helpers
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

def _validate_file(name, content):
    ext = os.path.splitext(name or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type; allowed: {sorted(ALLOWED_EXTENSIONS)}")
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB)")

def _build_init_kwargs(cfg):
    # Strict: only pass keys PPStructureV3.__init__ for this build supports
    sig = inspect.signature(PPStructureV3.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "args", "kwargs"}

    cand = {
        # Core
        "device": cfg["DEVICE"],
        "lang": LANG,
        "cpu_threads": cfg["CPU_THREADS"],
        "enable_mkldnn": ENABLE_MKLDNN,
        "mkldnn_cache_capacity": MKLDNN_CACHE_CAPACITY,
        "enable_hpi": ENABLE_HPI,

        # Feature toggles
        "use_doc_orientation_classify": cfg["USE_DOC_ORIENTATION_CLASSIFY"],
        "use_doc_unwarping": cfg["USE_DOC_UNWARPING"],
        "use_textline_orientation": cfg["USE_TEXTLINE_ORIENTATION"],
        "use_table_recognition": cfg["USE_TABLE_RECOGNITION"],
        "use_formula_recognition": cfg["USE_FORMULA_RECOGNITION"],
        "use_chart_recognition": cfg["USE_CHART_RECOGNITION"],
        "use_seal_recognition": cfg["USE_SEAL_RECOGNITION"],
        "use_region_detection": cfg["USE_REGION_DETECTION"],

        # Models
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,

        # Optional model dirs
        "layout_detection_model_dir": LAYOUT_DETECTION_MODEL_DIR,
        "text_detection_model_dir": TEXT_DETECTION_MODEL_DIR,
        "text_recognition_model_dir": TEXT_RECOGNITION_MODEL_DIR,

        # Layout thresholds
        "layout_threshold": LAYOUT_THRESHOLD,
        "layout_unclip_ratio": LAYOUT_UNCLIP_RATIO,
        "layout_nms": LAYOUT_NMS,
        "layout_merge_bboxes_mode": LAYOUT_MERGE_BBOXES_MODE,

        # Detection thresholds/sizing
        "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
        "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
        "text_det_db_thresh": TEXT_DET_DB_THRESH,
        "text_det_db_box_thresh": TEXT_DET_DB_BOX_THRESH,
        "text_det_db_unclip_ratio": TEXT_DET_DB_UNCLIP_RATIO,
        "text_det_db_score_mode": TEXT_DET_DB_SCORE_MODE,

        # Recognition filters/batching
        "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
        "text_rec_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
    }

    return {k: v for k, v in cand.items() if (k in allowed and _provided(v))}

def _build_predict_kwargs(save_dir_for_native):
    sig = inspect.signature(PPStructureV3.predict)
    allowed = set(sig.parameters.keys()) - {"self"}
    cand = {
        "save_path": save_dir_for_native,
        "page_num": PAGE_NUM,
    }
    return {k: v for k, v in cand.items() if (k in allowed and _provided(v))}

def _warmup(pipeline):
    # Warm up with an actual file path (more stable than ndarray on some stacks)
    from PIL import Image
    tiny = (255 * np.ones((64, 64, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        Image.fromarray(tiny).save(tf.name)
        tmp = tf.name
    try:
        pipeline.predict(input=tmp)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# =========================
# App and pipeline cache
# =========================
app = FastAPI(
    title="PP-StructureV3 (CPU, ARM64) Service",
    version="2.0.0",
    description="PP-StructureV3 FastAPI with strict parameter filtering, JSON/Markdown outputs, and PDF Markdown concatenation. Optimized for medical lab reports."
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
        LAYOUT_DETECTION_MODEL_NAME, LAYOUT_DETECTION_MODEL_DIR,
        TEXT_DETECTION_MODEL_NAME, TEXT_DETECTION_MODEL_DIR,
        TEXT_RECOGNITION_MODEL_NAME, TEXT_RECOGNITION_MODEL_DIR,
        LAYOUT_THRESHOLD, LAYOUT_UNCLIP_RATIO, LAYOUT_NMS, LAYOUT_MERGE_BBOXES_MODE,
        TEXT_DET_LIMIT_SIDE_LEN, TEXT_DET_LIMIT_TYPE, TEXT_DET_DB_THRESH, TEXT_DET_DB_BOX_THRESH,
        TEXT_DET_DB_UNCLIP_RATIO, TEXT_DET_DB_SCORE_MODE, TEXT_REC_SCORE_THRESH, TEXT_RECOGNITION_BATCH_SIZE
    )

def _get_pipeline(cfg):
    init_kwargs = _build_init_kwargs(cfg)
    return _get_pipeline_cached(_cache_key_from_cfg(cfg), json.dumps(init_kwargs))

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
            "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
            "use_region_detection": USE_REGION_DETECTION
        },
        "endpoints": {"POST /parse": "Multipart: files=..., options=<JSON string> (optional)"}
    }

@app.post("/parse")
async def parse(
    files: list[UploadFile] = File(..., description="One or more images or PDFs"),
    options: str = Form(default="")
):
    # Output flags (request-level)
    ret_json = RETURN_JSON_DEFAULT
    ret_md = RETURN_MARKDOWN_DEFAULT
    concat_md = CONCATENATE_MARKDOWN_DEFAULT
    save_dir = DEFAULT_SAVE_DIR

    # Build request cfg (subset of toggles can be overridden)
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

    # Optional overrides via JSON string
    if options:
        try:
            ov = json.loads(options)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid options JSON: {e}")

        # Output toggles
        if "return_json" in ov: ret_json = bool(ov["return_json"])
        if "return_markdown" in ov: ret_md = bool(ov["return_markdown"])
        if "concatenate_markdown" in ov: concat_md = bool(ov["concatenate_markdown"])
        if "save_dir" in ov: save_dir = str(ov["save_dir"])

        # Device/threads
        if "device" in ov: cfg["DEVICE"] = str(ov["device"])
        if "cpu_threads" in ov:
            try:
                cfg["CPU_THREADS"] = int(ov["cpu_threads"])
            except Exception:
                pass

        # Feature toggles
        for k in [
            "USE_DOC_ORIENTATION_CLASSIFY","USE_DOC_UNWARPING","USE_TEXTLINE_ORIENTATION",
            "USE_TABLE_RECOGNITION","USE_FORMULA_RECOGNITION","USE_CHART_RECOGNITION",
            "USE_SEAL_RECOGNITION","USE_REGION_DETECTION"
        ]:
            if k in ov: cfg[k] = bool(ov[k])

        # Model overrides (names/dirs)
        global LAYOUT_DETECTION_MODEL_NAME, TEXT_DETECTION_MODEL_NAME, TEXT_RECOGNITION_MODEL_NAME
        global LAYOUT_DETECTION_MODEL_DIR, TEXT_DETECTION_MODEL_DIR, TEXT_RECOGNITION_MODEL_DIR
        if "layout_detection_model_name" in ov: LAYOUT_DETECTION_MODEL_NAME = str(ov["layout_detection_model_name"])
        if "text_detection_model_name" in ov: TEXT_DETECTION_MODEL_NAME = str(ov["text_detection_model_name"])
        if "text_recognition_model_name" in ov: TEXT_RECOGNITION_MODEL_NAME = str(ov["text_recognition_model_name"])
        if "layout_detection_model_dir" in ov: LAYOUT_DETECTION_MODEL_DIR = str(ov["layout_detection_model_dir"])
        if "text_detection_model_dir" in ov: TEXT_DETECTION_MODEL_DIR = str(ov["text_detection_model_dir"])
        if "text_recognition_model_dir" in ov: TEXT_RECOGNITION_MODEL_DIR = str(ov["text_recognition_model_dir"])

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

            # Save per-page using native helpers and collect in-memory markdown objects
            for res in preds:
                if ret_json:
                    try:
                        res.save_to_json(save_path=native_dir)
                    except Exception:
                        pass
                if ret_md:
                    try:
                        res.save_to_markdown(save_path=native_dir)
                    except Exception:
                        pass

                md_obj = getattr(res, "markdown", None)
                if isinstance(md_obj, dict):
                    page_md_objs.append(md_obj)

            # Read saved JSON/MD files back to the response
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

            # Concatenate Markdown for the whole PDF/file
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

# app.py
# PP-StructureV3 FastAPI wrapper (official-like, signature-safe, tuned for medical lab reports on ARM64)
# - Only passes parameters present in the installed PPStructureV3 signatures (prevents native crashes)
# - Minimal, clean endpoint that mirrors the official usage: pipeline.predict(input=path)
# - Multi-page Markdown concatenation via native helper (with safe fallback) + optional base64 inlining of images
# - Sensible defaults for English medical lab reports; all overrides are optional

# Set thread env before importing numeric libs/Paddle (helps stability on some ARM64 builds)
import os 
import io
import json
import base64
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple, List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from paddleocr import PPStructureV3


# ================= Defaults (ARM64 CPU; optimized for English medical lab reports) =================
DEVICE = "cpu"
CPU_THREADS = 4
LANG = "en"

# High-impact toggles (keep stable modules on; heavy/fragile ones off)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False
USE_TABLE_RECOGNITION = True
USE_REGION_DETECTION = False
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False

# Core models (balanced for CPU)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# Optional model dirs (empty string => not passed)
LAYOUT_DETECTION_MODEL_DIR = ""
REGION_DETECTION_MODEL_DIR = ""
TEXT_DETECTION_MODEL_DIR = ""
TEXT_RECOGNITION_MODEL_DIR = ""

# Text detection (DB) tuning (helps tiny fonts in A4 scans)
TEXT_DET_LIMIT_SIDE_LEN = 1536
TEXT_DET_LIMIT_TYPE = ""            # let pipeline default
TEXT_DET_DB_THRESH = 0.30
TEXT_DET_DB_BOX_THRESH = 0.40
TEXT_DET_DB_UNCLIP_RATIO = 2.0
TEXT_DET_DB_SCORE_MODE = ""         # default (keep empty to not pass)

# Recognition filtering/batching
TEXT_REC_SCORE_THRESH = 0.60
TEXT_RECOGNITION_BATCH_SIZE = 8
TEXTLINE_ORIENTATION_BATCH_SIZE = 8

# Backend toggles (MKLDNN often offers limited benefit on ARM64; keep off unless validated)
ENABLE_MKLDNN = False
MKLDNN_CACHE_CAPACITY = 5
ENABLE_HPI = False                  # typically unavailable on ARM64

# Startup warmup (disable if your build crashes on first forward)
WARMUP_AT_START = False

# I/O policy
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1


# ================= Small utilities =================
def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_too_big(upload: UploadFile) -> bool:
    size_hdr = upload.headers.get("content-length")
    if size_hdr and size_hdr.isdigit():
        return int(size_hdr) > MAX_FILE_SIZE_MB * 1024 * 1024
    return False

def _provided(v: Any) -> bool:
    # Treat "", 0, -1, None as "not provided"; booleans always considered provided
    if isinstance(v, bool):
        return True
    if v is None:
        return False
    if isinstance(v, (int, float)) and v in (0, -1, -1.0):
        return False
    if isinstance(v, str) and v == "":
        return False
    return True

def _filter_by_signature(params: Dict[str, Any], fn) -> Dict[str, Any]:
    import inspect
    allowed = set(inspect.signature(fn).parameters.keys())
    return {k: v for k, v in params.items() if (k in allowed and _provided(v))}

def _map_version_variant_keys(cand: Dict[str, Any]) -> Dict[str, Any]:
    # Map chart_recognition <-> chart_parsing depending on installed signature
    import inspect
    params = set(inspect.signature(PPStructureV3.__init__).parameters.keys())
    out = dict(cand)
    if "use_chart_parsing" in params and "use_chart_recognition" not in params:
        if "use_chart_recognition" in out:
            out["use_chart_parsing"] = out.pop("use_chart_recognition")
    # Some builds name text recognition batch as text_rec_batch_size
    if "text_rec_batch_size" in params and "text_recognition_batch_size" in out:
        out["text_rec_batch_size"] = out.pop("text_recognition_batch_size")
    return out

def _open_image_any(val: Union[Image.Image, bytes, str]) -> Optional[Image.Image]:
    try:
        if isinstance(val, Image.Image):
            return val
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val)).convert("RGBA")
        if isinstance(val, str) and Path(val).exists():
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

def _collect_results(pipeline: PPStructureV3, outputs, inline_images: bool) -> Tuple[List[Dict[str, Any]], str]:
    page_json: List[Dict[str, Any]] = []
    md_list: List[Dict[str, Any]] = []
    merged_img_map: Dict[str, Union[Image.Image, bytes, str]] = {}

    for res in outputs:
        page_json.append(_page_json_with_fallback(res))
        md = getattr(res, "markdown", None)
        if isinstance(md, dict):
            md_list.append(md)
            imgs = md.get("markdown_images", {})
            if isinstance(imgs, dict):
                merged_img_map.update(imgs)

    merged_md = ""
    if md_list:
        # Native helper when available
        try:
            merged_md = pipeline.concatenate_markdown_pages(md_list)
        except AttributeError:
            try:
                px = getattr(pipeline, "paddlex_pipeline", None)
                if px and hasattr(px, "concatenate_markdown_pages"):
                    merged_md = px.concatenate_markdown_pages(md_list)
                else:
                    merged_md = ""
            except Exception:
                merged_md = ""
        except Exception:
            merged_md = ""

        if isinstance(merged_md, dict) and "markdown_texts" in merged_md:
            merged_md = merged_md["markdown_texts"]
        elif not isinstance(merged_md, str):
            # Safe fallback: join page markdown_text/markdown if present
            parts: List[str] = []
            for md in md_list:
                txt = md.get("markdown_text") or md.get("markdown") or ""
                if isinstance(txt, str) and txt:
                    parts.append(txt)
            merged_md = "\n\n<!-- Page Break -->\n\n".join(parts)

    if inline_images and merged_md:
        merged_md = _embed_images_in_markdown(merged_md, merged_img_map)

    return page_json, merged_md

def _warmup_pipeline(pipeline: PPStructureV3) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        Image.new("RGB", (64, 64), color="white").save(tf.name)
        p = tf.name
    try:
        pipeline.predict(input=p)
    except Exception:
        pass
    finally:
        try:
            os.remove(p)
        except Exception:
            pass


# ================= Signature-aware builders =================
def _build_init_defaults() -> Dict[str, Any]:
    cand: Dict[str, Any] = {
        # device/lang/threads/backend
        "device": DEVICE,
        "lang": LANG,
        "cpu_threads": CPU_THREADS,
        "enable_mkldnn": ENABLE_MKLDNN,
        "mkldnn_cache_capacity": MKLDNN_CACHE_CAPACITY,
        "enable_hpi": ENABLE_HPI,

        # High-value toggles
        "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": USE_DOC_UNWARPING,
        "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        "use_table_recognition": USE_TABLE_RECOGNITION,
        "use_region_detection": USE_REGION_DETECTION,
        "use_formula_recognition": USE_FORMULA_RECOGNITION,
        "use_chart_recognition": USE_CHART_RECOGNITION,
        "use_seal_recognition": USE_SEAL_RECOGNITION,

        # Core models
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,

        # Optional dirs
        "layout_detection_model_dir": LAYOUT_DETECTION_MODEL_DIR,
        "region_detection_model_dir": REGION_DETECTION_MODEL_DIR,
        "text_detection_model_dir": TEXT_DETECTION_MODEL_DIR,
        "text_recognition_model_dir": TEXT_RECOGNITION_MODEL_DIR,

        # Text detection (DB)
        "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
        "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
        "text_det_db_thresh": TEXT_DET_DB_THRESH,
        "text_det_db_box_thresh": TEXT_DET_DB_BOX_THRESH,
        "text_det_db_unclip_ratio": TEXT_DET_DB_UNCLIP_RATIO,
        "text_det_db_score_mode": TEXT_DET_DB_SCORE_MODE,

        # Recognition/batching
        "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
        "text_recognition_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
        "textline_orientation_batch_size": TEXTLINE_ORIENTATION_BATCH_SIZE,
    }
    cand = _map_version_variant_keys(cand)
    return _filter_by_signature({k: v for k, v in cand.items() if _provided(v)}, PPStructureV3.__init__)

def _build_variant_init_kwargs(overrides: Dict[str, Any]) -> Dict[str, Any]:
    base = _build_init_defaults()
    merged = dict(base)
    merged.update({k: v for k, v in overrides.items() if _provided(v)})
    merged = _map_version_variant_keys(merged)
    return _filter_by_signature(merged, PPStructureV3.__init__)

def _build_predict_kwargs(page_num: Optional[int]) -> Dict[str, Any]:
    cand: Dict[str, Any] = {}
    if page_num and page_num > 0:
        cand["page_num"] = page_num
    return _filter_by_signature(cand, PPStructureV3.predict)


# ================= App & lifecycle =================
_PIPELINE_LOCK = threading.Lock()
_PIPELINE_CACHE: Dict[Tuple[Tuple[str, Any], ...], PPStructureV3] = {}
def _cache_key_from_kwargs(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(d.items(), key=lambda kv: kv[0]))

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_kwargs = _build_init_defaults()
    pipe = PPStructureV3(**init_kwargs)
    if WARMUP_AT_START:
        _warmup_pipeline(pipe)
    _PIPELINE_CACHE[_cache_key_from_kwargs(init_kwargs)] = pipe

    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(
    title="PP-StructureV3 (Official-like) API",
    version="4.2.0",
    lifespan=lifespan,
    description="Signature-safe PP-StructureV3 wrapper tuned for medical lab reports on ARM64 CPU."
)

@app.get("/health")
def health():
    return {"status": "ok", "cache_size": len(_PIPELINE_CACHE)}

def _get_or_create_pipeline(overrides: Dict[str, Any]) -> PPStructureV3:
    init_kwargs = _build_variant_init_kwargs(overrides)
    key = _cache_key_from_kwargs(init_kwargs)
    pipe = _PIPELINE_CACHE.get(key)
    if pipe is not None:
        return pipe
    with _PIPELINE_LOCK:
        pipe = _PIPELINE_CACHE.get(key)
        if pipe is not None:
            return pipe
        pipe = PPStructureV3(**init_kwargs)
        _PIPELINE_CACHE[key] = pipe
        return pipe


# ================= Endpoint (mirrors official pattern) =================
@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown", "both"] = Query("both"),
    markdown_images: Literal["none", "inline"] = Query("none", description="Inline base64 images in Markdown"),

    # Minimal, safe overrides (kept aligned with public docs; all optional)
    device: Optional[str] = Query(None),
    cpu_threads: Optional[int] = Query(None, gt=0),
    enable_mkldnn: Optional[bool] = Query(None),

    use_doc_orientation_classify: Optional[bool] = Query(None),
    use_doc_unwarping: Optional[bool] = Query(None),
    use_textline_orientation: Optional[bool] = Query(None),
    use_table_recognition: Optional[bool] = Query(None),
    use_region_detection: Optional[bool] = Query(None),
    use_formula_recognition: Optional[bool] = Query(None),
    use_chart_recognition: Optional[bool] = Query(None),
    use_seal_recognition: Optional[bool] = Query(None),

    layout_detection_model_name: Optional[str] = Query(None),
    text_detection_model_name: Optional[str] = Query(None),
    text_recognition_model_name: Optional[str] = Query(None),

    layout_detection_model_dir: Optional[str] = Query(None),
    region_detection_model_dir: Optional[str] = Query(None),
    text_detection_model_dir: Optional[str] = Query(None),
    text_recognition_model_dir: Optional[str] = Query(None),

    text_det_limit_side_len: Optional[int] = Query(None, gt=0),
    text_det_limit_type: Optional[str] = Query(None),
    text_det_db_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_db_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_det_db_unclip_ratio: Optional[float] = Query(None, gt=0.0),
    text_det_db_score_mode: Optional[str] = Query(None),

    text_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    text_recognition_batch_size: Optional[int] = Query(None, gt=0),
    textline_orientation_batch_size: Optional[int] = Query(None, gt=0),

    page_num: Optional[int] = Query(None, gt=0),
):
    if _file_too_big(file):
        raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB)")
    if not _ext_ok(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type; allowed: {sorted(ALLOWED_EXTENSIONS)}")

    tmp_dir = tempfile.mkdtemp(prefix="ppsv3_")
    in_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Build safe overrides (only provided values, later filtered by signature)
        overrides: Dict[str, Any] = {}
        for k, v in dict(
            device=device,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,

            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_table_recognition=use_table_recognition,
            use_region_detection=use_region_detection,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_seal_recognition=use_seal_recognition,

            layout_detection_model_name=layout_detection_model_name,
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,

            layout_detection_model_dir=layout_detection_model_dir or "",
            region_detection_model_dir=region_detection_model_dir or "",
            text_detection_model_dir=text_detection_model_dir or "",
            text_recognition_model_dir=text_recognition_model_dir or "",

            text_det_limit_side_len=text_det_limit_side_len if text_det_limit_side_len is not None else "",
            text_det_limit_type=text_det_limit_type or "",
            text_det_db_thresh=text_det_db_thresh if text_det_db_thresh is not None else "",
            text_det_db_box_thresh=text_det_db_box_thresh if text_det_db_box_thresh is not None else "",
            text_det_db_unclip_ratio=text_det_db_unclip_ratio if text_det_db_unclip_ratio is not None else "",
            text_det_db_score_mode=text_det_db_score_mode or "",

            text_recognition_batch_size=text_recognition_batch_size if text_recognition_batch_size is not None else "",
            textline_orientation_batch_size=textline_orientation_batch_size if textline_orientation_batch_size is not None else "",
            text_rec_score_thresh=text_rec_score_thresh if text_rec_score_thresh is not None else "",
        ).items():
            if _provided(v):
                overrides[k] = v

        pipeline = _get_or_create_pipeline(overrides)
        predict_kwargs = _build_predict_kwargs(page_num)

        acquired = app.state.predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy")
        try:
            outputs = await run_in_threadpool(lambda: pipeline.predict(input=in_path, **predict_kwargs))
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

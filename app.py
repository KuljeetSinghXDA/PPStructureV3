import os
import io
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List, Union

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from paddleocr import PPStructureV3


# ================= Minimal, stable defaults (ARM64 CPU, medical lab reports) =================
DEVICE = "cpu"
CPU_THREADS = 4
LANG = "en"

# Subpipelines (keep only the safest, high-value toggles)
USE_DOC_ORIENTATION_CLASSIFY = True
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = True
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False
USE_REGION_DETECTION = False  # heavy and fragile on some ARM64 stacks; keep OFF

# Core models (good CPU accuracy for A4 lab reports)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# Backend knobs: keep OFF/lean on ARM64
ENABLE_MKLDNN = False
MKLDNN_CACHE_CAPACITY = 5
ENABLE_HPI = False  # usually unsupported on ARM64

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= Helpers =================
def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_too_big(upload: UploadFile) -> bool:
    size_hdr = upload.headers.get("content-length")
    if size_hdr and size_hdr.isdigit():
        return int(size_hdr) > MAX_FILE_SIZE_MB * 1024 * 1024
    return False

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
    # Replace image paths in the markdown with inline base64 (handy for single-file returns)
    for path, val in images_map.items():
        pil_img = _open_image_any(val)
        if pil_img is None:
            continue
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = buf.getvalue()
        import base64 as _b64
        data_uri = "data:image/png;base64," + _b64.b64encode(b64).decode("ascii")
        md_text = md_text.replace(f'src="{path}"', f'src="{data_uri}"')
        md_text = md_text.replace(f"({path})", f"({data_uri})")
    return md_text

def _page_json(res) -> Dict[str, Any]:
    j = getattr(res, "json", None)
    if isinstance(j, dict):
        return j
    # If not available in-memory, persist and read last JSON file
    with tempfile.TemporaryDirectory() as td:
        try:
            res.save_to_json(save_path=td)
            files = sorted(Path(td).glob("*.json"))
            if files:
                return json.loads(files[-1].read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _collect_results(pipeline: PPStructureV3, outputs, inline_images: bool):
    page_json: List[Dict[str, Any]] = []
    md_dicts: List[Dict[str, Any]] = []
    all_images: Dict[str, Union[Image.Image, bytes, str]] = {}

    for res in outputs:
        # JSON
        page_json.append(_page_json(res))
        # Markdown dict (structure may vary by version)
        md = getattr(res, "markdown", None)
        if isinstance(md, dict):
            md_dicts.append(md)
            imgs = md.get("markdown_images", {})
            if isinstance(imgs, dict):
                all_images.update(imgs)

    # Concatenate markdown using native helper when available
    merged_md = ""
    if md_dicts:
        try:
            merged_md = pipeline.concatenate_markdown_pages(md_dicts)
        except AttributeError:
            try:
                paddlex = getattr(pipeline, "paddlex_pipeline", None)
                if paddlex and hasattr(paddlex, "concatenate_markdown_pages"):
                    merged_md = paddlex.concatenate_markdown_pages(md_dicts)
            except Exception:
                merged_md = ""
        except Exception:
            merged_md = ""
        # Normalize return type
        if isinstance(merged_md, dict) and "markdown_texts" in merged_md:
            merged_md = merged_md["markdown_texts"]
        elif not isinstance(merged_md, str):
            merged_md = ""

    if inline_images and merged_md:
        merged_md = _embed_images_in_markdown(merged_md, all_images)

    return page_json, merged_md

def _build_init_kwargs_minimal() -> Dict[str, Any]:
    # Pass ONLY a minimal, stable subset to avoid native crashes
    import inspect
    allowed = set(inspect.signature(PPStructureV3.__init__).parameters.keys())
    cand = {
        "device": DEVICE,
        "lang": LANG,
        "cpu_threads": CPU_THREADS,
        "enable_mkldnn": ENABLE_MKLDNN,
        "mkldnn_cache_capacity": MKLDNN_CACHE_CAPACITY,
        # Safe feature toggles
        "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": USE_DOC_UNWARPING,
        "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        "use_table_recognition": USE_TABLE_RECOGNITION,
        "use_formula_recognition": USE_FORMULA_RECOGNITION,
        "use_chart_recognition": USE_CHART_RECOGNITION,
        "use_seal_recognition": USE_SEAL_RECOGNITION,
        "use_region_detection": USE_REGION_DETECTION,
        # Core models
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,
    }
    # Some builds expose 'use_chart_parsing' not 'use_chart_recognition'
    if "use_chart_parsing" in allowed and "use_chart_recognition" not in allowed:
        cand["use_chart_parsing"] = cand.pop("use_chart_recognition")
    # Finally filter to installed signature and drop any empty/None
    return {k: v for k, v in cand.items() if (k in allowed and v is not None)}

def _warmup_pipeline(pipeline: PPStructureV3) -> None:
    # Warm up with a real PNG path for stability
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


# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PPStructureV3(**_build_init_kwargs_minimal())
    _warmup_pipeline(app.state.pipeline)
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(
    title="PPStructureV3 /parse API",
    version="3.0.0",
    lifespan=lifespan,
    description="Minimal, stable wrapper for PP-StructureV3 on ARM64 CPU; tuned for medical lab reports."
)

@app.get("/health")
def health():
    return {"status": "ok"}


# ================= Endpoint (clean, “official-like”) =================
# Default returns both JSON and concatenated Markdown
@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown", "both"] = Query("both"),
    markdown_images: Literal["none", "inline"] = Query("none", description="Inline images as base64")
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

        acquired = app.state.predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy")

        try:
            # Call predict with ONLY input, no extra kwargs
            outputs = await run_in_threadpool(lambda: app.state.pipeline.predict(input=in_path))
        finally:
            app.state.predict_sem.release()

        inline_flag = markdown_images == "inline"
        page_json, merged_md = _collect_results(app.state.pipeline, outputs, inline_flag)

        if output_format == "json":
            return JSONResponse({"results": page_json, "pages": len(page_json)})
        elif output_format == "markdown":
            return PlainTextResponse(merged_md or "")
        else:
            return JSONResponse({"results": page_json, "markdown": merged_md or "", "pages": len(page_json)})

    except HTTPException:
        raise
    except Exception as e:
        # Return Python-level errors verbosely while we harden native config
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {str(e)}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

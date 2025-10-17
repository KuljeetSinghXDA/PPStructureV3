# Set thread env vars BEFORE importing any numeric libs/Paddle to avoid OpenMP/BLAS issues on ARM64
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "2")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
_os.environ.setdefault("MKL_NUM_THREADS", "2")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
# Keep MKLDNN off unless you explicitly enable it in the code below
_os.environ.setdefault("FLAGS_use_mkldnn", "0")

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

# Import AFTER env vars are set
from paddleocr import PPStructureV3


# ================= Minimal, official-like defaults (ARM64 CPU, medical PDFs/scans) =================
DEVICE = "cpu"
CPU_THREADS = 2         # keep small; ARM64 often crashes with high OpenMP fanout
LANG = "en"             # ensures English-recognition models are chosen internally

# I/O policy
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# Global state (lazy pipeline)
_PIPELINE: Optional[PPStructureV3] = None
_PIPELINE_LOCK = threading.Lock()


# ================= Small helpers =================
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
        if isinstance(val, str) and Path(val).exists():
            return Image.open(val).convert("RGBA")
    except (UnidentifiedImageError, OSError):
        return None
    return None

def _embed_images_in_markdown(md_text: str, images_map: Dict[str, Union[Image.Image, bytes, str]]) -> str:
    # Replace image paths in the markdown with inline base64
    import base64 as _b64
    for path, val in images_map.items():
        pil_img = _open_image_any(val)
        if pil_img is None:
            continue
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        data_uri = "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode("ascii")
        md_text = md_text.replace(f'src="{path}"', f'src="{data_uri}"')
        md_text = md_text.replace(f"({path})", f"({data_uri})")
    return md_text

def _page_json(res) -> Dict[str, Any]:
    # Prefer in-memory .json field when present, else persist and read back
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

def _collect_results(pipeline: PPStructureV3, outputs, inline_images: bool) -> tuple[list[dict], str]:
    page_json: List[Dict[str, Any]] = []
    md_dicts: List[Dict[str, Any]] = []
    all_images: Dict[str, Union[Image.Image, bytes, str]] = {}

    for res in outputs:
        page_json.append(_page_json(res))
        md = getattr(res, "markdown", None)
        if isinstance(md, dict):
            md_dicts.append(md)
            imgs = md.get("markdown_images", {})
            if isinstance(imgs, dict):
                all_images.update(imgs)

    merged_md = ""
    if md_dicts:
        # Prefer native concatenator; fall back to simple join if unavailable
        try:
            merged_md = pipeline.concatenate_markdown_pages(md_dicts)
        except AttributeError:
            try:
                paddlex = getattr(pipeline, "paddlex_pipeline", None)
                if paddlex and hasattr(paddlex, "concatenate_markdown_pages"):
                    merged_md = paddlex.concatenate_markdown_pages(md_dicts)
                else:
                    merged_md = ""
            except Exception:
                merged_md = ""
        except Exception:
            merged_md = ""

        if isinstance(merged_md, dict) and "markdown_texts" in merged_md:
            merged_md = merged_md["markdown_texts"]
        elif not isinstance(merged_md, str):
            # Fallback join by page if the helper isn’t present/compatible
            parts = []
            for md_obj in md_dicts:
                txt = md_obj.get("markdown_text", "") or md_obj.get("markdown", "")
                if isinstance(txt, str) and txt:
                    parts.append(txt)
            merged_md = "\n\n<!-- Page Break -->\n\n".join(parts)

    if inline_images and merged_md:
        merged_md = _embed_images_in_markdown(merged_md, all_images)

    return page_json, merged_md


# ================= App (build pipeline lazily, no warmup) =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Do NOT warm up or construct at startup — build lazily on first request
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(
    title="PP-StructureV3 (Official-like) API",
    version="3.1.0",
    lifespan=lifespan,
    description="Lean, stable PP-StructureV3 wrapper for ARM64 CPU with minimal parameters and official call pattern."
)

@app.get("/health")
def health():
    status = {"status": "ok"}
    if _PIPELINE is not None:
        status["pipeline_ready"] = True
    else:
        status["pipeline_ready"] = False
    return status


def _get_pipeline() -> PPStructureV3:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            # IMPORTANT: pass only minimal, stable keys
            # - device/lang/cpu_threads are safe and officially documented
            # - do NOT pass extra toggles, thresholds, or model names here
            _PIPELINE = PPStructureV3(
                device=DEVICE,
                lang=LANG,
                cpu_threads=CPU_THREADS
            )
    return _PIPELINE


# Endpoint: official-like usage. We call predict(input=path) only.
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
    in_path = str(Path(tmp_dir) / file.filename)
    try:
        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        pipe = _get_pipeline()

        acquired = app.state.predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy")

        try:
            outputs = await run_in_threadpool(lambda: pipe.predict(input=in_path))
        finally:
            app.state.predict_sem.release()

        inline_flag = markdown_images == "inline"
        page_json, merged_md = _collect_results(pipe, outputs, inline_flag)

        if output_format == "json":
            return JSONResponse({"results": page_json, "pages": len(page_json)})
        elif output_format == "markdown":
            return PlainTextResponse(merged_md or "")
        else:
            return JSONResponse({"results": page_json, "markdown": merged_md or "", "pages": len(page_json)})

    except HTTPException:
        raise
    except Exception as e:
        # If a Python-level error occurs, surface it
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {str(e)}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

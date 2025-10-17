import os
import io
import base64
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from paddleocr import PPStructureV3

# ================= Service defaults (kept to official interfaces only) =================
# Environment-driven init knobs (official PPStructureV3 __init__ parameters)
DEVICE = os.getenv("PPOCR_DEVICE", "cpu")
ENABLE_MKLDNN = bool(int(os.getenv("PPOCR_ENABLE_MKLDNN", "1")))   # If unavailable, Paddle ignores it (official behavior)
PRECISION = os.getenv("PPOCR_PRECISION", "fp32")
CPU_THREADS = int(os.getenv("PPOCR_CPU_THREADS", "8"))
PADDLEX_CONFIG = os.getenv("PPOCR_PADDLEX_CONFIG", None)

# Medical lab report bias: English recognition model; override via env if desired.
# Use official model names. Leave as None to stick with PaddleOCR defaults.
TEXT_RECOGNITION_MODEL_NAME = os.getenv("PPOCR_TEXT_RECOGNITION_MODEL_NAME", "en_PP-OCRv5_mobile_rec")

# I/O
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = int(os.getenv("PPOCR_MAX_PARALLEL", "1"))

# Modest concurrency guard
_sem = threading.Semaphore(MAX_PARALLEL_PREDICT)

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Instantiate PPStructureV3 with official init args only
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        precision=PRECISION,
        cpu_threads=CPU_THREADS,
        paddlex_config=PADDLEX_CONFIG,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME if TEXT_RECOGNITION_MODEL_NAME else None,
    )
    yield

app = FastAPI(title="PPStructureV3 /parse API (official-only)", version="3.0", lifespan=lifespan)

# ================= Helpers =================
def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_too_big(upload: UploadFile) -> bool:
    size_hdr = upload.headers.get("content-length")
    if size_hdr and size_hdr.isdigit():
        return int(size_hdr) > MAX_FILE_SIZE_MB * 1024 * 1024
    return False

def _to_b64_png(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ================= Endpoints =================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(
    # Input: provide either an uploaded file or a URL
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Query(None, description="HTTP/HTTPS URL to an image or PDF"),

    # Output shaping
    output_format: Literal["json", "markdown", "both"] = Query("both"),
    markdown_images: Literal["none", "separate"] = Query("none", description="Return merged markdown images separately as base64"),

    # Predict-time toggles (official predict parameters)
    use_doc_orientation_classify: Optional[bool] = Query(True, description="Enable doc orientation classifier"),
    use_doc_unwarping: Optional[bool] = Query(False, description="Enable doc unwarping"),
    use_textline_orientation: Optional[bool] = Query(True, description="Enable text line orientation classifier"),
    use_table_recognition: Optional[bool] = Query(True, description="Enable table recognition"),
    use_formula_recognition: Optional[bool] = Query(False),
    use_chart_recognition: Optional[bool] = Query(False),
    use_seal_recognition: Optional[bool] = Query(False),
    use_region_detection: Optional[bool] = Query(False),

    # Text det/rec tuning (optimized for small fonts in lab reports)
    text_det_limit_side_len: Optional[int] = Query(1536, gt=0),
    text_det_thresh: Optional[float] = Query(0.30, ge=0.0, le=1.0),
    text_det_box_thresh: Optional[float] = Query(0.40, ge=0.0, le=1.0),
    text_det_unclip_ratio: Optional[float] = Query(2.0, gt=0.0),
    text_rec_score_thresh: Optional[float] = Query(0.60, ge=0.0, le=1.0),

    # Table behavior (official predict parameters)
    use_ocr_results_with_table_cells: Optional[bool] = Query(True),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(False),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(False),
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(False),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(False),
    use_table_orientation_classify: Optional[bool] = Query(True),
):
    if not any([file, file_url]):
        raise HTTPException(status_code=400, detail="Provide one of: file or file_url")

    tmp_dir = None
    input_arg = None
    try:
        if file:
            if _file_too_big(file):
                raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB)")
            if not _ext_ok(file.filename):
                raise HTTPException(status_code=400, detail=f"Unsupported file type; allowed: {sorted(ALLOWED_EXTENSIONS)}")
            tmp_dir = tempfile.mkdtemp(prefix="ppsv3_")
            input_arg = str(Path(tmp_dir) / file.filename)
            with open(input_arg, "wb") as f:
                shutil.copyfileobj(file.file, f)
        else:
            # Official API: URLs are valid inputs to predict()
            input_arg = file_url

        # Build predict kwargs strictly from official predict parameters
        predict_kwargs: Dict[str, Any] = dict(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_seal_recognition=use_seal_recognition,
            use_region_detection=use_region_detection,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
            use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
            use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
            use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
            use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
            use_table_orientation_classify=use_table_orientation_classify,
        )

        acquired = _sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy")

        try:
            pipeline: PPStructureV3 = app.state.pipeline
            outputs = await run_in_threadpool(lambda: pipeline.predict(input=input_arg, **predict_kwargs))
        finally:
            _sem.release()

        # Collect JSON results and Markdown parts (official attributes)
        page_json: List[Dict[str, Any]] = []
        markdown_list: List[Dict[str, Any]] = []
        for res in outputs:
            # direct attributes from official API
            page_json.append(res.json)
            md = res.markdown or {}
            markdown_list.append(md)

        merged_md_text = ""
        merged_md_images: Dict[str, Any] = {}

        if output_format in ("markdown", "both"):
            # Official concatenation API
            ret = app.state.pipeline.concatenate_markdown_pages(markdown_list)
            if isinstance(ret, tuple) and len(ret) >= 2:
                merged_md_text, merged_md_images = ret[0], (ret[1] or {})
            else:
                # Some docs/examples show a single string return; we stick to official behavior here.
                merged_md_text = ret if isinstance(ret, str) else ""

        # Shape responses
        if output_format == "json":
            return JSONResponse({"results": page_json, "pages": len(page_json)})

        if output_format == "markdown":
            return PlainTextResponse(merged_md_text or "")

        # both
        if markdown_images == "separate" and merged_md_images:
            images_b64: Dict[str, str] = {}
            for k, v in merged_md_images.items():
                try:
                    images_b64[k] = _to_b64_png(v)
                except Exception:
                    pass
            return JSONResponse(
                {"results": page_json, "markdown": {"text": merged_md_text or "", "images": images_b64}, "pages": len(page_json)}
            )
        else:
            return JSONResponse({"results": page_json, "markdown": merged_md_text or "", "pages": len(page_json)})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {str(e)}")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

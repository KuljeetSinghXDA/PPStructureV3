# app/app/server.py
import base64
import io
import json
import os
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

# PaddleOCR PP-StructureV3 pipeline
from paddleocr import PPStructureV3

# ================= Core Configuration (safe CPU defaults) =================
DEVICE = "cpu"
CPU_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
ENABLE_HPI = False
ENABLE_MKLDNN = True
MKLDNN_CACHE_CAPACITY = 10
PRECISION = "fp32"

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ==================== Pydantic request models ====================
class LayoutUnclipType(BaseModel):
    # Accept float, list[float], or dict[str,float]
    value: Union[float, List[float], Dict[str, float]]

class ParseParams(BaseModel):
    # General output controls
    return_json: bool = True
    return_markdown: bool = True
    concat_markdown: bool = True
    visualize: Optional[bool] = False
    include_input_image: bool = False

    # JSON formatting
    format_json: bool = True
    indent: int = 2
    ensure_ascii: bool = False

    # Optional page range for PDFs (1-based inclusive)
    page_from: Optional[int] = None
    page_to: Optional[int] = None

    # Language (optional) - when set, PPStructureV3(lang=...) will be re-instantiated at startup only
    # For per-request we keep pipeline default; this param is present for future extension.
    lang: Optional[str] = None

    # Predict-time sub-pipeline toggles
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = None
    use_table_recognition: Optional[bool] = None
    use_formula_recognition: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_region_detection: Optional[bool] = None
    use_table_cells_ocr_results: Optional[bool] = None
    use_e2e_wired_table_rec_model: Optional[bool] = None
    use_e2e_wireless_table_rec_model: Optional[bool] = None

    # Layout detection thresholds/postprocess
    layout_threshold: Optional[Union[float, Dict[int, float]]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, List[float], Dict[str, float]]] = None
    layout_merge_bboxes_mode: Optional[Union[str, Dict[str, Any]]] = None

    # Text detection & recognition
    text_det_limit_side_len: Optional[int] = None
    text_det_limit_type: Optional[str] = None
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    text_rec_score_thresh: Optional[float] = None

    # Batch sizes (predict-time)
    text_recognition_batch_size: Optional[int] = None
    textline_orientation_batch_size: Optional[int] = None
    chart_recognition_batch_size: Optional[int] = None
    seal_text_recognition_batch_size: Optional[int] = None
    formula_recognition_batch_size: Optional[int] = None

    # Seal detection thresholds
    seal_det_limit_side_len: Optional[int] = None
    seal_det_limit_type: Optional[str] = None
    seal_det_thresh: Optional[float] = None
    seal_det_box_thresh: Optional[float] = None
    seal_det_unclip_ratio: Optional[float] = None
    seal_rec_score_thresh: Optional[float] = None

    # Visualization control: return which images; if None or empty -> default from pipeline
    # Valid keys roughly include: layout_det_res, overall_ocr_res, text_paragraphs_ocr_res,
    # formula_res_region1, table_cell_img, seal_res_region1
    return_image_keys: Optional[List[str]] = None

    @validator("text_det_limit_type", "seal_det_limit_type")
    def _valid_limit_type(cls, v):
        if v is not None and v not in ("min", "max"):
            raise ValueError("limit_type must be 'min' or 'max'")
        return v

class ParseResponse(BaseModel):
    pages: int
    filetype: str
    layoutParsingResults: List[Dict[str, Any]]
    concatenatedMarkdown: Optional[Dict[str, Any]] = None
    dataInfo: Dict[str, Any]

# ==================== Utilities ====================
def _ext_ok(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS

def _check_file_size(f: UploadFile):
    f.file.seek(0, io.SEEK_END)
    size = f.file.tell()
    f.file.seek(0)
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB).")

def _save_upload(tmpdir: Path, up: UploadFile) -> Path:
    dst = tmpdir / Path(up.filename).name
    with open(dst, "wb") as w:
        shutil.copyfileobj(up.file, w)
    return dst

def _pdf_slice(src: Path, page_from: Optional[int], page_to: Optional[int]) -> Path:
    """
    Slice a PDF to the given 1-based inclusive range.
    """
    if page_from is None and page_to is None:
        return src
    doc = fitz.open(src)
    n = len(doc)
    start = max(1, page_from or 1)
    end = min(n, page_to or n)
    if start > end:
        raise HTTPException(status_code=400, detail="Invalid page range.")
    out = fitz.open()
    for p in range(start - 1, end):
        out.insert_pdf(doc, from_page=p, to_page=p)
    sliced = src.parent / f"{src.stem}_p{start}-{end}.pdf"
    out.save(sliced)
    out.close()
    doc.close()
    return sliced

def _pil_to_b64(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ==================== App & Lifespan ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create a single PP-StructureV3 pipeline instance with CPU-only,
    MKL-DNN enabled, HPI off by default (can be toggled if needed).
    """
    pipeline = PPStructureV3(
        device=DEVICE,
        enable_hpi=ENABLE_HPI,
        enable_mkldnn=ENABLE_MKLDNN,
        mkldnn_cache_capacity=MKLDNN_CACHE_CAPACITY,
        cpu_threads=CPU_THREADS,
        precision=PRECISION,
        # Keep model names None to use pipeline defaults; request-level toggles are passed to predict().
        # If you want to pin specific models, you can set the *_model_name/_model_dir args here.
    )
    app.state.pipeline = pipeline
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield
    # Pipeline uses lazy-loaded models; no explicit close needed.

app = FastAPI(title="PP-StructureV3 /parse API", version="1.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="Image or PDF"),
    params_json: Optional[str] = Form(default=None, description="JSON string of ParseParams"),
    # Minimal toggles also exposed as query for convenience:
    return_markdown: Optional[bool] = Query(None),
    return_json: Optional[bool] = Query(None),
):
    if not _ext_ok(file.filename):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {Path(file.filename).suffix}")
    _check_file_size(file)

    # Merge params from form JSON + query fallbacks
    try:
        req = ParseParams(**(json.loads(params_json) if params_json else {}))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid params_json: {e}")

    if return_markdown is not None:
        req.return_markdown = return_markdown
    if return_json is not None:
        req.return_json = return_json

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        src_path = _save_upload(tmpdir, file)

        # Optional page-range slicing for PDFs
        in_path = src_path
        if src_path.suffix.lower() == ".pdf":
            in_path = _pdf_slice(src_path, req.page_from, req.page_to)

        # Semaphore to cap concurrent inferences
        sem = app.state.predict_sem
        sem.acquire()
        try:
            # Build predict() kwargs from request
            predict_kwargs: Dict[str, Any] = {}

            # Sub-pipeline toggles
            for k in (
                "use_doc_orientation_classify",
                "use_doc_unwarping",
                "use_textline_orientation",
                "use_table_recognition",
                "use_formula_recognition",
                "use_chart_recognition",
                "use_seal_recognition",
                "use_region_detection",
                "use_table_cells_ocr_results",
                "use_e2e_wired_table_rec_model",
                "use_e2e_wireless_table_rec_model",
            ):
                v = getattr(req, k)
                if v is not None:
                    predict_kwargs[k] = v

            # Layout & OCR thresholds
            for k in (
                "layout_threshold",
                "layout_nms",
                "layout_unclip_ratio",
                "layout_merge_bboxes_mode",
                "text_det_limit_side_len",
                "text_det_limit_type",
                "text_det_thresh",
                "text_det_box_thresh",
                "text_det_unclip_ratio",
                "text_rec_score_thresh",
                "seal_det_limit_side_len",
                "seal_det_limit_type",
                "seal_det_thresh",
                "seal_det_box_thresh",
                "seal_det_unclip_ratio",
                "seal_rec_score_thresh",
                "text_recognition_batch_size",
                "textline_orientation_batch_size",
                "chart_recognition_batch_size",
                "seal_text_recognition_batch_size",
                "formula_recognition_batch_size",
            ):
                v = getattr(req, k)
                if v is not None:
                    predict_kwargs[k] = v

            # Visualization control
            if req.visualize is not None:
                predict_kwargs["visualize"] = req.visualize

            # Run predict in threadpool to avoid blocking event loop
            def _run_predict():
                return list(app.state.pipeline.predict(input=str(in_path), **predict_kwargs))

            output_list = await run_in_threadpool(_run_predict)

        finally:
            sem.release()

        # Build response
        pages = len(output_list)
        result_pages: List[Dict[str, Any]] = []

        for res in output_list:
            page_obj: Dict[str, Any] = {}

            # JSON
            if req.return_json:
                # res.json is already a structured dict; pretty-print if requested
                if req.format_json:
                    page_obj["json"] = json.loads(json.dumps(res.json, ensure_ascii=req.ensure_ascii))
                else:
                    page_obj["json"] = res.json

            # Markdown
            if req.return_markdown:
                md = res.markdown  # dict: {markdown_texts, markdown_images, page_continuation_flags}
                # Convert PIL images to base64 if present
                md_images64 = []
                for im in md.get("markdown_images", []) or []:
                    md_images64.append(_pil_to_b64(im))
                page_obj["markdown"] = {
                    "markdown_texts": md.get("markdown_texts"),
                    "markdown_images_b64": md_images64,
                    "page_continuation_flags": md.get("page_continuation_flags"),
                }

            # Visualization images (optional)
            if req.visualize:
                imgs = res.img  # dict of PIL.Image
                selected_keys = req.return_image_keys or list(imgs.keys())
                vis_dict = {}
                for k in selected_keys:
                    im = imgs.get(k)
                    if im is not None:
                        vis_dict[k] = _pil_to_b64(im)
                page_obj["outputImages"] = vis_dict

            # Optionally include the input image (only when image input)
            if req.include_input_image and src_path.suffix.lower() != ".pdf":
                try:
                    from PIL import Image
                    im = Image.open(src_path)
                    page_obj["inputImage"] = _pil_to_b64(im)
                except Exception:
                    page_obj["inputImage"] = None

            # A pruned version of res.json without path/page (if available)
            if "res" in (res.json or {}):
                pruned = dict(res.json["res"])
                pruned.pop("input_path", None)
                pruned.pop("page_index", None)
                page_obj["prunedResult"] = pruned

            result_pages.append(page_obj)

        # Concatenate Markdown across pages if requested
        concatenated_md: Optional[Dict[str, Any]] = None
        if req.return_markdown and req.concat_markdown and pages > 1:
            # Prefer pipeline's helper if available; otherwise do a simple concat
            try:
                md_list = [r["markdown"] for r in result_pages if "markdown" in r]
                if hasattr(app.state.pipeline, "concatenate_markdown_pages"):
                    merged_texts, merged_images, flags = app.state.pipeline.concatenate_markdown_pages(md_list)  # type: ignore
                    concatenated_md = {
                        "markdown_texts": merged_texts,
                        "markdown_images_b64": [ _pil_to_b64(im) for im in (merged_images or []) ],
                        "page_continuation_flags": flags,
                    }
                else:
                    # Fallback: naive concatenation of texts and images
                    all_texts: List[str] = []
                    all_imgs_b64: List[str] = []
                    for md in md_list:
                        all_texts.extend(md.get("markdown_texts") or [])
                        all_imgs_b64.extend(md.get("markdown_images_b64") or [])
                    concatenated_md = {
                        "markdown_texts": all_texts,
                        "markdown_images_b64": all_imgs_b64,
                        "page_continuation_flags": None,
                    }
            except Exception:
                concatenated_md = None

        resp = ParseResponse(
            pages=pages,
            filetype=src_path.suffix.lower(),
            layoutParsingResults=result_pages,
            concatenatedMarkdown=concatenated_md,
            dataInfo={
                "filename": file.filename,
                "input_path": str(in_path.name),
                "page_range_used": {
                    "from": req.page_from,
                    "to": req.page_to,
                } if src_path.suffix.lower() == ".pdf" else None,
            },
        )
        return JSONResponse(content=json.loads(resp.json()))

# app/server.py
import os
import io
import base64
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# Import PP-StructureV3 pipeline
from paddleocr import PPStructureV3

# ================= Core Service Configuration =================
DEVICE = "cpu"                  # Arm64 CPU only
CPU_THREADS = 4
ENABLE_MKLDNN = True
ENABLE_HPI = False              # High-Performance Inference plugin (not needed for baseline CPU)
PRECISION = "fp32"              # CPU precision

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# Reasonable modern defaults (can be overridden per-request)
DEFAULTS = dict(
    # Major models
    layout_detection_model_name="PP-DocLayout-L",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    table_classification_model_name="PP-LCNet_x1_0_table_cls",
    chart_recognition_model_name="PP-Chart2Table",
    # Optional sub-pipelines toggles
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    use_region_detection=True,
    use_table_recognition=True,
    use_formula_recognition=False,
    use_chart_recognition=False,
    use_seal_recognition=False,
    # CPU accel
    enable_mkldnn=ENABLE_MKLDNN,
    enable_hpi=ENABLE_HPI,
    cpu_threads=CPU_THREADS,
    device=DEVICE,
    precision=PRECISION,
)

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a baseline pipeline instance (per-request overrides will be supplied to predict())
    app.state.pipeline = PPStructureV3(**DEFAULTS)
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PP-StructureV3 /parse API", version="1.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _check_ext_size(tmp_path: Path):
    ext = tmp_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file extension: {ext}")
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB), limit {MAX_FILE_SIZE_MB} MB")


def _save_upload_to_tmp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix
    if not suffix:
        suffix = ".bin"
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    with open(tmp_name, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    p = Path(tmp_name)
    _check_ext_size(p)
    return p


def _encode_pil_to_b64(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gather_predict_kwargs(
    # Model names/dirs
    layout_detection_model_name: Optional[str],
    layout_detection_model_dir: Optional[str],
    region_detection_model_name: Optional[str],
    region_detection_model_dir: Optional[str],
    doc_orientation_classify_model_name: Optional[str],
    doc_orientation_classify_model_dir: Optional[str],
    doc_unwarping_model_name: Optional[str],
    doc_unwarping_model_dir: Optional[str],
    text_detection_model_name: Optional[str],
    text_detection_model_dir: Optional[str],
    textline_orientation_model_name: Optional[str],
    textline_orientation_model_dir: Optional[str],
    text_recognition_model_name: Optional[str],
    text_recognition_model_dir: Optional[str],
    table_classification_model_name: Optional[str],
    table_classification_model_dir: Optional[str],
    wired_table_structure_recognition_model_name: Optional[str],
    wired_table_structure_recognition_model_dir: Optional[str],
    wireless_table_structure_recognition_model_name: Optional[str],
    wireless_table_structure_recognition_model_dir: Optional[str],
    wired_table_cells_detection_model_name: Optional[str],
    wired_table_cells_detection_model_dir: Optional[str],
    wireless_table_cells_detection_model_name: Optional[str],
    wireless_table_cells_detection_model_dir: Optional[str],
    table_orientation_classify_model_name: Optional[str],
    table_orientation_classify_model_dir: Optional[str],
    seal_text_detection_model_name: Optional[str],
    seal_text_detection_model_dir: Optional[str],
    seal_text_recognition_model_name: Optional[str],
    seal_text_recognition_model_dir: Optional[str],
    formula_recognition_model_name: Optional[str],
    formula_recognition_model_dir: Optional[str],
    chart_recognition_model_name: Optional[str],
    chart_recognition_model_dir: Optional[str],
    # Numeric thresholds/limits/batch sizes
    layout_threshold: Optional[float],
    layout_nms: Optional[bool],
    layout_unclip_ratio: Optional[float],
    layout_merge_bboxes_mode: Optional[str],
    text_det_limit_side_len: Optional[int],
    text_det_limit_type: Optional[str],
    text_det_thresh: Optional[float],
    text_det_box_thresh: Optional[float],
    text_det_unclip_ratio: Optional[float],
    textline_orientation_batch_size: Optional[int],
    text_recognition_batch_size: Optional[int],
    text_rec_score_thresh: Optional[float],
    chart_recognition_batch_size: Optional[int],
    formula_recognition_batch_size: Optional[int],
    seal_det_limit_side_len: Optional[int],
    seal_det_limit_type: Optional[str],
    seal_det_thresh: Optional[float],
    seal_det_box_thresh: Optional[float],
    seal_det_unclip_ratio: Optional[float],
    seal_text_recognition_batch_size: Optional[int],
    seal_rec_score_thresh: Optional[float],
    # Toggles
    use_doc_orientation_classify: Optional[bool],
    use_doc_unwarping: Optional[bool],
    use_textline_orientation: Optional[bool],
    use_region_detection: Optional[bool],
    use_table_recognition: Optional[bool],
    use_formula_recognition: Optional[bool],
    use_chart_recognition: Optional[bool],
    use_seal_recognition: Optional[bool],
    # End-to-end table flags
    use_e2e_wired_table_rec_model: Optional[bool],
    use_e2e_wireless_table_rec_model: Optional[bool],
    # Device/accel
    device: Optional[str],
    enable_mkldnn: Optional[bool],
    enable_hpi: Optional[bool],
    use_tensorrt: Optional[bool],
    precision: Optional[str],
    mkldnn_cache_capacity: Optional[int],
    cpu_threads: Optional[int],
    # PaddleX pipeline config path (advanced)
    paddlex_config: Optional[str],
) -> Dict[str, Any]:
    # Only include non-None overrides
    items = locals()
    kwargs: Dict[str, Any] = {}
    for k, v in items.items():
        if k == "kwargs":
            continue
        if v is not None:
            # Force CPU-only even if a caller passes GPU by mistake
            if k == "device":
                v = "cpu"
            kwargs[k] = v
    return kwargs


@app.post(
    "/parse",
    responses={
        200: {"description": "PP-StructureV3 inference result in JSON and/or Markdown"},
        400: {"description": "Bad request"},
        413: {"description": "Payload too large"},
        415: {"description": "Unsupported media type"},
        500: {"description": "Internal server error"},
    },
)
async def parse(
    # Input
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Query(default=None, description="Image/PDF HTTP(S) URL"),
    # Output control
    output_format: Literal["json", "markdown", "both"] = Query(default="json"),
    concat_pdf_to_single_markdown: bool = Query(default=True, description="If input is a multi-page PDF"),
    inline_markdown_images: bool = Query(default=False, description="Embed images in Markdown as base64 data URIs"),
    # Model overrides (names/dirs)
    layout_detection_model_name: Optional[str] = Query(default=None),
    layout_detection_model_dir: Optional[str] = Query(default=None),
    region_detection_model_name: Optional[str] = Query(default=None),
    region_detection_model_dir: Optional[str] = Query(default=None),
    doc_orientation_classify_model_name: Optional[str] = Query(default=None),
    doc_orientation_classify_model_dir: Optional[str] = Query(default=None),
    doc_unwarping_model_name: Optional[str] = Query(default=None),
    doc_unwarping_model_dir: Optional[str] = Query(default=None),
    text_detection_model_name: Optional[str] = Query(default=None),
    text_detection_model_dir: Optional[str] = Query(default=None),
    textline_orientation_model_name: Optional[str] = Query(default=None),
    textline_orientation_model_dir: Optional[str] = Query(default=None),
    text_recognition_model_name: Optional[str] = Query(default=None),
    text_recognition_model_dir: Optional[str] = Query(default=None),
    table_classification_model_name: Optional[str] = Query(default=None),
    table_classification_model_dir: Optional[str] = Query(default=None),
    wired_table_structure_recognition_model_name: Optional[str] = Query(default=None),
    wired_table_structure_recognition_model_dir: Optional[str] = Query(default=None),
    wireless_table_structure_recognition_model_name: Optional[str] = Query(default=None),
    wireless_table_structure_recognition_model_dir: Optional[str] = Query(default=None),
    wired_table_cells_detection_model_name: Optional[str] = Query(default=None),
    wired_table_cells_detection_model_dir: Optional[str] = Query(default=None),
    wireless_table_cells_detection_model_name: Optional[str] = Query(default=None),
    wireless_table_cells_detection_model_dir: Optional[str] = Query(default=None),
    table_orientation_classify_model_name: Optional[str] = Query(default=None),
    table_orientation_classify_model_dir: Optional[str] = Query(default=None),
    seal_text_detection_model_name: Optional[str] = Query(default=None),
    seal_text_detection_model_dir: Optional[str] = Query(default=None),
    seal_text_recognition_model_name: Optional[str] = Query(default=None),
    seal_text_recognition_model_dir: Optional[str] = Query(default=None),
    formula_recognition_model_name: Optional[str] = Query(default=None),
    formula_recognition_model_dir: Optional[str] = Query(default=None),
    chart_recognition_model_name: Optional[str] = Query(default=None),
    chart_recognition_model_dir: Optional[str] = Query(default=None),
    # Numeric thresholds and sizes
    layout_threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    layout_nms: Optional[bool] = Query(default=None),
    layout_unclip_ratio: Optional[float] = Query(default=None, gt=0.0),
    layout_merge_bboxes_mode: Optional[str] = Query(default=None),
    text_det_limit_side_len: Optional[int] = Query(default=None, gt=0),
    text_det_limit_type: Optional[str] = Query(default=None),
    text_det_thresh: Optional[float] = Query(default=None, gt=0.0),
    text_det_box_thresh: Optional[float] = Query(default=None, gt=0.0),
    text_det_unclip_ratio: Optional[float] = Query(default=None, gt=0.0),
    textline_orientation_batch_size: Optional[int] = Query(default=None, gt=0),
    text_recognition_batch_size: Optional[int] = Query(default=None, gt=0),
    text_rec_score_thresh: Optional[float] = Query(default=None, ge=0.0),
    chart_recognition_batch_size: Optional[int] = Query(default=None, gt=0),
    formula_recognition_batch_size: Optional[int] = Query(default=None, gt=0),
    seal_det_limit_side_len: Optional[int] = Query(default=None, gt=0),
    seal_det_limit_type: Optional[str] = Query(default=None),
    seal_det_thresh: Optional[float] = Query(default=None, gt=0.0),
    seal_det_box_thresh: Optional[float] = Query(default=None, gt=0.0),
    seal_det_unclip_ratio: Optional[float] = Query(default=None, gt=0.0),
    seal_text_recognition_batch_size: Optional[int] = Query(default=None, gt=0),
    seal_rec_score_thresh: Optional[float] = Query(default=None, ge=0.0),
    # Subpipeline toggles
    use_doc_orientation_classify: Optional[bool] = Query(default=None),
    use_doc_unwarping: Optional[bool] = Query(default=None),
    use_textline_orientation: Optional[bool] = Query(default=None),
    use_region_detection: Optional[bool] = Query(default=None),
    use_table_recognition: Optional[bool] = Query(default=None),
    use_formula_recognition: Optional[bool] = Query(default=None),
    use_chart_recognition: Optional[bool] = Query(default=None),
    use_seal_recognition: Optional[bool] = Query(default=None),
    # End-to-end table structure switches
    use_e2e_wired_table_rec_model: Optional[bool] = Query(default=None),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(default=None),
    # Device/accel (will be coerced to CPU)
    device: Optional[str] = Query(default=None),
    enable_mkldnn: Optional[bool] = Query(default=None),
    enable_hpi: Optional[bool] = Query(default=None),
    use_tensorrt: Optional[bool] = Query(default=None),
    precision: Optional[str] = Query(default=None),
    mkldnn_cache_capacity: Optional[int] = Query(default=None, ge=0),
    cpu_threads: Optional[int] = Query(default=None, ge=1),
    paddlex_config: Optional[str] = Query(default=None),
):
    # Validate input choice
    if (file is None) and (not url):
        raise HTTPException(status_code=400, detail="Provide either a file or a url")

    tmp_path: Optional[Path] = None
    input_ref: str
    try:
        if file is not None:
            tmp_path = _save_upload_to_tmp(file)
            input_ref = str(tmp_path)
        else:
            # URL path accepted directly by pipeline
            input_ref = url  # type: ignore

        predict_kwargs = _gather_predict_kwargs(
            layout_detection_model_name, layout_detection_model_dir,
            region_detection_model_name, region_detection_model_dir,
            doc_orientation_classify_model_name, doc_orientation_classify_model_dir,
            doc_unwarping_model_name, doc_unwarping_model_dir,
            text_detection_model_name, text_detection_model_dir,
            textline_orientation_model_name, textline_orientation_model_dir,
            text_recognition_model_name, text_recognition_model_dir,
            table_classification_model_name, table_classification_model_dir,
            wired_table_structure_recognition_model_name, wired_table_structure_recognition_model_dir,
            wireless_table_structure_recognition_model_name, wireless_table_structure_recognition_model_dir,
            wired_table_cells_detection_model_name, wired_table_cells_detection_model_dir,
            wireless_table_cells_detection_model_name, wireless_table_cells_detection_model_dir,
            table_orientation_classify_model_name, table_orientation_classify_model_dir,
            seal_text_detection_model_name, seal_text_detection_model_dir,
            seal_text_recognition_model_name, seal_text_recognition_model_dir,
            formula_recognition_model_name, formula_recognition_model_dir,
            chart_recognition_model_name, chart_recognition_model_dir,
            layout_threshold, layout_nms, layout_unclip_ratio, layout_merge_bboxes_mode,
            text_det_limit_side_len, text_det_limit_type, text_det_thresh, text_det_box_thresh, text_det_unclip_ratio,
            textline_orientation_batch_size, text_recognition_batch_size, text_rec_score_thresh,
            chart_recognition_batch_size, formula_recognition_batch_size,
            seal_det_limit_side_len, seal_det_limit_type, seal_det_thresh, seal_det_box_thresh, seal_det_unclip_ratio,
            seal_text_recognition_batch_size, seal_rec_score_thresh,
            use_doc_orientation_classify, use_doc_unwarping, use_textline_orientation,
            use_region_detection, use_table_recognition, use_formula_recognition,
            use_chart_recognition, use_seal_recognition,
            use_e2e_wired_table_rec_model, use_e2e_wireless_table_rec_model,
            device, enable_mkldnn, enable_hpi, use_tensorrt, precision, mkldnn_cache_capacity, cpu_threads,
            paddlex_config,
        )

        # Run the pipeline inside a concurrency gate (thread pool offload)
        async with app.state.predict_sem:
            output = await run_in_threadpool(app.state.pipeline.predict, input_ref, **predict_kwargs)

        # Collect results
        json_results: List[Dict[str, Any]] = []
        md_info_list: List[Dict[str, Any]] = []

        for res in output:
            # JSON result: try the documented attribute .json; fallback to dict-like if present
            if hasattr(res, "json"):
                json_results.append(res.json)  # type: ignore
            elif hasattr(res, "res"):
                json_results.append(res.res)  # type: ignore
            else:
                # Last resort: try a generic serializer
                json_results.append(json.loads(json.dumps(res, default=lambda o: getattr(o, "__dict__", str(o)))))

            # Markdown info dict for concatenation and images
            if hasattr(res, "markdown"):
                md_info_list.append(res.markdown)  # type: ignore

        # Prepare outputs
        if output_format == "json":
            return JSONResponse(content={"results": json_results})

        # Build Markdown text (single image or PDF pages)
        combined_md_text = ""
        markdown_images_out: Dict[str, str] = {}
        if md_info_list:
            if concat_pdf_to_single_markdown and hasattr(app.state.pipeline, "concatenate_markdown_pages"):
                combined_md_text = app.state.pipeline.concatenate_markdown_pages(md_info_list)
            else:
                # Fallback: join pages with a page break
                parts = []
                for mdi in md_info_list:
                    # compatible key names: try common ones
                    if isinstance(mdi, dict):
                        txt = mdi.get("markdown") or mdi.get("markdown_text") or ""
                    else:
                        txt = str(mdi)
                    parts.append(txt)
                combined_md_text = "\n\n---\n\n".join(parts)

            # Collect images
            if inline_markdown_images:
                # Replace image paths with embedded data URIs if available
                # Also return a mapping for client convenience
                for mdi in md_info_list:
                    if isinstance(mdi, dict):
                        imgs = mdi.get("markdown_images") or {}
                        for path, pil_img in imgs.items():
                            b64 = _encode_pil_to_b64(pil_img)
                            markdown_images_out[path] = f"data:image/png;base64,{b64}"
                # Caller can search and replace the image paths if needed; leaving original paths in the MD text

        if output_format == "markdown":
            return PlainTextResponse(content=combined_md_text, media_type="text/markdown")

        # both
        return JSONResponse(
            content={
                "results": json_results,
                "markdown": combined_md_text,
                "markdown_images": markdown_images_out if inline_markdown_images else {},
            }
        )
    finally:
        if file is not None:
            try:
                file.file.close()
            except Exception:
                pass
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

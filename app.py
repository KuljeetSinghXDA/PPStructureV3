import io
import os
import json
import shutil
import tempfile
import inspect
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, UploadFile, File, Body, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# PaddleOCR PP-StructureV3
from paddleocr import PPStructureV3  # installed in container


app = FastAPI(
    title="PP-StructureV3 (CPU, arm64) Service",
    version="1.2.0",
    description="FastAPI wrapper around PaddleOCR PP-StructureV3 with full parameter exposure, JSON and Markdown outputs, and PDF Markdown concatenation."
)

# -------------------------
# Options model (expose all documented knobs; defaults chosen for CPU + med-lab accuracy)
# All fields are Optional; when None, we do not pass them so native defaults apply.
# -------------------------
class ParseOptions(BaseModel):
    # General behavior
    return_json: bool = True
    return_markdown: bool = True
    concatenate_markdown: bool = True  # Single MD for whole PDF
    save_dir: Optional[str] = None

    # Device & perf
    device: str = "cpu"
    enable_mkldnn: Optional[bool] = None
    mkldnn_cache_capacity: Optional[int] = None
    cpu_threads: Optional[int] = 4

    # Language
    lang: Optional[str] = "en"

    # Module toggles (optional sub-pipelines)
    use_doc_preprocessor: Optional[bool] = None
    use_general_ocr: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_table_recognition: Optional[bool] = None
    use_formula_recognition: Optional[bool] = None
    use_chart_parsing: Optional[bool] = None

    # Preprocess toggles commonly recommended in 3.x
    use_doc_orientation_classify: Optional[bool] = False
    use_doc_unwarping: Optional[bool] = False
    # Improves rotated text lines; good for medical forms with rotated headers/sideways labels
    use_textline_orientation: Optional[bool] = True

    # Layout model selection + tuning
    layout_detection_model_name: Optional[str] = "PP-DocLayout-L"
    layout_detection_model_dir: Optional[str] = None
    layout_threshold: Optional[Union[float, Dict[int, float]]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[float] = None
    layout_merge_bboxes_mode: Optional[str] = None  # e.g., "all", "text_and_table"

    # Text detection tuning (DB-like params in docs)
    text_detection_model_name: Optional[str] = "PP-OCRv5_mobile_det"
    text_detection_model_dir: Optional[str] = None
    text_det_limit_side_len: Optional[int] = None
    text_det_db_thresh: Optional[float] = None
    text_det_db_box_thresh: Optional[float] = None
    text_det_db_unclip_ratio: Optional[float] = None
    text_det_db_score_mode: Optional[str] = None  # "slow"/"fast" if supported

    # Text recognition model selection + tuning
    text_recognition_model_name: Optional[str] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_dir: Optional[str] = None
    text_rec_score_thresh: Optional[float] = None

    # Table, formula, seal, chart model names/dirs (exposed but left None to use defaults)
    table_cell_det_model_name: Optional[str] = None
    table_cell_det_model_dir: Optional[str] = None
    table_structure_model_name: Optional[str] = None
    table_structure_model_dir: Optional[str] = None
    table_classifier_model_name: Optional[str] = None
    table_classifier_model_dir: Optional[str] = None

    formula_recognition_model_name: Optional[str] = None
    formula_recognition_model_dir: Optional[str] = None

    seal_recognition_model_name: Optional[str] = None
    seal_recognition_model_dir: Optional[str] = None

    chart_parsing_model_name: Optional[str] = None
    chart_parsing_model_dir: Optional[str] = None

    # PDF control (predict-level options sometimes exposed)
    page_num: Optional[int] = None  # limit pages for PDF if supported

    # Extra predict-time saving path (we save separately; this is only passed if supported)
    save_path: Optional[str] = None


# -------------------------
# Pipeline cache with safe arg filtering via introspection
# -------------------------
from functools import lru_cache

def _opts_to_init_kwargs(opts: ParseOptions) -> Dict[str, Any]:
    # Build kwargs only with parameters supported by current PPStructureV3.__init__
    sig = inspect.signature(PPStructureV3.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "args", "kwargs"}

    candidate: Dict[str, Any] = {
        # Device/lang
        "device": opts.device,
        "lang": opts.lang,
        "enable_mkldnn": opts.enable_mkldnn,
        "mkldnn_cache_capacity": opts.mkldnn_cache_capacity,
        "cpu_threads": opts.cpu_threads,
        # Subpipelines
        "use_doc_preprocessor": opts.use_doc_preprocessor,
        "use_general_ocr": opts.use_general_ocr,
        "use_seal_recognition": opts.use_seal_recognition,
        "use_table_recognition": opts.use_table_recognition,
        "use_formula_recognition": opts.use_formula_recognition,
        "use_chart_parsing": opts.use_chart_parsing,
        # Preprocess toggles
        "use_doc_orientation_classify": opts.use_doc_orientation_classify,
        "use_doc_unwarping": opts.use_doc_unwarping,
        "use_textline_orientation": opts.use_textline_orientation,
        # Layout
        "layout_detection_model_name": opts.layout_detection_model_name,
        "layout_detection_model_dir": opts.layout_detection_model_dir,
        "layout_threshold": opts.layout_threshold,
        "layout_nms": opts.layout_nms,
        "layout_unclip_ratio": opts.layout_unclip_ratio,
        "layout_merge_bboxes_mode": opts.layout_merge_bboxes_mode,
        # Text det
        "text_detection_model_name": opts.text_detection_model_name,
        "text_detection_model_dir": opts.text_detection_model_dir,
        "text_det_limit_side_len": opts.text_det_limit_side_len,
        "text_det_db_thresh": opts.text_det_db_thresh,
        "text_det_db_box_thresh": opts.text_det_db_box_thresh,
        "text_det_db_unclip_ratio": opts.text_det_db_unclip_ratio,
        "text_det_db_score_mode": opts.text_det_db_score_mode,
        # Text rec
        "text_recognition_model_name": opts.text_recognition_model_name,
        "text_recognition_model_dir": opts.text_recognition_model_dir,
        "text_rec_score_thresh": opts.text_rec_score_thresh,
        # Table
        "table_cell_det_model_name": opts.table_cell_det_model_name,
        "table_cell_det_model_dir": opts.table_cell_det_model_dir,
        "table_structure_model_name": opts.table_structure_model_name,
        "table_structure_model_dir": opts.table_structure_model_dir,
        "table_classifier_model_name": opts.table_classifier_model_name,
        "table_classifier_model_dir": opts.table_classifier_model_dir,
        # Formula
        "formula_recognition_model_name": opts.formula_recognition_model_name,
        "formula_recognition_model_dir": opts.formula_recognition_model_dir,
        # Seal
        "seal_recognition_model_name": opts.seal_recognition_model_name,
        "seal_recognition_model_dir": opts.seal_recognition_model_dir,
        # Chart
        "chart_parsing_model_name": opts.chart_parsing_model_name,
        "chart_parsing_model_dir": opts.chart_parsing_model_dir,
    }

    # Drop None and keys not supported by the installed version
    init_kwargs = {k: v for k, v in candidate.items() if (v is not None and k in allowed)}
    return init_kwargs

def _opts_to_predict_kwargs(opts: ParseOptions, save_dir_for_native: Optional[str]) -> Dict[str, Any]:
    # Pass only predict() parameters supported by installed version
    sig = inspect.signature(PPStructureV3.predict)
    allowed = set(sig.parameters.keys()) - {"self"}

    candidate: Dict[str, Any] = {
        "page_num": opts.page_num,
        "save_path": opts.save_path or save_dir_for_native,
        # We do not pass 'input' here; it is positional/keyword in call.
    }
    predict_kwargs = {k: v for k, v in candidate.items() if (v is not None and k in allowed)}
    return predict_kwargs

def _pipeline_cache_key(opts: ParseOptions) -> tuple:
    # Use fields that influence weights/graph; excludes return_json, return_markdown, concatenate_markdown, save_dir
    return tuple([
        opts.device, opts.lang,
        opts.enable_mkldnn, opts.mkldnn_cache_capacity, opts.cpu_threads,
        opts.use_doc_preprocessor, opts.use_general_ocr, opts.use_seal_recognition,
        opts.use_table_recognition, opts.use_formula_recognition, opts.use_chart_parsing,
        opts.use_doc_orientation_classify, opts.use_doc_unwarping, opts.use_textline_orientation,
        opts.layout_detection_model_name, opts.layout_detection_model_dir,
        json.dumps(opts.layout_threshold, sort_keys=True) if isinstance(opts.layout_threshold, dict) else opts.layout_threshold,
        opts.layout_nms, opts.layout_unclip_ratio, opts.layout_merge_bboxes_mode,
        opts.text_detection_model_name, opts.text_detection_model_dir,
        opts.text_det_limit_side_len, opts.text_det_db_thresh, opts.text_det_db_box_thresh,
        opts.text_det_db_unclip_ratio, opts.text_det_db_score_mode,
        opts.text_recognition_model_name, opts.text_recognition_model_dir, opts.text_rec_score_thresh,
        opts.table_cell_det_model_name, opts.table_cell_det_model_dir,
        opts.table_structure_model_name, opts.table_structure_model_dir,
        opts.table_classifier_model_name, opts.table_classifier_model_dir,
        opts.formula_recognition_model_name, opts.formula_recognition_model_dir,
        opts.seal_recognition_model_name, opts.seal_recognition_model_dir,
        opts.chart_parsing_model_name, opts.chart_parsing_model_dir,
    ])

@lru_cache(maxsize=8)
def get_pipeline_cached(key: tuple, init_kwargs_json: str) -> PPStructureV3:
    init_kwargs = json.loads(init_kwargs_json)
    return PPStructureV3(**init_kwargs)

def get_pipeline(opts: ParseOptions) -> PPStructureV3:
    init_kwargs = _opts_to_init_kwargs(opts)
    return get_pipeline_cached(_pipeline_cache_key(opts), json.dumps(init_kwargs))


# -------------------------
# Startup: preload default models and warm up to avoid first-request latency
# -------------------------
DEFAULT_OPTS = ParseOptions()

def _warmup(pipeline: PPStructureV3) -> None:
    # Small white image warmup triggers model download and graph init without heavy compute.
    import numpy as np
    tiny = (255 * np.ones((64, 64, 3), dtype=np.uint8))
    try:
        _ = pipeline.predict(input=tiny)
    except Exception:
        # Some older versions expect path; fallback to writing a temp PNG.
        import cv2, tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            cv2.imwrite(tf.name, tiny)
            try:
                _ = pipeline.predict(input=tf.name)
            finally:
                try:
                    os.remove(tf.name)
                except Exception:
                    pass

@app.on_event("startup")
def on_startup():
    pipe = get_pipeline(DEFAULT_OPTS)
    _warmup(pipe)


# -------------------------
# Health
# -------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# -------------------------
# /parse endpoint
# - Accepts multiple files.
# - Options can be JSON body or multipart form field 'options' containing JSON string.
# - Saves native JSON/Markdown via res.save_to_* and also returns them.
# - Concatenates multi-page PDF markdown if requested.
# -------------------------
@app.post("/parse")
async def parse(
    files: List[UploadFile] = File(..., description="One or more images or PDFs"),
    # Either send options as raw JSON body...
    options_body: Optional[ParseOptions] = Body(default=None),
    # ...or as a multipart field 'options' with JSON text (for curl -F)
    options_form: Optional[str] = Form(default=None),
    # Convenience query overrides for simple calls
    return_json: Optional[bool] = Query(default=None),
    return_markdown: Optional[bool] = Query(default=None),
    concatenate_markdown: Optional[bool] = Query(default=None)
):
    # Merge options
    if options_body is not None:
        opts = options_body
    elif options_form:
        opts = ParseOptions(**json.loads(options_form))
    else:
        opts = DEFAULT_OPTS

    # Apply query overrides if present
    if return_json is not None:
        opts.return_json = return_json
    if return_markdown is not None:
        opts.return_markdown = return_markdown
    if concatenate_markdown is not None:
        opts.concatenate_markdown = concatenate_markdown

    # Prepare save_dir
    if opts.save_dir:
        os.makedirs(opts.save_dir, exist_ok=True)

    pipeline = get_pipeline(opts)

    overall: Dict[str, Any] = {
        "engine": "PP-StructureV3",
        "device": opts.device,
        "results": []
    }

    with tempfile.TemporaryDirectory(prefix="ppstructv3_req_") as req_tmpdir:
        for uf in files:
            original_name = uf.filename or "upload"
            base, ext = os.path.splitext(original_name)
            file_tmpdir = tempfile.mkdtemp(prefix="file_", dir=req_tmpdir)
            tmp_path = os.path.join(file_tmpdir, f"input{ext or ''}")

            # Persist the upload
            content = await uf.read()
            with open(tmp_path, "wb") as f:
                f.write(content)

            # Native outputs dir
            native_out_dir = os.path.join(file_tmpdir, "native")
            os.makedirs(native_out_dir, exist_ok=True)

            # Run predict with version-compatible kwargs
            predict_kwargs = _opts_to_predict_kwargs(opts, save_dir_for_native=native_out_dir)
            preds = pipeline.predict(input=tmp_path, **predict_kwargs)

            # Save per-page outputs and collect them back
            page_json: List[Dict[str, Any]] = []
            page_markdown: List[str] = []
            page_md_objs: List[Dict[str, Any]] = []  # res.markdown dicts for concatenation

            for res in preds:
                if opts.return_json:
                    try:
                        res.save_to_json(save_path=native_out_dir)
                    except Exception:
                        pass
                if opts.return_markdown:
                    try:
                        res.save_to_markdown(save_path=native_out_dir)
                    except Exception:
                        pass

                # Capture in-memory markdown objects for concatenation, if present
                md_obj = getattr(res, "markdown", None)
                if isinstance(md_obj, dict):
                    page_md_objs.append(md_obj)

            # Read saved JSON/MD files to include in response
            if opts.return_json:
                for name in sorted(os.listdir(native_out_dir)):
                    if name.lower().endswith(".json"):
                        with open(os.path.join(native_out_dir, name), "r", encoding="utf-8") as jf:
                            try:
                                page_json.append(json.load(jf))
                            except Exception:
                                page_json.append({"raw_json": jf.read()})

            if opts.return_markdown:
                for name in sorted(os.listdir(native_out_dir)):
                    if name.lower().endswith(".md"):
                        with open(os.path.join(native_out_dir, name), "r", encoding="utf-8") as mf:
                            page_markdown.append(mf.read())

            # Concatenate multi-page markdown using native API if requested/available
            markdown_combined: Optional[str] = None
            if opts.return_markdown and opts.concatenate_markdown and page_md_objs:
                try:
                    merged = pipeline.concatenate_markdown_pages(page_md_objs)
                    # merged may be dict with 'markdown_texts' or a plain string depending on version
                    if isinstance(merged, dict) and "markdown_texts" in merged:
                        markdown_combined = merged["markdown_texts"]
                    elif isinstance(merged, str):
                        markdown_combined = merged
                    else:
                        # Try common nested forms
                        markdown_combined = merged.get("markdown", None) if isinstance(merged, dict) else None

                    if markdown_combined:
                        combined_fname = f"{base}_combined.md"
                        combined_path = os.path.join(native_out_dir, combined_fname)
                        with open(combined_path, "w", encoding="utf-8") as cf:
                            cf.write(markdown_combined)
                except Exception:
                    # If the installed version lacks concatenate_markdown_pages, skip
                    pass

            # Persist outputs to user-provided save_dir
            persisted_dir = None
            if opts.save_dir:
                persisted_dir = os.path.join(opts.save_dir, base)
                os.makedirs(persisted_dir, exist_ok=True)
                for name in os.listdir(native_out_dir):
                    shutil.copy2(os.path.join(native_out_dir, name), os.path.join(persisted_dir, name))

            overall["results"].append({
                "filename": original_name,
                "pages": len(preds),
                "json": page_json if opts.return_json else None,
                "markdown_pages": page_markdown if opts.return_markdown else None,
                "markdown_combined": markdown_combined if opts.return_markdown else None,
                "saved_to": persisted_dir
            })

    return JSONResponse(overall)


@app.get("/")
def index():
    return {
        "name": "PP-StructureV3 (FastAPI)",
        "defaults": {
            "layout_detection_model_name": "PP-DocLayout-L",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "use_textline_orientation": True
        },
        "endpoints": {
            "POST /parse": "Upload multi-file (images or PDFs). Returns JSON + Markdown; concatenates PDF Markdown."
        },
        "notes": [
            "All parameters documented for PP-StructureV3 are exposed and passed only if supported by your installed version.",
            "Set save_dir to persist native files; otherwise, they’re returned inline."
        ]
    }

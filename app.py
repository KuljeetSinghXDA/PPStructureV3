import os
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# PP-StructureV3 pipeline
from paddleocr import PPStructureV3  # type: ignore

app = FastAPI(
    title="PP-StructureV3 (CPU arm64) Service",
    version="1.1.0",
    description="FastAPI wrapper for PaddleOCR PP-StructureV3 with full parameter exposure, JSON + Markdown outputs, and native PDF Markdown concatenation."
)

# -----------------------------
# Options: expose every parameter in docs
# -----------------------------
# Notes:
# - We pass model/threshold knobs at initialization (native behavior).
# - At predict() time we pass runtime toggles/acceleration knobs that the docs allow.
# - Defaults are chosen to be “native” unless marked as tuned for lab reports.


class ParseOptions(BaseModel):
    # Output controls
    return_json: bool = True
    return_markdown: bool = True
    concat_markdown: bool = True  # single Markdown for whole PDF via native concatenate_markdown_pages()
    save_dir: Optional[str] = Field(default=None, description="If set, native JSON/MD files (and combined PDF MD) are saved here.")

    # -----------------------------
    # Initialization parameters (constructor) — full list
    # -----------------------------
    # Layout
    layout_detection_model_name: Optional[str] = Field(default="PP-DocLayout-L")  # requested default
    layout_detection_model_dir: Optional[str] = None
    layout_threshold: Optional[Union[float, Dict[int, float]]] = Field(default=None)  # you can use class-wise dict
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[float] = None
    layout_merge_bboxes_mode: Optional[str] = None  # large | small | union

    # Chart
    chart_recognition_model_name: Optional[str] = None
    chart_recognition_model_dir: Optional[str] = None
    chart_recognition_batch_size: Optional[int] = None

    # Region detection
    region_detection_model_name: Optional[str] = None
    region_detection_model_dir: Optional[str] = None

    # Doc preprocessor
    doc_orientation_classify_model_name: Optional[str] = None
    doc_orientation_classify_model_dir: Optional[str] = None
    doc_unwarping_model_name: Optional[str] = None
    doc_unwarping_model_dir: Optional[str] = None

    # Text detection (requested default model set below)
    text_detection_model_name: Optional[str] = Field(default="PP-OCRv5_mobile_det")
    text_detection_model_dir: Optional[str] = None
    text_det_limit_side_len: Optional[int] = Field(default=1536, description="Tuned: enlarge for small-font clinical tables.")
    text_det_limit_type: Optional[str] = None  # min | max
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None

    # Text line orientation
    textline_orientation_model_name: Optional[str] = None
    textline_orientation_model_dir: Optional[str] = None
    textline_orientation_batch_size: Optional[int] = None

    # Text recognition (requested default English model)
    text_recognition_model_name: Optional[str] = Field(default="en_PP-OCRv5_mobile_rec")
    text_recognition_model_dir: Optional[str] = None
    text_recognition_batch_size: Optional[int] = None
    text_rec_score_thresh: Optional[float] = Field(default=0.4, description="Tuned: filter noisy reads in lab tables.")

    # Table
    table_classification_model_name: Optional[str] = None
    table_classification_model_dir: Optional[str] = None
    wired_table_structure_recognition_model_name: Optional[str] = None
    wired_table_structure_recognition_model_dir: Optional[str] = None
    wireless_table_structure_recognition_model_name: Optional[str] = None
    wireless_table_structure_recognition_model_dir: Optional[str] = None
    wired_table_cells_detection_model_name: Optional[str] = None
    wired_table_cells_detection_model_dir: Optional[str] = None
    wireless_table_cells_detection_model_name: Optional[str] = None
    wireless_table_cells_detection_model_dir: Optional[str] = None
    table_orientation_classify_model_name: Optional[str] = None
    table_orientation_classify_model_dir: Optional[str] = None

    # Seal
    seal_text_detection_model_name: Optional[str] = None
    seal_text_detection_model_dir: Optional[str] = None
    seal_det_limit_side_len: Optional[int] = None
    seal_det_limit_type: Optional[str] = None
    seal_det_thresh: Optional[float] = None
    seal_det_box_thresh: Optional[float] = None
    seal_det_unclip_ratio: Optional[float] = None
    seal_text_recognition_model_name: Optional[str] = None
    seal_text_recognition_model_dir: Optional[str] = None
    seal_text_recognition_batch_size: Optional[int] = None
    seal_rec_score_thresh: Optional[float] = None

    # Formula
    formula_recognition_model_name: Optional[str] = None
    formula_recognition_model_dir: Optional[str] = None
    formula_recognition_batch_size: Optional[int] = None

    # Pipeline submodules toggles (init)
    use_doc_orientation_classify: Optional[bool] = Field(default=True)     # tuned: enable to fix rotated scans
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = Field(default=True)         # tuned: helps rotated narrow lines in tables
    use_seal_recognition: Optional[bool] = None
    use_table_recognition: Optional[bool] = None
    use_formula_recognition: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_region_detection: Optional[bool] = None

    # Device/acceleration (init)
    device: Optional[str] = Field(default="cpu")
    enable_hpi: Optional[bool] = None
    use_tensorrt: Optional[bool] = None
    precision: Optional[str] = None
    enable_mkldnn: Optional[bool] = None   # default True in library; we keep None to leave native default
    mkldnn_cache_capacity: Optional[int] = None
    cpu_threads: Optional[int] = None
    paddlex_config: Optional[str] = None

    # -----------------------------
    # Predict-time overrides (docs-supported)
    # -----------------------------
    pred_use_doc_orientation_classify: Optional[bool] = None
    pred_use_doc_unwarping: Optional[bool] = None
    pred_use_textline_orientation: Optional[bool] = None
    pred_use_seal_recognition: Optional[bool] = None
    pred_use_table_recognition: Optional[bool] = None
    pred_use_formula_recognition: Optional[bool] = None
    pred_use_chart_recognition: Optional[bool] = None
    pred_use_region_detection: Optional[bool] = None
    pred_device: Optional[str] = None
    pred_enable_hpi: Optional[bool] = None
    pred_use_tensorrt: Optional[bool] = None
    pred_precision: Optional[str] = None
    pred_enable_mkldnn: Optional[bool] = None
    pred_mkldnn_cache_capacity: Optional[int] = None
    pred_cpu_threads: Optional[int] = None


def _if_set(d: Dict[str, Any], k: str, v: Any):
    if v is not None:
        d[k] = v


def build_init_kwargs(opts: ParseOptions) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}

    # Layout
    _if_set(kw, "layout_detection_model_name", opts.layout_detection_model_name)
    _if_set(kw, "layout_detection_model_dir", opts.layout_detection_model_dir)
    _if_set(kw, "layout_threshold", opts.layout_threshold)
    _if_set(kw, "layout_nms", opts.layout_nms)
    _if_set(kw, "layout_unclip_ratio", opts.layout_unclip_ratio)
    _if_set(kw, "layout_merge_bboxes_mode", opts.layout_merge_bboxes_mode)

    # Chart
    _if_set(kw, "chart_recognition_model_name", opts.chart_recognition_model_name)
    _if_set(kw, "chart_recognition_model_dir", opts.chart_recognition_model_dir)
    _if_set(kw, "chart_recognition_batch_size", opts.chart_recognition_batch_size)

    # Region detection
    _if_set(kw, "region_detection_model_name", opts.region_detection_model_name)
    _if_set(kw, "region_detection_model_dir", opts.region_detection_model_dir)

    # Doc preprocessor
    _if_set(kw, "doc_orientation_classify_model_name", opts.doc_orientation_classify_model_name)
    _if_set(kw, "doc_orientation_classify_model_dir", opts.doc_orientation_classify_model_dir)
    _if_set(kw, "doc_unwarping_model_name", opts.doc_unwarping_model_name)
    _if_set(kw, "doc_unwarping_model_dir", opts.doc_unwarping_model_dir)

    # Text det
    _if_set(kw, "text_detection_model_name", opts.text_detection_model_name)
    _if_set(kw, "text_detection_model_dir", opts.text_detection_model_dir)
    _if_set(kw, "text_det_limit_side_len", opts.text_det_limit_side_len)
    _if_set(kw, "text_det_limit_type", opts.text_det_limit_type)
    _if_set(kw, "text_det_thresh", opts.text_det_thresh)
    _if_set(kw, "text_det_box_thresh", opts.text_det_box_thresh)
    _if_set(kw, "text_det_unclip_ratio", opts.text_det_unclip_ratio)

    # Text line orientation
    _if_set(kw, "textline_orientation_model_name", opts.textline_orientation_model_name)
    _if_set(kw, "textline_orientation_model_dir", opts.textline_orientation_model_dir)
    _if_set(kw, "textline_orientation_batch_size", opts.textline_orientation_batch_size)

    # Text rec
    _if_set(kw, "text_recognition_model_name", opts.text_recognition_model_name)
    _if_set(kw, "text_recognition_model_dir", opts.text_recognition_model_dir)
    _if_set(kw, "text_recognition_batch_size", opts.text_recognition_batch_size)
    _if_set(kw, "text_rec_score_thresh", opts.text_rec_score_thresh)

    # Table
    _if_set(kw, "table_classification_model_name", opts.table_classification_model_name)
    _if_set(kw, "table_classification_model_dir", opts.table_classification_model_dir)
    _if_set(kw, "wired_table_structure_recognition_model_name", opts.wired_table_structure_recognition_model_name)
    _if_set(kw, "wired_table_structure_recognition_model_dir", opts.wired_table_structure_recognition_model_dir)
    _if_set(kw, "wireless_table_structure_recognition_model_name", opts.wireless_table_structure_recognition_model_name)
    _if_set(kw, "wireless_table_structure_recognition_model_dir", opts.wireless_table_structure_recognition_model_dir)
    _if_set(kw, "wired_table_cells_detection_model_name", opts.wired_table_cells_detection_model_name)
    _if_set(kw, "wired_table_cells_detection_model_dir", opts.wired_table_cells_detection_model_dir)
    _if_set(kw, "wireless_table_cells_detection_model_name", opts.wireless_table_cells_detection_model_name)
    _if_set(kw, "wireless_table_cells_detection_model_dir", opts.wireless_table_cells_detection_model_dir)
    _if_set(kw, "table_orientation_classify_model_name", opts.table_orientation_classify_model_name)
    _if_set(kw, "table_orientation_classify_model_dir", opts.table_orientation_classify_model_dir)

    # Seal
    _if_set(kw, "seal_text_detection_model_name", opts.seal_text_detection_model_name)
    _if_set(kw, "seal_text_detection_model_dir", opts.seal_text_detection_model_dir)
    _if_set(kw, "seal_det_limit_side_len", opts.seal_det_limit_side_len)
    _if_set(kw, "seal_det_limit_type", opts.seal_det_limit_type)
    _if_set(kw, "seal_det_thresh", opts.seal_det_thresh)
    _if_set(kw, "seal_det_box_thresh", opts.seal_det_box_thresh)
    _if_set(kw, "seal_det_unclip_ratio", opts.seal_det_unclip_ratio)
    _if_set(kw, "seal_text_recognition_model_name", opts.seal_text_recognition_model_name)
    _if_set(kw, "seal_text_recognition_model_dir", opts.seal_text_recognition_model_dir)
    _if_set(kw, "seal_text_recognition_batch_size", opts.seal_text_recognition_batch_size)
    _if_set(kw, "seal_rec_score_thresh", opts.seal_rec_score_thresh)

    # Formula
    _if_set(kw, "formula_recognition_model_name", opts.formula_recognition_model_name)
    _if_set(kw, "formula_recognition_model_dir", opts.formula_recognition_model_dir)
    _if_set(kw, "formula_recognition_batch_size", opts.formula_recognition_batch_size)

    # Submodules (init)
    _if_set(kw, "use_doc_orientation_classify", opts.use_doc_orientation_classify)
    _if_set(kw, "use_doc_unwarping", opts.use_doc_unwarping)
    _if_set(kw, "use_textline_orientation", opts.use_textline_orientation)
    _if_set(kw, "use_seal_recognition", opts.use_seal_recognition)
    _if_set(kw, "use_table_recognition", opts.use_table_recognition)
    _if_set(kw, "use_formula_recognition", opts.use_formula_recognition)
    _if_set(kw, "use_chart_recognition", opts.use_chart_recognition)
    _if_set(kw, "use_region_detection", opts.use_region_detection)

    # Device/accel (init)
    _if_set(kw, "device", opts.device)
    _if_set(kw, "enable_hpi", opts.enable_hpi)
    _if_set(kw, "use_tensorrt", opts.use_tensorrt)
    _if_set(kw, "precision", opts.precision)
    _if_set(kw, "enable_mkldnn", opts.enable_mkldnn)
    _if_set(kw, "mkldnn_cache_capacity", opts.mkldnn_cache_capacity)
    _if_set(kw, "cpu_threads", opts.cpu_threads)
    _if_set(kw, "paddlex_config", opts.paddlex_config)

    return kw


def build_predict_kwargs(opts: ParseOptions) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    _if_set(kw, "use_doc_orientation_classify", opts.pred_use_doc_orientation_classify)
    _if_set(kw, "use_doc_unwarping", opts.pred_use_doc_unwarping)
    _if_set(kw, "use_textline_orientation", opts.pred_use_textline_orientation)
    _if_set(kw, "use_seal_recognition", opts.pred_use_seal_recognition)
    _if_set(kw, "use_table_recognition", opts.pred_use_table_recognition)
    _if_set(kw, "use_formula_recognition", opts.pred_use_formula_recognition)
    _if_set(kw, "use_chart_recognition", opts.pred_use_chart_recognition)
    _if_set(kw, "use_region_detection", opts.pred_use_region_detection)
    _if_set(kw, "device", opts.pred_device)
    _if_set(kw, "enable_hpi", opts.pred_enable_hpi)
    _if_set(kw, "use_tensorrt", opts.pred_use_tensorrt)
    _if_set(kw, "precision", opts.pred_precision)
    _if_set(kw, "enable_mkldnn", opts.pred_enable_mkldnn)
    _if_set(kw, "mkldnn_cache_capacity", opts.pred_mkldnn_cache_capacity)
    _if_set(kw, "cpu_threads", opts.pred_cpu_threads)
    return kw


# Lightweight pipeline cache keyed by init kwargs (so weights aren’t reloaded each call)
from functools import lru_cache

def _pipeline_key(init_kwargs: Dict[str, Any]) -> tuple:
    # Make a tuple with stable order
    return tuple(sorted(init_kwargs.items(), key=lambda x: x[0]))

@lru_cache(maxsize=4)
def get_pipeline_cached(key: tuple) -> PPStructureV3:
    init_kwargs = dict(key)
    return PPStructureV3(**init_kwargs)  # type: ignore

def get_pipeline(opts: ParseOptions) -> PPStructureV3:
    init_kwargs = build_init_kwargs(opts)
    return get_pipeline_cached(_pipeline_key(init_kwargs))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/parse")
async def parse(
    files: List[UploadFile] = File(..., description="One or more images or PDFs."),
    options: ParseOptions = Body(default=ParseOptions()),
):
    pipeline = get_pipeline(options)
    predict_kwargs = build_predict_kwargs(options)

    overall = {
        "engine": "PP-StructureV3",
        "device": options.pred_device or options.device or "cpu",
        "results": []
    }

    if options.save_dir:
        os.makedirs(options.save_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ppstructv3_req_") as req_tmp:
        for uf in files:
            fname = uf.filename or "upload"
            stem = Path(fname).stem
            suffix = Path(fname).suffix
            work_dir = Path(req_tmp) / stem
            work_dir.mkdir(parents=True, exist_ok=True)
            input_path = work_dir / f"input{suffix}"
            with open(input_path, "wb") as f:
                f.write(await uf.read())

            # Run predict (native API)
            preds = pipeline.predict(str(input_path), **predict_kwargs)

            # Save native per-page outputs
            per_page_json: List[Dict[str, Any]] = []
            per_page_md: List[str] = []

            native_dir = work_dir / "native"
            native_dir.mkdir(parents=True, exist_ok=True)

            for res in preds:
                if options.return_json:
                    res.save_to_json(save_path=str(native_dir))
                if options.return_markdown:
                    res.save_to_markdown(save_path=str(native_dir))

            # Collect saved per-page outputs back into the API response
            if options.return_json:
                for name in sorted(os.listdir(native_dir)):
                    if name.lower().endswith(".json"):
                        with open(native_dir / name, "r", encoding="utf-8") as jf:
                            try:
                                per_page_json.append(json.load(jf))
                            except Exception:
                                per_page_json.append({"raw_json": jf.read()})

            if options.return_markdown:
                for name in sorted(os.listdir(native_dir)):
                    if name.lower().endswith(".md"):
                        with open(native_dir / name, "r", encoding="utf-8") as mf:
                            per_page_md.append(mf.read())

            # Native single-Markdown concatenation for PDFs (or multi-page images)
            combined_md_text = None
            combined_md_file = None
            if options.return_markdown and options.concat_markdown and len(preds) > 1:
                md_list = [res.markdown for res in preds]  # native markdown dicts
                combined_md_text = pipeline.concatenate_markdown_pages(md_list)
                if isinstance(combined_md_text, tuple):
                    # Some versions return (markdown_text, markdown_images); normalize
                    md_text, md_images = combined_md_text
                else:
                    md_text, md_images = combined_md_text, []
                combined_md_text = md_text

                # Persist combined Markdown (and images) if save_dir provided
                if options.save_dir:
                    out_dir = Path(options.save_dir) / stem
                    out_dir.mkdir(parents=True, exist_ok=True)
                    combined_md_file = out_dir / f"{stem}.md"
                    with open(combined_md_file, "w", encoding="utf-8") as f:
                        f.write(md_text)
                    # Save any page-level images that concatenation returns
                    if md_images:
                        for item in md_images:
                            if isinstance(item, dict):
                                for p, img in item.items():
                                    p = p.lstrip("/").lstrip("./")
                                    fp = out_dir / p
                                    fp.parent.mkdir(parents=True, exist_ok=True)
                                    img.save(fp)

            # Optionally persist per-page native files
            persisted_dir = None
            if options.save_dir:
                persisted_dir = str((Path(options.save_dir) / stem).resolve())
                os.makedirs(persisted_dir, exist_ok=True)
                for name in os.listdir(native_dir):
                    shutil.copy2(native_dir / name, Path(persisted_dir) / name)

            overall["results"].append({
                "filename": fname,
                "pages": len(preds),
                "json_pages": per_page_json if options.return_json else None,
                "markdown_pages": per_page_md if options.return_markdown else None,
                "markdown_combined": combined_md_text if options.return_markdown and options.concat_markdown else None,
                "saved_to": persisted_dir,
                "combined_markdown_file": str(combined_md_file) if combined_md_file else None
            })

    return JSONResponse(overall)


@app.get("/")
def index():
    return {
        "name": "PP-StructureV3 (FastAPI)",
        "endpoints": {
            "POST /parse": "Upload multiple files (images or PDFs) and get JSON + Markdown.",
            "GET /healthz": "Health check"
        }
    }

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3  # Native 3.x pipeline API

# -----------------------------------------------------------------------------
# Centralized configuration: all knobs for PP-StructureV3 live here.
# Only edit this block to change models/parameters/behavior.
# Parameter names and defaults follow the official PP-StructureV3 docs. 
# https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html
# -----------------------------------------------------------------------------

@dataclass
class PipelineInitConfig:
    # Language and device
    lang: Optional[str] = "en"             # English-centric for US medical lab reports
    device: Optional[str] = "cpu"          # Force CPU

    # Layout detection
    layout_detection_model_name: Optional[str] = "PP-DocLayout-L"
    layout_detection_model_dir: Optional[str] = None
    layout_threshold: Optional[Union[float, Dict[int, float]]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]]] = None
    layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None  # large|small|union

    # Region detection (doc regions for submodules)
    region_detection_model_name: Optional[str] = None
    region_detection_model_dir: Optional[str] = None

    # Doc preprocessor (orientation/unwarping) [left at defaults per requirement]
    doc_orientation_classify_model_name: Optional[str] = None
    doc_orientation_classify_model_dir: Optional[str] = None
    doc_unwarping_model_name: Optional[str] = None
    doc_unwarping_model_dir: Optional[str] = None

    # General OCR subline (text detection + recognition)
    text_detection_model_name: Optional[str] = "PP-OCRv5_mobile_det"
    text_detection_model_dir: Optional[str] = None
    text_det_limit_side_len: Optional[int] = None     # default 960 (native)
    text_det_limit_type: Optional[str] = None         # min|max
    text_det_thresh: Optional[float] = None           # default 0.3
    text_det_box_thresh: Optional[float] = None       # default 0.6
    text_det_unclip_ratio: Optional[float] = None     # default 2.0

    textline_orientation_model_name: Optional[str] = None
    textline_orientation_model_dir: Optional[str] = None
    textline_orientation_batch_size: Optional[int] = None

    text_recognition_model_name: Optional[str] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_dir: Optional[str] = None
    text_recognition_batch_size: Optional[int] = None
    text_rec_score_thresh: Optional[float] = None     # default 0.0 (no threshold)

    # Table recognition subline (keep others default; set SLANet_plus)
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

    # Seal recognition subline (leave defaults)
    seal_text_detection_model_name: Optional[str] = None
    seal_text_detection_model_dir: Optional[str] = None
    seal_det_limit_side_len: Optional[int] = None     # default 736
    seal_det_limit_type: Optional[str] = None         # min|max
    seal_det_thresh: Optional[float] = None           # default 0.2
    seal_det_box_thresh: Optional[float] = None       # default 0.6
    seal_det_unclip_ratio: Optional[float] = None     # default 0.5
    seal_text_recognition_model_name: Optional[str] = None
    seal_text_recognition_model_dir: Optional[str] = None
    seal_text_recognition_batch_size: Optional[int] = None
    seal_rec_score_thresh: Optional[float] = None

    # Formula recognition subline (leave defaults)
    formula_recognition_model_name: Optional[str] = None
    formula_recognition_model_dir: Optional[str] = None
    formula_recognition_batch_size: Optional[int] = None

    # Chart parsing module (leave defaults)
    chart_recognition_model_name: Optional[str] = None
    chart_recognition_model_dir: Optional[str] = None
    chart_recognition_batch_size: Optional[int] = None

    # Module toggles (None = use pipeline default behavior)
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_table_recognition: Optional[bool] = None
    use_formula_recognition: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_region_detection: Optional[bool] = None

    # CPU/HW inference settings
    enable_hpi: bool = False
    use_tensorrt: bool = False
    precision: str = "fp32"
    enable_mkldnn: bool = True  # CPU acceleration
    mkldnn_cache_capacity: int = 10
    cpu_threads: int = 4

    # Optional external PDX config
    paddlex_config: Optional[str] = None

@dataclass
class PredictOverrideConfig:
    """
    Per-call overrides for predict(); keep None to obey instantiation defaults.
    Exposed here for completeness and easy tuning for domain docs.
    """
    # Sub-pipeline toggles
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_table_recognition: Optional[bool] = None
    use_formula_recognition: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_region_detection: Optional[bool] = None

    # Layout & OCR thresholds (None = leave to pipeline)
    layout_threshold: Optional[Union[float, Dict[int, float]]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]]] = None
    layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None

    text_det_limit_side_len: Optional[int] = None
    text_det_limit_type: Optional[str] = None
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None

    text_recognition_batch_size: Optional[int] = None
    text_rec_score_thresh: Optional[float] = None

    # Table recognition behavior
    use_table_orientation_classify: Optional[bool] = None  # default True in pipeline
    use_ocr_results_with_table_cells: Optional[bool] = None  # default True
    use_e2e_wired_table_rec_model: Optional[bool] = None     # default False
    use_e2e_wireless_table_rec_model: Optional[bool] = None  # default True


# Default service configuration tuned for Medical Lab Reports (English-heavy)
PIPELINE_INIT = PipelineInitConfig()
PREDICT_OVERRIDE = PredictOverrideConfig(
    # Keep module toggles None to honor native defaults.
    # For medical lab reports, tables are key: defaults already enable table rec.
    # You may set use_table_orientation_classify=True if your inputs are often rotated.
    # Example (uncomment to force):
    # use_table_orientation_classify=True
)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(title="PP-StructureV3 CPU Service", version="1.0.0")
_pipeline: Optional[PPStructureV3] = None


def _create_pipeline(cfg: PipelineInitConfig) -> PPStructureV3:
    # Map dataclass -> kwargs filtering out None to keep true native defaults.
    kwargs = {k: v for k, v in asdict(cfg).items() if v is not None}
    return PPStructureV3(**kwargs)  # Native 3.x API. Docs and params: see link in header.


@app.on_event("startup")
def _startup() -> None:
    global _pipeline
    _pipeline = _create_pipeline(PIPELINE_INIT)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok", "device": PIPELINE_INIT.device, "mkldnn": PIPELINE_INIT.enable_mkldnn}


def _save_uploads_to_tmp(files: List[UploadFile]) -> List[Path]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    tmp_root = Path(tempfile.mkdtemp(prefix="ppstructv3_"))
    saved: List[Path] = []
    for f in files:
        suffix = Path(f.filename or f"file-{uuid.uuid4().hex}").suffix or ""
        dst = tmp_root / (Path(f.filename).name if f.filename else f"file-{uuid.uuid4().hex}{suffix}")
        with open(dst, "wb") as out:
            out.write(f.file.read())
        saved.append(dst)
    return saved


def _pil_image_to_data_uri(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _group_results_by_input_path(res_list) -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = {}
    for res in res_list:
        ipath = res.json.get("res", {}).get("input_path") or res.json.get("input_path") or "unknown"
        groups.setdefault(ipath, []).append(res)
    return groups


@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Accept multiple images and/or PDFs and return per-file JSON and Markdown.
    - JSON: native PP-StructureV3 JSON (page-level list per file)
    - Markdown: concatenated markdown for multi-page PDFs; single md for images
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    saved_paths = _save_uploads_to_tmp(files)
    # Build predict() kwargs with None filtered to honor pipeline defaults
    pred_kwargs = {k: v for k, v in asdict(PREDICT_OVERRIDE).items() if v is not None}

    # Run inference (pipeline accepts list of paths)
    results = _pipeline.predict(input=[str(p) for p in saved_paths], **pred_kwargs)

    # Group per original input to concat PDF pages to one markdown
    grouped = _group_results_by_input_path(results)

    api_payload: List[Dict[str, Any]] = []
    for ipath, res_objs in grouped.items():
        # Page-level JSONs
        page_jsons: List[Dict[str, Any]] = [r.json for r in res_objs]

        # Markdown: concatenate all pages with pipeline helper
        markdown_list = [r.markdown for r in res_objs]  # dicts with markdown_texts/markdown_images
        full_markdown_text = _pipeline.concatenate_markdown_pages(markdown_list)

        # Collect images as data URIs for transport (caller can embed or write to files)
        md_images_merged: Dict[str, str] = {}
        for md in markdown_list:
            md_images = md.get("markdown_images", {}) or {}
            for rel_path, pil_image in md_images.items():
                md_images_merged[rel_path] = _pil_image_to_data_uri(pil_image)

        api_payload.append(
            {
                "input_path": ipath,
                "filename": Path(ipath).name,
                "markdown": full_markdown_text,
                "markdown_images_data_uri": md_images_merged,
                "pages": page_jsons,
            }
        )

    return JSONResponse(
        content={
            "results": api_payload,
            "model_settings": {
                "layout_detection_model_name": PIPELINE_INIT.layout_detection_model_name,
                "text_detection_model_name": PIPELINE_INIT.text_detection_model_name,
                "text_recognition_model_name": PIPELINE_INIT.text_recognition_model_name,
                "wireless_table_structure_recognition_model_name": PIPELINE_INIT.wireless_table_structure_recognition_model_name,
                "device": PIPELINE_INIT.device,
                "enable_mkldnn": PIPELINE_INIT.enable_mkldnn,
                "cpu_threads": PIPELINE_INIT.cpu_threads,
                "lang": PIPELINE_INIT.lang,
            },
            "note": "Parameters and models follow native PP-StructureV3. Modify in app.py only.",
        }
    )

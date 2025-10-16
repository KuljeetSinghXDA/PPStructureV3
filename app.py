from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import tempfile
import shutil
import os
import json
from paddleocr import PPStructureV3

# ------------------------------------------------------------------------------------
# Configuration: edit only this section to change models/behavior as required.
# ------------------------------------------------------------------------------------

@dataclass
class PPStructureV3Config:
    # Device and performance
    device: str = "cpu"                      # "cpu" or "gpu:0"
    enable_hpi: bool = False                 # High-performance inference (backend-dependent)
    use_tensorrt: bool = False               # GPU-only; ignored on CPU
    precision: str = "fp32"                  # "fp32" on CPU
    enable_mkldnn: bool = True               # CPU acceleration
    mkldnn_cache_capacity: int = 10
    cpu_threads: int = 4

    # Language and general OCR behavior
    lang: str = "en"                         # English focus for medical lab reports

    # Core model choices per requirements
    layout_detection_model_name: str = "PP-DocLayout-L"
    text_detection_model_name: str = "PP-OCRv5_mobile_det"
    text_recognition_model_name: str = "en_PP-OCRv5_mobile_rec"

    # Optional local model dirs (None pulls official weights)
    layout_detection_model_dir: Optional[str] = None
    text_detection_model_dir: Optional[str] = None
    text_recognition_model_dir: Optional[str] = None

    # Sub-pipeline toggles
    use_doc_orientation_classify: bool = True      # helpful for scans
    use_doc_unwarping: bool = True                 # helpful for camera/skew
    use_textline_orientation: bool = False
    use_seal_recognition: bool = False
    use_table_recognition: bool = True
    use_formula_recognition: bool = True
    use_chart_recognition: bool = False            # disable chart VLM
    use_region_detection: bool = True

    # Detection params
    text_det_limit_side_len: int = 960
    text_det_limit_type: str = "max"
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.6
    text_det_unclip_ratio: float = 2.0

    # Layout params
    layout_threshold: float = 0.5
    layout_nms: bool = True
    layout_unclip_ratio: float = 1.0
    layout_merge_bboxes_mode: str = "large"

    # Batching and thresholds
    text_recognition_batch_size: int = 1
    text_rec_score_thresh: float = 0.0

    # Optional specialized modules (keep None to use defaults or skip)
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

    formula_recognition_model_name: Optional[str] = None
    formula_recognition_model_dir: Optional[str] = None
    formula_recognition_batch_size: int = 1

    seal_text_detection_model_name: Optional[str] = None
    seal_text_detection_model_dir: Optional[str] = None
    seal_det_limit_side_len: int = 736
    seal_det_limit_type: str = "min"
    seal_det_thresh: float = 0.2
    seal_det_box_thresh: float = 0.6
    seal_det_unclip_ratio: float = 0.5
    seal_text_recognition_model_name: Optional[str] = None
    seal_text_recognition_model_dir: Optional[str] = None
    seal_text_recognition_batch_size: int = 1
    seal_rec_score_thresh: float = 0.0

    chart_recognition_model_name: Optional[str] = None
    chart_recognition_model_dir: Optional[str] = None
    chart_recognition_batch_size: int = 1

    # Markdown combination behavior
    combine_markdown_pages: bool = True

def build_pipeline(cfg: PPStructureV3Config) -> PPStructureV3:
    kwargs = dict(
        device=cfg.device,
        enable_hpi=cfg.enable_hpi,
        use_tensorrt=cfg.use_tensorrt,
        precision=cfg.precision,
        enable_mkldnn=cfg.enable_mkldnn,
        mkldnn_cache_capacity=cfg.mkldnn_cache_capacity,
        cpu_threads=cfg.cpu_threads,
        lang=cfg.lang,

        # core models
        layout_detection_model_name=cfg.layout_detection_model_name,
        text_detection_model_name=cfg.text_detection_model_name,
        text_recognition_model_name=cfg.text_recognition_model_name,

        # optional local dirs
        layout_detection_model_dir=cfg.layout_detection_model_dir,
        text_detection_model_dir=cfg.text_detection_model_dir,
        text_recognition_model_dir=cfg.text_recognition_model_dir,

        # toggles
        use_doc_orientation_classify=cfg.use_doc_orientation_classify,
        use_doc_unwarping=cfg.use_doc_unwarping,
        use_textline_orientation=cfg.use_textline_orientation,
        use_seal_recognition=cfg.use_seal_recognition,
        use_table_recognition=cfg.use_table_recognition,
        use_formula_recognition=cfg.use_formula_recognition,
        use_chart_recognition=cfg.use_chart_recognition,
        use_region_detection=cfg.use_region_detection,

        # det params
        text_det_limit_side_len=cfg.text_det_limit_side_len,
        text_det_limit_type=cfg.text_det_limit_type,
        text_det_thresh=cfg.text_det_thresh,
        text_det_box_thresh=cfg.text_det_box_thresh,
        text_det_unclip_ratio=cfg.text_det_unclip_ratio,

        # layout params
        layout_threshold=cfg.layout_threshold,
        layout_nms=cfg.layout_nms,
        layout_unclip_ratio=cfg.layout_unclip_ratio,
        layout_merge_bboxes_mode=cfg.layout_merge_bboxes_mode,

        # batching
        text_recognition_batch_size=cfg.text_recognition_batch_size,
        text_rec_score_thresh=cfg.text_rec_score_thresh,

        # optional module overrides
        table_classification_model_name=cfg.table_classification_model_name,
        table_classification_model_dir=cfg.table_classification_model_dir,
        wired_table_structure_recognition_model_name=cfg.wired_table_structure_recognition_model_name,
        wired_table_structure_recognition_model_dir=cfg.wired_table_structure_recognition_model_dir,
        wireless_table_structure_recognition_model_name=cfg.wireless_table_structure_recognition_model_name,
        wireless_table_structure_recognition_model_dir=cfg.wireless_table_structure_recognition_model_dir,
        wired_table_cells_detection_model_name=cfg.wired_table_cells_detection_model_name,
        wired_table_cells_detection_model_dir=cfg.wired_table_cells_detection_model_dir,
        wireless_table_cells_detection_model_name=cfg.wireless_table_cells_detection_model_name,
        wireless_table_cells_detection_model_dir=cfg.wireless_table_cells_detection_model_dir,
        table_orientation_classify_model_name=cfg.table_orientation_classify_model_name,
        table_orientation_classify_model_dir=cfg.table_orientation_classify_model_dir,

        formula_recognition_model_name=cfg.formula_recognition_model_name,
        formula_recognition_model_dir=cfg.formula_recognition_model_dir,
        formula_recognition_batch_size=cfg.formula_recognition_batch_size,

        seal_text_detection_model_name=cfg.seal_text_detection_model_name,
        seal_text_detection_model_dir=cfg.seal_text_detection_model_dir,
        seal_det_limit_side_len=cfg.seal_det_limit_side_len,
        seal_det_limit_type=cfg.seal_det_limit_type,
        seal_det_thresh=cfg.seal_det_thresh,
        seal_det_box_thresh=cfg.seal_det_box_thresh,
        seal_det_unclip_ratio=cfg.seal_det_unclip_ratio,
        seal_text_recognition_model_name=cfg.seal_text_recognition_model_name,
        seal_text_recognition_model_dir=cfg.seal_text_recognition_model_dir,
        seal_text_recognition_batch_size=cfg.seal_text_recognition_batch_size,
        seal_rec_score_thresh=cfg.seal_rec_score_thresh,

        chart_recognition_model_name=cfg.chart_recognition_model_name,
        chart_recognition_model_dir=cfg.chart_recognition_model_dir,
        chart_recognition_batch_size=cfg.chart_recognition_batch_size,
    )
    # Drop only keys whose value is None
    return PPStructureV3(**{k: v for k, v in kwargs.items() if v is not None})

# Instantiate configuration and set CPU threading env before pipeline init
CONFIG = PPStructureV3Config()
PIPELINE = build_pipeline(CONFIG)

app = FastAPI(title="PP-StructureV3 Parser", version="1.2.1")

def _save_and_collect_outputs(res_list, work_dir: str) -> List[Dict[str, Any]]:
    """
    Save native JSON and Markdown via official result methods and load for response.
    """
    for res in res_list:
        res.save_to_json(save_path=work_dir)
        res.save_to_markdown(save_path=work_dir)

    json_files = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.lower().endswith(".json")]
    md_files = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.lower().endswith(".md")]
    md_index = {os.path.splitext(os.path.basename(p))[0]: p for p in md_files}

    pages = []
    for jf in sorted(json_files):
        key = os.path.splitext(os.path.basename(jf))[0]
        with open(jf, "r", encoding="utf-8") as f:
            j = json.load(f)
        md_content = None
        if key in md_index:
            with open(md_index[key], "r", encoding="utf-8") as mf:
                md_content = mf.read()
        page_idx = None
        try:
            page_idx = j.get("res", {}).get("page_index", None)
        except Exception:
            page_idx = None

        pages.append({"page_index": page_idx, "json": j, "markdown": md_content})
    return pages

def _combine_markdown_pages_from_res(res_list) -> str:
    """
    Produce a single Markdown string using the documented API with robust fallback.
    """
    md_list = []
    for res in res_list:
        if hasattr(res, "markdown"):
            if isinstance(res.markdown, str):
                md_list.append(res.markdown)
            else:
                md_list.append(res.markdown.get("markdown_texts", ""))

    # Prefer official convenience method, then robust fallback, else manual join
    try:
        return PIPELINE.concatenate_markdown_pages(md_list)
    except Exception:
        try:
            return PIPELINE.paddlex_pipeline.concatenate_markdown_pages(md_list)
        except Exception:
            return "\n\n".join([m for m in md_list if m])

@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)):
    """
    Accept multiple image/PDF files and return structured JSON and Markdown,
    with combined Markdown per file when enabled.
    """
    results: List[Dict[str, Any]] = []
    temp_root = tempfile.mkdtemp(prefix="ppstructv3_")
    try:
        for uf in files:
            # Persist upload to a temp path
            file_suffix = os.path.splitext(uf.filename or "")[1] or ""
            tmp_path = os.path.join(temp_root, next(tempfile._get_candidate_names()) + file_suffix)
            with open(tmp_path, "wb") as out:
                content = await uf.read()
                out.write(content)

            # Predict; keep chart parsing off via the supported flag
            preds = PIPELINE.predict(
                tmp_path,
                use_chart_recognition=False
            )

            # Save native outputs and collect into response
            file_work = os.path.join(temp_root, next(tempfile._get_candidate_names()))
            os.makedirs(file_work, exist_ok=True)
            pages = _save_and_collect_outputs(preds, file_work)

            combined_md = None
            if CONFIG.combine_markdown_pages:
                combined_md = _combine_markdown_pages_from_res(preds)

            results.append({
                "filename": uf.filename,
                "pages": pages,
                "markdown_combined": combined_md,
                "config_used": asdict(CONFIG)
            })

        return JSONResponse(content={"results": results})
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

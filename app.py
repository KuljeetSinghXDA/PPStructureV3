from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import tempfile
import shutil
import os
import json
from paddleocr import PPStructureV3

# ------------------------------------------------------------------------------------
# Configuration: all models, parameters, and toggles live here
# Everything can be tuned in-code without relying on environment variables or CLI flags
# ------------------------------------------------------------------------------------

@dataclass
class PPStructureV3Config:
    # Device and performance
    device: str = "cpu"                      # "cpu" or e.g., "gpu:0" when GPU present
    enable_hpi: bool = False                 # High performance inference toggle
    use_tensorrt: bool = False               # GPU-only; no-op on CPU
    precision: str = "fp32"                  # "fp32" on CPU
    enable_mkldnn: bool = True               # MKL-DNN acceleration on CPU
    mkldnn_cache_capacity: int = 10
    cpu_threads: int = 4

    # Language and general OCR behavior
    lang: str = "en"                         # English focus for medical lab reports

    # Core model choices per user request
    layout_detection_model_name: str = "PP-DocLayout-L"
    text_detection_model_name: str = "PP-OCRv5_mobile_det"
    # The v5 mobile recognition model is multi-scenario; setting lang="en" specializes it to English
    text_recognition_model_name: str = "PP-OCRv5_mobile_rec"

    # Optional local dirs if using downloaded models (keep None to pull officially)
    layout_detection_model_dir: str = None
    text_detection_model_dir: str = None
    text_recognition_model_dir: str = None

    # Sub-pipeline toggles (kept aligned with native defaults)
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    use_seal_recognition: bool = False
    use_table_recognition: bool = True
    use_formula_recognition: bool = True
    use_chart_recognition: bool = False
    use_region_detection: bool = True

    # Detection params (native defaults retained unless you know a domain-specific need)
    text_det_limit_side_len: int = 960
    text_det_limit_type: str = "max"         # "min" or "max"
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.6
    text_det_unclip_ratio: float = 2.0

    # Layout detection optional params (native defaults)
    layout_threshold: float = 0.5
    layout_nms: bool = True
    layout_unclip_ratio: float = 1.0
    layout_merge_bboxes_mode: str = "large"

    # Batch sizes and thresholds
    text_recognition_batch_size: int = 1
    text_rec_score_thresh: float = 0.0

    # Table recognition model names/dirs (default to official pipeline defaults)
    table_classification_model_name: str = None
    table_classification_model_dir: str = None
    wired_table_structure_recognition_model_name: str = None
    wired_table_structure_recognition_model_dir: str = None
    wireless_table_structure_recognition_model_name: str = None
    wireless_table_structure_recognition_model_dir: str = None
    wired_table_cells_detection_model_name: str = None
    wired_table_cells_detection_model_dir: str = None
    wireless_table_cells_detection_model_name: str = None
    wireless_table_cells_detection_model_dir: str = None
    table_orientation_classify_model_name: str = None
    table_orientation_classify_model_dir: str = None

    # Formula, seal, chart modules (names and dirs if overriding)
    formula_recognition_model_name: str = None
    formula_recognition_model_dir: str = None
    formula_recognition_batch_size: int = 1
    seal_text_detection_model_name: str = None
    seal_text_detection_model_dir: str = None
    seal_det_limit_side_len: int = 736
    seal_det_limit_type: str = "min"
    seal_det_thresh: float = 0.2
    seal_det_box_thresh: float = 0.6
    seal_det_unclip_ratio: float = 0.5
    seal_text_recognition_model_name: str = None
    seal_text_recognition_model_dir: str = None
    seal_text_recognition_batch_size: int = 1
    seal_rec_score_thresh: float = 0.0
    chart_recognition_model_name: str = None
    chart_recognition_model_dir: str = None
    chart_recognition_batch_size: int = None

def build_pipeline(cfg: PPStructureV3Config) -> PPStructureV3:
    kwargs = dict(
        # Device and performance
        device=cfg.device,
        enable_hpi=cfg.enable_hpi,
        use_tensorrt=cfg.use_tensorrt,
        precision=cfg.precision,
        enable_mkldnn=cfg.enable_mkldnn,
        mkldnn_cache_capacity=cfg.mkldnn_cache_capacity,
        cpu_threads=cfg.cpu_threads,

        # Language
        lang=cfg.lang,

        # Core models
        layout_detection_model_name=cfg.layout_detection_model_name,
        text_detection_model_name=cfg.text_detection_model_name,
        text_recognition_model_name=cfg.text_recognition_model_name,

        # Optional local model dirs
        layout_detection_model_dir=cfg.layout_detection_model_dir,
        text_detection_model_dir=cfg.text_detection_model_dir,
        text_recognition_model_dir=cfg.text_recognition_model_dir,

        # Feature toggles
        use_doc_orientation_classify=cfg.use_doc_orientation_classify,
        use_doc_unwarping=cfg.use_doc_unwarping,
        use_textline_orientation=cfg.use_textline_orientation,
        use_seal_recognition=cfg.use_seal_recognition,
        use_table_recognition=cfg.use_table_recognition,
        use_formula_recognition=cfg.use_formula_recognition,
        use_chart_recognition=cfg.use_chart_recognition,
        use_region_detection=cfg.use_region_detection,

        # Text detection params
        text_det_limit_side_len=cfg.text_det_limit_side_len,
        text_det_limit_type=cfg.text_det_limit_type,
        text_det_thresh=cfg.text_det_thresh,
        text_det_box_thresh=cfg.text_det_box_thresh,
        text_det_unclip_ratio=cfg.text_det_unclip_ratio,

        # Layout params
        layout_threshold=cfg.layout_threshold,
        layout_nms=cfg.layout_nms,
        layout_unclip_ratio=cfg.layout_unclip_ratio,
        layout_merge_bboxes_mode=cfg.layout_merge_bboxes_mode,

        # Batching and thresholds
        text_recognition_batch_size=cfg.text_recognition_batch_size,
        text_rec_score_thresh=cfg.text_rec_score_thresh,

        # Table models (if overriding)
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

        # Formula models
        formula_recognition_model_name=cfg.formula_recognition_model_name,
        formula_recognition_model_dir=cfg.formula_recognition_model_dir,
        formula_recognition_batch_size=cfg.formula_recognition_batch_size,

        # Seal models
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

        # Chart models
        chart_recognition_model_name=cfg.chart_recognition_model_name,
        chart_recognition_model_dir=cfg.chart_recognition_model_dir,
        chart_recognition_batch_size=cfg.chart_recognition_batch_size,
    )
    return PPStructureV3(**{k: v for k, v in kwargs.items() if v is not None})

# Instantiate configuration and pipeline at startup
CONFIG = PPStructureV3Config()
PIPELINE = build_pipeline(CONFIG)

app = FastAPI(title="PP-StructureV3 Parser", version="1.0.0")

def _save_and_collect_outputs(res_list, work_dir: str) -> List[Dict[str, Any]]:
    """
    For each page result:
      - Save native JSON and Markdown using pipeline result methods
      - Load and return their content per page
    """
    outputs: List[Dict[str, Any]] = []

    # Save files
    for res in res_list:
        res.save_to_json(save_path=work_dir)
        res.save_to_markdown(save_path=work_dir)

    # Collect JSON and Markdown files, map by page_index where possible
    # Strategy: find all JSON files, load them, then try to pair with a .md file
    page_entries = []
    json_files = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.lower().endswith(".json")]
    md_files = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.lower().endswith(".md")]

    # Build index by a best-effort key from filename stem
    def stem(p): return os.path.splitext(os.path.basename(p))[0]
    md_index = {stem(p): p for p in md_files}

    for jf in sorted(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            j = json.load(f)
        key = stem(jf)
        md_content = None
        if key in md_index:
            with open(md_index[key], "r", encoding="utf-8") as mf:
                md_content = mf.read()
        # Extract page index if provided
        page_idx = None
        try:
            page_idx = j.get("res", {}).get("page_index", None)
        except Exception:
            page_idx = None

        page_entries.append({
            "page_index": page_idx,
            "json": j,
            "markdown": md_content
        })

    return page_entries

@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)):
    """
    Accept multiple image/PDF files and return structured JSON and Markdown
    directly from PP-StructureV3 outputs.
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

            # Run pipeline prediction
            preds = PIPELINE.predict(tmp_path)

            # Save native outputs and collect into response
            file_work = os.path.join(temp_root, next(tempfile._get_candidate_names()))
            os.makedirs(file_work, exist_ok=True)
            pages = _save_and_collect_outputs(preds, file_work)

            results.append({
                "filename": uf.filename,
                "pages": pages,
                "config_used": asdict(CONFIG)
            })

        return JSONResponse(content={"results": results})
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

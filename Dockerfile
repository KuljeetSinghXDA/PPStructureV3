from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Literal, Optional
from pathlib import Path
import tempfile
import shutil
import os
import json
import threading

from paddleocr import PPStructureV3

# Service constants
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ==============================
# Supported PP-StructureV3 params (None => defaults)
# TUNED FOR MEDICAL LAB REPORTS
# ==============================

# Backend/config toggles
DEVICE = "cpu"
ENABLE_MKLDNN = True
ENABLE_HPI = False
USE_TENSORRT = False
PRECISION = None
MKLDNN_CACHE_CAPACITY = 10

# Threads
CPU_THREADS = 4

# Note: paddlex_config will be auto-built from constants below
PADDLEX_CONFIG: Optional[Dict[str, Any]] = None

# Subpipeline toggles
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False
USE_REGION_DETECTION = True

# Model names
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
REGION_DETECTION_MODEL_NAME = None
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
TABLE_CLASSIFICATION_MODEL_NAME = None
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = None
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = None
WIRED_TABLE_CELLS_DET_MODEL_NAME = None
WIRELESS_TABLE_CELLS_DET_MODEL_NAME = None
TABLE_ORIENTATION_CLASSIFY_MODEL_NAME = None
FORMULA_RECOGNITION_MODEL_NAME = None
DOC_ORIENTATION_CLASSIFY_MODEL_NAME = None
DOC_UNWARPING_MODEL_NAME = None
TEXTLINE_ORIENTATION_MODEL_NAME = None
SEAL_TEXT_DETECTION_MODEL_NAME = None
SEAL_TEXT_RECOGNITION_MODEL_NAME = None
CHART_RECOGNITION_MODEL_NAME = None

# Model dirs - All None to use default downloads
LAYOUT_DETECTION_MODEL_DIR = None
REGION_DETECTION_MODEL_DIR = None
TEXT_DETECTION_MODEL_DIR = None
TEXT_RECOGNITION_MODEL_DIR = None
TABLE_CLASSIFICATION_MODEL_DIR = None
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = None
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = None
WIRED_TABLE_CELLS_DET_MODEL_DIR = None
WIRELESS_TABLE_CELLS_DET_MODEL_DIR = None
TABLE_ORIENTATION_CLASSIFY_MODEL_DIR = None
FORMULA_RECOGNITION_MODEL_DIR = None
DOC_ORIENTATION_CLASSIFY_MODEL_DIR = None
DOC_UNWARPING_MODEL_DIR = None
TEXTLINE_ORIENTATION_MODEL_DIR = None
SEAL_TEXT_DETECTION_MODEL_DIR = None
SEAL_TEXT_RECOGNITION_MODEL_DIR = None
CHART_RECOGNITION_MODEL_DIR = None

# Layout thresholds/controls
LAYOUT_THRESHOLD = None
LAYOUT_NMS = None
LAYOUT_UNCLIP_RATIO = None
LAYOUT_MERGE_BBOXES_MODE = None
# Optional category config placeholder (list of labels) if your layout model supports it
LAYOUT_CATEGORIES = None  # e.g., ["title","text","figure","table"]

# Text detection tuning
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
# Extra postprocess control
DET_USE_DILATION = None  # boolean

# Region detection tuning (separate from layout)
REGION_DET_LIMIT_SIDE_LEN = None
REGION_DET_LIMIT_TYPE = None
REGION_DET_THRESH = None
REGION_DET_BOX_THRESH = None
REGION_DET_UNCLIP_RATIO = None

# Seal detection tuning
SEAL_DET_LIMIT_SIDE_LEN = None
SEAL_DET_LIMIT_TYPE = None
SEAL_DET_THRESH = None
SEAL_DET_BOX_THRESH = None
SEAL_DET_UNCLIP_RATIO = None

# Recognition controls
REC_CHAR_DICT_PATH = None                  # path to custom charset file
REC_IMAGE_SHAPE = None                     # e.g., "3,48,320"
USE_SPACE_CHAR = None                      # bool
MAX_TEXT_LENGTH = None                     # int
USE_ANGLE_CLS = None                       # bool (line-level angle classifier)
CLS_THRESH = None                          # float (angle classifier threshold)
DROP_SCORE = None                          # alias for rec filtering if the build uses this name

# Recognition thresholds/batches
TEXT_REC_SCORE_THRESH = None               # top-level; also mapped to rec.drop_score if set
TEXT_RECOGNITION_BATCH_SIZE = None
TEXTLINE_ORIENTATION_BATCH_SIZE = None
FORMULA_RECOGNITION_BATCH_SIZE = None
CHART_RECOGNITION_BATCH_SIZE = None
SEAL_TEXT_RECOGNITION_BATCH_SIZE = None
SEAL_REC_SCORE_THRESH = None

# Table module options
TABLE_ALGORITHM = None                     # e.g., "SLANeXt_wired" or model-compatible string
TABLE_CHAR_DICT_PATH = None
TABLE_MAX_LEN = None                       # long-side resize for table crops

# PDF/page handling
PDF_RENDER_DPI = None                      # e.g., 300–400
PAGE_INDICES = None                        # list of ints, e.g., [0,1,2]

# Predict-time table recognition defaults
# Must be False to avoid PaddleOCR 3.2.0 bug per community notes
DEFAULT_USE_OCR_RESULTS_WITH_TABLE_CELLS = False
DEFAULT_USE_E2E_WIRED_TABLE_REC_MODEL = None
DEFAULT_USE_E2E_WIRELESS_TABLE_REC_MODEL = None
DEFAULT_USE_WIRED_TABLE_CELLS_TRANS_TO_HTML = None
DEFAULT_USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML = None
DEFAULT_USE_TABLE_ORIENTATION_CLASSIFY = False

def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_exceeds_limit(tmp_path: Path) -> bool:
    try:
        return tmp_path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024
    except Exception:
        return False

app = FastAPI(title="PP-StructureV3 API (ARM64, native)", version="3.2.0")

def _build_paddlex_config_from_constants() -> Optional[Dict[str, Any]]:
    cfg: Dict[str, Any] = {}

    # Recognition submodule
    rec_cfg: Dict[str, Any] = {}
    if REC_CHAR_DICT_PATH is not None:
        rec_cfg["rec_char_dict_path"] = REC_CHAR_DICT_PATH
    if REC_IMAGE_SHAPE is not None:
        rec_cfg["rec_image_shape"] = REC_IMAGE_SHAPE
    if USE_SPACE_CHAR is not None:
        rec_cfg["use_space_char"] = USE_SPACE_CHAR
    if MAX_TEXT_LENGTH is not None:
        rec_cfg["max_text_length"] = MAX_TEXT_LENGTH
    if USE_ANGLE_CLS is not None:
        rec_cfg["use_angle_cls"] = USE_ANGLE_CLS
    if CLS_THRESH is not None:
        rec_cfg["cls_thresh"] = CLS_THRESH
    # Map global text_rec_score_thresh to rec.drop_score for older naming if provided
    if DROP_SCORE is not None:
        rec_cfg["drop_score"] = DROP_SCORE
    elif TEXT_REC_SCORE_THRESH is not None:
        rec_cfg["drop_score"] = TEXT_REC_SCORE_THRESH

    if rec_cfg:
        cfg["rec"] = rec_cfg

    # Detection submodule
    det_cfg: Dict[str, Any] = {}
    if TEXT_DET_LIMIT_SIDE_LEN is not None:
        det_cfg["limit_side_len"] = TEXT_DET_LIMIT_SIDE_LEN
    if TEXT_DET_LIMIT_TYPE is not None:
        det_cfg["limit_type"] = TEXT_DET_LIMIT_TYPE
    if TEXT_DET_THRESH is not None:
        det_cfg["thresh"] = TEXT_DET_THRESH
    if TEXT_DET_BOX_THRESH is not None:
        det_cfg["box_thresh"] = TEXT_DET_BOX_THRESH
    if TEXT_DET_UNCLIP_RATIO is not None:
        det_cfg["unclip_ratio"] = TEXT_DET_UNCLIP_RATIO
    if DET_USE_DILATION is not None:
        det_cfg["use_dilation"] = DET_USE_DILATION

    if det_cfg:
        cfg["det"] = det_cfg

    # Region detector submodule
    region_cfg: Dict[str, Any] = {}
    if REGION_DET_LIMIT_SIDE_LEN is not None:
        region_cfg["limit_side_len"] = REGION_DET_LIMIT_SIDE_LEN
    if REGION_DET_LIMIT_TYPE is not None:
        region_cfg["limit_type"] = REGION_DET_LIMIT_TYPE
    if REGION_DET_THRESH is not None:
        region_cfg["thresh"] = REGION_DET_THRESH
    if REGION_DET_BOX_THRESH is not None:
        region_cfg["box_thresh"] = REGION_DET_BOX_THRESH
    if REGION_DET_UNCLIP_RATIO is not None:
        region_cfg["unclip_ratio"] = REGION_DET_UNCLIP_RATIO

    if region_cfg:
        cfg["region"] = region_cfg

    # Table structure submodule
    table_cfg: Dict[str, Any] = {}
    if TABLE_ALGORITHM is not None:
        table_cfg["table_algorithm"] = TABLE_ALGORITHM
    if TABLE_CHAR_DICT_PATH is not None:
        table_cfg["table_char_dict_path"] = TABLE_CHAR_DICT_PATH
    if TABLE_MAX_LEN is not None:
        table_cfg["table_max_len"] = TABLE_MAX_LEN

    if table_cfg:
        cfg["table"] = table_cfg

    # Layout submodule
    layout_cfg: Dict[str, Any] = {}
    if LAYOUT_THRESHOLD is not None:
        layout_cfg["threshold"] = LAYOUT_THRESHOLD
    if LAYOUT_NMS is not None:
        layout_cfg["nms"] = LAYOUT_NMS
    if LAYOUT_UNCLIP_RATIO is not None:
        layout_cfg["unclip_ratio"] = LAYOUT_UNCLIP_RATIO
    if LAYOUT_MERGE_BBOXES_MODE is not None:
        layout_cfg["merge_bboxes_mode"] = LAYOUT_MERGE_BBOXES_MODE
    if LAYOUT_CATEGORIES is not None:
        layout_cfg["categories"] = LAYOUT_CATEGORIES

    if layout_cfg:
        cfg["layout"] = layout_cfg

    # Reader/input controls (PDF rasterization, page selection)
    reader_cfg: Dict[str, Any] = {}
    if PDF_RENDER_DPI is not None:
        reader_cfg["pdf2img_dpi"] = PDF_RENDER_DPI
    if PAGE_INDICES is not None:
        reader_cfg["page_indices"] = PAGE_INDICES

    if reader_cfg:
        cfg["reader"] = reader_cfg

    return cfg or None

def _build_init_kwargs() -> Dict[str, Any]:
    # Build paddlex_config dynamically unless user supplied one
    px_cfg = PADDLEX_CONFIG if PADDLEX_CONFIG else _build_paddlex_config_from_constants()

    params = dict(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        use_tensorrt=USE_TENSORRT,
        precision=PRECISION,
        mkldnn_cache_capacity=MKLDNN_CACHE_CAPACITY,
        cpu_threads=CPU_THREADS,
        paddlex_config=px_cfg,

        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
        use_seal_recognition=USE_SEAL_RECOGNITION,
        use_region_detection=USE_REGION_DETECTION,

        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        layout_detection_model_dir=LAYOUT_DETECTION_MODEL_DIR,
        region_detection_model_name=REGION_DETECTION_MODEL_NAME,
        region_detection_model_dir=REGION_DETECTION_MODEL_DIR,

        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_detection_model_dir=TEXT_DETECTION_MODEL_DIR,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        text_recognition_model_dir=TEXT_RECOGNITION_MODEL_DIR,

        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        table_classification_model_dir=TABLE_CLASSIFICATION_MODEL_DIR,

        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_dir=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_dir=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,

        wired_table_cells_detection_model_name=WIRED_TABLE_CELLS_DET_MODEL_NAME,
        wired_table_cells_detection_model_dir=WIRED_TABLE_CELLS_DET_MODEL_DIR,
        wireless_table_cells_detection_model_name=WIRELESS_TABLE_CELLS_DET_MODEL_NAME,
        wireless_table_cells_detection_model_dir=WIRELESS_TABLE_CELLS_DET_MODEL_DIR,

        table_orientation_classify_model_name=TABLE_ORIENTATION_CLASSIFY_MODEL_NAME,
        table_orientation_classify_model_dir=TABLE_ORIENTATION_CLASSIFY_MODEL_DIR,

        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        formula_recognition_model_dir=FORMULA_RECOGNITION_MODEL_DIR,

        doc_orientation_classify_model_name=DOC_ORIENTATION_CLASSIFY_MODEL_NAME,
        doc_orientation_classify_model_dir=DOC_ORIENTATION_CLASSIFY_MODEL_DIR,
        doc_unwarping_model_name=DOC_UNWARPING_MODEL_NAME,
        doc_unwarping_model_dir=DOC_UNWARPING_MODEL_DIR,
        textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL_NAME,
        textline_orientation_model_dir=TEXTLINE_ORIENTATION_MODEL_DIR,

        seal_text_detection_model_name=SEAL_TEXT_DETECTION_MODEL_NAME,
        seal_text_detection_model_dir=SEAL_TEXT_DETECTION_MODEL_DIR,
        seal_det_limit_side_len=SEAL_DET_LIMIT_SIDE_LEN,
        seal_det_limit_type=SEAL_DET_LIMIT_TYPE,
        seal_det_thresh=SEAL_DET_THRESH,
        seal_det_box_thresh=SEAL_DET_BOX_THRESH,
        seal_det_unclip_ratio=SEAL_DET_UNCLIP_RATIO,

        seal_text_recognition_model_name=SEAL_TEXT_RECOGNITION_MODEL_NAME,
        seal_text_recognition_model_dir=SEAL_TEXT_RECOGNITION_MODEL_DIR,
        seal_text_recognition_batch_size=SEAL_TEXT_RECOGNITION_BATCH_SIZE,
        seal_rec_score_thresh=SEAL_REC_SCORE_THRESH,

        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        chart_recognition_model_dir=CHART_RECOGNITION_MODEL_DIR,
        chart_recognition_batch_size=CHART_RECOGNITION_BATCH_SIZE,

        layout_threshold=LAYOUT_THRESHOLD,
        layout_nms=LAYOUT_NMS,
        layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
        layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,

        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,

        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        textline_orientation_batch_size=TEXTLINE_ORIENTATION_BATCH_SIZE,
        formula_recognition_batch_size=FORMULA_RECOGNITION_BATCH_SIZE,
    )
    return {k: v for k, v in params.items() if v is not None}

_init_kwargs = _build_init_kwargs()
pipeline = PPStructureV3(**_init_kwargs)
predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)

def _concat_markdown_pages(markdown_list: List[Dict[str, Any]]) -> str:
    # Prefer top-level helper; fallback to PaddleX pipeline where needed
    if hasattr(pipeline, "concatenate_markdown_pages"):
        return pipeline.concatenate_markdown_pages(markdown_list)
    if hasattr(pipeline, "paddlex_pipeline"):
        return pipeline.paddlex_pipeline.concatenate_markdown_pages(markdown_list)
    # Final fallback: minimal join (least preferred)
    return "\n\n".join([md.get("text", "") if isinstance(md, dict) else str(md) for md in markdown_list])

def predict_collect_one(path: Path,
                        use_ocr_results_with_table_cells,
                        use_e2e_wired_table_rec_model,
                        use_e2e_wireless_table_rec_model,
                        use_wired_table_cells_trans_to_html,
                        use_wireless_table_cells_trans_to_html,
                        use_table_orientation_classify) -> Dict[str, Any]:
    kwargs = {}
    kwargs["use_ocr_results_with_table_cells"] = (
        use_ocr_results_with_table_cells
        if use_ocr_results_with_table_cells is not None
        else DEFAULT_USE_OCR_RESULTS_WITH_TABLE_CELLS
    )
    kwargs["use_e2e_wired_table_rec_model"] = (
        use_e2e_wired_table_rec_model
        if use_e2e_wired_table_rec_model is not None
        else DEFAULT_USE_E2E_WIRED_TABLE_REC_MODEL
    )
    kwargs["use_e2e_wireless_table_rec_model"] = (
        use_e2e_wireless_table_rec_model
        if use_e2e_wireless_table_rec_model is not None
        else DEFAULT_USE_E2E_WIRELESS_TABLE_REC_MODEL
    )
    kwargs["use_wired_table_cells_trans_to_html"] = (
        use_wired_table_cells_trans_to_html
        if use_wired_table_cells_trans_to_html is not None
        else DEFAULT_USE_WIRED_TABLE_CELLS_TRANS_TO_HTML
    )
    kwargs["use_wireless_table_cells_trans_to_html"] = (
        use_wireless_table_cells_trans_to_html
        if use_wireless_table_cells_trans_to_html is not None
        else DEFAULT_USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML
    )
    kwargs["use_table_orientation_classify"] = (
        use_table_orientation_classify
        if use_table_orientation_classify is not None
        else DEFAULT_USE_TABLE_ORIENTATION_CLASSIFY
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    outputs = pipeline.predict(str(path), **kwargs)

    pages: List[Dict[str, Any]] = []
    markdown_list: List[Dict[str, Any]] = []
    markdown_images_list: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        for res in outputs:
            # Save page-level JSON and MD (useful for debugging/compat)
            res.save_to_json(save_path=str(out_dir))
            res.save_to_markdown(save_path=str(out_dir))
            # Collect structured markdown dict and images for merged output
            md_info = getattr(res, "markdown", {}) or {}
            markdown_list.append(md_info)
            markdown_images_list.append(md_info.get("markdown_images", {}) or {})

        # Re-attach per-page artifacts
        json_files = sorted(out_dir.glob("*.json"))
        md_files = sorted(out_dir.glob("*.md"))
        for i in range(max(len(json_files), len(md_files))):
            page_json = {}
            page_md = ""
            if i < len(json_files):
                try:
                    page_json = json.loads(json_files[i].read_text(encoding="utf-8"))
                except Exception:
                    page_json = {}
            if i < len(md_files):
                page_md = md_files[i].read_text(encoding="utf-8")
            pages.append({"page_index": i, "json": page_json, "markdown": page_md})

    merged_markdown = _concat_markdown_pages(markdown_list)

    return {
        "pages": pages,
        "merged_markdown": merged_markdown,
        "markdown_images": markdown_images_list,
        "page_count": len(outputs),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(
    files: List[UploadFile] = File(...),
    output_format: Literal["json", "markdown", "both"] = Query("both"),
    use_ocr_results_with_table_cells = Query(None),
    use_e2e_wired_table_rec_model = Query(None),
    use_e2e_wireless_table_rec_model = Query(None),
    use_wired_table_cells_trans_to_html = Query(None),
    use_wireless_table_cells_trans_to_html = Query(None),
    use_table_orientation_classify = Query(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    outputs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        acquired = predict_sem.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy")
        try:
            for uf in files:
                if not _ext_ok(uf.filename or ""):
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {uf.filename}")
                suffix = Path(uf.filename or "").suffix or ".bin"
                target = tmpdir / (Path(uf.filename or f"upload{len(outputs)}{suffix}").name)
                with target.open("wb") as w:
                    shutil.copyfileobj(uf.file, w)
                if _file_exceeds_limit(target):
                    raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB): {uf.filename}")
                file_res = predict_collect_one(
                    target,
                    use_ocr_results_with_table_cells,
                    use_e2e_wired_table_rec_model,
                    use_e2e_wireless_table_rec_model,
                    use_wired_table_cells_trans_to_html,
                    use_wireless_table_cells_trans_to_html,
                    use_table_orientation_classify,
                )
                outputs.append({"filename": uf.filename, **file_res})
        finally:
            predict_sem.release()

    if output_format == "json":
        return JSONResponse({"files": outputs})
    elif output_format == "markdown":
        combined_md = ""
        for f in outputs:
            combined_md += f"# {f['filename']}\n\n"
            combined_md += f.get("merged_markdown", "").strip() + "\n\n"
        return combined_md
    else:
        return JSONResponse({"files": outputs})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

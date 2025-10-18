# Build on ARM64 host or use: docker build --platform=linux/arm64/v8
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system dependencies for OpenCV (even in "headless" contexts, PaddleOCR may pull GUI build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# PaddlePaddle CPU and PaddleOCR 3.2.0 doc-parser stack + FastAPI runtime deps
RUN python -m pip install --upgrade pip && \
    python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    python -m pip install "paddleocr[doc-parser]==3.2.0" fastapi "uvicorn[standard]" python-multipart pymupdf

# Embed the FastAPI app with heredoc
RUN cat > /app.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Literal, Optional
from pathlib import Path
import tempfile
import shutil
import json
import threading
from contextlib import contextmanager

import fitz  # PyMuPDF
from paddleocr import PPStructureV3

# ==============================
# File handling / service constants
# ==============================
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ==============================
# Accuracy-critical parameters (Medical lab reports)
# ==============================

# 0) Toggle pre-OCR rasterization (PDF upscaling)
#    False => pass PDFs directly to PP-StructureV3 (native multi-page handling)
USE_PDF_RASTERIZATION = False  # set True to re-enable PyMuPDF upscaling

# 1) Input fidelity (pre-OCR rasterization)
PDF_RASTER_DPI = 400  # 300–400 recommended

# 2) Layout detection
LAYOUT_THRESHOLD = None
LAYOUT_NMS = None
LAYOUT_UNCLIP_RATIO = None
LAYOUT_MERGE_BBOXES_MODE = None

# 3) Text detection (DB-style)
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None

# 4) Text recognition (decoder / charset / orientation)
REC_CHAR_DICT_PATH = None
REC_IMAGE_SHAPE = None                     # e.g., "3,48,320"
USE_SPACE_CHAR = None
MAX_TEXT_LENGTH = None
USE_ANGLE_CLS = None
CLS_THRESH = None
DROP_SCORE = None
TEXT_REC_SCORE_THRESH = None               # mapped to rec.drop_score if set

# 5) Table structure recognition
TABLE_ALGORITHM = None
TABLE_CHAR_DICT_PATH = None
TABLE_MAX_LEN = None

# 6) Accuracy-relevant pipeline toggles
USE_TABLE_RECOGNITION = True
USE_REGION_DETECTION = True
USE_TEXTLINE_ORIENTATION = False
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False

# ==============================
# Other runtime / infra parameters
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

# PaddleX config passthrough (auto-built below if None)
PADDLEX_CONFIG: Optional[Dict[str, Any]] = None

# Optional modules
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False

# Model names
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
REGION_DETECTION_MODEL_NAME = None
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
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

# Model dirs
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

# Seal detection tuning
SEAL_DET_LIMIT_SIDE_LEN = None
SEAL_DET_LIMIT_TYPE = None
SEAL_DET_THRESH = None
SEAL_DET_BOX_THRESH = None
SEAL_DET_UNCLIP_RATIO = None

# Recognition batch sizes
TEXT_RECOGNITION_BATCH_SIZE = None
TEXTLINE_ORIENTATION_BATCH_SIZE = None
FORMULA_RECOGNITION_BATCH_SIZE = None
CHART_RECOGNITION_BATCH_SIZE = None
SEAL_TEXT_RECOGNITION_BATCH_SIZE = None
SEAL_REC_SCORE_THRESH = None

# Predict-time table recognition defaults
DEFAULT_USE_OCR_RESULTS_WITH_TABLE_CELLS = False
DEFAULT_USE_E2E_WIRED_TABLE_REC_MODEL = None
DEFAULT_USE_E2E_WIRELESS_TABLE_REC_MODEL = None
DEFAULT_USE_WIRED_TABLE_CELLS_TRANS_TO_HTML = None
DEFAULT_USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML = None
DEFAULT_USE_TABLE_ORIENTATION_CLASSIFY = False

# Diagnostics
DEBUG_SAVE_ARTIFACTS = False

app = FastAPI(title="PP-StructureV3 API (ARM64, native)", version="3.2.0")

def _build_paddlex_config_from_constants() -> Optional[Dict[str, Any]]:
    cfg: Dict[str, Any] = {}

    # Recognition
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
    if DROP_SCORE is not None:
        rec_cfg["drop_score"] = DROP_SCORE
    elif TEXT_REC_SCORE_THRESH is not None:
        rec_cfg["drop_score"] = TEXT_REC_SCORE_THRESH
    if rec_cfg:
        cfg["rec"] = rec_cfg

    # Detection
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
    if det_cfg:
        cfg["det"] = det_cfg

    # Table
    table_cfg: Dict[str, Any] = {}
    if TABLE_ALGORITHM is not None:
        table_cfg["table_algorithm"] = TABLE_ALGORITHM
    if TABLE_CHAR_DICT_PATH is not None:
        table_cfg["table_char_dict_path"] = TABLE_CHAR_DICT_PATH
    if TABLE_MAX_LEN is not None:
        table_cfg["table_max_len"] = TABLE_MAX_LEN
    if table_cfg:
        cfg["table"] = table_cfg

    # Layout
    layout_cfg: Dict[str, Any] = {}
    if LAYOUT_THRESHOLD is not None:
        layout_cfg["threshold"] = LAYOUT_THRESHOLD
    if LAYOUT_NMS is not None:
        layout_cfg["nms"] = LAYOUT_NMS
    if LAYOUT_UNCLIP_RATIO is not None:
        layout_cfg["unclip_ratio"] = LAYOUT_UNCLIP_RATIO
    if LAYOUT_MERGE_BBOXES_MODE is not None:
        layout_cfg["merge_bboxes_mode"] = LAYOUT_MERGE_BBOXES_MODE
    if layout_cfg:
        cfg["layout"] = layout_cfg

    return cfg or None

def _build_init_kwargs() -> Dict[str, Any]:
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
    if hasattr(pipeline, "concatenate_markdown_pages"):
        return pipeline.concatenate_markdown_pages(markdown_list)
    if hasattr(pipeline, "paddlex_pipeline"):
        return pipeline.paddlex_pipeline.concatenate_markdown_pages(markdown_list)
    return "\n\n".join([md.get("text", "") if isinstance(md, dict) else str(md) for md in markdown_list])

@contextmanager
def rasterized_pdf_paths(pdf_path: Path, dpi: int):
    with tempfile.TemporaryDirectory(prefix="pdf_pages_") as td:
        out_dir = Path(td)
        doc = fitz.open(pdf_path)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        image_paths: List[Path] = []
        for i in range(len(doc)):
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            img_path = out_dir / f"page_{i:04d}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
        yield image_paths

def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_exceeds_limit(tmp_path: Path) -> bool:
    try:
        return tmp_path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024
    except Exception:
        return False

def predict_collect_one(path: Path,
                        use_ocr_results_with_table_cells,
                        use_e2e_wired_table_rec_model,
                        use_e2e_wireless_table_rec_model,
                        use_wired_table_cells_trans_to_html,
                        use_wireless_table_cells_trans_to_html,
                        use_table_orientation_classify) -> Dict[str, Any]:
    kwargs = {
        "use_ocr_results_with_table_cells": (
            use_ocr_results_with_table_cells
            if use_ocr_results_with_table_cells is not None
            else DEFAULT_USE_OCR_RESULTS_WITH_TABLE_CELLS
        ),
        "use_e2e_wired_table_rec_model": (
            use_e2e_wired_table_rec_model
            if use_e2e_wired_table_rec_model is not None
            else DEFAULT_USE_E2E_WIRED_TABLE_REC_MODEL
        ),
        "use_e2e_wireless_table_rec_model": (
            use_e2e_wireless_table_rec_model
            if use_e2e_wireless_table_rec_model is not None
            else DEFAULT_USE_E2E_WIRELESS_TABLE_REC_MODEL
        ),
        "use_wired_table_cells_trans_to_html": (
            use_wired_table_cells_trans_to_html
            if use_wired_table_cells_trans_to_html is not None
            else DEFAULT_USE_WIRED_TABLE_CELLS_TRANS_TO_HTML
        ),
        "use_wireless_table_cells_trans_to_html": (
            use_wireless_table_cells_trans_to_html
            if use_wireless_table_cells_trans_to_html is not None
            else DEFAULT_USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML
        ),
        "use_table_orientation_classify": (
            use_table_orientation_classify
            if use_table_orientation_classify is not None
            else DEFAULT_USE_TABLE_ORIENTATION_CLASSIFY
        ),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    is_pdf = path.suffix.lower() == ".pdf"
    pages: List[Dict[str, Any]] = []
    markdown_list: List[Dict[str, Any]] = []
    markdown_images_list: List[Dict[str, Any]] = []

    if is_pdf and not USE_PDF_RASTERIZATION:
        # Let PP-StructureV3 handle multi-page PDFs natively
        outputs = pipeline.predict(str(path), **kwargs)
        for res in outputs:
            if DEBUG_SAVE_ARTIFACTS:
                with tempfile.TemporaryDirectory() as out_dir:
                    res.save_to_json(save_path=str(out_dir))
                    res.save_to_markdown(save_path=str(out_dir))
            md_info = getattr(res, "markdown", {}) or {}
            markdown_list.append(md_info)
            markdown_images_list.append(md_info.get("markdown_images", {}) or {})
            page_json = {}
            if hasattr(res, "to_dict"):
                try:
                    page_json = res.to_dict()
                except Exception:
                    page_json = {}
            elif hasattr(res, "to_json"):
                try:
                    j = res.to_json()
                    page_json = json.loads(j) if isinstance(j, str) else (j or {})
                except Exception:
                    page_json = {}
            page_md = md_info.get("text", "")
            pages.append({"page_index": len(pages), "json": page_json, "markdown": page_md})
    elif is_pdf and USE_PDF_RASTERIZATION:
        # Rasterize PDFs to high-DPI PNGs before OCR
        with rasterized_pdf_paths(path, PDF_RASTER_DPI) as input_paths:
            for p in input_paths:
                outputs = pipeline.predict(str(p), **kwargs)
                for res in outputs:
                    if DEBUG_SAVE_ARTIFACTS:
                        with tempfile.TemporaryDirectory() as out_dir:
                            res.save_to_json(save_path=str(out_dir))
                            res.save_to_markdown(save_path=str(out_dir))
                    md_info = getattr(res, "markdown", {}) or {}
                    markdown_list.append(md_info)
                    markdown_images_list.append(md_info.get("markdown_images", {}) or {})
                    page_json = {}
                    if hasattr(res, "to_dict"):
                        try:
                            page_json = res.to_dict()
                        except Exception:
                            page_json = {}
                    elif hasattr(res, "to_json"):
                        try:
                            j = res.to_json()
                            page_json = json.loads(j) if isinstance(j, str) else (j or {})
                        except Exception:
                            page_json = {}
                    page_md = md_info.get("text", "")
                    pages.append({"page_index": len(pages), "json": page_json, "markdown": page_md})
    else:
        # Single image input path
        outputs = pipeline.predict(str(path), **kwargs)
        for res in outputs:
            if DEBUG_SAVE_ARTIFACTS:
                with tempfile.TemporaryDirectory() as out_dir:
                    res.save_to_json(save_path=str(out_dir))
                    res.save_to_markdown(save_path=str(out_dir))
            md_info = getattr(res, "markdown", {}) or {}
            markdown_list.append(md_info)
            markdown_images_list.append(md_info.get("markdown_images", {}) or {})
            page_json = {}
            if hasattr(res, "to_dict"):
                try:
                    page_json = res.to_dict()
                except Exception:
                    page_json = {}
            elif hasattr(res, "to_json"):
                try:
                    j = res.to_json()
                    page_json = json.loads(j) if isinstance(j, str) else (j or {})
                except Exception:
                    page_json = {}
            page_md = md_info.get("text", "")
            pages.append({"page_index": len(pages), "json": page_json, "markdown": page_md})

    merged_markdown = _concat_markdown_pages(markdown_list)

    return {
        "pages": pages,
        "merged_markdown": merged_markdown,
        "markdown_images": markdown_images_list,
        "page_count": len(pages),
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
    acquired = predict_sem.acquire(timeout=600)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for uf in files:
                if not (uf.filename and _ext_ok(uf.filename)):
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {uf.filename}")
                target = tmpdir / Path(uf.filename).name
                with target.open("wb") as w:
                    shutil.copyfileobj(uf.file, w)
                if target.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
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
EOF

WORKDIR /

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

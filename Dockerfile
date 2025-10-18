# Build on ARM64 host or use: docker build --platform=linux/arm64/v8
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 
#    OMP_NUM_THREADS=4

# System deps for PDFs and image backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# PaddlePaddle stable CPU (ARM64) and PaddleOCR 3.2.0 doc-parser group
# Note: Ensure stable CPU index provides cp313 aarch64 wheels in your region mirror
RUN python -m pip install --upgrade pip && \
    python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    python -m pip install "paddleocr[doc-parser]==3.2.0" fastapi "uvicorn[standard]" python-multipart pymupdf

# Embedded FastAPI app:
# - One long-lived PP-StructureV3 with requested models and accuracy tuning
# - All supported init args exposed (None => library defaults)
# - All documented table predict-time flags exposed (None => default)
# - Native JSON/Markdown outputs; HPI not enabled for ARM64 stability
RUN cat > /app.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Literal
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
# TUNED FOR MEDICAL LAB REPORTS: tables, small fonts, dense layouts
# ==============================

# Backend/config toggles
DEVICE = "cpu"                  # Explicit CPU device
ENABLE_MKLDNN = True            # Enable CPU optimization for better performance
ENABLE_HPI = False              # Keep False on ARM64 to avoid PaddleX HPI plugin
USE_TENSORRT = False             # Not applicable for CPU
PRECISION = None                # FP32 default for CPU
MKLDNN_CACHE_CAPACITY = 10      # Standard cache for MKL-DNN

# Threads
CPU_THREADS = 4                 # Ampere A1 4 OCPU single-process

# Optional PaddleX config passthrough (kept None)
PADDLEX_CONFIG = None

# Subpipeline toggles - TUNED FOR LAB REPORTS
USE_DOC_ORIENTATION_CLASSIFY = False      # Enable to handle rotated scans
USE_DOC_UNWARPING = False                # Keep False on ARM64 for stability
USE_TEXTLINE_ORIENTATION = False          # Enable for mixed orientation text
USE_TABLE_RECOGNITION = True             # CRITICAL: Enable for lab report tables
USE_FORMULA_RECOGNITION = False           # Not typically needed for lab reports
USE_CHART_RECOGNITION = False             # Not typically needed for lab reports
USE_SEAL_RECOGNITION = False              # Not typically needed for lab reports
USE_REGION_DETECTION = True              # Let layout detection handle regions

# Model names (requested three set; others None) - DO NOT CHANGE
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

# Text detection tuning
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None

# Seal detection tuning
SEAL_DET_LIMIT_SIDE_LEN = None
SEAL_DET_LIMIT_TYPE = None
SEAL_DET_THRESH = None
SEAL_DET_BOX_THRESH = None
SEAL_DET_UNCLIP_RATIO = None

# Recognition thresholds/batches
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None
TEXTLINE_ORIENTATION_BATCH_SIZE = None
FORMULA_RECOGNITION_BATCH_SIZE = None
CHART_RECOGNITION_BATCH_SIZE = None
SEAL_TEXT_RECOGNITION_BATCH_SIZE = None
SEAL_REC_SCORE_THRESH = None

# Predict-time table recognition defaults - TUNED FOR LAB REPORTS
# CRITICAL: use_table_orientation_classify MUST be False to avoid PaddleOCR 3.2.0 UnboundLocalError bug
DEFAULT_USE_OCR_RESULTS_WITH_TABLE_CELLS = None   # Use OCR results for better cell text accuracy
DEFAULT_USE_E2E_WIRED_TABLE_REC_MODEL = None      # Use default (None lets library decide)
DEFAULT_USE_E2E_WIRELESS_TABLE_REC_MODEL = None   # Use default (None lets library decide)
DEFAULT_USE_WIRED_TABLE_CELLS_TRANS_TO_HTML = None    # Use default (None lets library decide)
DEFAULT_USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML = None # Use default (None lets library decide)
DEFAULT_USE_TABLE_ORIENTATION_CLASSIFY = False    # MUST be False to avoid library bug at line 1310

# ==============================
# END OF CONFIGURATION
# ==============================

def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_exceeds_limit(tmp_path: Path) -> bool:
    try:
        return tmp_path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024
    except Exception:
        return False

app = FastAPI(title="PP-StructureV3 API (ARM64, native)", version="3.2.0")

# Init kwargs builder (filters None so library defaults apply)
def _build_init_kwargs() -> Dict[str, Any]:
    params = dict(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        use_tensorrt=USE_TENSORRT,
        precision=PRECISION,
        mkldnn_cache_capacity=MKLDNN_CACHE_CAPACITY,
        cpu_threads=CPU_THREADS,
        paddlex_config=PADDLEX_CONFIG,

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

# Build final init kwargs
_init_kwargs = _build_init_kwargs()

# Initialize pipeline with all tuning applied via _init_kwargs (no overrides below)
pipeline = PPStructureV3(**_init_kwargs)

# Gate to enforce one-at-a-time inference
predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)

def predict_collect_one(path: Path,
                        use_ocr_results_with_table_cells,
                        use_e2e_wired_table_rec_model,
                        use_e2e_wireless_table_rec_model,
                        use_wired_table_cells_trans_to_html,
                        use_wireless_table_cells_trans_to_html,
                        use_table_orientation_classify) -> Dict[str, Any]:
    kwargs = {}
    
    # Apply user override or use default from top configuration section
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
    
    # Filter out None values to let library use its internal defaults
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    outputs = pipeline.predict(str(path), **kwargs)
    pages = []
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        json_files, md_files = [], []
        for res in outputs:
            res.save_to_json(save_path=str(out_dir))
            res.save_to_markdown(save_path=str(out_dir))
        for p in sorted(out_dir.glob("*.json")):
            json_files.append(p)
        for p in sorted(out_dir.glob("*.md")):
            md_files.append(p)
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
    return {"pages": pages}

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
            for pidx, page in enumerate(f["pages"]):
                combined_md += f"## Page {pidx+1}\n\n{page['markdown']}\n\n"
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

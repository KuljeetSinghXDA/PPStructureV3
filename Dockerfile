# Build on ARM64 host or use: docker build --platform=linux/arm64/v8
FROM --platform=linux/arm64/v8 python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4

# System deps for PDFs and image backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# PaddlePaddle nightly CPU (ARM64) and PaddleOCR 3.3.0 doc-parser group
RUN python -m pip install --upgrade pip && \
    python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    python -m pip install "paddleocr[doc-parser]==3.3.0" fastapi "uvicorn[standard]" python-multipart pymupdf

# Embedded FastAPI app:
# - One long-lived PP-StructureV3 with requested models and accuracy tuning
# - All supported init args exposed (None => library defaults)
# - All documented table predict-time flags exposed (None => default)
# - Native JSON/Markdown outputs; HPI disabled for ARM64 stability
RUN cat > /app.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Literal
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

def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _file_exceeds_limit(tmp_path: Path) -> bool:
    try:
        return tmp_path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024
    except Exception:
        return False

app = FastAPI(title="PP-StructureV3 API (ARM64, native)", version="3.3.0")

# ==============================
# Supported PP-StructureV3 params (None => defaults)
# ==============================
# Backend/config toggles
DEVICE: Optional[str] = None
ENABLE_MKLDNN: Optional[bool] = None
ENABLE_HPI: Optional[bool] = None      # Keep None/False on ARM64 to avoid PaddleX HPI plugin
USE_TENSORRT: Optional[bool] = None
PRECISION: Optional[str] = None
MKLDNN_CACHE_CAPACITY: Optional[int] = None

# Threads
CPU_THREADS: Optional[int] = 4         # Ampere A1 4 OCPU single-process

# Optional PaddleX config passthrough (kept None)
PADDLEX_CONFIG: Optional[str] = None

# Subpipeline toggles
USE_DOC_ORIENTATION_CLASSIFY: Optional[bool] = None
USE_DOC_UNWARPING: Optional[bool] = None
USE_TEXTLINE_ORIENTATION: Optional[bool] = None
USE_TABLE_RECOGNITION: Optional[bool] = None
USE_FORMULA_RECOGNITION: Optional[bool] = None
USE_CHART_RECOGNITION: Optional[bool] = None
USE_SEAL_RECOGNITION: Optional[bool] = None
USE_REGION_DETECTION: Optional[bool] = None

# Model names (requested three set; others None)
LAYOUT_DETECTION_MODEL_NAME: Optional[str] = "PP-DocLayout-L"
REGION_DETECTION_MODEL_NAME: Optional[str] = None
TEXT_DETECTION_MODEL_NAME: Optional[str] = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME: Optional[str] = "en_PP-OCRv5_mobile_rec"
TABLE_CLASSIFICATION_MODEL_NAME: Optional[str] = None
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME: Optional[str] = None
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME: Optional[str] = None
WIRED_TABLE_CELLS_DET_MODEL_NAME: Optional[str] = None
WIRELESS_TABLE_CELLS_DET_MODEL_NAME: Optional[str] = None
TABLE_ORIENTATION_CLASSIFY_MODEL_NAME: Optional[str] = None
FORMULA_RECOGNITION_MODEL_NAME: Optional[str] = None
DOC_ORIENTATION_CLASSIFY_MODEL_NAME: Optional[str] = None
DOC_UNWARPING_MODEL_NAME: Optional[str] = None
TEXTLINE_ORIENTATION_MODEL_NAME: Optional[str] = None
SEAL_TEXT_DETECTION_MODEL_NAME: Optional[str] = None
SEAL_TEXT_RECOGNITION_MODEL_NAME: Optional[str] = None
CHART_RECOGNITION_MODEL_NAME: Optional[str] = None

# Model dirs
LAYOUT_DETECTION_MODEL_DIR: Optional[str] = None
REGION_DETECTION_MODEL_DIR: Optional[str] = None
TEXT_DETECTION_MODEL_DIR: Optional[str] = None
TEXT_RECOGNITION_MODEL_DIR: Optional[str] = None
TABLE_CLASSIFICATION_MODEL_DIR: Optional[str] = None
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR: Optional[str] = None
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR: Optional[str] = None
WIRED_TABLE_CELLS_DET_MODEL_DIR: Optional[str] = None
WIRELESS_TABLE_CELLS_DET_MODEL_DIR: Optional[str] = None
TABLE_ORIENTATION_CLASSIFY_MODEL_DIR: Optional[str] = None
FORMULA_RECOGNITION_MODEL_DIR: Optional[str] = None
DOC_ORIENTATION_CLASSIFY_MODEL_DIR: Optional[str] = None
DOC_UNWARPING_MODEL_DIR: Optional[str] = None
TEXTLINE_ORIENTATION_MODEL_DIR: Optional[str] = None
SEAL_TEXT_DETECTION_MODEL_DIR: Optional[str] = None
SEAL_TEXT_RECOGNITION_MODEL_DIR: Optional[str] = None
CHART_RECOGNITION_MODEL_DIR: Optional[str] = None

# Layout thresholds/controls
LAYOUT_THRESHOLD: Optional[float] = None
LAYOUT_NMS: Optional[bool] = None
LAYOUT_UNCLIP_RATIO: Optional[float] = None
LAYOUT_MERGE_BBOXES_MODE: Optional[str] = None

# Text detection tuning (explicit accuracy tuning applied below)
TEXT_DET_LIMIT_SIDE_LEN: Optional[int] = None
TEXT_DET_LIMIT_TYPE: Optional[str] = None
TEXT_DET_THRESH: Optional[float] = None
TEXT_DET_BOX_THRESH: Optional[float] = None
TEXT_DET_UNCLIP_RATIO: Optional[float] = None

# Seal detection tuning
SEAL_DET_LIMIT_SIDE_LEN: Optional[int] = None
SEAL_DET_LIMIT_TYPE: Optional[str] = None
SEAL_DET_THRESH: Optional[float] = None
SEAL_DET_BOX_THRESH: Optional[float] = None
SEAL_DET_UNCLIP_RATIO: Optional[float] = None

# Recognition thresholds/batches
TEXT_REC_SCORE_THRESH: Optional[float] = None
TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None
TEXTLINE_ORIENTATION_BATCH_SIZE: Optional[int] = None
FORMULA_RECOGNITION_BATCH_SIZE: Optional[int] = None
CHART_RECOGNITION_BATCH_SIZE: Optional[int] = None
SEAL_TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None
SEAL_REC_SCORE_THRESH: Optional[float] = None

# Chart batch (explicit, already above)
# CHART_RECOGNITION_BATCH_SIZE present

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

# Build final init kwargs and overlay accuracy tuning explicitly
_init_kwargs = _build_init_kwargs()

pipeline = PPStructureV3(
    **_init_kwargs,
    # Accuracy-oriented detection tuning for medical lab reports
    text_det_limit_side_len=1920,
    text_det_limit_type="max",
    text_det_thresh=0.20,
    text_det_box_thresh=0.30,
    text_det_unclip_ratio=2.5
)

# Gate to enforce one-at-a-time inference
predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)

def predict_collect_one(path: Path,
                        use_ocr_results_with_table_cells: Optional[bool],
                        use_e2e_wired_table_rec_model: Optional[bool],
                        use_e2e_wireless_table_rec_model: Optional[bool],
                        use_wired_table_cells_trans_to_html: Optional[bool],
                        use_wireless_table_cells_trans_to_html: Optional[bool],
                        use_table_orientation_classify: Optional[bool]) -> Dict[str, Any]:
    kwargs = {}
    if use_ocr_results_with_table_cells is not None:
        kwargs["use_ocr_results_with_table_cells"] = use_ocr_results_with_table_cells
    if use_e2e_wired_table_rec_model is not None:
        kwargs["use_e2e_wired_table_rec_model"] = use_e2e_wired_table_rec_model
    if use_e2e_wireless_table_rec_model is not None:
        kwargs["use_e2e_wireless_table_rec_model"] = use_e2e_wireless_table_rec_model
    if use_wired_table_cells_trans_to_html is not None:
        kwargs["use_wired_table_cells_trans_to_html"] = use_wired_table_cells_trans_to_html
    if use_wireless_table_cells_trans_to_html is not None:
        kwargs["use_wireless_table_cells_trans_to_html"] = use_wireless_table_cells_trans_to_html
    if use_table_orientation_classify is not None:
        kwargs["use_table_orientation_classify"] = use_table_orientation_classify

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
    # Optional predict-time table flags
    use_ocr_results_with_table_cells: Optional[bool] = Query(None),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None),
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None),
    use_table_orientation_classify: Optional[bool] = Query(None),
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
        # Concatenate per-file per-page markdowns into a single markdown string
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

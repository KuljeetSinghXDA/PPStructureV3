# fastapi_app.py
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple, Literal

import uvicorn
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

# Models: PaddleOCR PP-StructureV3
from paddleocr import PPStructureV3

# ================= Service Configuration =================
DEVICE = "cpu"
CPU_THREADS = 4
ENABLE_MKLDNN = True
ENABLE_HPI = False

# Subpipeline toggles (explicit booleans)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model overrides (explicit names)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (high recall but CPU-friendly)
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None

# I/O and limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1  # tuned for CPU-only

# PDF rasterization
PDF_DPI_DEFAULT = 300  # 300–400 DPI recommended for small text

# ================= FastAPI App and Lifespan =================
app = FastAPI(title="PPStructureV3 OCR API", version="2.0.0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    # Initialize a single, long-lived pipeline (recommended in docs/tech report)
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
    )
    yield

app.router.lifespan_context = lifespan

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= Helpers =================
def _suffix(name: str) -> str:
    return Path(name or "").suffix.lower()

def _validate_upload(file: UploadFile) -> str:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Missing file")
    ext = _suffix(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
    return ext

def _parse_page_range(page_range_str: Optional[str], num_pages: int) -> List[int]:
    if not page_range_str:
        return list(range(num_pages))
    selected = set()
    for part in page_range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a else 1
            end = int(b) if b else num_pages
            if start > end:
                raise HTTPException(status_code=400, detail="Invalid page range: start > end")
            selected.update(range(start - 1, end))
        else:
            selected.add(int(part) - 1)
    return sorted([i for i in selected if 0 <= i < num_pages])

def _rasterize_pdf_pymupdf(pdf_path: Path, out_dir: Path, dpi: int, page_indices: Optional[List[int]] = None) -> List[Path]:
    import fitz
    doc = fitz.open(pdf_path.as_posix())
    try:
        n = doc.page_count
        if page_indices is None:
            page_indices = list(range(n))
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        img_paths: List[Path] = []
        for idx in page_indices:
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=mat)
            out_path = out_dir / f"page_{idx+1:04d}.png"
            pix.save(out_path.as_posix())
            img_paths.append(out_path)
        return img_paths
    finally:
        doc.close()

# ================= /parse Endpoint =================
@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="Supported: " + ", ".join(sorted(ALLOWED_EXTENSIONS))),
    output_format: Literal["json", "markdown"] = Query("json"),
    page_range: Optional[str] = Query(None, description="1-based ranges for PDFs, e.g., '1-3,5'"),
    dpi: int = Query(PDF_DPI_DEFAULT, ge=150, le=600, description="PDF rasterization DPI"),
    use_chart_recognition: Optional[bool] = Query(None, description="Predict-time toggle if supported"),
):
    ext = _validate_upload(file)

    # Persist upload in a temp directory via streaming
    temp_dir = tempfile.mkdtemp(prefix="ppsv3_")
    try:
        src_path = Path(temp_dir) / ("upload" + ext)
        size = 0
        with open(src_path, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large (>{MAX_FILE_SIZE_MB}MB)")
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Prepare inputs for pipeline.predict
        inputs: List[str] = []
        if ext == ".pdf":
            # High-DPI pre-render
            out_dir = Path(temp_dir) / "pages"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Resolve pages
            import fitz
            with fitz.open(src_path.as_posix()) as doc:
                pages = _parse_page_range(page_range, doc.page_count)
            img_paths = _rasterize_pdf_pymupdf(src_path, out_dir, dpi=dpi, page_indices=pages)
            if not img_paths:
                return JSONResponse({"filename": file.filename, "message": "No pages selected"}, status_code=200)
            inputs = [p.as_posix() for p in img_paths]
        else:
            inputs = [src_path.as_posix()]

        # Predict-time kwargs (only documented/supported flags)
        predict_kwargs = {}
        if use_chart_recognition is not None:
            predict_kwargs["use_chart_recognition"] = use_chart_recognition

        # Call OCR under concurrency guard
        with app.state.predict_sem:
            try:
                results = await run_in_threadpool(app.state.pipeline.predict, input=inputs, **predict_kwargs)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

        if not isinstance(results, (list, tuple)):
            results = [results]

        # Format output
        if output_format == "markdown":
            parts: List[str] = []
            for res in results:
                if hasattr(res, "markdown"):
                    md = res.markdown
                    parts.append(md if isinstance(md, str) else str(md))
                elif hasattr(res, "to_markdown"):
                    parts.append(res.to_markdown())
                else:
                    parts.append(str(getattr(res, "text", "")))
            concatenated = "\n\n---\n\n".join(parts)
            return PlainTextResponse(concatenated, media_type="text/markdown")

        # Default: json
        pages_out: List[dict] = []
        for idx, res in enumerate(results, start=1):
            if hasattr(res, "to_dict"):
                d = res.to_dict()
            elif hasattr(res, "json"):
                d = res.json
            elif isinstance(res, dict):
                d = res
            else:
                d = {"result": str(res)}
            pages_out.append({"index": idx, "data": d})

        return JSONResponse({"filename": file.filename, "pages": pages_out})

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Optional: run with `uvicorn fastapi_app:app --host 0.0.0.0 --port 8000`
    uvicorn.run(app, host="0.0.0.0", port=8000)

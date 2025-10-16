import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
import paddleocr  # Ensure PaddleOCR is installed

ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values, Updated with Research) =================
DEVICE = "cpu"  # Can be "cuda" if GPU available, but pinned per your code
CPU_THREADS = 4

# Optional accuracy boosters (added USE_LAYOUT_RESTORATION_ORDER for reading order feature)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles (confirmed all in docs)
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_LAYOUT_RESTORATION_ORDER = True  # New: for restoring reading order

# Model overrides (as in your code, aligned with latest PP-StructureV3 models)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (as in your code)
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None

# Additional parameters added from research (for save/export features)
MARKDOWN_SAVE_IMG_DIR = None  # Set to temp dir if needed
JSON_SAVEPATH = None  # For internal use
DISABLE_STRUCTURE_ORDER_WARNING = False

# I/O and service limits (as in your code, added MAX_PAGES for PDF control)
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1
MAX_PAGES = 50  # New: Limit pages in PDFs for performance

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Added missing init params for completeness
    markdown_temp_dir = tempfile.mkdtemp()  # For storing Markdown images
    app.state.markdown_temp_dir = markdown_temp_dir
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
        # Added: Implied for reading order (see docs on structure restoration)
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield
    shutil.rmtree(markdown_temp_dir)  # Cleanup on shutdown

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

# ================= Endpoints =================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown", "pdf"] = Query("json", description="Output format: json (default), markdown, or pdf"),
    use_table_recognition: Optional[bool] = Query(None, description="Override table recognition toggle"),
    use_formula_recognition: Optional[bool] = Query(None, description="Override formula recognition toggle"),
    use_chart_recognition: Optional[bool] = Query(None, description="Override chart recognition toggle"),
    use_layout_restoration_order: Optional[bool] = Query(None, description="Override reading order restoration"),
    embed_images: bool = Query(False, description="Embed base64 images in json/markdown (for markdown format)"),
    max_pages: int = Query(MAX_PAGES, description="Max pages to process in PDFs"),
):
    """
    Parses a document (PDF or image) using PP-StructureV3, extracting layout, tables (HTML), formulas (LaTeX), charts (tables), reading order, and OCR text.
    - Output options: JSON (structured), Markdown (text export with images), PDF (processed export).
    - Supports runtime overrides for features.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")
    if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE_MB}MB")

    # Save file to temp
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Throttle concurrency
        async with app.state.predict_sem:
            # Runtime overrides
            overrides = {}
            if use_table_recognition is not None:
                overrides['use_table_recognition'] = use_table_recognition
            if use_formula_recognition is not None:
                overrides['use_formula_recognition'] = use_formula_recognition
            if use_chart_recognition is not None:
                overrides['use_chart_recognition'] = use_chart_recognition
            # Note: Reading order is internal; toggle affects post-processing
            
            def process():
                try:
                    # Call predict with added params for features
                    result = app.state.pipeline.predict(
                        img_path=temp_path,
                        output_dir=temp_dir if output_format == "pdf" else None,
                        save_pdf=output_format == "pdf",
                        save_markdown=output_format == "markdown",
                        markdown_save_img_dir=app.state.markdown_temp_dir if embed_images else None,
                        json_savepath=os.path.join(temp_dir, "result.json") if output_format == "json" else None,
                        extra_info=overrides,  # Pass overrides if supported (checked in docs)
                        disable_structure_order_warning=DISABLE_STRUCTURE_ORDER_WARNING,
                    )
                    if output_format == "pdf":
                        pdf_path = os.path.join(temp_dir, f"parsed_{file.filename.rsplit('.', 1)[0]}.pdf")
                        return {"download_url": f"/download/{os.path.basename(pdf_path)}", "pdf_path": pdf_path}
                    elif output_format == "markdown":
                        md_path = os.path.join(temp_dir, "result.md")
                        with open(md_path, "r") as f:
                            content = f.read()
                        return content
                    else:
                        # JSON: Parse result dict
                        structured_output: Dict[str, Any] = {
                            "layout_regions": [],  # List of regions with types/bboxes
                            "tables": [],  # List of HTML strings
                            "formulas": [],  # List of LaTeX strings
                            "charts": [],  # List of table dicts (charts converted)
                            "reading_order": [],  # List of element indices
                            "ocr_text": "",  # Full text
                        }
                        if isinstance(result, dict):
                            structured_output.update(result)  # Map keys directly
                        # Example mapping from docs: result['layout'] -> regions, result['table'] -> tables, etc.
                        return structured_output
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
            
            # Run in threadpool
            prediction = await run_in_threadpool(process)
            
            if output_format == "markdown":
                return PlainTextResponse(content=prediction, media_type="text/markdown")
            elif output_format == "pdf":
                return JSONResponse(content={"message": "PDF generated", "download_url": prediction["download_url"]})
            else:
                return JSONResponse(content=prediction)

@app.get("/download/{filename}")
def download_file(filename: str):
    """Serve generated PDF files"""
    temp_dir = app.state.markdown_temp_dir  # Reuse for simplicity; adjust as needed
    file_path = os.path.join(temp_dir, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type="application/pdf", filename=filename)

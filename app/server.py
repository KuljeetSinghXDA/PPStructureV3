import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes

ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values) =================
DEVICE = "cpu"
CPU_THREADS = 4

# Optional accuracy boosters
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model overrides
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None


# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= Helper Functions =================
def validate_file(file: UploadFile) -> None:
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    file.file.seek(0, 2)  # Seek to end
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)  # Reset
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit.")

def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    try:
        return convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

async def process_image_with_pipeline(image: Image.Image, request: Request, overrides: dict) -> dict:
    with app.state.predict_sem:
        pipeline = app.state.pipeline
        # Override pipeline parameters if provided (keeps defaults otherwise)
        for key, value in overrides.items():
            if hasattr(pipeline, key):
                setattr(pipeline, key, value)
        # Run OCR: Use return_md=True for Markdown output, covering all features
        # Note: PPStructureV3 returns dict with 'layout', 'tables', 'formulas', 'charts', 'md'
        try:
            result = await run_in_threadpool(lambda: pipeline(image=image, return_md=True))
            # Structure output to cover all features
            output = {
                "status": "success",
                "layout": result.get("layout", []),  # List of regions (type, bbox, etc.)
                "tables": result.get("tables", []),  # Extracted table data
                "formulas": result.get("formulas", []),  # Recognized formulas
                "charts": result.get("charts", []),  # Chart extractions
                "markdown": result.get("md", "")  # Full Markdown doc with reading order
            }
            return output
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse", response_model=None)
async def parse_document(
    request: Request,
    file: UploadFile = File(...),
    use_doc_orientation_classify: Optional[bool] = Query(USE_DOC_ORIENTATION_CLASSIFY, description="Enable document orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(USE_DOC_UNWARPING, description="Enable document unwarping"),
    use_textline_orientation: Optional[bool] = Query(USE_TEXTLINE_ORIENTATION, description="Enable textline orientation detection"),
    use_table_recognition: Optional[bool] = Query(USE_TABLE_RECOGNITION, description="Enable table recognition"),
    use_formula_recognition: Optional[bool] = Query(USE_FORMULA_RECOGNITION, description="Enable formula recognition"),
    use_chart_recognition: Optional[bool] = Query(USE_CHART_RECOGNITION, description="Enable chart recognition")
) -> JSONResponse | PlainTextResponse:
    """
    Parse a document (image or PDF) using PP-StructureV3 pipeline.
    - Supports layout detection (20 categories), table/formula/chart recognition, reading order, and Markdown output.
    - For PDFs, processes first page (extend for multi-page if needed).
    - Returns JSON with structured results and Markdown; use Accept: text/markdown for Markdown-only.
    """
    # Validate file
    validate_file(file)
    
    # Read file
    file_bytes = await file.read()
    
    # Handle PDF: Convert to first page image
    file_ext = Path(file.filename).suffix.lower()
    if file_ext == ".pdf":
        images = pdf_to_images(file_bytes)
        if not images:
            raise HTTPException(status_code=400, detail="PDF has no pages.")
        image = images[0]  # Process first page
    else:
        # For images, open directly
        image = Image.open(BytesIO(file_bytes))
    
    # Prepare overrides from query params
    overrides = {
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "use_textline_orientation": use_textline_orientation,
        "use_table_recognition": use_table_recognition,
        "use_formula_recognition": use_formula_recognition,
        "use_chart_recognition": use_chart_recognition,
    }
    
    # Process with pipeline
    result = await process_image_with_pipeline(image, request, overrides)
    
    # Return based on Accept header
    accept = request.headers.get("accept", "application/json")
    if "text/markdown" in accept:
        return PlainTextResponse(content=result["markdown"], media_type="text/markdown")
    return JSONResponse(content=result)

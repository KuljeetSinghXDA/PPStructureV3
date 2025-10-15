import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

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
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
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
    # Cleanup is handled by Python's garbage collector

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

def _is_allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

async def _parse_document(
    pipeline,
    file_path: str,
    output_format: Literal["json", "markdown"]
) -> str:
    """
    Parse a document using the PPStructureV3 pipeline.
    
    Args:
        pipeline: The initialized PPStructureV3 pipeline.
        file_path: Path to the input file.
        output_format: The desired output format ('json' or 'markdown').
    
    Returns:
        The parsed document as a string in the specified format.
    """
    try:
        # Run the prediction
        results = await run_in_threadpool(pipeline.predict, file_path)
        
        if not results:
            raise HTTPException(status_code=400, detail="No content could be parsed from the document.")
        
        # For now, we handle the first result. PPStructureV3 can return multiple results for multi-page PDFs.
        # A more robust implementation would aggregate all pages.
        result = results[0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "output")
            
            if output_format == "json":
                result.save_to_json(save_path=temp_path)
                output_file = temp_path + ".json"
            else:  # markdown
                result.save_to_markdown(save_path=temp_path)
                output_file = temp_path + ".md"
            
            with open(output_file, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        # It's important to catch and re-raise as HTTPException for proper error handling in FastAPI
        raise HTTPException(status_code=500, detail=f"Document parsing failed: {str(e)}")

@app.post("/parse")
async def parse_endpoint(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown"] = Query("markdown", description="Output format for the parsed document.")
):
    """
    Parse a document (PDF or image) and return its structured content.
    
    - **file**: The document to parse. Must be a PDF, JPG, JPEG, PNG, or BMP.
    - **output_format**: The format for the response. Can be 'json' or 'markdown'.
    """
    # --- Input Validation ---
    if not _is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    file_size = await file.read()
    if len(file_size) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit."
        )
    
    # --- File Handling ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        temp_file.write(file_size)
        temp_file_path = temp_file.name

    try:
        # --- Prediction with Concurrency Control ---
        with app.state.predict_sem:
            parsed_content = await _parse_document(
                pipeline=app.state.pipeline,
                file_path=temp_file_path,
                output_format=output_format
            )
        
        # --- Prepare Response ---
        if output_format == "json":
            try:
                json_content = json.loads(parsed_content)
                return JSONResponse(content=json_content)
            except json.JSONDecodeError:
                # Fallback in case the JSON is malformed, return as plain text
                return PlainTextResponse(content=parsed_content, media_type="application/json")
        else:
            return PlainTextResponse(content=parsed_content, media_type="text/markdown")
            
    finally:
        # Ensure the temporary file is always cleaned up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/health")
def health():
    return {"status": "ok"}

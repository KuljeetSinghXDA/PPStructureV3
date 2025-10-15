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

# ================= Configuration (Pinned for v3.2.0) =================
# Core Settings
ENABLE_HPI = False
ENABLE_MKLDNN = True
DEVICE = "cpu"
CPU_THREADS = 4

# Accuracy Boosters (Disabled for speed)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline Toggles
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model Overrides (Using latest v3.2.0 models)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/Recognition Parameters (Using defaults by passing None)
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None

# I/O and Service Limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1


# ================= App & Lifespan Management =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize and clean up the PPStructureV3 pipeline.
    """
    # Initialize the PPStructureV3 pipeline with the configured parameters.
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
    # Semaphore to control concurrent predictions, as the pipeline is not thread-safe.
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield
    # Cleanup: The pipeline object is automatically garbage collected.


# Import must be after configuration for correct initialization.
from paddleocr import PPStructureV3

app = FastAPI(
    title="PPStructureV3 /parse API",
    description="An API endpoint for parsing document images and PDFs into structured Markdown and JSON using PaddleOCR's PP-StructureV3.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_class=JSONResponse)
def health():
    """Health check endpoint."""
    return {"status": "ok"}


def _is_allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


async def _process_file_in_thread(
    pipeline, temp_file_path: str, output_format: Literal["markdown", "json"]
) -> str:
    """
    Runs the PPStructureV3 prediction in a separate thread and returns the result as a string.
    """
    def _run_pipeline():
        results = pipeline.predict(input=temp_file_path)

        # PPStructureV3 returns a list of result objects.
        # For a single file, we typically get one result object.
        if not results:
            raise HTTPException(status_code=500, detail="Pipeline returned no results.")

        result_obj = results[0]

        # Use a temporary directory to save the output file from the pipeline's method.
        with tempfile.TemporaryDirectory() as temp_output_dir:
            output_path = Path(temp_output_dir) / "output"
            if output_format == "markdown":
                result_obj.save_to_markdown(save_path=str(output_path))
                output_file = output_path.with_suffix('.md')
            else: # json
                result_obj.save_to_json(save_path=str(output_path))
                output_file = output_path.with_suffix('.json')

            if not output_file.exists():
                raise HTTPException(status_code=500, detail=f"Failed to generate {output_format} output.")

            return output_file.read_text(encoding='utf-8')

    return await run_in_threadpool(_run_pipeline)


@app.post("/parse", response_class=PlainTextResponse)
async def parse_document(
    file: UploadFile = File(..., description="The document file to parse. Supports PDF, JPG, JPEG, PNG, BMP."),
    output_format: Literal["markdown", "json"] = Query(
        default="markdown",
        description="The desired output format."
    )
):
    """
    Parses an uploaded document (PDF or image) and returns its structured content.

    This endpoint uses PaddleOCR's PP-StructureV3 pipeline to perform layout analysis,
    OCR, and table recognition, converting the input into either Markdown or JSON.
    """
    # --- 1. Validate the uploaded file ---
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

    # --- 2. Save the file to a temporary location ---
    file_extension = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_size)
        temp_file_path = temp_file.name

    # Ensure cleanup of the temporary file
    try:
        # --- 3. Acquire semaphore and run prediction ---
        pipeline = app.state.pipeline
        predict_sem = app.state.predict_sem

        if not predict_sem.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Server is busy. Please try again later.")

        try:
            result_content = await _process_file_in_thread(pipeline, temp_file_path, output_format)
        finally:
            predict_sem.release()

        return PlainTextResponse(content=result_content, media_type="text/plain")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error `e` in a real application
        raise HTTPException(status_code=500, detail=f"Internal server error during processing: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

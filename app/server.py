go a deep research, take your time and plan /parse endpoint script which should be pinnacle for ppstructurev3. Then generate script /parse endpoint, do not touch  or provide suggestions for other code other than /parse endpoint. 

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
USE_DOC_ORIENTATION_CLASSIFY = None
USE_DOC_UNWARPING = None
USE_TEXTLINE_ORIENTATION = None

# Subpipeline toggles
USE_TABLE_RECOGNITION = None
USE_FORMULA_RECOGNITION = None
USE_CHART_RECOGNITION = None

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

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="The document image or PDF file to process."),
    lang: Literal["en", "ch", "ch_tra", "ko", "ja"] = Query("en", description="Language of the text in the document."),
    mode: Literal["structure", "html", "markdown"] = Query("structure", description="Output format for table and formula results.")
):
    """
    Analyzes an uploaded document (image or PDF) using PPStructureV3 to extract layout,
    text, tables, and formulas. Respects concurrency limits set by MAX_PARALLEL_PREDICT.
    """
    pipeline = app.state.pipeline
    semaphore = app.state.predict_sem
    temp_dir = None

    # --- 1. Validation ---
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Must be one of: {list(ALLOWED_EXTENSIONS)}"
        )

    # Read the file chunk by chunk to check size limit
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the limit of {MAX_FILE_SIZE_MB}MB."
        )
    
    # --- 2. Temporary Storage & Setup ---
    try:
        # Create a temporary directory to handle the uploaded file securely
        temp_dir = tempfile.mkdtemp()
        temp_file_path = Path(temp_dir) / file.filename

        # Write the file contents to the temporary path
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)
        
        # --- 3. Synchronous Prediction Helper ---
        # Define a synchronous function to run in the threadpool. This function
        # will handle the blocking semaphore acquisition and the CPU-intensive predict call.
        def _predict_with_semaphore(
            file_path: Path, 
            ppstructure_pipeline: PPStructureV3, 
            sem: threading.Semaphore, 
            input_lang: str,
            input_mode: str
        ) -> List[dict]:
            """Acquires semaphore, runs PPStructureV3 predict, and releases semaphore."""
            try:
                # Blocks until a slot is available based on MAX_PARALLEL_PREDICT
                with sem:
                    # The PPStructureV3 predict call. return_ocr_info is standard for detail.
                    # lang and mode are passed as key runtime parameters.
                    results = ppstructure_pipeline.predict(
                        str(file_path),
                        lang=input_lang,
                        mode=input_mode,
                        return_ocr_info=True
                    )
                    return results
            except Exception as e:
                # Log or handle prediction-specific errors here
                print(f"Prediction error: {e}")
                # Re-raise the exception to be caught by run_in_threadpool wrapper
                raise

        # --- 4. Execution in Threadpool ---
        # Run the synchronous helper function in the threadpool to prevent blocking the event loop
        prediction_results = await run_in_threadpool(
            _predict_with_semaphore,
            temp_file_path,
            pipeline,
            semaphore,
            lang,
            mode
        )

        # --- 5. Response ---
        # PPStructureV3 returns a list of dictionaries (one per page/element)
        return JSONResponse(content=prediction_results)

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        # Catch any other unexpected errors during file I/O or prediction
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during processing: {e}")
    finally:
        # --- 6. Cleanup ---
        # Ensure the temporary directory and all its contents are removed
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                # Log cleanup errors but don't fail the request
                print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")

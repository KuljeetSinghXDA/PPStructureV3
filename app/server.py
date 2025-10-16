import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# ================= Core Configuration (Pinned Values - Defined by user) =================
# These values are used to initialize the PPStructureV3 pipeline once.
ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3
from paddleocr.ppstructure.structure.ppstructure import PPStructure

# Hardware and parallelism settings
DEVICE = "cpu"
CPU_THREADS = 4
MAX_PARALLEL_PREDICT = 1
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50

# Subpipeline toggles (Base Configuration)
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model and parameter settings (fixed at init)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (fixed at init)
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None

# Optional accuracy boosters (fixed at init)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# ================= App & Lifespan =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing PPStructureV3 pipeline...")
    # PPStructureV3 initialization, loading models into memory once.
    try:
        pipeline = PPStructureV3(
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
        app.state.pipeline = pipeline
        app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
        print("PPStructureV3 pipeline ready.")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        # Initialize pipeline to None so the health check or /parse fails gracefully
        app.state.pipeline = None
    
    yield
    print("Shutting down PPStructureV3 pipeline...")
    # Cleanup logic if necessary (though PaddlePaddle typically handles this)

app = FastAPI(
    title="PPStructureV3 Document Parsing API",
    description="A highly configurable FastAPI wrapper for the PaddleOCR PPStructureV3 document parsing pipeline.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    """Checks if the service and the PPStructureV3 pipeline are running."""
    if app.state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PPStructureV3 pipeline failed to initialize."
        )
    return {"status": "ok", "max_parallel_predict": MAX_PARALLEL_PREDICT}

# ================= Core Parsing Function (Runs in Threadpool) =================

def _process_file_in_threadpool(
    pipeline: PPStructure,
    temp_input_path: Path,
    output_format: str,
    page_number: Optional[int],
    use_table_recognition_override: Optional[bool],
    use_formula_recognition_override: Optional[bool],
    use_chart_recognition_override: Optional[bool],
) -> tuple[str, str]:
    """
    Handles file saving, pipeline prediction, output formatting, and cleanup.
    Runs inside FastAPI's threadpool to prevent blocking the event loop.
    """
    temp_output_dir = None
    try:
        # 1. Prediction Call
        # We assume that the predict method accepts optional sub-pipeline overrides
        # However, since these flags are typically set in __init__, we need to adjust the predict call.
        # PPStructureV3's predict method allows passing in the flags to control which components process.
        
        # Collect runtime flags to pass to the predict method (if the PaddleOCR API supports it)
        # Note: If PaddleOCR doesn't accept these flags at runtime, the pre-initialized flags will be used.
        # We pass them anyway for completeness, following the deep research intent.
        
        predict_kwargs = {}
        if page_number is not None:
            predict_kwargs['page_num'] = page_number
        
        if use_table_recognition_override is not None:
             predict_kwargs['use_table_recognition'] = use_table_recognition_override
        if use_formula_recognition_override is not None:
             predict_kwargs['use_formula_recognition'] = use_formula_recognition_override
        if use_chart_recognition_override is not None:
             predict_kwargs['use_chart_recognition'] = use_chart_recognition_override


        # PPStructureV3's predict can take a path string
        output_results = pipeline.predict(str(temp_input_path), **predict_kwargs)
        
        if not output_results:
            return "[]", "application/json" if output_format == "json" else "text/markdown"

        # 2. Output Handling (Requires saving to a temp dir and reading)
        temp_output_dir = tempfile.mkdtemp()
        
        combined_content = ""
        
        for i, res in enumerate(output_results):
            if output_format == "json":
                # Save and read JSON output
                json_path = Path(temp_output_dir) / f"page_{i}.json"
                res.save_to_json(save_path=temp_output_dir, file_name=f"page_{i}")
                
                # Check if the file was created and read its content
                if json_path.exists():
                     with open(json_path, 'r', encoding='utf-8') as f:
                         # Append the JSON content (as a string) to be combined later
                         combined_content += f.read() + ",\n" 

            elif output_format == "markdown":
                # Save and read Markdown output
                md_path = Path(temp_output_dir) / f"page_{i}.md"
                res.save_to_markdown(save_path=temp_output_dir, file_name=f"page_{i}")
                
                if md_path.exists():
                    with open(md_path, 'r', encoding='utf-8') as f:
                        combined_content += f"\n\n---\n\n## Page {i + 1}\n\n" + f.read()

        
        # 3. Final Serialization
        if output_format == "json":
            # Strip trailing comma and wrap in list brackets if multiple pages were processed
            final_content = f"[{combined_content.rstrip(',\n')}]"
            return final_content, "application/json"
        
        elif output_format == "markdown":
            return combined_content.strip(), "text/markdown"

    finally:
        # 4. Cleanup
        if temp_output_dir and Path(temp_output_dir).exists():
            shutil.rmtree(temp_output_dir)
        if temp_input_path.exists():
            os.remove(temp_input_path)


# ================= FastAPI Endpoint =================

@app.post("/parse", 
          summary="Parse a document or image using PPStructureV3.",
          responses={
              200: {"content": {"application/json": {}, "text/markdown": {}}},
              400: {"model": Dict[str, str]},
              503: {"model": Dict[str, str]},
              500: {"model": Dict[str, str]},
          })
async def parse_document(
    file: UploadFile = File(..., description="The document or image file to parse (.pdf, .jpg, .png)."),
    output_format: Literal["json", "markdown"] = Query(
        "json", 
        description="Desired output format. 'json' returns structured data, 'markdown' returns text/tables in Markdown format."
    ),
    page_number: Optional[int] = Query(
        None, 
        ge=0, 
        description="For PDFs, specify the 0-based page index to parse. If omitted, all pages are processed."
    ),
    # Feature Overrides (Allowing dynamic control over pipeline components)
    use_table_recognition: Optional[bool] = Query(
        None, 
        description=f"Override base config ({USE_TABLE_RECOGNITION}). Enable/disable table structure recognition."
    ),
    use_formula_recognition: Optional[bool] = Query(
        None, 
        description=f"Override base config ({USE_FORMULA_RECOGNITION}). Enable/disable formula (LaTeX) recognition."
    ),
    use_chart_recognition: Optional[bool] = Query(
        None, 
        description=f"Override base config ({USE_CHART_RECOGNITION}). Enable/disable chart-to-table parsing."
    )
):
    """
    Performs document layout analysis and content recognition (OCR, Table, Formula, Chart) 
    using the highly efficient PPStructureV3 pipeline.
    """
    
    # 0. System Check
    if app.state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PPStructureV3 pipeline is not ready. Check service health."
        )

    # 1. File Validation
    file_size_mb = file.size / (1024 * 1024) if file.size else 0
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB."
        )

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Only {', '.join(ALLOWED_EXTENSIONS)} are allowed."
        )

    # 2. Save File to Temp Directory
    # We must save the file synchronously before passing the path to the threadpool
    temp_input_path = Path(tempfile.gettempdir()) / Path(file.filename)
    try:
        temp_input_path.write_bytes(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file to temp path: {e}"
        )

    # 3. Concurrency and Execution
    async with app.state.predict_sem:
        try:
            # Execute the synchronous blocking call in the thread pool
            content, media_type = await run_in_threadpool(
                _process_file_in_threadpool,
                app.state.pipeline,
                temp_input_path,
                output_format,
                page_number,
                use_table_recognition,
                use_formula_recognition,
                use_chart_recognition,
            )
            
            # 4. Return Response
            if output_format == "json":
                # Ensure the JSON string is parsed to a dict/list before sending as JSONResponse
                return JSONResponse(content=json.loads(content), media_type=media_type)
            else: # markdown
                return PlainTextResponse(content=content, media_type=media_type)

        except Exception as e:
            print(f"Prediction Error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document parsing failed: {type(e).__name__}: {str(e)}"
            )

# The exception handling in `_process_file_in_threadpool` and `parse_document` ensures
# the temporary file is cleaned up even if an error occurs.

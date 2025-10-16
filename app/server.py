# app/server.py
import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.concurrency import run_in_threadpool

# This is a community-supported HPI; may require extra dependencies if enabled.
# See: https://github.com/HolidayPony/PaddleOCR-HPI
ENABLE_HPI = False

# MKL-DNN is a performance library for Intel CPUs.
# It is safe to enable on ARM64, but will have no effect.
ENABLE_MKLDNN = True


from paddleocr import PPStructureV3

# ================= Core Configuration (Default Values) =================
# These values are used to initialize the pipeline but can be overridden
# by query parameters in the /parse endpoint for per-request customization.

DEVICE = "cpu"
CPU_THREADS = os.cpu_count() or 4

# --- Optional Pre-processing Modules ---
# These can improve accuracy on skewed or rotated documents but add latency.
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# --- Sub-pipeline Toggles ---
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_KIE = False # Key Information Extraction

# --- Model Overrides ---
# Using the latest large/server models by default for max accuracy.
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_server_rec" # Change 'en' for other languages
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S" # Faster version
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"
KIE_MODEL_NAME = "kie_ppstructure_SER_en"

# --- Detection/Recognition Parameters ---
LAYOUT_THRESHOLD = 0.5
TEXT_DET_THRESH = 0.3
TEXT_DET_BOX_THRESH = 0.6
TEXT_DET_UNCLIP_RATIO = 1.6
TEXT_DET_LIMIT_SIDE_LEN = 960
TEXT_DET_LIMIT_TYPE = 'max'
TEXT_REC_SCORE_THRESH = 0.5
TEXT_RECOGNITION_BATCH_SIZE = 6

# --- I/O and Service Limits ---
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 100
MAX_PARALLEL_PREDICT = 1 # Set to 1 to avoid high memory usage with large models.

# ================= App & Lifespan =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the PPStructureV3 pipeline on application startup.
    This is a long-running operation and should only be done once.
    """
    print("Initializing PP-StructureV3 pipeline...")
    # Initialize with default models. Most parameters can be overridden at prediction time.
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
        layout_model_dir=LAYOUT_DETECTION_MODEL_NAME, # Use updated param names
        det_model_dir=TEXT_DETECTION_MODEL_NAME,
        rec_model_dir=TEXT_RECOGNITION_MODEL_NAME,
        table_model_dir=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME, # Combines wired/wireless
        table_char_type='en', # Assumes english table content
        formula_model_dir=FORMULA_RECOGNITION_MODEL_NAME,
        ser_model_dir=KIE_MODEL_NAME,
        # The following params are typically set at prediction time
        use_angle_cls=USE_DOC_ORIENTATION_CLASSIFY,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        lang='en' # Default language
    )
    # Semaphore to limit concurrent predictions and prevent OOM errors.
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    print("Pipeline initialized successfully.")
    yield
    # Clean up resources if needed (though not strictly necessary here)
    del app.state.pipeline


app = FastAPI(
    title="PP-StructureV3 Document Analysis API",
    description="A FastAPI wrapper for PaddleOCR's PP-StructureV3, exposing all major features for document parsing, table/formula recognition, and layout recovery.",
    version="1.1.0",
    lifespan=lifespan
)


def format_to_markdown(result_data: list) -> str:
    """
    Formats the structured JSON output from PP-StructureV3 into Markdown.
    """
    md_string = ""
    for item in result_data:
        region_type = item.get('type', 'text').lower()
        content = item.get('res', '')
        
        if region_type == 'title':
            md_string += f"# {content}\n\n"
        elif region_type == 'text':
            md_string += f"{content}\n\n"
        elif region_type == 'figure':
            md_string += f"![figure]({item.get('img_path', 'image')})\n*Caption: {content}*\n\n"
        elif region_type == 'table':
            # `res` for a table is an HTML string
            md_string += f"### Table\n{content}\n\n"
        elif region_type == 'formula':
            # `res` for a formula is a LaTeX string
            md_string += f"### Formula\n```latex\n{content}\n```\n\n"
        elif region_type == 'header':
            md_string += f"**Header:** {content}\n\n"
        elif region_type == 'footer':
            md_string += f"**Footer:** {content}\n\n"
        else:
            md_string += f"**[{region_type.capitalize()}]**\n{content}\n\n"
            
    return md_string.strip()


@app.get("/health", summary="Check service health")
def health():
    """A simple health check endpoint."""
    return {"status": "ok", "pipeline_initialized": hasattr(app.state, 'pipeline')}

@app.post(
    "/parse",
    summary="Parse a document image or PDF",
    description="Upload a document file (PDF, PNG, JPG, BMP) to perform comprehensive layout analysis and OCR."
)
async def parse(
    file: UploadFile = File(..., description="The document file to parse."),

    # --- Major Feature Toggles ---
    output_format: Literal["json", "markdown"] = Query("json", description="The desired format for the structured text output."),
    recovery: bool = Query(False, description="Enable layout recovery to generate a .docx file and .xlsx for tables. Overrides 'output_format'."),
    page_num: int = Query(0, description="For PDF files, the page number to process (0-indexed). 0 means all pages."),
    
    # --- Sub-pipeline Toggles ---
    use_table_recognition: bool = Query(USE_TABLE_RECOGNITION, description="Enable table detection and structure recognition."),
    use_formula_recognition: bool = Query(USE_FORMULA_RECOGNITION, description="Enable mathematical formula detection and recognition."),
    use_chart_recognition: bool = Query(USE_CHART_RECOGNITION, description="Enable chart detection and recognition."),
    use_kie: bool = Query(USE_KIE, description="Enable Key Information Extraction for form-like documents."),
    
    # --- Fine-tuning Parameters ---
    return_ocr_result_in_table: bool = Query(True, description="If True, OCR results for text inside tables are returned."),
    
    # --- OCR Pre-processing ---
    use_doc_orientation_classify: bool = Query(USE_DOC_ORIENTATION_CLASSIFY, description="Enable document orientation classification (0, 90, 180, 270 degrees)."),
    use_textline_orientation: bool = Query(USE_TEXTLINE_ORIENTATION, description="Enable text line orientation classification (horizontal vs. vertical)."),
    
    # --- Detection Thresholds ---
    layout_threshold: Optional[float] = Query(LAYOUT_THRESHOLD, description="Threshold for layout detection confidence.", ge=0, le=1),
    text_det_limit_side_len: Optional[int] = Query(TEXT_DET_LIMIT_SIDE_LEN, description="Resize image to this size for the long side before text detection."),
):
    """
    **Main endpoint to parse a document.** It provides extensive control over the `PP-StructureV3` pipeline.
    
    - **Basic Usage:** Upload a file to get a structured JSON of the content.
    - **Markdown Output:** Set `output_format=markdown`.
    - **DOCX Recovery:** Set `recovery=true`. The API will return a `.docx` file instead of JSON/Markdown.
    - **PDFs:** By default, it processes all pages. Use `page_num` to specify a single page.
    """
    # 1. File validation
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Please use one of: {ALLOWED_EXTENSIONS}")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB.")

    # 2. Process the file in a thread-safe manner
    temp_dir = None
    try:
        # PPStructure works best with file paths, so we write the uploaded content to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir="./") as temp_file:
            temp_file.write(contents)
            temp_filepath = temp_file.name

        # Create a temporary directory for outputs if recovery is enabled
        if recovery:
            temp_dir = tempfile.mkdtemp(dir="./")

        # Acquire semaphore to limit concurrency
        async with app.state.predict_sem:
            print(f"Starting parsing for file: {file.filename}")
            
            # 3. Call the pipeline using run_in_threadpool to avoid blocking the event loop
            result = await run_in_threadpool(
                app.state.pipeline,
                img_path=temp_filepath,
                page_num=page_num,
                # Pass all runtime parameters to the prediction call
                return_ocr_result_in_table=return_ocr_result_in_table,
                use_visual_backbone=use_kie, # `use_visual_backbone` controls KIE
                recovery=recovery,
                output_dir=temp_dir,
                # Sub-pipelines
                table=use_table_recognition,
                formula=use_formula_recognition,
                chart=use_chart_recognition,
                # Thresholds & settings
                layout_threshold=layout_threshold,
                ocr_dict={
                    "use_angle_cls": use_doc_orientation_classify,
                    "use_textline_orientation": use_textline_orientation,
                    "det_limit_side_len": text_det_limit_side_len,
                 }
            )
            print(f"Finished parsing for file: {file.filename}")

        # 4. Format and return the response
        if recovery and temp_dir:
            # If recovery is on, the model saves files to `temp_dir`.
            # We need to find the .docx file and return it.
            output_files = os.listdir(temp_dir)
            doc_file = next((f for f in output_files if f.endswith('.docx')), None)
            if doc_file:
                doc_path = os.path.join(temp_dir, doc_file)
                return FileResponse(
                    path=doc_path,
                    filename=f"{Path(file.filename).stem}.docx",
                    media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
            else:
                 raise HTTPException(status_code=500, detail="Document recovery was enabled, but no DOCX file was generated.")

        elif output_format == "markdown":
            markdown_content = format_to_markdown(result)
            return PlainTextResponse(content=markdown_content)
            
        else: # Default is JSON
            return JSONResponse(content=result)

    except Exception as e:
        # Broad exception to catch errors from the Paddle pipeline
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
    
    finally:
        # Clean up temporary files and directories
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

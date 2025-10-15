import os
import tempfile
import threading
import json
import shutil
import uuid
import traceback
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# ================= Configuration Flags =================
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

# ================= Request/Response Models =================

class ParseRequest(BaseModel):
    """Request model for parse endpoint"""
    use_doc_orientation_classify: Optional[bool] = Field(None, description="Enable document orientation classification")
    use_doc_unwarping: Optional[bool] = Field(None, description="Enable document unwarping")
    use_textline_orientation: Optional[bool] = Field(None, description="Enable textline orientation classification")
    use_table_recognition: Optional[bool] = Field(None, description="Enable table recognition")
    use_formula_recognition: Optional[bool] = Field(None, description="Enable formula recognition")
    use_chart_recognition: Optional[bool] = Field(None, description="Enable chart recognition")
    use_seal_recognition: Optional[bool] = Field(None, description="Enable seal text recognition")
    use_region_detection: Optional[bool] = Field(None, description="Enable region detection")
    
    layout_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Layout detection threshold")
    text_det_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Text detection pixel threshold")
    text_det_box_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Text detection box threshold")
    text_det_unclip_ratio: Optional[float] = Field(None, gt=0.0, description="Text detection unclip ratio")
    text_rec_score_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Text recognition score threshold")
    
    visualize: Optional[bool] = Field(True, description="Return visualization images")
    return_markdown: Optional[bool] = Field(True, description="Return markdown format")
    return_json: Optional[bool] = Field(True, description="Return JSON format")

class ParseResponse(BaseModel):
    """Response model for parse endpoint"""
    log_id: str = Field(..., description="Request UUID")
    error_code: int = Field(0, description="Error code (0 = success)")
    error_msg: str = Field("Success", description="Error message")
    result: Optional[Dict[str, Any]] = Field(None, description="Parsing results")

# ================= Helper Functions =================

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
        )

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """Save uploaded file to temporary directory"""
    try:
        suffix = Path(upload_file.filename).suffix
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = Path(tmp_file.name)
        
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        return tmp_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    finally:
        upload_file.file.close()

def process_pipeline_results(output: List, params: ParseRequest, log_id: str) -> Dict[str, Any]:
    """Process pipeline output into response format"""
    result = {
        "layout_parsing_results": [],
        "data_info": {
            "log_id": log_id,
            "num_pages": len(output)
        }
    }
    
    for idx, res in enumerate(output):
        page_result = {
            "page_index": idx
        }
        
        # Add pruned result (JSON format)
        if params.return_json:
            try:
                res_json = res.json
                # Remove input_path and page_index from pruned result as per API spec
                if 'res' in res_json:
                    pruned = dict(res_json['res'])
                    pruned.pop('input_path', None)
                    pruned.pop('page_index', None)
                    page_result["pruned_result"] = pruned
            except Exception as e:
                page_result["pruned_result"] = {"error": f"Failed to generate JSON: {str(e)}"}
        
        # Add markdown result
        if params.return_markdown:
            try:
                md_data = res.markdown
                page_result["markdown"] = {
                    "text": md_data.get("text", ""),
                    "is_start": md_data.get("is_start", True),
                    "is_end": md_data.get("is_end", True)
                }
                # Note: Images are handled separately in visualization
            except Exception as e:
                page_result["markdown"] = {"error": f"Failed to generate markdown: {str(e)}"}
        
        # Add visualization images
        if params.visualize:
            try:
                img_data = res.img
                if img_data:
                    page_result["output_images"] = {}
                    for img_name, img_obj in img_data.items():
                        # Images are PIL Image objects, convert to base64 if needed
                        # For now, just indicate they're available
                        page_result["output_images"][img_name] = f"Image available: {img_name}"
            except Exception as e:
                page_result["output_images"] = {"error": f"Failed to get images: {str(e)}"}
        
        result["layout_parsing_results"].append(page_result)
    
    return result

# ================= App & Lifespan =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup pipeline"""
    print("Initializing PPStructureV3 pipeline...")
    
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
    
    print("Pipeline initialized successfully")
    yield
    print("Shutting down pipeline...")

app = FastAPI(
    title="PPStructureV3 Document Parser API",
    description="High-performance document parsing and OCR service using PaddleOCR PPStructureV3",
    version="3.2.0",
    lifespan=lifespan
)

# ================= Endpoints =================

@app.get("/health", response_class=PlainTextResponse)
async def health():
    """Health check endpoint"""
    return "ok"

@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root endpoint"""
    return "PPStructureV3 Document Parser API - Use POST /parse to process documents"

@app.post("/parse", response_model=ParseResponse)
async def parse_document(
    file: UploadFile = File(..., description="Document file (PDF, JPG, PNG, BMP)"),
    use_doc_orientation_classify: Optional[bool] = Query(None, description="Enable document orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(None, description="Enable document unwarping"),
    use_textline_orientation: Optional[bool] = Query(None, description="Enable textline orientation classification"),
    use_table_recognition: Optional[bool] = Query(None, description="Enable table recognition"),
    use_formula_recognition: Optional[bool] = Query(None, description="Enable formula recognition"),
    use_chart_recognition: Optional[bool] = Query(None, description="Enable chart recognition"),
    use_seal_recognition: Optional[bool] = Query(None, description="Enable seal text recognition"),
    use_region_detection: Optional[bool] = Query(None, description="Enable region detection"),
    layout_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Layout detection threshold"),
    text_det_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Text detection pixel threshold"),
    text_det_box_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Text detection box threshold"),
    text_det_unclip_ratio: Optional[float] = Query(None, gt=0.0, description="Text detection unclip ratio"),
    text_rec_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Text recognition score threshold"),
    visualize: Optional[bool] = Query(True, description="Return visualization images"),
    return_markdown: Optional[bool] = Query(True, description="Return markdown format"),
    return_json: Optional[bool] = Query(True, description="Return JSON format")
):
    """
    Parse document (PDF or image) and extract structured content.
    
    This endpoint processes documents through the PPStructureV3 pipeline and returns:
    - Layout analysis results
    - OCR text extraction
    - Table recognition (optional)
    - Formula recognition (optional)
    - Chart parsing (optional)
    - Markdown conversion (optional)
    - Visualization images (optional)
    
    **Supported file formats:** PDF, JPG, JPEG, PNG, BMP
    
    **Maximum file size:** 50 MB
    """
    log_id = str(uuid.uuid4())
    tmp_path = None
    
    try:
        # Validate file
        validate_file(file)
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Seek back to start
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
            )
        
        # Save uploaded file
        tmp_path = await run_in_threadpool(save_upload_file_tmp, file)
        
        # Prepare prediction parameters
        predict_params = ParseRequest(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_seal_recognition=use_seal_recognition,
            use_region_detection=use_region_detection,
            layout_threshold=layout_threshold,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            visualize=visualize,
            return_markdown=return_markdown,
            return_json=return_json
        )
        
        # Acquire semaphore for rate limiting
        app.state.predict_sem.acquire()
        
        try:
            # Build predict kwargs (only include non-None values)
            predict_kwargs = {
                "input": str(tmp_path)
            }
            
            # Add optional parameters if they are not None
            if use_doc_orientation_classify is not None:
                predict_kwargs["use_doc_orientation_classify"] = use_doc_orientation_classify
            if use_doc_unwarping is not None:
                predict_kwargs["use_doc_unwarping"] = use_doc_unwarping
            if use_textline_orientation is not None:
                predict_kwargs["use_textline_orientation"] = use_textline_orientation
            if use_table_recognition is not None:
                predict_kwargs["use_table_recognition"] = use_table_recognition
            if use_formula_recognition is not None:
                predict_kwargs["use_formula_recognition"] = use_formula_recognition
            if use_chart_recognition is not None:
                predict_kwargs["use_chart_recognition"] = use_chart_recognition
            if use_seal_recognition is not None:
                predict_kwargs["use_seal_recognition"] = use_seal_recognition
            if use_region_detection is not None:
                predict_kwargs["use_region_detection"] = use_region_detection
            if layout_threshold is not None:
                predict_kwargs["layout_threshold"] = layout_threshold
            if text_det_thresh is not None:
                predict_kwargs["text_det_thresh"] = text_det_thresh
            if text_det_box_thresh is not None:
                predict_kwargs["text_det_box_thresh"] = text_det_box_thresh
            if text_det_unclip_ratio is not None:
                predict_kwargs["text_det_unclip_ratio"] = text_det_unclip_ratio
            if text_rec_score_thresh is not None:
                predict_kwargs["text_rec_score_thresh"] = text_rec_score_thresh
            
            # Run prediction in threadpool
            output = await run_in_threadpool(
                app.state.pipeline.predict,
                **predict_kwargs
            )
            
        finally:
            app.state.predict_sem.release()
        
        # Process results
        result = await run_in_threadpool(
            process_pipeline_results,
            output,
            predict_params,
            log_id
        )
        
        return ParseResponse(
            log_id=log_id,
            error_code=0,
            error_msg="Success",
            result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Internal server error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {tmp_path}: {e}")

@app.get("/info")
async def get_info():
    """Get API and model information"""
    return {
        "api_version": "3.2.0",
        "paddleocr_version": "3.2.0",
        "models": {
            "layout_detection": LAYOUT_DETECTION_MODEL_NAME,
            "text_detection": TEXT_DETECTION_MODEL_NAME,
            "text_recognition": TEXT_RECOGNITION_MODEL_NAME,
            "wired_table_structure": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
            "wireless_table_structure": WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
            "table_classification": TABLE_CLASSIFICATION_MODEL_NAME,
            "formula_recognition": FORMULA_RECOGNITION_MODEL_NAME,
            "chart_recognition": CHART_RECOGNITION_MODEL_NAME
        },
        "features": {
            "doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
            "doc_unwarping": USE_DOC_UNWARPING,
            "textline_orientation": USE_TEXTLINE_ORIENTATION,
            "table_recognition": USE_TABLE_RECOGNITION,
            "formula_recognition": USE_FORMULA_RECOGNITION,
            "chart_recognition": USE_CHART_RECOGNITION
        },
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "max_parallel_requests": MAX_PARALLEL_PREDICT,
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        },
        "device": DEVICE,
        "cpu_threads": CPU_THREADS
    }

# ================= Main Entry Point =================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )

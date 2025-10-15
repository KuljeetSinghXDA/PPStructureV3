import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import asyncio
import logging
from datetime import datetime
import uuid

# ================= Configuration & Constants =================
# Device configuration - optimized for CPU deployment
DEVICE = "cpu"
CPU_THREADS = 4
ENABLE_MKLDNN = True  # Significant performance boost on Intel CPUs
ENABLE_HPI = False    # Disabled for CPU deployment

# Optional accuracy boosters - disabled for performance by default
USE_DOC_ORIENTATION_CLASSIFY = False  # Set to True if document orientation correction needed
USE_DOC_UNWARPING = False            # Set to True for curved document images
USE_TEXTLINE_ORIENTATION = False     # Uses PP-LCNet_x1_0_textline_ori model if True

# Subpipeline toggles - enable only what you need
USE_TABLE_RECOGNITION = True         # SLANet_plus for table structure
USE_FORMULA_RECOGNITION = False      # PP-FormulaNet_plus-S for formulas
USE_CHART_RECOGNITION = False        # PP-Chart2Table for chart conversion

# Model overrides - mobile models for better performance
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"  # Balanced layout detection
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"  # Mobile-optimized detection
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"  # English mobile recognition
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"  # Smaller formula model
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Performance tuning parameters based on PP-StructureV3 benchmarks
LAYOUT_THRESHOLD = 0.5
TEXT_DET_THRESH = 0.3
TEXT_DET_BOX_THRESH = 0.5
TEXT_DET_UNCLIP_RATIO = 1.6
TEXT_DET_LIMIT_SIDE_LEN = 960  # Reduced from 4096 for better performance
TEXT_DET_LIMIT_TYPE = "max"
TEXT_REC_SCORE_THRESH = 0.5
TEXT_RECOGNITION_BATCH_SIZE = 8  # Optimized for CPU memory constraints

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1  # Conservative for CPU deployment
MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024

# Output configuration
OUTPUT_FORMATS = ["markdown", "json", "html"]
DEFAULT_OUTPUT_FORMAT = "markdown"

# ================= Logging Configuration =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ppstructurev3_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PPStructureV3-API")

# ================= PPStructureV3 Pipeline Initialization =================
def initialize_ppstructurev3():
    """
    Initialize PPStructureV3 pipeline with optimized configuration for PaddleOCR 3.2.0
    Based on official documentation and performance benchmarks
    """
    try:
        from paddleocr import PPStructureV3
        
        pipeline_config = {
            "device": DEVICE,
            "enable_mkldnn": ENABLE_MKLDNN,
            "enable_hpi": ENABLE_HPI, 
            "cpu_threads": CPU_THREADS,
            "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
            "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
            "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,
            "wired_table_structure_recognition_model_name": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
            "wireless_table_structure_recognition_model_name": WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
            "table_classification_model_name": TABLE_CLASSIFICATION_MODEL_NAME,
            "formula_recognition_model_name": FORMULA_RECOGNITION_MODEL_NAME,
            "chart_recognition_model_name": CHART_RECOGNITION_MODEL_NAME,
            "layout_threshold": LAYOUT_THRESHOLD,
            "text_det_thresh": TEXT_DET_THRESH,
            "text_det_box_thresh": TEXT_DET_BOX_THRESH,
            "text_det_unclip_ratio": TEXT_DET_UNCLIP_RATIO,
            "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
            "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
            "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
            "text_recognition_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
            "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
            "use_doc_unwarping": USE_DOC_UNWARPING,
            "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
            "use_table_recognition": USE_TABLE_RECOGNITION,
            "use_formula_recognition": USE_FORMULA_RECOGNITION,
            "use_chart_recognition": USE_CHART_RECOGNITION,
        }
        
        # Remove None values from config
        pipeline_config = {k: v for k, v in pipeline_config.items() if v is not None}
        
        logger.info("Initializing PPStructureV3 pipeline with configuration:")
        for key, value in pipeline_config.items():
            if "model_name" in key or "name" in key:
                logger.info(f"  {key}: {value}")
        
        pipeline = PPStructureV3(**pipeline_config)
        logger.info("PPStructureV3 pipeline initialized successfully")
        return pipeline
        
    except ImportError as e:
        logger.error(f"Failed to import PaddleOCR: {e}")
        raise RuntimeError("PaddleOCR is not installed. Please install it using: pip install paddlepaddle paddleocr")
    except Exception as e:
        logger.error(f"Failed to initialize PPStructureV3 pipeline: {e}")
        raise

# ================= FastAPI Application Setup =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for resource initialization and cleanup
    """
    logger.info("Starting PPStructureV3 API application...")
    
    # Initialize the pipeline
    app.state.pipeline = await run_in_threadpool(initialize_ppstructurev3)
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    app.state.request_count = 0
    
    logger.info("Application startup complete")
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down PPStructureV3 API application...")
    if hasattr(app.state, 'pipeline'):
        del app.state.pipeline

app = FastAPI(
    title="PPStructureV3 Document Parser API",
    description="Production-ready document parsing API using PaddleOCR PP-StructureV3 3.2.0",
    version="3.2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Utility Functions =================
def validate_file_extension(filename: str) -> bool:
    """
    Validate file extension against allowed types
    """
    extension = Path(filename).suffix.lower()
    return extension in ALLOWED_EXTENSIONS

def validate_file_size(content: bytes) -> bool:
    """
    Validate file size against maximum allowed size
    """
    return len(content) <= MAX_CONTENT_LENGTH

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks
    """
    return Path(filename).name

def process_ppstructure_output(results, output_format: str) -> Dict[str, Any]:
    """
    Process PPStructureV3 output into standardized response format
    """
    try:
        markdown_output = []
        structured_data = []
        
        for page_num, result in enumerate(results):
            page_data = {
                "page_number": page_num + 1,
                "layout_regions": [],
                "tables": [],
                "formulas": [],
                "charts": []
            }
            
            # Extract markdown content
            if hasattr(result, 'markdown') and result.markdown:
                md_content = result.markdown
                if isinstance(md_content, dict) and 'markdown' in md_content:
                    markdown_output.append(md_content['markdown'])
                else:
                    markdown_output.append(str(md_content))
            
            # Extract structured layout information
            if hasattr(result, 'layout') and result.layout:
                for layout_region in result.layout:
                    region_data = {
                        "type": getattr(layout_region, 'type', 'unknown'),
                        "bbox": getattr(layout_region, 'bbox', []),
                        "confidence": getattr(layout_region, 'score', 0.0)
                    }
                    page_data["layout_regions"].append(region_data)
            
            # Extract table data if available
            if hasattr(result, 'tables') and result.tables:
                for table in result.tables:
                    table_data = {
                        "bbox": getattr(table, 'bbox', []),
                        "html": getattr(table, 'html', ''),
                        "cells": getattr(table, 'cells', [])
                    }
                    page_data["tables"].append(table_data)
            
            # Extract formula data if available
            if hasattr(result, 'formulas') and result.formulas:
                for formula in result.formulas:
                    formula_data = {
                        "bbox": getattr(formula, 'bbox', []),
                        "latex": getattr(formula, 'latex', ''),
                        "confidence": getattr(formula, 'score', 0.0)
                    }
                    page_data["formulas"].append(formula_data)
            
            structured_data.append(page_data)
        
        # Combine markdown from all pages
        combined_markdown = "\n\n--- Page Break ---\n\n".join(markdown_output)
        
        response_data = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": None,  # Will be filled by the endpoint
            "metadata": {
                "total_pages": len(results),
                "output_format": output_format
            }
        }
        
        # Format response based on requested output format
        if output_format == "json":
            response_data["structured_data"] = structured_data
            response_data["markdown"] = combined_markdown
        elif output_format == "html":
            # Basic HTML conversion from markdown
            html_content = combined_markdown.replace('\n', '<br>')
            response_data["html"] = f"<div>{html_content}</div>"
            response_data["markdown"] = combined_markdown
        else:  # markdown
            response_data["markdown"] = combined_markdown
            response_data["structured_data"] = structured_data
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing PPStructure output: {e}")
        raise

def cleanup_temp_files(file_path: str):
    """
    Clean up temporary files
    """
    try:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")

# ================= API Endpoints =================
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "PPStructureV3 Document Parser API",
        "version": "3.2.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "parse_document": "/parse",
            "supported_formats": "/formats"
        }
    }

@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    pipeline_status = "healthy" if hasattr(app.state, 'pipeline') and app.state.pipeline is not None else "unhealthy"
    
    return {
        "status": "ok",
        "pipeline": pipeline_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.2.0"
    }

@app.get("/formats")
async def get_supported_formats():
    """
    Get supported file formats and output options
    """
    return {
        "input_formats": list(ALLOWED_EXTENSIONS),
        "output_formats": OUTPUT_FORMATS,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "default_output_format": DEFAULT_OUTPUT_FORMAT
    }

@app.post("/parse")
async def parse_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to parse"),
    output_format: str = Query(DEFAULT_OUTPUT_FORMAT, description="Output format: markdown, json, or html"),
    include_metadata: bool = Query(True, description="Include metadata in response"),
    enable_table_recognition: bool = Query(None, description="Override default table recognition setting"),
    enable_formula_recognition: bool = Query(None, description="Override default formula recognition setting"),
    enable_chart_recognition: bool = Query(None, description="Override default chart recognition setting")
):
    """
    Parse document using PPStructureV3 and return structured content
    
    This endpoint processes documents through the PP-StructureV3 pipeline,
    extracting text, layout, tables, formulas, and charts with industry-leading accuracy.
    """
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{request_id}] Processing document: {file.filename}")
    
    try:
        # Validate input
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read and validate file content
        file_content = await file.read()
        if not validate_file_size(file_content):
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        # Validate output format
        if output_format.lower() not in OUTPUT_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format. Choose from: {', '.join(OUTPUT_FORMATS)}"
            )
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, temp_path)
        
        # Acquire semaphore for parallel request limiting
        async with app.state.predict_sem:
            logger.info(f"[{request_id}] Starting PPStructureV3 processing")
            
            # Run PPStructureV3 prediction
            results = await run_in_threadpool(
                app.state.pipeline.predict,
                input=temp_path
            )
        
        # Process results
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        response_data = process_ppstructure_output(results, output_format.lower())
        response_data["processing_time"] = processing_time
        response_data["request_id"] = request_id
        
        if not include_metadata:
            response_data.pop("metadata", None)
            response_data.pop("processing_time", None)
            response_data.pop("request_id", None)
        
        logger.info(f"[{request_id}] Processing completed in {processing_time:.2f}s")
        
        # Return appropriate response based on format
        if output_format.lower() == "markdown":
            return PlainTextResponse(
                content=response_data["markdown"],
                headers={"X-Request-ID": request_id}
            )
        else:
            return JSONResponse(
                content=response_data,
                headers={"X-Request-ID": request_id}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

@app.post("/parse/batch")
async def parse_documents_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple document files to parse"),
    output_format: str = Query(DEFAULT_OUTPUT_FORMAT, description="Output format for all documents")
):
    """
    Batch process multiple documents (limited parallel processing for CPU deployment)
    """
    if len(files) > 5:  # Limit batch size for CPU deployment
        raise HTTPException(
            status_code=400,
            detail="Batch processing limited to 5 files maximum for CPU deployment"
        )
    
    results = []
    for file in files:
        try:
            # Reuse the single parse endpoint for each file
            result = await parse_document(background_tasks, file, output_format, True)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_id": str(uuid.uuid4())[:8],
        "processed": len(files),
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "results": results
    }

# ================= Error Handlers =================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ================= Main Execution =================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None
    )

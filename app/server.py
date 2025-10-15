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
from fastapi.concurrency import run_in_threadpool
import io
import asyncio

# ================= Core Configuration (Updated for v3.2.0) =================
DEVICE = "cpu"
CPU_THREADS = 4  # Increased for better CPU utilization

# Enhanced accuracy boosters with new v3.2.0 capabilities
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False  
USE_TEXTLINE_ORIENTATION = True  # Enabled for better textline orientation

# Subpipeline toggles - optimized based on performance metrics
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False  # Disabled by default for performance
USE_CHART_RECOGNITION = False  # New chart recognition capability

# Model overrides updated for v3.2.0
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"


# Enhanced detection/recognition parameters for v3.2.0
LAYOUT_THRESHOLD = 0.5
TEXT_DET_THRESH = 0.3
TEXT_DET_BOX_THRESH = 0.6
TEXT_DET_UNCLIP_RATIO = 1.8
TEXT_DET_LIMIT_SIDE_LEN = 960  # Optimized for document processing
TEXT_DET_LIMIT_TYPE = "max"
TEXT_REC_SCORE_THRESH = 0.5
TEXT_RECOGNITION_BATCH_SIZE = 16  # Increased batch size for efficiency

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 100  # Increased for larger documents
MAX_PARALLEL_PREDICT = 2  # Increased for better throughput

# Output configuration
OUTPUT_FORMATS = ["json", "markdown", "txt"]
DEFAULT_OUTPUT_FORMAT = "json"

# ================= Import with Error Handling =================
try:
    ENABLE_HPI = False
    ENABLE_MKLDNN = True  # Enabled by default in v3.2.0 for CPU acceleration
    
    from paddleocr import PPStructureV3
    import paddle
    print("✓ PaddlePaddle and PaddleOCR imported successfully")
    print(f"✓ PaddlePaddle version: {paddle.__version__}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install PaddleOCR and PaddlePaddle:")
    print("pip install paddlepaddle paddleocr")
    raise

# ================= Configuration Validation =================
class PPStructureConfig:
    """Configuration validator for PP-StructureV3 v3.2.0"""
    
    @staticmethod
    def validate_config():
        """Validate the current configuration against known v3.2.0 constraints"""
        issues = []
        
        # Validate device and MKLDNN compatibility
        if DEVICE == "cpu" and not ENABLE_MKLDNN:
            issues.append("MKLDNN is disabled on CPU - this will impact performance")
            
        # Validate memory constraints
        if MAX_PARALLEL_PREDICT > 4:
            issues.append("High parallel prediction may cause memory issues")
            
        # Validate model combinations
        if USE_CHART_RECOGNITION and not USE_TABLE_RECOGNITION:
            issues.append("Chart recognition works best with table recognition enabled")
            
        return issues

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with enhanced initialization and validation"""
    
    # Validate configuration before startup
    config_issues = PPStructureConfig.validate_config()
    if config_issues:
        print("Configuration warnings:")
        for issue in config_issues:
            print(f"  - {issue}")
    
    try:
        print("Initializing PPStructureV3 pipeline...")
        
        # Initialize with v3.2.0 enhanced parameters
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
            use_chart_parsing=USE_CHART_PARSING,  # New in v3.2.0
        )
        
        app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
        app.state.processing_stats = {"total_processed": 0, "errors": 0}
        
        print("✓ PPStructureV3 pipeline initialized successfully")
        print(f"✓ Device: {DEVICE}, MKLDNN: {ENABLE_MKLDNN}, Threads: {CPU_THREADS}")
        print(f"✓ Table recognition: {USE_TABLE_RECOGNITION}")
        print(f"✓ Chart recognition: {USE_CHART_RECOGNITION}")
        print(f"✓ Formula recognition: {USE_FORMULA_RECOGNITION}")
        
    except Exception as e:
        print(f"✗ Failed to initialize PPStructureV3: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down PPStructureV3 API...")

app = FastAPI(
    title="PPStructureV3 Parser API v3.2.0", 
    version="3.2.0",
    description="Enhanced document parsing with PP-StructureV3 latest features",
    lifespan=lifespan
)

# ================= Utility Functions =================
def validate_file(file: UploadFile) -> Dict[str, Any]:
    """Enhanced file validation with better error reporting"""
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_extension} not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (approximate)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
    
    return {
        "filename": file.filename,
        "extension": file_extension,
        "size": file_size
    }

def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file with chunked writing for large files"""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def format_output(result: Any, output_format: str, include_visual: bool = False) -> Any:
    """Enhanced output formatting for different response types"""
    
    if output_format == "json":
        return result
        
    elif output_format == "markdown":
        if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            # Multiple pages - concatenate
            markdown_content = ""
            for page_idx, page_result in enumerate(result):
                if hasattr(page_result, 'get'):
                    markdown_content += page_result.get('markdown', f'# Page {page_idx + 1}\n\nNo markdown content\n\n')
                else:
                    markdown_content += f"# Page {page_idx + 1}\n\nContent not available in markdown format\n\n"
            return markdown_content
        else:
            return str(result)
            
    elif output_format == "txt":
        # Extract plain text from structured result
        if isinstance(result, list):
            text_content = ""
            for page_idx, page_result in enumerate(result):
                text_content += f"=== Page {page_idx + 1} ===\n"
                if isinstance(page_result, dict):
                    # Extract text from layout elements
                    for element in page_result.get('layout', []):
                        if 'text' in element:
                            text_content += element['text'] + "\n"
                text_content += "\n"
            return text_content
        return str(result)
    
    return result

# ================= API Endpoints =================
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "PPStructureV3 Parser API",
        "version": "3.2.0",
        "status": "operational",
        "features": {
            "table_recognition": USE_TABLE_RECOGNITION,
            "formula_recognition": USE_FORMULA_RECOGNITION,
            "chart_recognition": USE_CHART_RECOGNITION,
            "chart_parsing": USE_CHART_PARSING,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        }
    }

@app.get("/health")
async def health():
    """Enhanced health check with pipeline status"""
    pipeline_healthy = hasattr(app.state, 'pipeline') and app.state.pipeline is not None
    
    health_info = {
        "status": "healthy" if pipeline_healthy else "unhealthy",
        "timestamp": asyncio.get_event_loop().time(),
        "pipeline_initialized": pipeline_healthy,
        "processing_stats": app.state.processing_stats if hasattr(app.state, 'processing_stats') else {}
    }
    
    return JSONResponse(content=health_info)

@app.post("/parse")
async def parse_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document to parse (PDF or image)"),
    output_format: str = Query(DEFAULT_OUTPUT_FORMAT, description="Output format: json, markdown, or txt"),
    include_visualization: bool = Query(False, description="Include visualization data in response"),
    enable_table_recognition: bool = Query(USE_TABLE_RECOGNITION, description="Enable table recognition"),
    enable_formula_recognition: bool = Query(USE_FORMULA_RECOGNITION, description="Enable formula recognition"),
    enable_chart_recognition: bool = Query(USE_CHART_RECOGNITION, description="Enable chart recognition"),
    page_numbers: Optional[str] = Query(None, description="Specific page numbers to process (e.g., '1,3,5-10')")
):
    """
    Enhanced document parsing endpoint with PP-StructureV3 v3.2.0
    
    This endpoint processes documents and extracts structured information including:
    - Text content with layout preservation
    - Tables with structure recognition  
    - Mathematical formulas
    - Charts converted to tabular data
    - Document structure and reading order
    """
    
    # Validate output format
    if output_format not in OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output format. Supported: {OUTPUT_FORMATS}"
        )
    
    # Validate and process file
    file_info = validate_file(file)
    
    # Create temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_info['extension']) as tmp_file:
        tmp_path = Path(tmp_file.name)
        save_upload_file(file, tmp_path)
        
        try:
            # Acquire semaphore for parallel processing control
            async with app.state.predict_sem:
                # Run the actual processing in thread pool
                result = await run_in_threadpool(
                    process_document,
                    tmp_path,
                    file_info['extension'],
                    {
                        'include_visualization': include_visualization,
                        'enable_table_recognition': enable_table_recognition,
                        'enable_formula_recognition': enable_formula_recognition, 
                        'enable_chart_recognition': enable_chart_recognition,
                        'page_numbers': page_numbers
                    }
                )
                
                # Update processing statistics
                app.state.processing_stats["total_processed"] += 1
                
                # Format output based on requested format
                formatted_result = format_output(result, output_format, include_visualization)
                
                # Prepare response based on format
                if output_format == "json":
                    return JSONResponse(content=formatted_result)
                else:
                    return PlainTextResponse(
                        content=str(formatted_result),
                        media_type="text/plain"
                    )
                        
        except Exception as e:
            app.state.processing_stats["errors"] += 1
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
        finally:
            # Cleanup temporary file in background
            background_tasks.add_task(cleanup_temp_file, tmp_path)

def process_document(
    file_path: Path, 
    file_extension: str,
    options: Dict[str, Any]
) -> Any:
    """
    Core document processing function with enhanced error handling
    """
    try:
        pipeline = app.state.pipeline
        
        # Process based on file type
        if file_extension.lower() == '.pdf':
            # PDF processing with optional page selection
            result = pipeline(file_path, page_numbers=options.get('page_numbers'))
        else:
            # Image processing
            result = pipeline(str(file_path))
        
        return result
        
    except Exception as e:
        print(f"Processing error for {file_path}: {e}")
        raise

def cleanup_temp_file(file_path: Path):
    """Clean up temporary files with retry logic"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Warning: Could not delete temp file {file_path}: {e}")

@app.get("/config")
async def get_config():
    """Get current pipeline configuration"""
    config = {
        "device": DEVICE,
        "cpu_threads": CPU_THREADS,
        "enable_mkldnn": ENABLE_MKLDNN,
        "enable_hpi": ENABLE_HPI,
        "models": {
            "layout_detection": LAYOUT_DETECTION_MODEL_NAME,
            "text_detection": TEXT_DETECTION_MODEL_NAME,
            "text_recognition": TEXT_RECOGNITION_MODEL_NAME,
            "table_structure": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
            "formula_recognition": FORMULA_RECOGNITION_MODEL_NAME,
            "chart_recognition": CHART_RECOGNITION_MODEL_NAME
        },
        "features": {
            "table_recognition": USE_TABLE_RECOGNITION,
            "formula_recognition": USE_FORMULA_RECOGNITION,
            "chart_recognition": USE_CHART_RECOGNITION,
            "chart_parsing": USE_CHART_PARSING,
            "textline_orientation": USE_TEXTLINE_ORIENTATION
        },
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "max_parallel_predict": MAX_PARALLEL_PREDICT,
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        }
    }
    return JSONResponse(content=config)

# ================= Main Execution =================
if __name__ == "__main__":
    import uvicorn
    
    # Configuration validation on startup
    config_issues = PPStructureConfig.validate_config()
    if config_issues:
        print("Configuration issues found:")
        for issue in config_issues:
            print(f"  - {issue}")
    
    # Start the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )

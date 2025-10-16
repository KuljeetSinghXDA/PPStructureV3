"""
PP-StructureV3 FastAPI Server with CPU Support
Optimized for medical lab reports with configurable models
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import traceback

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - All settings configurable here
# =============================================================================

class PPStructureConfig:
    """Configuration class for PP-StructureV3 settings"""
    
    # Model Configuration - Change these models as needed
    LAYOUT_DETECTION_MODEL = "PP-DocLayout-L"  # High accuracy layout detection for medical docs
    TEXT_DETECTION_MODEL = "PP-OCRv5_mobile_det"  # Mobile detection model for CPU
    TEXT_RECOGNITION_MODEL = "en_PP-OCRv5_mobile_rec"  # English recognition for lab reports
    TABLE_RECOGNITION_MODEL = "SLANet_plus"  # Enhanced table recognition
    
    # Advanced Configuration
    FORMULA_RECOGNITION_MODEL = "PP-FormulaNet-M"  # Formula recognition (if needed)
    SEAL_DETECTION_MODEL = "PP-OCRv4_server_seal_det"  # Seal detection
    
    # Pipeline Settings
    USE_DOC_ORIENTATION_CLASSIFY = False  # Disable for speed on CPU
    USE_DOC_UNWARPING = False  # Disable for speed on CPU  
    USE_TEXTLINE_ORIENTATION = True  # Keep for accuracy
    USE_CHART_RECOGNITION = True  # Enable for medical charts
    USE_FORMULA_RECOGNITION = False  # Disable unless needed for medical formulas
    USE_SEAL_RECOGNITION = False  # Disable unless needed
    
    # Performance Settings for CPU
    DEVICE = "cpu"  # CPU only
    ENABLE_MKLDNN = True  # Enable Intel MKL-DNN for CPU optimization
    CPU_THREADS = 4  # Number of CPU threads
    
    # Text Detection Settings
    TEXT_DET_LIMIT_SIDE_LEN = 1200  # Reduced for CPU performance
    TEXT_DET_LIMIT_TYPE = "max"
    TEXT_DET_MAX_SIDE_LIMIT = 2400  # Reduced from default 4096 for CPU
    TEXT_DET_THRESH = 0.3
    TEXT_DET_BOX_THRESH = 0.6
    TEXT_DET_UNCLIP_RATIO = 1.5
    
    # Text Recognition Settings  
    TEXT_REC_BATCH_SIZE = 1  # Small batch size for CPU
    TEXT_REC_SCORE_THRESH = 0.5
    
    # Table Recognition Settings
    TABLE_MAX_LEN = 480  # Reduced for CPU performance
    
    # Output Settings
    SAVE_LOG_PATH = "./output/"
    OUTPUT_FORMAT = ["json", "markdown"]  # Both JSON and Markdown output
    
    # File Upload Settings
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    UPLOAD_DIR = Path("./uploads")
    OUTPUT_DIR = Path("./output") 
    TEMP_DIR = Path("./temp")

# Initialize configuration
config = PPStructureConfig()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ProcessingResult(BaseModel):
    """Response model for processing results"""
    success: bool
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    markdown_content: Optional[str] = None
    processing_time: float
    timestamp: str
    file_count: int

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: str

# =============================================================================
# PP-STRUCTUREV3 WRAPPER CLASS
# =============================================================================

class PPStructureV3Processor:
    """PP-StructureV3 processor with optimized configuration"""
    
    def __init__(self):
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize PP-StructureV3 pipeline with custom configuration"""
        try:
            from paddleocr import PPStructureV3
            
            logger.info("Initializing PP-StructureV3 with optimized CPU configuration...")
            
            # Initialize pipeline with medical-optimized configuration
            self.pipeline = PPStructureV3(
                # Model configuration
                layout_detection_model_name=config.LAYOUT_DETECTION_MODEL,
                text_detection_model_name=config.TEXT_DETECTION_MODEL,
                text_recognition_model_name=config.TEXT_RECOGNITION_MODEL,
                table_recognition_model_name=config.TABLE_RECOGNITION_MODEL,
                formula_recognition_model_name=config.FORMULA_RECOGNITION_MODEL,
                seal_detection_model_name=config.SEAL_DETECTION_MODEL,
                
                # Pipeline features
                use_doc_orientation_classify=config.USE_DOC_ORIENTATION_CLASSIFY,
                use_doc_unwarping=config.USE_DOC_UNWARPING,
                use_textline_orientation=config.USE_TEXTLINE_ORIENTATION,
                use_chart_recognition=config.USE_CHART_RECOGNITION,
                use_formula_recognition=config.USE_FORMULA_RECOGNITION,
                use_seal_recognition=config.USE_SEAL_RECOGNITION,
                
                # Performance settings
                device=config.DEVICE,
                enable_mkldnn=config.ENABLE_MKLDNN,
                cpu_threads=config.CPU_THREADS,
                
                # Text detection parameters
                text_det_limit_side_len=config.TEXT_DET_LIMIT_SIDE_LEN,
                text_det_limit_type=config.TEXT_DET_LIMIT_TYPE,
                max_side_limit=config.TEXT_DET_MAX_SIDE_LIMIT,
                text_det_thresh=config.TEXT_DET_THRESH,
                text_det_box_thresh=config.TEXT_DET_BOX_THRESH,
                text_det_unclip_ratio=config.TEXT_DET_UNCLIP_RATIO,
                
                # Text recognition parameters  
                text_rec_batch_size=config.TEXT_REC_BATCH_SIZE,
                text_rec_score_thresh=config.TEXT_REC_SCORE_THRESH,
                
                # Table recognition parameters
                table_max_len=config.TABLE_MAX_LEN,
            )
            
            logger.info("PP-StructureV3 pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PP-StructureV3: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple files with PP-StructureV3"""
        try:
            results = []
            markdown_content = ""
            
            for file_path in file_paths:
                logger.info(f"Processing file: {file_path}")
                
                # Process single file
                result = self.pipeline.predict(file_path)
                
                # Extract results for each page/image
                for res in result:
                    # Convert result to serializable format
                    result_dict = {
                        "input_path": res.input_path if hasattr(res, 'input_path') else file_path,
                        "page_index": res.page_index if hasattr(res, 'page_index') else None,
                    }
                    
                    # Extract layout information
                    if hasattr(res, 'layout'):
                        result_dict["layout"] = res.layout
                    
                    # Extract OCR text results
                    if hasattr(res, 'ocr_result'):
                        result_dict["ocr_result"] = res.ocr_result
                    
                    # Extract table results
                    if hasattr(res, 'table_result'):
                        result_dict["table_result"] = res.table_result
                    
                    # Extract chart results
                    if hasattr(res, 'chart_result'):
                        result_dict["chart_result"] = res.chart_result
                    
                    # Extract formula results
                    if hasattr(res, 'formula_result'):
                        result_dict["formula_result"] = res.formula_result
                    
                    # Extract markdown content
                    if hasattr(res, 'markdown'):
                        result_dict["markdown"] = res.markdown
                        markdown_content += f"# Document: {Path(file_path).name}\n\n"
                        markdown_content += res.markdown + "\n\n---\n\n"
                    
                    results.append(result_dict)
            
            return {
                "results": results,
                "markdown_content": markdown_content,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "results": [],
                "markdown_content": "",
                "success": False,
                "error": str(e)
            }

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="PP-StructureV3 Document Parser API",
    description="High-accuracy document parsing API optimized for medical lab reports using PP-StructureV3",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = None

# =============================================================================
# UTILITY FUNCTIONS  
# =============================================================================

def ensure_directories():
    """Ensure required directories exist"""
    for directory in [config.UPLOAD_DIR, config.OUTPUT_DIR, config.TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in config.ALLOWED_EXTENSIONS

async def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    """Save uploaded files and return file paths"""
    file_paths = []
    
    for file in files:
        if not is_allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed: {file.filename}. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file.filename}. Max size: {config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = config.UPLOAD_DIR / filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        file_paths.append(str(file_path))
        await file.seek(0)  # Reset file pointer
    
    return file_paths

def cleanup_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global processor
    
    logger.info("Starting PP-StructureV3 FastAPI server...")
    ensure_directories()
    
    try:
        processor = PPStructureV3Processor()
        logger.info("PP-StructureV3 processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PP-StructureV3 Document Parser API",
        "version": "1.0.0",
        "status": "active",
        "configuration": {
            "layout_model": config.LAYOUT_DETECTION_MODEL,
            "text_detection_model": config.TEXT_DETECTION_MODEL,
            "text_recognition_model": config.TEXT_RECOGNITION_MODEL,
            "table_model": config.TABLE_RECOGNITION_MODEL,
            "device": config.DEVICE,
            "optimized_for": "Medical Lab Reports"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/config")
async def get_configuration():
    """Get current configuration"""
    return {
        "models": {
            "layout_detection": config.LAYOUT_DETECTION_MODEL,
            "text_detection": config.TEXT_DETECTION_MODEL,
            "text_recognition": config.TEXT_RECOGNITION_MODEL,
            "table_recognition": config.TABLE_RECOGNITION_MODEL,
            "formula_recognition": config.FORMULA_RECOGNITION_MODEL,
            "seal_detection": config.SEAL_DETECTION_MODEL
        },
        "features": {
            "doc_orientation_classify": config.USE_DOC_ORIENTATION_CLASSIFY,
            "doc_unwarping": config.USE_DOC_UNWARPING,
            "textline_orientation": config.USE_TEXTLINE_ORIENTATION,
            "chart_recognition": config.USE_CHART_RECOGNITION,
            "formula_recognition": config.USE_FORMULA_RECOGNITION,
            "seal_recognition": config.USE_SEAL_RECOGNITION
        },
        "performance": {
            "device": config.DEVICE,
            "cpu_threads": config.CPU_THREADS,
            "enable_mkldnn": config.ENABLE_MKLDNN,
            "text_det_max_side_limit": config.TEXT_DET_MAX_SIDE_LIMIT,
            "batch_size": config.TEXT_REC_BATCH_SIZE
        }
    }

@app.post("/parse", response_model=ProcessingResult)
async def parse_documents(
    files: List[UploadFile] = File(..., description="Document files to process (PDF, images)"),
    output_format: str = Form("both", description="Output format: 'json', 'markdown', or 'both'")
):
    """
    Parse documents using PP-StructureV3
    
    - **files**: Multiple document files (PDF, PNG, JPG, etc.)  
    - **output_format**: Choose output format ('json', 'markdown', or 'both')
    
    Returns structured JSON data and/or Markdown content with high accuracy parsing
    optimized for medical lab reports.
    """
    start_time = asyncio.get_event_loop().time()
    file_paths = []
    
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if not processor:
            raise HTTPException(status_code=500, detail="Processor not initialized")
        
        # Validate output format
        if output_format not in ["json", "markdown", "both"]:
            raise HTTPException(status_code=400, detail="Invalid output_format. Use 'json', 'markdown', or 'both'")
        
        logger.info(f"Processing {len(files)} files with output format: {output_format}")
        
        # Save uploaded files
        file_paths = await save_uploaded_files(files)
        
        # Process files with PP-StructureV3
        result = await processor.process_files(file_paths)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Processing failed: {result.get('error', 'Unknown error')}")
        
        # Calculate processing time
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Prepare response based on output format
        response_data = {
            "success": True,
            "message": f"Successfully processed {len(files)} files",
            "file_count": len(files),
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        if output_format in ["json", "both"]:
            response_data["results"] = result["results"]
        
        if output_format in ["markdown", "both"]:
            response_data["markdown_content"] = result["markdown_content"]
        
        return ProcessingResult(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in parse_documents: {str(e)}")
        logger.error(traceback.format_exc())
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal processing error",
                details=str(e),
                timestamp=datetime.now().isoformat()
            ).dict()
        )
    finally:
        # Clean up uploaded files
        cleanup_files(file_paths)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# =============================================================================
# MAIN APPLICATION RUNNER
# =============================================================================

if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    
    # Configure uvicorn for production
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for CPU optimization
        loop="asyncio",
        log_level="info",
        access_log=True,
        reload=False  # Disable reload for production
    )

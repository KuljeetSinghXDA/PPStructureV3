"""
PP-StructureV3 FastAPI Service
Version: 3.2.0
Configurable document parsing service with all PP-StructureV3 features
"""

import os
import json
import base64
import asyncio
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
from paddleocr import PPStructureV3
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# Configuration - All PP-StructureV3 parameters can be configured here
# ============================================================================

class PPStructureConfig:
    """Configuration for PP-StructureV3 - Modify parameters here"""
    
    # Device configuration
    DEVICE: str = "cpu"  # Options: "cpu", "gpu", "gpu:0,1,2,3" for multi-GPU
    USE_GPU: bool = False
    GPU_MEM: int = 8000  # GPU memory in MB
    
    # Model selection - Optimized for medical lab reports
    LAYOUT_MODEL_NAME: str = "PP-DocLayout-L"  # Best accuracy layout model
    TEXT_DETECTION_MODEL_NAME: str = "PP-OCRv5_mobile_det"  # OCR detection model
    TEXT_RECOGNITION_MODEL_NAME: str = "en_PP-OCRv5_mobile_rec"  # English OCR recognition
    TABLE_RECOGNITION_MODEL_NAME: str = "SLANet_plus"  # Best table recognition
    FORMULA_RECOGNITION_MODEL_NAME: str = "PP-FormulaNet-L"  # Formula recognition
    SEAL_RECOGNITION_MODEL_NAME: str = None  # Auto-select seal model
    
    # Feature toggles
    USE_DOC_PREPROCESSOR: bool = True  # Enable document preprocessing
    USE_DOC_ORIENTATION_CLASSIFY: bool = True  # Auto-rotate documents
    USE_DOC_UNWARPING: bool = True  # Unwarp curved documents
    USE_TEXTLINE_ORIENTATION: bool = True  # Correct text line orientation
    USE_SEAL_RECOGNITION: bool = True  # Detect and recognize seals
    USE_TABLE_RECOGNITION: bool = True  # Extract tables
    USE_FORMULA_RECOGNITION: bool = True  # Extract formulas
    USE_CHART_RECOGNITION: bool = True  # Extract charts
    USE_REGION_DETECTION: bool = True  # Detect document regions
    
    # OCR parameters
    DET_DB_THRESH: float = 0.3  # Text detection threshold
    DET_DB_BOX_THRESH: float = 0.6  # Box detection threshold
    DET_DB_UNCLIP_RATIO: float = 1.5  # Unclip ratio for detection
    MAX_SIDE_LEN: int = 4000  # Max side length for detection (4096 for best accuracy)
    USE_ANGLE_CLS: bool = True  # Enable angle classification
    REC_BATCH_NUM: int = 6  # Recognition batch size
    
    # Layout analysis parameters
    LAYOUT_SCORE_THRESHOLD: float = 0.3  # Layout detection confidence threshold
    LAYOUT_NMS_THRESHOLD: float = 0.5  # NMS threshold for layout
    
    # Table recognition parameters
    TABLE_MAX_LEN: int = 512  # Maximum table length
    TABLE_BATCH_NUM: int = 1  # Table batch processing size
    
    # Formula recognition parameters
    FORMULA_BATCH_NUM: int = 1  # Formula batch processing size
    
    # Recovery parameters  
    RECOVERY_TO_MARKDOWN: bool = True  # Convert to markdown format
    RECOVERY_TO_JSON: bool = True  # Convert to JSON format
    
    # Performance parameters
    DROP_SCORE: float = 0.5  # Drop predictions below this score
    ENABLE_MKLDNN: bool = True  # Enable MKL-DNN for CPU acceleration
    CPU_THREADS: int = 10  # Number of CPU threads
    WARMUP: bool = False  # Warmup model before inference
    
    # Output parameters
    SAVE_VISUALIZATIONS: bool = True  # Save visualization images
    OUTPUT_DIR: str = "/app/outputs"  # Output directory
    UPLOAD_DIR: str = "/app/uploads"  # Upload directory

config = PPStructureConfig()

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="PP-StructureV3 API Service",
    description="Advanced document parsing with PaddleOCR PP-StructureV3",
    version="3.2.0"
)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Global PP-StructureV3 instance
ppstructure_model = None

# ============================================================================
# Data Models
# ============================================================================

class ParseRequest(BaseModel):
    """Request model for parsing configuration"""
    device: Optional[str] = Field(default=config.DEVICE, description="Device type")
    use_gpu: Optional[bool] = Field(default=config.USE_GPU, description="Use GPU")
    max_side_len: Optional[int] = Field(default=config.MAX_SIDE_LEN, description="Max side length")
    use_doc_preprocessor: Optional[bool] = Field(default=config.USE_DOC_PREPROCESSOR)
    use_table_recognition: Optional[bool] = Field(default=config.USE_TABLE_RECOGNITION)
    use_formula_recognition: Optional[bool] = Field(default=config.USE_FORMULA_RECOGNITION)
    use_chart_recognition: Optional[bool] = Field(default=config.USE_CHART_RECOGNITION)
    recovery_to_markdown: Optional[bool] = Field(default=config.RECOVERY_TO_MARKDOWN)
    recovery_to_json: Optional[bool] = Field(default=config.RECOVERY_TO_JSON)

class ParseResponse(BaseModel):
    """Response model for parsing results"""
    status: str
    message: str
    results: List[Dict[str, Any]]
    markdown: Optional[str] = None
    processing_time: float
    metadata: Dict[str, Any]

# ============================================================================
# PP-StructureV3 Initialization
# ============================================================================

def initialize_ppstructure(custom_config: Optional[Dict] = None) -> PPStructureV3:
    """Initialize PP-StructureV3 with configuration"""
    
    params = {
        # Device settings
        "device": custom_config.get("device", config.DEVICE) if custom_config else config.DEVICE,
        "use_gpu": custom_config.get("use_gpu", config.USE_GPU) if custom_config else config.USE_GPU,
        "gpu_mem": config.GPU_MEM,
        
        # Model selection
        "layout_model_name": config.LAYOUT_MODEL_NAME,
        "text_detection_model_name": config.TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": config.TEXT_RECOGNITION_MODEL_NAME,
        "table_recognition_model_name": config.TABLE_RECOGNITION_MODEL_NAME,
        "formula_recognition_model_name": config.FORMULA_RECOGNITION_MODEL_NAME,
        "seal_recognition_model_name": config.SEAL_RECOGNITION_MODEL_NAME,
        
        # Feature toggles
        "use_doc_preprocessor": custom_config.get("use_doc_preprocessor", config.USE_DOC_PREPROCESSOR) if custom_config else config.USE_DOC_PREPROCESSOR,
        "use_doc_orientation_classify": config.USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": config.USE_DOC_UNWARPING,
        "use_textline_orientation": config.USE_TEXTLINE_ORIENTATION,
        "use_seal_recognition": config.USE_SEAL_RECOGNITION,
        "use_table_recognition": custom_config.get("use_table_recognition", config.USE_TABLE_RECOGNITION) if custom_config else config.USE_TABLE_RECOGNITION,
        "use_formula_recognition": custom_config.get("use_formula_recognition", config.USE_FORMULA_RECOGNITION) if custom_config else config.USE_FORMULA_RECOGNITION,
        "use_chart_recognition": custom_config.get("use_chart_recognition", config.USE_CHART_RECOGNITION) if custom_config else config.USE_CHART_RECOGNITION,
        "use_region_detection": config.USE_REGION_DETECTION,
        
        # OCR parameters
        "det_db_thresh": config.DET_DB_THRESH,
        "det_db_box_thresh": config.DET_DB_BOX_THRESH,
        "det_db_unclip_ratio": config.DET_DB_UNCLIP_RATIO,
        "max_side_len": custom_config.get("max_side_len", config.MAX_SIDE_LEN) if custom_config else config.MAX_SIDE_LEN,
        "use_angle_cls": config.USE_ANGLE_CLS,
        "rec_batch_num": config.REC_BATCH_NUM,
        
        # Layout parameters
        "layout_score_threshold": config.LAYOUT_SCORE_THRESHOLD,
        "layout_nms_threshold": config.LAYOUT_NMS_THRESHOLD,
        
        # Table parameters
        "table_max_len": config.TABLE_MAX_LEN,
        "table_batch_num": config.TABLE_BATCH_NUM,
        
        # Formula parameters
        "formula_batch_num": config.FORMULA_BATCH_NUM,
        
        # Recovery parameters
        "recovery_to_markdown": custom_config.get("recovery_to_markdown", config.RECOVERY_TO_MARKDOWN) if custom_config else config.RECOVERY_TO_MARKDOWN,
        
        # Performance parameters
        "drop_score": config.DROP_SCORE,
        "enable_mkldnn": config.ENABLE_MKLDNN,
        "cpu_threads": config.CPU_THREADS,
        "warmup": config.WARMUP,
    }
    
    return PPStructureV3(**params)

# ============================================================================
# Document Processing Functions
# ============================================================================

def process_single_file(
    file_path: str,
    ppstructure: PPStructureV3,
    recovery_to_json: bool = True,
    recovery_to_markdown: bool = True
) -> Dict[str, Any]:
    """Process a single document file"""
    
    try:
        # Run PP-StructureV3 prediction
        result = ppstructure.predict(
            input_path=file_path,
            input_type='img' if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) else 'pdf',
            do_visualize=config.SAVE_VISUALIZATIONS
        )
        
        # Extract structured data
        structured_result = {
            "file_name": os.path.basename(file_path),
            "status": "success",
            "layout_elements": [],
            "ocr_results": [],
            "tables": [],
            "formulas": [],
            "charts": [],
            "markdown": "",
            "json_data": {}
        }
        
        if result and hasattr(result, 'res'):
            res_data = result.res if hasattr(result, 'res') else result
            
            # Process layout detection results
            if hasattr(res_data, 'layout_det_res') and res_data.layout_det_res:
                layout_data = res_data.layout_det_res
                if hasattr(layout_data, 'boxes'):
                    for box in layout_data.boxes:
                        structured_result["layout_elements"].append({
                            "type": getattr(box, 'label', 'unknown'),
                            "score": float(getattr(box, 'score', 0)),
                            "bbox": getattr(box, 'coordinate', []).tolist() if hasattr(box, 'coordinate') else []
                        })
            
            # Process OCR results
            if hasattr(res_data, 'rec_texts') and res_data.rec_texts:
                for idx, text in enumerate(res_data.rec_texts):
                    score = res_data.rec_scores[idx] if hasattr(res_data, 'rec_scores') and idx < len(res_data.rec_scores) else 0
                    structured_result["ocr_results"].append({
                        "text": text,
                        "confidence": float(score)
                    })
            
            # Process tables
            if hasattr(res_data, 'table_res') and res_data.table_res:
                for table in res_data.table_res:
                    if hasattr(table, 'html'):
                        structured_result["tables"].append({
                            "html": table.html,
                            "bbox": getattr(table, 'bbox', [])
                        })
            
            # Process formulas
            if hasattr(res_data, 'formula_res') and res_data.formula_res:
                for formula in res_data.formula_res:
                    structured_result["formulas"].append({
                        "latex": getattr(formula, 'text', ''),
                        "score": float(getattr(formula, 'score', 0))
                    })
            
            # Get markdown output
            if recovery_to_markdown and hasattr(res_data, 'markdown'):
                structured_result["markdown"] = res_data.markdown
            elif recovery_to_markdown:
                # Generate markdown from results
                markdown_content = generate_markdown_from_results(structured_result)
                structured_result["markdown"] = markdown_content
            
            # Get JSON output
            if recovery_to_json:
                structured_result["json_data"] = {
                    "layout": structured_result["layout_elements"],
                    "text": structured_result["ocr_results"],
                    "tables": structured_result["tables"],
                    "formulas": structured_result["formulas"],
                    "charts": structured_result["charts"]
                }
        
        return structured_result
        
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "status": "error",
            "error_message": str(e),
            "layout_elements": [],
            "ocr_results": [],
            "tables": [],
            "formulas": [],
            "charts": [],
            "markdown": "",
            "json_data": {}
        }

def generate_markdown_from_results(results: Dict) -> str:
    """Generate markdown from structured results"""
    markdown = []
    
    # Add title
    markdown.append(f"# Document Analysis: {results.get('file_name', 'Unknown')}\n")
    
    # Add OCR text
    if results["ocr_results"]:
        markdown.append("## Text Content\n")
        for item in results["ocr_results"]:
            if item["confidence"] > 0.5:  # Only include high-confidence text
                markdown.append(f"{item['text']}\n")
    
    # Add tables
    if results["tables"]:
        markdown.append("\n## Tables\n")
        for idx, table in enumerate(results["tables"], 1):
            markdown.append(f"\n### Table {idx}\n")
            markdown.append(table.get("html", ""))
    
    # Add formulas
    if results["formulas"]:
        markdown.append("\n## Formulas\n")
        for idx, formula in enumerate(results["formulas"], 1):
            markdown.append(f"\n**Formula {idx}:** `{formula['latex']}`\n")
    
    return "\n".join(markdown)

async def process_files_async(
    file_paths: List[str],
    custom_config: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """Process multiple files asynchronously"""
    
    # Initialize model with custom config if provided
    ppstructure = initialize_ppstructure(custom_config)
    
    # Process files
    results = []
    for file_path in file_paths:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            process_single_file,
            file_path,
            ppstructure,
            custom_config.get("recovery_to_json", True) if custom_config else True,
            custom_config.get("recovery_to_markdown", True) if custom_config else True
        )
        results.append(result)
    
    return results

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global ppstructure_model
    print("Initializing PP-StructureV3 model...")
    ppstructure_model = initialize_ppstructure()
    print("Model initialization complete!")
    
    # Create directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PP-StructureV3 API",
        "version": "3.2.0",
        "status": "running",
        "endpoints": {
            "/parse": "Parse documents (POST)",
            "/health": "Health check (GET)",
            "/config": "View current configuration (GET)"
        },
        "models": {
            "layout": config.LAYOUT_MODEL_NAME,
            "text_detection": config.TEXT_DETECTION_MODEL_NAME,
            "text_recognition": config.TEXT_RECOGNITION_MODEL_NAME,
            "table_recognition": config.TABLE_RECOGNITION_MODEL_NAME
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ppstructure_model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/config")
async def get_configuration():
    """Get current configuration"""
    return {
        "device": config.DEVICE,
        "models": {
            "layout": config.LAYOUT_MODEL_NAME,
            "text_detection": config.TEXT_DETECTION_MODEL_NAME,
            "text_recognition": config.TEXT_RECOGNITION_MODEL_NAME,
            "table_recognition": config.TABLE_RECOGNITION_MODEL_NAME,
            "formula_recognition": config.FORMULA_RECOGNITION_MODEL_NAME
        },
        "features": {
            "doc_preprocessor": config.USE_DOC_PREPROCESSOR,
            "table_recognition": config.USE_TABLE_RECOGNITION,
            "formula_recognition": config.USE_FORMULA_RECOGNITION,
            "chart_recognition": config.USE_CHART_RECOGNITION,
            "seal_recognition": config.USE_SEAL_RECOGNITION
        },
        "parameters": {
            "max_side_len": config.MAX_SIDE_LEN,
            "det_db_thresh": config.DET_DB_THRESH,
            "det_db_box_thresh": config.DET_DB_BOX_THRESH
        }
    }

@app.post("/parse")
async def parse_documents(
    files: List[UploadFile] = File(...),
    request_config: Optional[str] = None  # JSON string with custom config
):
    """
    Parse multiple documents with PP-StructureV3
    
    Returns both JSON and Markdown outputs
    """
    
    start_time = datetime.utcnow()
    
    # Parse custom configuration if provided
    custom_config = None
    if request_config:
        try:
            custom_config = json.loads(request_config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid configuration JSON")
    
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per request")
    
    # Save uploaded files
    saved_files = []
    try:
        for file in files:
            # Validate file type
            allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
                )
            
            # Save file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(config.UPLOAD_DIR, safe_filename)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            saved_files.append(file_path)
        
        # Process files
        results = await process_files_async(saved_files, custom_config)
        
        # Combine results
        all_markdown = []
        all_json_results = []
        
        for result in results:
            all_json_results.append(result)
            if result.get("markdown"):
                all_markdown.append(result["markdown"])
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare response
        response_data = ParseResponse(
            status="success",
            message=f"Successfully processed {len(files)} file(s)",
            results=all_json_results,
            markdown="\n\n---\n\n".join(all_markdown) if all_markdown else None,
            processing_time=processing_time,
            metadata={
                "files_processed": len(files),
                "timestamp": start_time.isoformat(),
                "configuration": custom_config or "default"
            }
        )
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up uploaded files (optional - you may want to keep them)
        for file_path in saved_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

@app.post("/parse-base64")
async def parse_base64_images(
    images: List[str],
    request_config: Optional[Dict] = None
):
    """
    Parse base64-encoded images
    
    Accepts a list of base64-encoded image strings
    """
    
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    start_time = datetime.utcnow()
    saved_files = []
    
    try:
        # Decode and save images
        for idx, image_data in enumerate(images):
            try:
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                
                # Save as temporary file
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(config.UPLOAD_DIR, f"{timestamp}_image_{idx}.png")
                
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                
                saved_files.append(file_path)
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image at index {idx}: {str(e)}")
        
        # Process files
        results = await process_files_async(saved_files, request_config)
        
        # Prepare response
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "status": "success",
            "results": results,
            "processing_time": processing_time,
            "files_processed": len(images)
        }
        
    finally:
        # Clean up
        for file_path in saved_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "pp_structure_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )

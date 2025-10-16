"""
PPStructureV3 FastAPI Server
Optimized for Medical Lab Reports with ARM64 CPU support
All configurations are adjustable in the CONFIG section below
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import aiofiles

from paddleocr import PPStructureV3

# ============================================================================
# CONFIGURATION SECTION - Modify all settings here
# ============================================================================

CONFIG = {
    # Device Configuration
    "device": "cpu",  # Use "cpu" for ARM64, or "gpu:0" if GPU available
    "cpu_threads": 8,  # Number of CPU threads
    "enable_mkldnn": True,  # Enable MKL-DNN acceleration for CPU
    "mkldnn_cache_capacity": 10,
    
    # Model Selection
    "layout_detection_model_name": "PP-DocLayout-L",  # High precision layout detection
    "text_detection_model_name": "PP-OCRv5_mobile_det",  # Mobile text detection
    "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",  # English recognition
    
    # Optional Model Names (can be customized)
    "doc_orientation_classify_model_name": "PP-LCNet_x1_0_doc_ori",
    "doc_unwarping_model_name": "UVDoc",
    "textline_orientation_model_name": "PP-LCNet_x0_25_textline_ori",
    "table_classification_model_name": "PP-LCNet_x1_0_table_cls",
    "wired_table_structure_recognition_model_name": "SLANeXt_wired",
    "wireless_table_structure_recognition_model_name": "SLANeXt_wireless",
    "wired_table_cells_detection_model_name": "RT-DETR-L_wired_table_cell_det",
    "wireless_table_cells_detection_model_name": "RT-DETR-L_wireless_table_cell_det",
    "seal_text_detection_model_name": "PP-OCRv4_mobile_seal_det",
    "seal_text_recognition_model_name": "PP-OCRv5_mobile_rec",
    "formula_recognition_model_name": "PP-FormulaNet_plus-L",
    "chart_recognition_model_name": "PP-Chart2Table",
    
    # Feature Toggles
    "use_doc_orientation_classify": True,  # Enable document orientation classification
    "use_doc_unwarping": False,  # Document image rectification (slower)
    "use_textline_orientation": True,  # Text line orientation classification
    "use_seal_recognition": True,  # Seal/stamp text recognition
    "use_table_recognition": True,  # Table structure recognition
    "use_formula_recognition": True,  # Mathematical formula recognition
    "use_chart_recognition": True,  # Chart parsing and understanding
    "use_region_detection": True,  # Document region detection
    
    # Layout Detection Parameters
    "layout_threshold": 0.5,  # Score threshold (0-1)
    "layout_nms": True,  # Non-Maximum Suppression
    "layout_unclip_ratio": 1.0,  # Unclip ratio for detected boxes
    "layout_merge_bboxes_mode": "large",  # Merging mode for detection boxes
    
    # Text Detection Parameters
    "text_det_limit_side_len": 960,  # Image side length limitation
    "text_det_limit_type": "max",  # "min" or "max"
    "text_det_thresh": 0.3,  # Pixel threshold for detection
    "text_det_box_thresh": 0.6,  # Box threshold
    "text_det_unclip_ratio": 2.0,  # Expansion ratio
    
    # Text Recognition Parameters
    "text_recognition_batch_size": 6,  # Batch size for text recognition
    "text_rec_score_thresh": 0.5,  # Recognition score threshold (higher for medical accuracy)
    
    # Seal Detection Parameters (for medical stamps)
    "seal_det_limit_side_len": 736,
    "seal_det_limit_type": "min",
    "seal_det_thresh": 0.2,
    "seal_det_box_thresh": 0.6,
    "seal_det_unclip_ratio": 0.5,
    "seal_text_recognition_batch_size": 6,
    "seal_rec_score_thresh": 0.5,
    
    # Table Recognition Parameters
    "table_orientation_classify_batch_size": 1,
    "table_classification_batch_size": 1,
    
    # Formula Recognition Parameters
    "formula_recognition_batch_size": 1,
    
    # Chart Recognition Parameters
    "chart_recognition_batch_size": 1,
    
    # Text Line Orientation Parameters
    "textline_orientation_batch_size": 6,
    
    # Output Settings
    "output_format": ["json", "markdown"],  # Output formats
    "save_results_locally": True,  # Save results to disk
    "output_directory": "/app/outputs",  # Output directory
    
    # Processing Settings
    "max_file_size_mb": 50,  # Maximum file size in MB
    "supported_formats": [".png", ".jpg", ".jpeg", ".bmp", ".pdf", ".tiff", ".tif"],
    "language": "en",  # Language for text recognition
}

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="PPStructureV3 Document Parser API",
    description="High-accuracy document parsing optimized for medical lab reports using PaddleOCR PPStructureV3",
    version="1.0.0"
)

# Global pipeline instance
pipeline: Optional[PPStructureV3] = None


def initialize_pipeline():
    """Initialize PPStructureV3 pipeline with configured settings"""
    global pipeline
    
    print("Initializing PPStructureV3 pipeline...")
    print(f"Device: {CONFIG['device']}")
    print(f"Layout Model: {CONFIG['layout_detection_model_name']}")
    print(f"Text Detection Model: {CONFIG['text_detection_model_name']}")
    print(f"Text Recognition Model: {CONFIG['text_recognition_model_name']}")
    
    pipeline = PPStructureV3(
        # Device configuration
        device=CONFIG["device"],
        cpu_threads=CONFIG["cpu_threads"],
        enable_mkldnn=CONFIG["enable_mkldnn"],
        mkldnn_cache_capacity=CONFIG["mkldnn_cache_capacity"],
        
        # Model names
        layout_detection_model_name=CONFIG["layout_detection_model_name"],
        text_detection_model_name=CONFIG["text_detection_model_name"],
        text_recognition_model_name=CONFIG["text_recognition_model_name"],
        doc_orientation_classify_model_name=CONFIG["doc_orientation_classify_model_name"],
        textline_orientation_model_name=CONFIG["textline_orientation_model_name"],
        table_classification_model_name=CONFIG["table_classification_model_name"],
        wired_table_structure_recognition_model_name=CONFIG["wired_table_structure_recognition_model_name"],
        wireless_table_structure_recognition_model_name=CONFIG["wireless_table_structure_recognition_model_name"],
        wired_table_cells_detection_model_name=CONFIG["wired_table_cells_detection_model_name"],
        wireless_table_cells_detection_model_name=CONFIG["wireless_table_cells_detection_model_name"],
        seal_text_detection_model_name=CONFIG["seal_text_detection_model_name"],
        seal_text_recognition_model_name=CONFIG["seal_text_recognition_model_name"],
        formula_recognition_model_name=CONFIG["formula_recognition_model_name"],
        chart_recognition_model_name=CONFIG["chart_recognition_model_name"],
        
        # Feature toggles
        use_doc_orientation_classify=CONFIG["use_doc_orientation_classify"],
        use_doc_unwarping=CONFIG["use_doc_unwarping"],
        use_textline_orientation=CONFIG["use_textline_orientation"],
        use_seal_recognition=CONFIG["use_seal_recognition"],
        use_table_recognition=CONFIG["use_table_recognition"],
        use_formula_recognition=CONFIG["use_formula_recognition"],
        use_chart_recognition=CONFIG["use_chart_recognition"],
        use_region_detection=CONFIG["use_region_detection"],
        
        # Layout detection parameters
        layout_threshold=CONFIG["layout_threshold"],
        layout_nms=CONFIG["layout_nms"],
        layout_unclip_ratio=CONFIG["layout_unclip_ratio"],
        layout_merge_bboxes_mode=CONFIG["layout_merge_bboxes_mode"],
        
        # Text detection parameters
        text_det_limit_side_len=CONFIG["text_det_limit_side_len"],
        text_det_limit_type=CONFIG["text_det_limit_type"],
        text_det_thresh=CONFIG["text_det_thresh"],
        text_det_box_thresh=CONFIG["text_det_box_thresh"],
        text_det_unclip_ratio=CONFIG["text_det_unclip_ratio"],
        
        # Text recognition parameters
        text_recognition_batch_size=CONFIG["text_recognition_batch_size"],
        text_rec_score_thresh=CONFIG["text_rec_score_thresh"],
        
        # Seal detection parameters
        seal_det_limit_side_len=CONFIG["seal_det_limit_side_len"],
        seal_det_limit_type=CONFIG["seal_det_limit_type"],
        seal_det_thresh=CONFIG["seal_det_thresh"],
        seal_det_box_thresh=CONFIG["seal_det_box_thresh"],
        seal_det_unclip_ratio=CONFIG["seal_det_unclip_ratio"],
        seal_text_recognition_batch_size=CONFIG["seal_text_recognition_batch_size"],
        seal_rec_score_thresh=CONFIG["seal_rec_score_thresh"],
        
        # Other batch sizes
        textline_orientation_batch_size=CONFIG["textline_orientation_batch_size"],
        formula_recognition_batch_size=CONFIG["formula_recognition_batch_size"],
        chart_recognition_batch_size=CONFIG["chart_recognition_batch_size"],
        
        # Language
        lang=CONFIG["language"],
    )
    
    print("Pipeline initialized successfully!")
    return pipeline


def convert_result_to_dict(result) -> Dict[str, Any]:
    """Convert PPStructureV3 result object to dictionary"""
    try:
        # Access the result dictionary from the result object
        if hasattr(result, 'res'):
            return result.res
        elif hasattr(result, '__dict__'):
            return result.__dict__
        else:
            return {"raw_result": str(result)}
    except Exception as e:
        return {"error": f"Failed to convert result: {str(e)}"}


def generate_markdown_from_result(result_dict: Dict[str, Any], filename: str) -> str:
    """Generate Markdown format from parsing result"""
    markdown_lines = [
        f"# Document Analysis Report",
        f"\n**File:** {filename}",
        f"\n**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n---\n"
    ]
    
    # Extract layout detection results
    if "layout_det_res" in result_dict and result_dict["layout_det_res"]:
        layout_res = result_dict["layout_det_res"]
        if "boxes" in layout_res:
            markdown_lines.append(f"\n## Layout Detection")
            markdown_lines.append(f"\nDetected {len(layout_res['boxes'])} layout regions:\n")
            
            for idx, box in enumerate(layout_res["boxes"], 1):
                label = box.get("label", "unknown")
                score = box.get("score", 0.0)
                markdown_lines.append(f"{idx}. **{label.upper()}** (confidence: {score:.2%})")
    
    # Extract OCR results
    if "ocr_res" in result_dict and result_dict["ocr_res"]:
        ocr_res = result_dict["ocr_res"]
        markdown_lines.append(f"\n## Text Recognition Results\n")
        
        if "rec_text" in ocr_res:
            for idx, text in enumerate(ocr_res["rec_text"], 1):
                if text.strip():
                    markdown_lines.append(f"{idx}. {text}")
    
    # Extract table results
    if "table_res" in result_dict and result_dict["table_res"]:
        markdown_lines.append(f"\n## Table Detection")
        markdown_lines.append(f"\nDetected {len(result_dict['table_res'])} table(s)")
    
    # Extract formula results
    if "formula_res" in result_dict and result_dict["formula_res"]:
        markdown_lines.append(f"\n## Formula Recognition")
        markdown_lines.append(f"\nDetected {len(result_dict['formula_res'])} formula(s)")
    
    # Extract seal results
    if "seal_res" in result_dict and result_dict["seal_res"]:
        markdown_lines.append(f"\n## Seal/Stamp Detection")
        markdown_lines.append(f"\nDetected {len(result_dict['seal_res'])} seal(s)")
    
    return "\n".join(markdown_lines)


async def save_file(upload_file: UploadFile, directory: str) -> str:
    """Save uploaded file to disk"""
    file_path = os.path.join(directory, upload_file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    return file_path


def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in CONFIG["supported_formats"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(CONFIG['supported_formats'])}"
        )
    return True


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    initialize_pipeline()
    
    # Create output directories
    os.makedirs(CONFIG["output_directory"], exist_ok=True)
    os.makedirs("/app/uploads", exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PPStructureV3 Document Parser",
        "version": "1.0.0",
        "status": "running",
        "configuration": {
            "device": CONFIG["device"],
            "layout_model": CONFIG["layout_detection_model_name"],
            "text_detection_model": CONFIG["text_detection_model_name"],
            "text_recognition_model": CONFIG["text_recognition_model_name"],
            "features_enabled": {
                "doc_orientation": CONFIG["use_doc_orientation_classify"],
                "doc_unwarping": CONFIG["use_doc_unwarping"],
                "textline_orientation": CONFIG["use_textline_orientation"],
                "seal_recognition": CONFIG["use_seal_recognition"],
                "table_recognition": CONFIG["use_table_recognition"],
                "formula_recognition": CONFIG["use_formula_recognition"],
                "chart_recognition": CONFIG["use_chart_recognition"],
            }
        },
        "endpoints": {
            "parse": "/parse - POST endpoint for document parsing (accepts multiple files)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {"configuration": CONFIG}


@app.post("/parse")
async def parse_documents(files: List[UploadFile] = File(...)):
    """
    Parse multiple document files using PPStructureV3
    
    Accepts multiple files and returns JSON and Markdown results for each file
    
    Args:
        files: List of uploaded files (images or PDFs)
    
    Returns:
        JSON response containing parsing results in both JSON and Markdown formats
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    
    for file in files:
        try:
            # Validate file
            validate_file(file)
            
            # Save file temporarily
            file_path = await save_file(file, "/app/uploads")
            
            print(f"Processing file: {file.filename}")
            
            # Run PPStructureV3 prediction
            output = pipeline.predict(input=file_path)
            
            # Process each result (for PDFs, there might be multiple pages)
            file_results = []
            for idx, res in enumerate(output):
                # Convert result to dictionary
                result_dict = convert_result_to_dict(res)
                
                # Generate markdown
                markdown_content = generate_markdown_from_result(result_dict, file.filename)
                
                # Prepare response
                page_result = {
                    "file": file.filename,
                    "page_index": idx,
                    "json": result_dict,
                    "markdown": markdown_content
                }
                
                # Save results locally if configured
                if CONFIG["save_results_locally"]:
                    output_base = os.path.join(
                        CONFIG["output_directory"],
                        f"{Path(file.filename).stem}_page{idx}"
                    )
                    
                    # Save JSON
                    if "json" in CONFIG["output_format"]:
                        json_path = f"{output_base}.json"
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result_dict, f, indent=2, ensure_ascii=False)
                        page_result["json_file"] = json_path
                    
                    # Save Markdown
                    if "markdown" in CONFIG["output_format"]:
                        md_path = f"{output_base}.md"
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(markdown_content)
                        page_result["markdown_file"] = md_path
                
                file_results.append(page_result)
            
            results.append({
                "file": file.filename,
                "status": "success",
                "pages": len(file_results),
                "results": file_results
            })
            
            # Clean up uploaded file
            os.remove(file_path)
            
        except Exception as e:
            results.append({
                "file": file.filename,
                "status": "error",
                "error": str(e)
            })
            print(f"Error processing {file.filename}: {str(e)}")
    
    return JSONResponse(content={
        "status": "completed",
        "processed_files": len(files),
        "results": results,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

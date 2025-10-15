import os
import tempfile
import threading
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

# ================= Core Configuration (Pinned Values per Official Docs) =================
# Device and threading configuration
DEVICE = "cpu"
CPU_THREADS = 4  # Optimal for most CPU-based deployments per Paddle docs

# Subpipeline toggles (match official model capabilities)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model names MUST match official model zoo (v3.2.0)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (set to None for default values per docs)
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
MAX_PARALLEL_PREDICT = 1  # Critical for CPU resource management

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ppstructure-api")

# ================= Helper Functions =================
def validate_file(file: UploadFile) -> None:
    """Validate file extension and size per configuration"""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, 
            f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    file_size = len(file.file.read())
    file.file.seek(0)  # Reset file pointer
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            400, 
            f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def convert_to_standard_format(
    page_result: Dict[str, Any],
    use_table_recognition: bool,
    use_formula_recognition: bool
) -> List[Dict[str, Any]]:
    """Convert raw PP-StructureV3 results to standardized block format"""
    blocks = []
    layout_regions = page_result.get("layout_result", [])
    text_lines = page_result.get("text_result", [])
    table_results = page_result.get("table_result", []) if use_table_recognition else []
    formula_results = page_result.get("formula_result", []) if use_formula_recognition else []

    # Match text lines to layout regions
    text_by_region = {}
    for text_line in text_lines:
        max_iou = 0
        best_region_idx = -1
        for idx, region in enumerate(layout_regions):
            iou = compute_iou(text_line["bbox"], region["bbox"])
            if iou > max_iou:
                max_iou = iou
                best_region_idx = idx
        
        if best_region_idx >= 0 and max_iou > 0.1:  # Minimum overlap threshold
            if best_region_idx not in text_by_region:
                text_by_region[best_region_idx] = []
            text_by_region[best_region_idx].append(text_line)

    # Process each layout region
    for idx, region in enumerate(layout_regions):
        bbox = region["bbox"]
        confidence = region["score"]
        label = region["label"]
        
        # Standardize label names per PP-StructureV3 documentation
        if label in ["text", "title", "header", "footer"]:
            block_type = "text"
        elif label == "figure":
            block_type = "image"
        elif label == "equation":
            block_type = "formula"
        else:
            block_type = label  # table, etc.

        block = {
            "type": block_type,
            "bbox": bbox,
            "confidence": confidence,
            "content": {}
        }

        # Add content based on block type
        if block_type == "text":
            if idx in text_by_region:
                # Sort text lines by vertical position then horizontal
                sorted_lines = sorted(
                    text_by_region[idx], 
                    key=lambda x: (x["bbox"][1], x["bbox"][0])
                )
                block["content"]["text"] = " ".join(line["text"] for line in sorted_lines)
            else:
                block["content"]["text"] = ""
        
        elif block_type == "table" and table_results:
            best_match = None
            max_iou = 0
            for table in table_results:
                iou = compute_iou(bbox, table["bbox_2d"])
                if iou > max_iou:
                    max_iou = iou
                    best_match = table
            
            if best_match and max_iou > 0.5:
                block["content"]["html"] = best_match.get("html", "")
                block["content"]["text"] = best_match.get("text_content", "")
        
        elif block_type == "formula" and formula_results:
            best_match = None
            max_iou = 0
            for formula in formula_results:
                iou = compute_iou(bbox, formula["bbox_2d"])
                if iou > max_iou:
                    max_iou = iou
                    best_match = formula
            
            if best_match and max_iou > 0.5:
                block["content"]["latex"] = best_match.get("latex", "")
                block["content"]["text"] = best_match.get("text_content", "")

        blocks.append(block)
    
    return blocks

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize PP-StructureV3 pipeline and resource management"""
    logger.info("Initializing PP-StructureV3 pipeline (v3.2.0)...")
    try:
        app.state.pipeline = PPStructureV3(
            device=DEVICE,
            enable_mkldnn=True,  # Enabled by default per Paddle docs
            enable_hpi=False,    # Not needed for CPU deployment
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
        logger.info("Pipeline initialized successfully")
        app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
        yield
    except Exception as e:
        logger.exception("Failed to initialize pipeline")
        raise RuntimeError(f"Pipeline initialization failed: {str(e)}") from e
    finally:
        logger.info("Shutting down PP-StructureV3 pipeline")

app = FastAPI(
    title="PP-StructureV3 /parse API",
    version="1.0.0",
    description="Document parsing API powered by PP-StructureV3 (v3.2.0)",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "PP-StructureV3", "version": "3.2.0"}

@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    """
    Parse document using PP-StructureV3
    
    Args:
        file: PDF or image file (up to 50MB)
    
    Returns:
        JSON structure containing parsed document content
    """
    validate_file(file)
    
    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Process with resource management
        with app.state.predict_sem:
            logger.info(f"Processing {file.filename} (size: {os.path.getsize(temp_path) / (1024*1024):.2f}MB)")
            results = await run_in_threadpool(app.state.pipeline, str(temp_path))
            
            # Convert to standard format
            parsed_results = []
            for page_result in results:
                blocks = convert_to_standard_format(
                    page_result,
                    use_table_recognition=USE_TABLE_RECOGNITION,
                    use_formula_recognition=USE_FORMULA_RECOGNITION
                )
                parsed_results.append({
                    "page_number": page_result["page_id"] + 1,
                    "blocks": blocks
                })
            
            logger.info(f"Successfully parsed {len(parsed_results)} pages")
            return {"pages": parsed_results}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

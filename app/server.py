import os
import tempfile
import threading
import json
import shutil
import base64
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

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

# ================= Response Models =================
class MarkdownOutput(BaseModel):
    text: str
    images: Dict[str, str]  # path -> base64 encoded image
    isStart: bool
    isEnd: bool

class PageResult(BaseModel):
    page_index: Optional[int]
    json_result: Dict[str, Any]
    markdown: MarkdownOutput

class ParseResponse(BaseModel):
    success: bool
    message: str
    total_pages: int
    results: List[PageResult]
    combined_markdown: Optional[str] = None

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize PPStructureV3 pipeline with all configured parameters
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
    # Cleanup (if needed)

app = FastAPI(
    title="PPStructureV3 Parse API", 
    version="3.2.0",
    description="Document parsing API using PPStructureV3 - converts PDFs and images to structured JSON and Markdown",
    lifespan=lifespan
)

# ================= Helper Functions =================
def validate_file(file: UploadFile) -> None:
    """Validate uploaded file extension and size"""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

def encode_image_to_base64(image) -> str:
    """Convert PIL Image to base64 string"""
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_result(result) -> Dict[str, Any]:
    """
    Process a single PPStructureV3 result object into structured format
    Extracts JSON and Markdown outputs
    """
    # Get JSON result
    json_result = result.json if hasattr(result, 'json') else {}
    
    # Get Markdown result
    markdown_dict = result.markdown if hasattr(result, 'markdown') else {}
    
    # Convert markdown images to base64 if they exist
    markdown_images = {}
    if 'markdown_images' in markdown_dict:
        for img_path, img_obj in markdown_dict['markdown_images'].items():
            markdown_images[img_path] = encode_image_to_base64(img_obj)
    
    markdown_output = MarkdownOutput(
        text=markdown_dict.get('markdown_text', ''),
        images=markdown_images,
        isStart=markdown_dict.get('isStart', True),
        isEnd=markdown_dict.get('isEnd', True)
    )
    
    return {
        'json_result': json_result,
        'markdown': markdown_output
    }

async def parse_document(
    pipeline,
    file_path: str,
    use_doc_orientation_classify: Optional[bool] = None,
    use_doc_unwarping: Optional[bool] = None,
    use_textline_orientation: Optional[bool] = None,
    use_table_recognition: Optional[bool] = None,
    use_formula_recognition: Optional[bool] = None,
    use_chart_recognition: Optional[bool] = None,
    combine_markdown: bool = True
) -> ParseResponse:
    """
    Parse document using PPStructureV3 pipeline
    
    Args:
        pipeline: PPStructureV3 instance
        file_path: Path to the file to parse
        use_doc_orientation_classify: Enable document orientation classification
        use_doc_unwarping: Enable document unwarping
        use_textline_orientation: Enable text line orientation
        use_table_recognition: Enable table recognition
        use_formula_recognition: Enable formula recognition
        use_chart_recognition: Enable chart recognition
        combine_markdown: Combine all pages into single markdown (for PDFs)
    
    Returns:
        ParseResponse with results
    """
    try:
        # Prepare prediction parameters
        predict_params = {
            'input': file_path,
        }
        
        # Add optional parameters if specified
        if use_doc_orientation_classify is not None:
            predict_params['use_doc_orientation_classify'] = use_doc_orientation_classify
        if use_doc_unwarping is not None:
            predict_params['use_doc_unwarping'] = use_doc_unwarping
        if use_textline_orientation is not None:
            predict_params['use_textline_orientation'] = use_textline_orientation
        if use_table_recognition is not None:
            predict_params['use_table_recognition'] = use_table_recognition
        if use_formula_recognition is not None:
            predict_params['use_formula_recognition'] = use_formula_recognition
        if use_chart_recognition is not None:
            predict_params['use_chart_recognition'] = use_chart_recognition
        
        # Run prediction in thread pool
        output = await run_in_threadpool(
            lambda: list(pipeline.predict(**predict_params))
        )
        
        # Process each page result
        page_results = []
        markdown_list = []
        
        for idx, result in enumerate(output):
            processed = process_result(result)
            
            page_result = PageResult(
                page_index=idx,
                json_result=processed['json_result'],
                markdown=processed['markdown']
            )
            page_results.append(page_result)
            
            # Collect markdown for concatenation
            if combine_markdown and hasattr(result, 'markdown'):
                markdown_list.append(result.markdown)
        
        # Combine markdown pages if requested and multiple pages exist
        combined_markdown = None
        if combine_markdown and len(markdown_list) > 1:
            combined_markdown = await run_in_threadpool(
                lambda: pipeline.concatenate_markdown_pages(markdown_list)
            )
        
        return ParseResponse(
            success=True,
            message="Document parsed successfully",
            total_pages=len(page_results),
            results=page_results,
            combined_markdown=combined_markdown
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during document parsing: {str(e)}"
        )

# ================= Endpoints =================
@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "version": "3.2.0", "pipeline": "PPStructureV3"}

@app.post("/parse", response_model=ParseResponse)
async def parse_endpoint(
    file: UploadFile = File(..., description="PDF or image file to parse"),
    use_doc_orientation_classify: Optional[bool] = Query(
        None, 
        description="Enable document orientation classification"
    ),
    use_doc_unwarping: Optional[bool] = Query(
        None,
        description="Enable document image unwarping"
    ),
    use_textline_orientation: Optional[bool] = Query(
        None,
        description="Enable text line orientation classification"
    ),
    use_table_recognition: Optional[bool] = Query(
        None,
        description="Enable table recognition subpipeline"
    ),
    use_formula_recognition: Optional[bool] = Query(
        None,
        description="Enable formula recognition subpipeline"
    ),
    use_chart_recognition: Optional[bool] = Query(
        None,
        description="Enable chart parsing module"
    ),
    combine_markdown: bool = Query(
        True,
        description="Combine all pages into single markdown (for multi-page PDFs)"
    ),
    save_outputs: bool = Query(
        False,
        description="Save JSON and Markdown files to disk"
    ),
    output_dir: Optional[str] = Query(
        None,
        description="Directory to save output files (if save_outputs=True)"
    )
):
    """
    Parse endpoint for document structure extraction
    
    Accepts PDF or image files and returns:
    - Structured JSON with layout, text, tables, formulas, etc.
    - Markdown representation with embedded images
    - Per-page results for multi-page documents
    
    The endpoint uses all parameters configured during pipeline initialization
    including table recognition, model selections, and detection thresholds.
    """
    # Validate file
    validate_file(file)
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)  # Reset to beginning
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size_mb:.2f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)"
        )
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        file_ext = Path(file.filename).suffix
        temp_file_path = Path(temp_dir) / f"input{file_ext}"
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Acquire semaphore for prediction
        app.state.predict_sem.acquire()
        
        try:
            # Parse document
            response = await parse_document(
                pipeline=app.state.pipeline,
                file_path=str(temp_file_path),
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                use_table_recognition=use_table_recognition,
                use_formula_recognition=use_formula_recognition,
                use_chart_recognition=use_chart_recognition,
                combine_markdown=combine_markdown
            )
            
            # Optionally save outputs to disk
            if save_outputs:
                save_dir = Path(output_dir) if output_dir else Path("output")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save per-page results
                for page_result in response.results:
                    page_idx = page_result.page_index
                    
                    # Save JSON
                    json_path = save_dir / f"page_{page_idx}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(page_result.json_result, f, ensure_ascii=False, indent=2)
                    
                    # Save Markdown
                    md_path = save_dir / f"page_{page_idx}.md"
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(page_result.markdown.text)
                    
                    # Save markdown images
                    for img_path, img_base64 in page_result.markdown.images.items():
                        full_img_path = save_dir / img_path
                        full_img_path.parent.mkdir(parents=True, exist_ok=True)
                        img_data = base64.b64decode(img_base64)
                        with open(full_img_path, "wb") as f:
                            f.write(img_data)
                
                # Save combined markdown if available
                if response.combined_markdown:
                    combined_path = save_dir / "combined.md"
                    with open(combined_path, "w", encoding="utf-8") as f:
                        f.write(response.combined_markdown)
            
            return response
            
        finally:
            app.state.predict_sem.release()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "PPStructureV3 Parse API",
        "version": "3.2.0",
        "description": "Document parsing service using PaddleOCR PPStructureV3",
        "endpoints": {
            "/health": "Health check",
            "/parse": "Parse documents (POST with file upload)",
            "/docs": "Interactive API documentation",
        },
        "features": {
            "layout_detection": True,
            "table_recognition": USE_TABLE_RECOGNITION,
            "formula_recognition": USE_FORMULA_RECOGNITION,
            "chart_recognition": USE_CHART_RECOGNITION,
            "text_ocr": True,
            "markdown_export": True,
            "json_export": True,
        },
        "models": {
            "layout_detection": LAYOUT_DETECTION_MODEL_NAME,
            "text_detection": TEXT_DETECTION_MODEL_NAME,
            "text_recognition": TEXT_RECOGNITION_MODEL_NAME,
            "table_structure": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

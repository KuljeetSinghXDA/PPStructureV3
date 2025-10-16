import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.concurrency import run_in_threadpool

# For PDF multi-page support (complements pipeline; install via pip install pdf2image)
try:
    from pdf2image import convert_from_bytes
except ImportError:
    raise ImportError("Install pdf2image for PDF support: pip install pdf2image")

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

# Additional features (added from research)
RETURN_MARKDOWN = True  # Enable Markdown export for full reconstruction
RETURN_HTML = False    # For tables/charts
CROP_REGION = None     # Optional: dict with bbox for sub-region processing

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
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
        return_markdown=RETURN_MARKDOWN,
        return_html=RETURN_HTML,
        crop_region=CROP_REGION,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= /parse Endpoint (Added from Research) =================
@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    return_format: Literal["json", "markdown", "html"] = Query("json", description="Output format: json (detailed), markdown (full doc), html (tables/charts)"),
    use_doc_orientation_classify: Optional[bool] = Query(USE_DOC_ORIENTATION_CLASSIFY, description="Enable doc orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(USE_DOC_UNWARPING, description="Enable doc unwarping"),
    use_textline_orientation: Optional[bool] = Query(USE_TEXTLINE_ORIENTATION, description="Enable textline orientation"),
    use_table_recognition: Optional[bool] = Query(USE_TABLE_RECOGNITION, description="Enable table recognition (outputs HTML)"),
    use_formula_recognition: Optional[bool] = Query(USE_FORMULA_RECOGNITION, description="Enable formula recognition"),
    use_chart_recognition: Optional[bool] = Query(USE_CHART_RECOGNITION, description="Enable chart recognition (outputs CSV/HTML)"),
    page: Optional[int] = Query(None, ge=1, description="Specific page for PDFs (default: all)"),
):
    """
    Parse a document using PP-StructureV3, covering all features:
    - Layout detection (titles, tables, formulas, charts, etc.)
    - Table recognition (structured HTML)
    - Formula recognition (printed/handwritten formulas)
    - Chart recognition (tabular conversion)
    - Reading order restoration
    - Markdown export for full reconstruction
    Returns in specified format, with confidence scores and bounding boxes.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE_MB} MB")
    
    # Prepare images
    images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        if ext == ".pdf":
            pdf_pages = convert_from_bytes(content)
            if page:
                if page > len(pdf_pages):
                    raise HTTPException(status_code=400, detail=f"Page {page} out of range (total: {len(pdf_pages)})")
                images = [str(Path(temp_dir) / f"page_{page}.png")]
                pdf_pages[page-1].save(images[0], "PNG")
            else:
                for i, img in enumerate(pdf_pages):
                    img_path = str(Path(temp_dir) / f"page_{i+1}.png")
                    img.save(img_path, "PNG")
                    images.append(img_path)
        else:
            img_path = str(Path(temp_dir) / f"input{ext}")
            with open(img_path, "wb") as f:
                f.write(content)
            images = [img_path]
    
    # Run pipeline (with semaphore for concurrency)
    async with app.state.predict_sem:
        results = []
        for img_path in images:
            try:
                pipeline_args = {
                    "use_doc_orientation_classify": use_doc_orientation_classify,
                    "use_doc_unwarping": use_doc_unwarping,
                    "use_textline_orientation": use_textline_orientation,
                    "use_table_recognition": use_table_recognition,
                    "use_formula_recognition": use_formula_recognition,
                    "use_chart_recognition": use_chart_recognition,
                }
                result = await run_in_threadpool(app.state.pipeline, img_path, **pipeline_args)
                results.append(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Pipeline error on image {img_path}: {str(e)}")
    
    # Format output
    if return_format == "json":
        return JSONResponse(content={"pages": results})
    elif return_format == "markdown":
        markdown_output = ""
        for res in results:
            markdown_output += res.get("markdown", "# Error: Markdown not available\n") + "\n"
        return PlainTextResponse(content=markdown_output)
    elif return_format == "html":
        html_output = "<html><body>"
        for res in results:
            html_output += res.get("html", "<p>HTML not available</p>")
        html_output += "</body></html>"
        return HTMLResponse(content=html_output)

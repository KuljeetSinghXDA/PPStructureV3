import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

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
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}
    
@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown", "excel"] = Query("json", description="Output format for the structured result"),
    lang: Optional[str] = Query("en", description="Language for OCR (e.g., 'en', 'ch', 'fr')"),
    recovery: bool = Query(False, description="Enable recovery mode for distorted documents"),
    save_folder: Optional[str] = Query(None, description="Optional folder to save intermediate results like images or tables")
):
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Validate file size (approximate check via content length if available, otherwise read in chunks)
    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit")
    
    # Read file contents
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit")
    
    # Acquire semaphore for parallel prediction control
    if not app.state.predict_sem.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Server busy: Maximum parallel predictions reached")
    
    try:
        # Create temporary file for input
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_input:
            tmp_input.write(contents)
            tmp_input_path = tmp_input.name
        
        # Prepare save folder if specified
        temp_save_dir = None
        if save_folder:
            temp_save_dir = tempfile.mkdtemp(dir=save_folder)
        else:
            temp_save_dir = tempfile.mkdtemp()
        
        # Dynamically override language if not English (PPStructureV3 supports lang parameter)
        pipeline_kwargs = {}
        if lang != "en":
            pipeline_kwargs["lang"] = lang
        
        # Run prediction in threadpool to avoid blocking the event loop
        result = await run_in_threadpool(
            app.state.pipeline,
            img=tmp_input_path,
            recovery=recovery,
            save_path=temp_save_dir,
            **pipeline_kwargs
        )
        
        # Process result based on output format
        if output_format == "json":
            # PPStructureV3 returns a list of dicts; serialize directly
            output_data = json.dumps(result, ensure_ascii=False, indent=2)
            response = PlainTextResponse(content=output_data, media_type="application/json")
        elif output_format == "markdown":
            # Convert structure to markdown (custom logic for text, tables, etc.)
            md_content = convert_to_markdown(result)
            response = PlainTextResponse(content=md_content, media_type="text/markdown")
        elif output_format == "excel" and result_has_tables(result):
            # Save tables to Excel if present
            excel_path = os.path.join(temp_save_dir, "output.xlsx")
            save_to_excel(result, excel_path)
            response = JSONResponse(content={"message": "Excel generated", "excel_path": excel_path})
        else:
            response = JSONResponse(content=result)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup
        app.state.predict_sem.release()
        os.unlink(tmp_input_path)
        if not save_folder and temp_save_dir:
            shutil.rmtree(temp_save_dir, ignore_errors=True)

def result_has_tables(result):
    """Check if result contains table elements."""
    return any(block.get("type") == "table" for page in result for block in page)

def convert_to_markdown(result: List[dict]) -> str:
    """Convert PPStructureV3 result to Markdown format."""
    md_lines = []
    for page in result:
        for block in page:
            block_type = block.get("type", "text")
            if block_type == "text":
                md_lines.append(block.get("text", ""))
            elif block_type == "title":
                md_lines.append(f"# {block.get('text', '')}")
            elif block_type == "table":
                # Convert table html to markdown table
                html = block.get("res_html", "")
                md_lines.append(html_to_markdown_table(html))
            elif block_type == "figure":
                md_lines.append(f"![Figure]({block.get('img_path', '')})")
            md_lines.append("\n")
    return "\n".join(md_lines)

def html_to_markdown_table(html: str) -> str:
    """Simple HTML table to Markdown conversion."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return ""
    
    md = []
    rows = table.find_all("tr")
    for i, row in enumerate(rows):
        cols = row.find_all(["td", "th"])
        cols_text = [col.get_text(strip=True) for col in cols]
        md.append("| " + " | ".join(cols_text) + " |")
        if i == 0:
            md.append("| " + " | ".join(["---"] * len(cols)) + " |")
    return "\n".join(md)

def save_to_excel(result: List[dict], excel_path: str):
    """Save tables from result to Excel using pandas/openpyxl."""
    import pandas as pd
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        table_idx = 1
        for page_idx, page in enumerate(result):
            for block_idx, block in enumerate(page):
                if block.get("type") == "table":
                    html = block.get("res_html", "")
                    df = html_table_to_dataframe(html)
                    df.to_excel(writer, sheet_name=f"Page{page_idx+1}_Table{table_idx}", index=False)
                    table_idx += 1

def html_table_to_dataframe(html: str) -> pd.DataFrame:
    """Convert HTML table to pandas DataFrame."""
    import pandas as pd
    from io import StringIO
    return pd.read_html(StringIO(html))[0]

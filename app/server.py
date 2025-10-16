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
import numpy as np
from PIL import Image
import asyncio  # Add this import for asyncio.Semaphore

# Explicit imports for PaddleOCR components
ENABLE_HPI = False  # High-performance inference (disable if causing issues)
ENABLE_MKLDNN = True  # MKL-DNN acceleration for CPU

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values) =================
DEVICE = "cpu"  # Pin to CPU; set to "gpu" if using CUDA
CPU_THREADS = 4  # Adjust based on hardware

# Optional accuracy boosters (disabled for speed; enable per docs if needed)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles (per PP-StructureV3 docs)
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = True  # Enabled for math parsing
USE_CHART_RECOGNITION = True   # Enabled for chart handling

# Model overrides (latest from docs: paddlepaddle.github.io)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"  # Improved from M; see module_usage/layout_detection.html
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"  # See pipeline_usage/formula_recognition.html
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters (None for defaults)
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
MAX_PARALLEL_PREDICT = 1  # Semaphore for CPU bound tasks

# ================= Helper Functions =================
def process_file(file_path: str, pipeline: PPStructureV3) -> dict:
    """
    Process a file (image/PDF) using PP-StructureV3.
    Handles images by loading as NumPy array; PDFs are processed natively.
    Returns the raw JSON-like dict output (per docs).
    """
    if file_path.lower().endswith('.pdf'):
        # Direct PDF processing (PP-StructureV3 handles multi-page internally)
        result = pipeline(file_path)
    else:
        # Load image, convert to NumPy, process
        img = Image.open(file_path).convert("RGB")
        img_np = np.array(img)
        result = pipeline(img_np)
    
    return result  # Assumed dict with 'results' list, etc. (based on doc examples)

def json_to_markdown(data: dict) -> str:
    """
    Convert PP-StructureV3 JSON output to Markdown.
    - Sorts by vertical position (bbox[1]).
    - Text: Detects headers (short lines), paras (long lines).
    - Tables: Converts to Markdown table (assumes content is HTML).
    - Formulas/Charts: Renders as code blocks.
    - Basic; full fidelity requires more parsing (e.g., LaTeX math syntax).
    """
    md_lines = []
    results = data.get("results", [])
    
    # Sort by bounding box Y-coordinate for top-down order
    sorted_results = sorted(results, key=lambda x: x.get('bbox', [0, 0, 0, 0])[1])
    
    for item in sorted_results:
        item_type = item.get('type', 'text').lower()  # Default to text
        content = item.get('content', item.get('text', ''))
        bbox = item.get('bbox', [0, 0, 100, 20])  # Fallback bbox
        
        if item_type in ['text', 'paragraph']:
            # Simple header/para detection: if line count < 3 and < 100 chars, treat as header
            lines = content.split('\n')
            if len(lines) <= 2 and len(content.strip()) < 100:
                md_lines.append(f"# {content.strip()}")
            else:
                md_lines.append(content.strip())
            md_lines.append("")  # Blank line
        elif item_type == 'table':
            # Assume content is HTML table; basic conversion to MD table
            html_table = content
            # Minimal HTML->MD: Split by rows/cells (rough approximation)
            md_table_lines = []
            rows = html_table.replace('<table>', '').replace('</table>', '').split('<tr>')
            for row in rows:
                if '<td>' in row or '<th>' in row:
                    cells = []
                    cols = row.split('</td>')[:-1]  # Chop off empty
                    for col in cols:
                        cell_text = col.split('>')[-1].strip()
                        cells.append(cell_text)
                    md_table_lines.append('| ' + ' | '.join(cells) + ' |')
                    if len(md_table_lines) == 1:  # Add header separator
                        md_table_lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
            md_lines.extend(md_table_lines)
            md_lines.append("")
        elif item_type == 'formula':
            # Render formulas as LaTeX code block (assumes content is LaTeX)
            md_lines.append(f"$$ {content.strip()} $$")
            md_lines.append("")
        elif item_type == 'chart':
            # Render as tabular data or code block (docs suggest chart->table conversion)
            md_lines.append("```chart")
            md_lines.append(content.strip() or "Chart data placeholder")
            md_lines.append("```")
            md_lines.append("")
        else:
            # Fallback: plain text
            md_lines.append(content.strip())
            md_lines.append("")
    
    return '\n'.join(md_lines).strip()

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
    app.state.predict_sem = asyncio.Semaphore(value=MAX_PARALLEL_PREDICT)  # Changed to asyncio.Semaphore
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(file: UploadFile = File(...)) -> JSONResponse:
    """
    Parse a document (image/PDF) using PP-StructureV3.
    Returns JSON output directly, plus a Markdown representation.
    """
    # Acquire semaphore for concurrency control (now async-compatible)
    async with app.state.predict_sem:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")
        
        # Read and validate file size
        content = await file.read()
        file_size = len(content)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE_MB}MB")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process in threadpool (CPU-bound task)
            pp_result = await run_in_threadpool(process_file, tmp_path, app.state.pipeline)
            
            # Generate Markdown from JSON
            markdown_output = json_to_markdown(pp_result)
            
            # Return combined JSON response
            return JSONResponse(content={
                "json": pp_result,  # Raw PP-StructureV3 output
                "markdown": markdown_output  # Generated Markdown string
            })
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

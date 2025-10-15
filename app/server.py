import os
import tempfile
import threading
import json
import shutil
import base64
import io
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image

ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values for Pinnacle Accuracy) =================
DEVICE = "cpu"
CPU_THREADS = 8  # Increased for better CPU utilization

# Accuracy boosters enabled for pinnacle performance
USE_DOC_ORIENTATION_CLASSIFY = True
USE_DOC_UNWARPING = True
USE_TEXTLINE_ORIENTATION = True

# All subpipelines enabled for comprehensive parsing
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = True
USE_CHART_RECOGNITION = True

# Model overrides for highest accuracy (server/large models where applicable)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_server_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANeXt_wired_structure_rec"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANeXt_wireless_structure_rec"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-L"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Optimized detection/recognition parameters for accuracy
LAYOUT_THRESHOLD = None  # Default 0.5
TEXT_DET_THRESH = None   # Default 0.3
TEXT_DET_BOX_THRESH = None  # Default 0.6
TEXT_DET_UNCLIP_RATIO = None  # Default 2.0
TEXT_DET_LIMIT_SIDE_LEN = 4096  # Higher limit for better accuracy on large docs
TEXT_DET_LIMIT_TYPE = "max"
TEXT_REC_SCORE_THRESH = None  # Default 0.0 (no threshold for completeness)
TEXT_RECOGNITION_BATCH_SIZE = 1  # Conservative batch for CPU stability

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
        lang="en",  # English for optimal text recognition
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
    file: UploadFile = File(..., description="Document file (image or PDF)"),
    output_format: Literal["json", "markdown"] = Query("json", description="Output format"),
    visualize: bool = Query(False, description="Include base64-encoded visualization images")
):
    # Validate file
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: " + ", ".join(ALLOWED_EXTENSIONS))
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    # Create temp file
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)

    sem = app.state.predict_sem
    if not sem.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Too many concurrent requests. Max parallel: 1")

    try:
        # Run prediction in threadpool
        results = await run_in_threadpool(app.state.pipeline.predict, str(tmp_path))
        
        # Handle single or multi-page output
        if len(results) == 0:
            raise HTTPException(status_code=500, detail="No results from pipeline")

        output = {}
        
        if output_format == "json":
            output["pages"] = []
            for i, res in enumerate(results):
                page_res = res.json["res"].copy()  # Deep copy to avoid mutations
                
                if visualize:
                    vis_dir = tempfile.mkdtemp()
                    try:
                        res.save_to_img(save_path=vis_dir)
                        page_res["visualizations"] = {}
                        for vis_file in Path(vis_dir).iterdir():
                            if vis_file.suffix.lower() == ".png":
                                with Image.open(vis_file) as img:
                                    page_res["visualizations"][vis_file.name] = pil_to_base64(img)
                    finally:
                        shutil.rmtree(vis_dir, ignore_errors=True)
                
                output["pages"].append(page_res)
        
        elif output_format == "markdown":
            if len(results) > 1:
                # Merge multi-page markdown
                from paddleocr import PPStructureV3
                merged_md = PPStructureV3.concatenate_markdown_pages([r.markdown for r in results])
                output["text"] = merged_md["markdown_texts"]
                output["images"] = {k: pil_to_base64(v) for k, v in merged_md["markdown_images"].items()}
                output["page_continuation_flags"] = merged_md["page_continuation_flags"]
            else:
                # Single page
                md = results[0].markdown
                output["text"] = md["markdown_texts"]
                output["images"] = {k: pil_to_base64(v) for k, v in md["markdown_images"].items()}
                output["page_continuation_flags"] = md["page_continuation_flags"]
        
        return JSONResponse(content=output)
    
    finally:
        # Cleanup
        sem.release()
        os.unlink(tmp_path)

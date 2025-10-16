import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

ENABLE_HPI = False
ENABLE_MKLDNN = True  # Enabled for CPU optimization; ignored on ARM64 if inapplicable

from paddleocr import PPStructureV3

# ================= Core Configuration (Latest PP-StructureV3 Models for ARM64 CPU) =================
DEVICE = "cpu"
CPU_THREADS = 4  # Suitable for ARM64 cores

# Optional accuracy boosters (defaults from PP-StructureV3 doc)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles (updated defaults from PP-StructureV3 doc)
USE_SEAL_RECOGNITION = False
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = True
USE_CHART_RECOGNITION = False
USE_REGION_DETECTION = True

# Model overrides (latest from PP-StructureV3 / PaddleOCR 3.0)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "en_PP-OCRv5_mobile_det"  # Mobile for CPU efficiency on ARM64
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-L"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"
SEAL_TEXT_DETECTION_MODEL_NAME = "PP-OCRv4_server_seal_det"
SEAL_TEXT_RECOGNITION_MODEL_NAME = "ch_PP-OCRv4_server_rec"  # Primarily for Chinese seals; toggle off for EN
REGION_DETECTION_MODEL_NAME = None  # Uses default official model
DOC_ORIENTATION_CLASSIFY_MODEL_NAME = None  # Default PP-LCNet_x1_0_doc_ori
DOC_UNWARPING_MODEL_NAME = None  # Default UVDoc
TEXTLINE_ORIENTATION_MODEL_NAME = None  # Default PP-LCNet_x1_0_textline_ori
TABLE_ORIENTATION_CLASSIFY_MODEL_NAME = None  # Default official
WIRED_TABLE_CELLS_DETECTION_MODEL_NAME = None  # Default official
WIRELESS_TABLE_CELLS_DETECTION_MODEL_NAME = None  # Default official

# Detection/recognition parameters (None for defaults from PP-StructureV3 doc)
LAYOUT_THRESHOLD = None  # Default 0.5
LAYOUT_NMS = None  # Default True
LAYOUT_UNCLIP_RATIO = None  # Default 1.0
LAYOUT_MERGE_BBOXES_MODE = None  # Default "large"
TEXT_DET_THRESH = None  # Default 0.3
TEXT_DET_BOX_THRESH = None  # Default 0.6
TEXT_DET_UNCLIP_RATIO = None  # Default 2.0
TEXT_DET_LIMIT_SIDE_LEN = None  # Default 960
TEXT_DET_LIMIT_TYPE = None  # Default "max"
TEXT_REC_SCORE_THRESH = None  # Default 0.0
TEXT_RECOGNITION_BATCH_SIZE = None  # Default 1
SEAL_DET_LIMIT_SIDE_LEN = None  # Default 736
SEAL_DET_LIMIT_TYPE = None  # Default "min"
SEAL_DET_THRESH = None  # Default 0.2
SEAL_DET_BOX_THRESH = None  # Default 0.6
SEAL_DET_UNCLIP_RATIO = None  # Default 0.5
SEAL_REC_SCORE_THRESH = None  # Default 0.0
CHART_RECOGNITION_BATCH_SIZE = None  # Default 1
FORMULA_RECOGNITION_BATCH_SIZE = None  # Default 1
TEXTLINE_ORIENTATION_BATCH_SIZE = None  # Default 1
SEAL_TEXT_RECOGNITION_BATCH_SIZE = None  # Default 1

# Additional table-specific params (from PP-StructureV3 doc) - these are for predict, not init
USE_WIRED_TABLE_CELLS_TRANS_TO_HTML = None  # Default True if table enabled
USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML = None  # Default True if table enabled
USE_TABLE_ORIENTATION_CLASSIFY = None  # Default True if table enabled
USE_OCR_RESULTS_WITH_TABLE_CELLS = None  # Default True if table enabled
USE_E2E_WIRED_TABLE_REC_MODEL = None  # Default False
USE_E2E_WIRELESS_TABLE_REC_MODEL = None  # Default False

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
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
        # Layout
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        layout_threshold=LAYOUT_THRESHOLD,
        layout_nms=LAYOUT_NMS,
        layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
        layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,
        # Region
        region_detection_model_name=REGION_DETECTION_MODEL_NAME,
        use_region_detection=USE_REGION_DETECTION,
        # Preprocessing
        doc_orientation_classify_model_name=DOC_ORIENTATION_CLASSIFY_MODEL_NAME,
        doc_unwarping_model_name=DOC_UNWARPING_MODEL_NAME,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        # Textline orientation
        textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL_NAME,
        textline_orientation_batch_size=TEXTLINE_ORIENTATION_BATCH_SIZE,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        # OCR
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        # Table
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wired_table_cells_detection_model_name=WIRED_TABLE_CELLS_DETECTION_MODEL_NAME,
        wireless_table_cells_detection_model_name=WIRELESS_TABLE_CELLS_DETECTION_MODEL_NAME,
        table_orientation_classify_model_name=TABLE_ORIENTATION_CLASSIFY_MODEL_NAME,
        use_table_recognition=USE_TABLE_RECOGNITION,
        # Formula
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        formula_recognition_batch_size=FORMULA_RECOGNITION_BATCH_SIZE,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        # Chart
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        chart_recognition_batch_size=CHART_RECOGNITION_BATCH_SIZE,
        use_chart_recognition=USE_CHART_RECOGNITION,
        # Seal
        seal_text_detection_model_name=SEAL_TEXT_DETECTION_MODEL_NAME,
        seal_det_limit_side_len=SEAL_DET_LIMIT_SIDE_LEN,
        seal_det_limit_type=SEAL_DET_LIMIT_TYPE,
        seal_det_thresh=SEAL_DET_THRESH,
        seal_det_box_thresh=SEAL_DET_BOX_THRESH,
        seal_det_unclip_ratio=SEAL_DET_UNCLIP_RATIO,
        seal_text_recognition_model_name=SEAL_TEXT_RECOGNITION_MODEL_NAME,
        seal_text_recognition_batch_size=SEAL_TEXT_RECOGNITION_BATCH_SIZE,
        seal_rec_score_thresh=SEAL_REC_SCORE_THRESH,
        use_seal_recognition=USE_SEAL_RECOGNITION,
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="3.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse", summary="Parse document using PP-StructureV3")
async def parse_document(
    file: UploadFile = File(..., description="Upload a document (PDF, image)"),
    output_format: Literal["json", "markdown"] = Query("json", description="Output format: json or markdown"),
    # Toggle overrides (passed to predict)
    use_doc_orientation_classify: Optional[bool] = Query(None, description="Override for document orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(None, description="Override for document unwarping"),
    use_textline_orientation: Optional[bool] = Query(None, description="Override for textline orientation"),
    use_seal_recognition: Optional[bool] = Query(None, description="Override for seal recognition"),
    use_table_recognition: Optional[bool] = Query(None, description="Override for table recognition"),
    use_formula_recognition: Optional[bool] = Query(None, description="Override for formula recognition"),
    use_chart_recognition: Optional[bool] = Query(None, description="Override for chart recognition"),
    use_region_detection: Optional[bool] = Query(None, description="Override for region detection"),
    # Threshold/Param overrides (passed to predict)
    layout_threshold: Optional[float] = Query(None, description="Override layout threshold"),
    layout_nms: Optional[bool] = Query(None, description="Override layout NMS"),
    layout_unclip_ratio: Optional[float] = Query(None, description="Override layout unclip ratio"),
    layout_merge_bboxes_mode: Optional[str] = Query(None, description="Override layout merge mode ('large', 'small', 'union')"),
    text_det_limit_side_len: Optional[int] = Query(None, description="Override text det side len limit"),
    text_det_limit_type: Optional[str] = Query(None, description="Override text det limit type ('min', 'max')"),
    text_det_thresh: Optional[float] = Query(None, description="Override text det thresh"),
    text_det_box_thresh: Optional[float] = Query(None, description="Override text det box thresh"),
    text_det_unclip_ratio: Optional[float] = Query(None, description="Override text det unclip ratio"),
    text_rec_score_thresh: Optional[float] = Query(None, description="Override text rec score thresh"),
    seal_det_limit_side_len: Optional[int] = Query(None, description="Override seal det side len limit"),
    seal_det_limit_type: Optional[str] = Query(None, description="Override seal det limit type"),
    seal_det_thresh: Optional[float] = Query(None, description="Override seal det thresh"),
    seal_det_box_thresh: Optional[float] = Query(None, description="Override seal det box thresh"),
    seal_det_unclip_ratio: Optional[float] = Query(None, description="Override seal det unclip ratio"),
    seal_rec_score_thresh: Optional[float] = Query(None, description="Override seal rec score thresh"),
    # Table-specific overrides
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None, description="Override wired table to HTML"),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None, description="Override wireless table to HTML"),
    use_table_orientation_classify: Optional[bool] = Query(None, description="Override table orientation classify"),
    use_ocr_results_with_table_cells: Optional[bool] = Query(None, description="Override OCR with table cells"),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None, description="Override E2E wired table rec"),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None, description="Override E2E wireless table rec"),
):
    # File validation
    filename = Path(file.filename).name.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported file extension. Allowed: " + ", ".join(ALLOWED_EXTENSIONS))
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")
    
    # Temp file handling
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Collect overrides dict (filter None values)
        overrides: Dict[str, Any] = {}
        locals_dict = locals()
        for key in [
            "use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation",
            "use_seal_recognition", "use_table_recognition", "use_formula_recognition",
            "use_chart_recognition", "use_region_detection",
            "layout_threshold", "layout_nms", "layout_unclip_ratio", "layout_merge_bboxes_mode",
            "text_det_limit_side_len", "text_det_limit_type", "text_det_thresh", "text_det_box_thresh",
            "text_det_unclip_ratio", "text_rec_score_thresh",
            "seal_det_limit_side_len", "seal_det_limit_type", "seal_det_thresh", "seal_det_box_thresh",
            "seal_det_unclip_ratio", "seal_rec_score_thresh",
            "use_wired_table_cells_trans_to_html", "use_wireless_table_cells_trans_to_html",
            "use_table_orientation_classify", "use_ocr_results_with_table_cells",
            "use_e2e_wired_table_rec_model", "use_e2e_wireless_table_rec_model",
        ]:
            if locals_dict[key] is not None:
                overrides[key] = locals_dict[key]
        
        # Run prediction in threadpool with semaphore
        async with app.state.predict_sem:
            output = await run_in_threadpool(app.state.pipeline.predict, tmp_path, **overrides)
        
        # Process output based on format
        if output_format == "json":
            # Collect pruned results per page
            json_data = []
            for i, res in enumerate(output):
                pruned = res.get("prunedResult", res)  # Fallback to full res if no pruned
                json_data.append({"page_index": i, "result": pruned})
            return JSONResponse(content=json_data)
        
        elif output_format == "markdown":
            # Collect markdown text per page
            md_parts = []
            for res in output:
                md_dict = res.get("markdown", {})
                md_text = md_dict.get("text", "")
                md_parts.append(md_text)
            md_content = "\n\n--- PAGE BREAK ---\n\n".join(md_parts)
            return PlainTextResponse(content=md_content, media_type="text/markdown")
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    raise HTTPException(status_code=500, detail="Processing failed")

import os
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
import base64
from PIL import Image

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
    file: UploadFile = File(..., description="Document file (image or PDF)"),
    # Module toggles
    use_doc_orientation_classify: Optional[bool] = Query(None, description="Enable document orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(None, description="Enable document unwarping"),
    use_textline_orientation: Optional[bool] = Query(None, description="Enable textline orientation classification"),
    use_seal_recognition: Optional[bool] = Query(None, description="Enable seal recognition"),
    use_table_recognition: Optional[bool] = Query(None, description="Enable table recognition"),
    use_formula_recognition: Optional[bool] = Query(None, description="Enable formula recognition"),
    use_chart_recognition: Optional[bool] = Query(None, description="Enable chart recognition"),
    use_region_detection: Optional[bool] = Query(None, description="Enable region detection"),
    # Table-specific
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None, description="Enable wired table to HTML conversion"),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None, description="Enable wireless table to HTML conversion"),
    use_table_orientation_classify: Optional[bool] = Query(None, description="Enable table orientation classification"),
    use_ocr_results_with_table_cells: Optional[bool] = Query(None, description="Use OCR with table cells"),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None, description="Use E2E wired table model"),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None, description="Use E2E wireless table model"),
    # Thresholds and limits (simple float/int for ease; dicts not supported in query)
    layout_threshold: Optional[float] = Query(None, description="Layout detection threshold (0-1)"),
    text_det_thresh: Optional[float] = Query(None, description="Text detection pixel threshold"),
    text_det_box_thresh: Optional[float] = Query(None, description="Text detection box threshold"),
    text_det_unclip_ratio: Optional[float] = Query(None, description="Text detection unclip ratio"),
    text_det_limit_side_len: Optional[int] = Query(None, description="Text detection side length limit"),
    text_det_limit_type: Optional[Literal["min", "max"]] = Query(None, description="Text detection limit type"),
    text_rec_score_thresh: Optional[float] = Query(None, description="Text recognition score threshold"),
    # Visualization
    visualize: bool = Query(False, description="Include base64-encoded images in response"),
):
    # Validate file
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Prepare overrides dict
        overrides: Dict[str, Any] = {}
        locals_dict = locals()
        for param in [
            'use_doc_orientation_classify', 'use_doc_unwarping', 'use_textline_orientation',
            'use_seal_recognition', 'use_table_recognition', 'use_formula_recognition',
            'use_chart_recognition', 'use_region_detection',
            'use_wired_table_cells_trans_to_html', 'use_wireless_table_cells_trans_to_html',
            'use_table_orientation_classify', 'use_ocr_results_with_table_cells',
            'use_e2e_wired_table_rec_model', 'use_e2e_wireless_table_rec_model',
            'layout_threshold', 'text_det_thresh', 'text_det_box_thresh', 'text_det_unclip_ratio',
            'text_det_limit_side_len', 'text_det_limit_type', 'text_rec_score_thresh'
        ]:
            if locals_dict[param] is not None:
                overrides[param] = locals_dict[param]
        
        def do_predict() -> List[Any]:
            app.state.predict_sem.acquire()
            try:
                return app.state.pipeline.predict(tmp_path, **overrides)
            finally:
                app.state.predict_sem.release()
        
        output = await run_in_threadpool(do_predict)
        
        # Process results
        pages: List[Dict[str, Any]] = []
        for idx, res in enumerate(output):
            # Pruned result (remove input_path and page_index)
            page_dict = res.json.copy()
            page_dict.pop('input_path', None)
            page_dict.pop('page_index', None)
            
            # Add markdown info
            md_info = res.markdown
            page_dict['markdown'] = {
                'text': md_info['markdown_texts'],
                'is_start': md_info['page_continuation_flags'][0],
                'is_end': md_info['page_continuation_flags'][1],
            }
            
            if visualize:
                # Visualization images
                viz_images = {}
                for key, img in res.img.items():
                    if isinstance(img, Image.Image):
                        buf = BytesIO()
                        img.save(buf, format='PNG')
                        viz_images[f'{key}.png'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                page_dict['visualization_images'] = viz_images
                
                # Markdown images
                md_images = {}
                for rel_path, img in md_info.get('markdown_images', {}).items():
                    if isinstance(img, Image.Image):
                        buf = BytesIO()
                        img.save(buf, format='PNG')
                        md_images[rel_path] = base64.b64encode(buf.getvalue()).decode('utf-8')
                page_dict['markdown']['images'] = md_images
            
            pages.append(page_dict)
        
        response_data = {
            'result': {
                'layoutParsingResults': pages
            },
            'dataInfo': {
                'input_file': file.filename,
                'num_pages': len(pages)
            }
        }
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

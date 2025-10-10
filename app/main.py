import os
import tempfile
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from paddleocr import PPStructureV3

def getenv_bool(key: str, default: str = "False") -> bool:
    return os.getenv(key, default).lower() in ("1", "true", "yes", "on")

def getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

app = FastAPI(title="PP-StructureV3 Service", version="1.0")

# Read env for pipeline configuration
use_doc_orientation = getenv_bool("USE_DOC_ORIENTATION", "False")
use_unwarp = getenv_bool("USE_UNWARP", "False")
use_textline_ori = getenv_bool("USE_TEXTLINE_ORI", "False")
use_region_det = getenv_bool("USE_REGION_DET", "True")
use_table = getenv_bool("USE_TABLE", "True")
use_formula = getenv_bool("USE_FORMULA", "False")
use_chart = getenv_bool("USE_CHART", "False")
use_seal = getenv_bool("USE_SEAL", "False")

device = os.getenv("DEVICE", "cpu")
enable_mkldnn = getenv_bool("ENABLE_MKLDNN", "True")
enable_hpi = getenv_bool("ENABLE_HPI", "False")
use_tensorrt = getenv_bool("USE_TENSORRT", "False")
precision = os.getenv("PRECISION", "fp32")
cpu_threads = getenv_int("CPU_THREADS", 4)

# Language (apply English-only tip)
ocr_lang = os.getenv("OCR_LANG", "en")  # PPStructureV3(lang="en") selects English recognizer. [web:16]

# Model names
layout_model = os.getenv("LAYOUT_MODEL") or None
region_model = os.getenv("REGION_MODEL") or None
text_det_model = os.getenv("TEXT_DET_MODEL") or None

# If TEXT_REC_MODEL is empty, leave as None so lang=en takes effect
_text_rec_env = os.getenv("TEXT_REC_MODEL", "").strip()
text_rec_model = _text_rec_env if _text_rec_env else None  # None => use `lang` selection. [web:16]

table_cls_model = os.getenv("TABLE_CLS_MODEL") or None
table_struct_wired = os.getenv("TABLE_STRUCT_WIRED") or None
table_struct_wireless = os.getenv("TABLE_STRUCT_WIRELESS") or None
table_cell_det_wired = os.getenv("TABLE_CELL_DET_WIRED") or None
table_cell_det_wireless = os.getenv("TABLE_CELL_DET_WIRELESS") or None
formula_model = os.getenv("FORMULA_MODEL") or None
chart_model = os.getenv("CHART_MODEL") or None
seal_det_model = os.getenv("SEAL_DET_MODEL") or None

# Thresholds / batch sizes
layout_threshold = getenv_float("LAYOUT_THRESHOLD", 0.5)
layout_nms = getenv_bool("LAYOUT_NMS", "True")
layout_unclip_ratio = getenv_float("LAYOUT_UNCLIP_RATIO", 1.0)
layout_merge_mode = os.getenv("LAYOUT_MERGE_MODE", "large")

text_det_limit_side_len = getenv_int("TEXT_DET_LIMIT_SIDE_LEN", 960)
text_det_limit_type = os.getenv("TEXT_DET_LIMIT_TYPE", "max")
text_det_thresh = getenv_float("TEXT_DET_THRESH", 0.3)
text_det_box_thresh = getenv_float("TEXT_DET_BOX_THRESH", 0.6)
text_det_unclip_ratio = getenv_float("TEXT_DET_UNCLIP_RATIO", 2.0)

text_rec_batch = getenv_int("TEXT_REC_BATCH", 2)
text_rec_score_thresh = getenv_float("TEXT_REC_SCORE_THRESH", 0.0)

textline_ori_batch = getenv_int("TEXTLINE_ORI_BATCH", 1)

chart_batch = getenv_int("CHART_BATCH", 1)
formula_batch = getenv_int("FORMULA_BATCH", 1)

seal_det_limit_side_len = getenv_int("SEAL_DET_LIMIT_SIDE_LEN", 736)
seal_det_limit_type = os.getenv("SEAL_DET_LIMIT_TYPE", "min")
seal_det_thresh = getenv_float("SEAL_DET_THRESH", 0.2)
seal_det_box_thresh = getenv_float("SEAL_DET_BOX_THRESH", 0.6)
seal_det_unclip_ratio = getenv_float("SEAL_DET_UNCLIP_RATIO", 0.5)

# Instantiate PP-StructureV3 pipeline with explicit flags and model names
pipeline = PPStructureV3(
    # Language selection (forces English recognizer when model name is not set)
    lang=ocr_lang,  # e.g., "en". [web:16]

    # Module toggles
    use_doc_orientation_classify=use_doc_orientation,
    use_doc_unwarping=use_unwarp,
    use_textline_orientation=use_textline_ori,
    use_region_detection=use_region_det,
    use_table_recognition=use_table,
    use_formula_recognition=use_formula,
    use_chart_recognition=use_chart,
    use_seal_recognition=use_seal,

    # Device / backend
    device=device,
    enable_mkldnn=enable_mkldnn,
    enable_hpi=enable_hpi,
    use_tensorrt=use_tensorrt,
    precision=precision,
    cpu_threads=cpu_threads,

    # Layout & region
    layout_detection_model_name=layout_model,
    layout_threshold=layout_threshold,
    layout_nms=layout_nms,
    layout_unclip_ratio=layout_unclip_ratio,
    layout_merge_bboxes_mode=layout_merge_mode,
    region_detection_model_name=region_model,

    # OCR
    text_detection_model_name=text_det_model,
    text_det_limit_side_len=text_det_limit_side_len,
    text_det_limit_type=text_det_limit_type,
    text_det_thresh=text_det_thresh,
    text_det_box_thresh=text_det_box_thresh,
    text_det_unclip_ratio=text_det_unclip_ratio,
    text_recognition_model_name=text_rec_model,  # None => pick English by `lang`. [web:16]
    text_recognition_batch_size=text_rec_batch,
    text_rec_score_thresh=text_rec_score_thresh,

    # Textline orientation
    textline_orientation_batch_size=textline_ori_batch,

    # Tables
    table_classification_model_name=table_cls_model,
    wired_table_structure_recognition_model_name=table_struct_wired,
    wireless_table_structure_recognition_model_name=table_struct_wireless,
    wired_table_cells_detection_model_name=table_cell_det_wired,
    wireless_table_cells_detection_model_name=table_cell_det_wireless,

    # Formula
    formula_recognition_model_name=formula_model,
    formula_recognition_batch_size=formula_batch,

    # Chart
    chart_recognition_model_name=chart_model,
    chart_recognition_batch_size=chart_batch,

    # Seals
    seal_text_detection_model_name=seal_det_model,
    seal_det_limit_side_len=seal_det_limit_side_len,
    seal_det_limit_type=seal_det_limit_type,
    seal_det_thresh=seal_det_thresh,
    seal_det_box_thresh=seal_det_box_thresh,
    seal_det_unclip_ratio=seal_det_unclip_ratio,
)

class ParseResponse(BaseModel):
    pages: int
    results: list

@app.post("/v1/parse/pdf", response_model=ParseResponse)
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            output = pipeline.predict(tmp.name)
            results = []
            for res in output:
                results.append(res.json)
            return JSONResponse(content={"pages": len(results), "results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse error: {e}")

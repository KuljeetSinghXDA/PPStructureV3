import os
import io
import multiprocessing
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from paddleocr import PPStructureV3
from PIL import Image

def _cpu_threads():
    cpus = multiprocessing.cpu_count()
    return max(2, min(8, cpus))

# Environment (read from .env or container env)
OCR_LANG = os.getenv("OCR_LANG", "en")  # PP-StructureV3 supports setting lang to select recognition models
USE_DOC_ORI = os.getenv("USE_DOC_ORI", "false").lower() == "true"
USE_UNWARP = os.getenv("USE_UNWARP", "false").lower() == "true"
USE_TEXTLINE_ORI = os.getenv("USE_TEXTLINE_ORI", "false").lower() == "true"

# Chart recognition (optional)
USE_CHART_RECOGNITION = os.getenv("USE_CHART_RECOGNITION", "false").lower() == "true"
CHART_RECOGNITION_MODEL_NAME = os.getenv("CHART_RECOGNITION_MODEL_NAME", "PP-Chart2Table")

# Table structure recognition (optional)
USE_E2E_WIRED_TABLE_REC_MODEL = os.getenv("USE_E2E_WIRED_TABLE_REC_MODEL", "false").lower() == "true"
USE_E2E_WIRELESS_TABLE_REC_MODEL = os.getenv("USE_E2E_WIRELESS_TABLE_REC_MODEL", "false").lower() == "true"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = os.getenv("WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME", "SLANeXt_wired")
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = os.getenv("WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME", "SLANeXt_wireless")

CPU_THREADS = int(os.getenv("CPU_THREADS", str(_cpu_threads())))  # Optional: PP-StructureV3 runs on CPU

# Cache directory hint (official models download here when first used)
OFFICIAL_DIR = Path("/root/.paddlex/official_models")

app = FastAPI()

@app.on_event("startup")
def load_pipeline():
    # Instantiate PP-StructureV3 pipeline; optional modules controlled by env
    app.state.pps = PPStructureV3(
        lang=OCR_LANG,
        use_doc_orientation_classify=USE_DOC_ORI,
        use_doc_unwarping=USE_UNWARP,
        use_textline_orientation=USE_TEXTLINE_ORI,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        device="cpu"
    )
    print(f"[startup] PP-StructureV3 ready | lang={OCR_LANG} | charts={USE_CHART_RECOGNITION}({CHART_RECOGNITION_MODEL_NAME}) "
          f"| table_wired={USE_E2E_WIRED_TABLE_REC_MODEL}({WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME}) "
          f"| table_wireless={USE_E2E_WIRELESS_TABLE_REC_MODEL}({WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME}) "
          f"| cpu_threads={CPU_THREADS}")

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "pipeline": "PP-StructureV3",
        "lang": OCR_LANG,
        "charts_enabled": USE_CHART_RECOGNITION,
        "chart_model": CHART_RECOGNITION_MODEL_NAME,
        "wired_table_e2e": USE_E2E_WIRED_TABLE_REC_MODEL,
        "wired_table_model": WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        "wireless_table_e2e": USE_E2E_WIRELESS_TABLE_REC_MODEL,
        "wireless_table_model": WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME
    }

def _bytes_to_ndarray(b: bytes):
    with Image.open(io.BytesIO(b)) as im:
        return np.array(im.convert("RGB"))

@app.post("/ocr")
async def ocr_endpoint(files: List[UploadFile] = File(...)):
    # Kept endpoint name for backward compatibility; now returns PP-StructureV3 structured results
    results = []
    for f in files:
        content = await f.read()
        img = _bytes_to_ndarray(content)
        preds = app.state.pps.predict(
            input=img,
            use_chart_recognition=USE_CHART_RECOGNITION,
            use_e2e_wired_table_rec_model=USE_E2E_WIRED_TABLE_REC_MODEL,
            use_e2e_wireless_table_rec_model=USE_E2E_WIRELESS_TABLE_REC_MODEL
        )
        for res in preds:
            item = {}
            # Prefer standardized dict payload when available
            if hasattr(res, "json") and isinstance(res.json, dict):
                item = res.json
            else:
                try:
                    item = res.to_dict()  # Fallback if provided by the SDK
                except Exception:
                    item = {"summary": str(res)}
            item["filename"] = f.filename
            results.append(item)
    return JSONResponse({"results": results})

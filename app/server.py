from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os, tempfile
from paddleocr import PPStructureV3  # PP-StructureV3 pipeline (CPU, lang, mkldnn, threads) [web:23]

# Runtime config (set in Dokploy Environment)
OCR_LANG = os.getenv("OCR_LANG", "en")             # English only [web:23]
CPU_THREADS = int(os.getenv("CPU_THREADS", "2"))   # Start conservative after segfault; can raise to 4 later [web:23]
# Default MKLDNN OFF for stability on current Arm build; can flip on via env if stable
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "false").lower() == "true"  # [web:23]

# Flagship model names so the latest checkpoints are fetched automatically [web:23]
LAYOUT_MODEL_NAME = os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayout_plus-L")
WIRED_TABLE_STRUCT_MODEL_NAME = os.getenv("WIRED_TABLE_STRUCT_MODEL_NAME", "SLANeXt_wired")
TEXT_DET_MODEL_NAME = os.getenv("TEXT_DET_MODEL_NAME", "PP-OCRv5_server_det")
TEXT_REC_MODEL_NAME = os.getenv("TEXT_REC_MODEL_NAME", "PP-OCRv5_server_rec")

pp = PPStructureV3(
    device="cpu",
    enable_mkldnn=ENABLE_MKLDNN,     # default false; enable later if stable [web:23]
    cpu_threads=CPU_THREADS,         # conservative start [web:23]
    lang=OCR_LANG,
    layout_detection_model_name=LAYOUT_MODEL_NAME,
    wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCT_MODEL_NAME,
    text_detection_model_name=TEXT_DET_MODEL_NAME,
    text_recognition_model_name=TEXT_REC_MODEL_NAME,
    # Turn OFF orientation classify to avoid the module that initialized before crash [web:23]
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
    use_doc_unwarping=False,
    use_formula_recognition=False,
    use_chart_recognition=False,
)  # [web:23]

def run_pps_v3(pipeline, input_path: str):
    if callable(pipeline):
        return pipeline(input_path)  # __call__ if present [web:23]
    if hasattr(pipeline, "predict"):
        return pipeline.predict(input_path)  # predict() if provided [web:23]
    if hasattr(pipeline, "process"):
        return pipeline.process(input_path)  # process() if provided [web:23]
    if hasattr(pipeline, "infer"):
        return pipeline.infer(input_path)    # infer() if provided [web:23]
    raise RuntimeError("Unsupported PPStructureV3 API surface")  # [web:23]

app = FastAPI()  # [web:23]

@app.get("/health")
def health():
    return {"status": "ok"}  # [web:23]

@app.post("/parse")
async def parse_doc(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}")  # [web:23]
    fd, tmp_path = tempfile.mkstemp(prefix="ppsv3_", suffix=suffix, dir="/tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())  # [web:23]
        result = run_pps_v3(pp, tmp_path)     # JSON only; no save_path means no files written [web:23]
        return JSONResponse(result)           # [web:23]
    finally:
        try:
            os.unlink(tmp_path)               # Always delete inputs after inference [web:23]
        except FileNotFoundError:
            pass                               # [web:23]

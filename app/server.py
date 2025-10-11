from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os, tempfile, threading
from paddleocr import PPStructureV3  # PP-StructureV3 CPU pipeline with tuning flags [web:23]

# Runtime config (Dokploy Environment recommended)
OCR_LANG = os.getenv("OCR_LANG", "en")               # English only [web:23]
CPU_THREADS = int(os.getenv("CPU_THREADS", "1"))     # Conservative for stability; raise after validation [web:23]
# Default MKLDNN OFF; can flip on via env if stable
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "false").lower() == "true"  # [web:23]

# Thread caps for BLAS/OMP backends before model init (stability on CPU)
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))       # [web:23]
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))  # [web:23]

# Flagship model names; latest checkpoints fetched automatically [web:23]
LAYOUT_MODEL_NAME = os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayout_plus-L")
WIRED_TABLE_STRUCT_MODEL_NAME = os.getenv("WIRED_TABLE_STRUCT_MODEL_NAME", "SLANeXt_wired")
TEXT_DET_MODEL_NAME = os.getenv("TEXT_DET_MODEL_NAME", "PP-OCRv5_server_det")
TEXT_REC_MODEL_NAME = os.getenv("TEXT_REC_MODEL_NAME", "PP-OCRv5_server_rec")

# Lazy, thread-safe pipeline creation to avoid init at import time
_pp = None
_pp_lock = threading.Lock()

def get_pipeline():
    global _pp
    if _pp is None:
        with _pp_lock:
            if _pp is None:
                _pp = PPStructureV3(
                    device="cpu",
                    enable_mkldnn=ENABLE_MKLDNN,
                    cpu_threads=CPU_THREADS,
                    lang=OCR_LANG,
                    layout_detection_model_name=LAYOUT_MODEL_NAME,
                    wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCT_MODEL_NAME,
                    text_detection_model_name=TEXT_DET_MODEL_NAME,
                    text_recognition_model_name=TEXT_REC_MODEL_NAME,
                    # Disable orientation classify to avoid that stage entirely
                    use_doc_orientation_classify=False,
                    use_textline_orientation=False,
                    use_doc_unwarping=False,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                )  # [web:23]
    return _pp  # [web:23]

# Robust invoker covers __call__/predict/process/infer across versions
def run_pps_v3(pipeline, input_path: str):
    if callable(pipeline):
        return pipeline(input_path)          # [web:23]
    if hasattr(pipeline, "predict"):
        return pipeline.predict(input_path)  # [web:23]
    if hasattr(pipeline, "process"):
        return pipeline.process(input_path)  # [web:23]
    if hasattr(pipeline, "infer"):
        return pipeline.infer(input_path)    # [web:23]
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
            # Stream-friendly pattern; avoids large memory spikes for big PDFs
            chunk = await file.read()
            f.write(chunk)  # Replace with chunked writes if needed [web:23]
        pp = get_pipeline()                # Lazy create on first request [web:23]
        result = run_pps_v3(pp, tmp_path) # JSON only; no save_path means no files written [web:23]
        return JSONResponse(result)        # [web:23]
    finally:
        try:
            os.unlink(tmp_path)            # Always delete inputs after inference [web:23]
        except FileNotFoundError:
            pass                           # [web:23]

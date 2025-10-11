from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from contextlib import asynccontextmanager
import os, tempfile, threading
from paddleocr import PPStructureV3  # PP-StructureV3 CPU pipeline (lang, mkldnn, threads, model names) [web:23]

# Runtime config (Dokploy Environment)
OCR_LANG = os.getenv("OCR_LANG", "en")                 # English only [web:23]
CPU_THREADS = int(os.getenv("CPU_THREADS", "1"))       # Conservative for stability; raise after validation [web:23]
# Default MKLDNN OFF; can enable later via env if stable on this ARM build
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "false").lower() == "true"  # [web:23]

# Thread caps for BLAS/OMP backends before model init (stability on CPU)
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))        # [web:23]
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))   # [web:23]

# Flagship model names so latest checkpoints are fetched automatically [web:23]
LAYOUT_MODEL_NAME = os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayout_plus-L")          # [web:23]
WIRED_TABLE_STRUCT_MODEL_NAME = os.getenv("WIRED_TABLE_STRUCT_MODEL_NAME", "SLANeXt_wired")  # [web:23]
TEXT_DET_MODEL_NAME = os.getenv("TEXT_DET_MODEL_NAME", "PP-OCRv5_server_det")      # [web:23]
TEXT_REC_MODEL_NAME = os.getenv("TEXT_REC_MODEL_NAME", "PP-OCRv5_server_rec")      # [web:23]

# Lazy, thread-safe pipeline with startup pre-warm
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
                    # Disable modules that previously triggered initialization near crash sites
                    use_doc_orientation_classify=False,
                    use_textline_orientation=False,
                    use_doc_unwarping=False,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                )  # [web:23]
    return _pp  # [web:23]

# Robust invoker to cover __call__/predict/process/infer across PaddleOCR releases
def run_pps_v3(pipeline, input_path: str):
    if callable(pipeline):
        return pipeline(input_path)          # __call__ if present [web:23]
    if hasattr(pipeline, "predict"):
        return pipeline.predict(input_path)  # predict() if provided [web:23]
    if hasattr(pipeline, "process"):
        return pipeline.process(input_path)  # process() if provided [web:23]
    if hasattr(pipeline, "infer"):
        return pipeline.infer(input_path)    # infer() if provided [web:23]
    raise RuntimeError("Unsupported PPStructureV3 API surface")  # [web:23]

@asynccontextmanager
async def lifespan(app):
    # Pre-warm models at startup so creation/download happens before first request
    _ = get_pipeline()  # [web:23]
    yield  # [web:23]

app = FastAPI(lifespan=lifespan)  # [web:23]

@app.get("/health")
def health():
    return {"status": "ok"}  # [web:23]

@app.post("/parse")
async def parse_doc(file: UploadFile = File(...)):
    # Preserve suffix so the pipeline recognizes pdf/jpg/jpeg/png/bmp [web:23]
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}"
        )  # [web:23]
    fd, tmp_path = tempfile.mkstemp(prefix="ppsv3_", suffix=suffix, dir="/tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            # Stream the upload to avoid memory spikes on large PDFs [web:23]
            while True:
                chunk = await file.read(1 << 20)  # 1 MiB [web:23]
                if not chunk:
                    break  # [web:23]
                f.write(chunk)  # [web:23]
        pp = get_pipeline()                 # Ensure pipeline exists (warm or hot) [web:23]
        result = run_pps_v3(pp, tmp_path)  # JSON only; no save_path means no files written [web:23]
        return JSONResponse(result)        # [web:23]
    finally:
        try:
            os.unlink(tmp_path)            # Always delete inputs after inference [web:23]
        except FileNotFoundError:
            pass                           # [web:23]

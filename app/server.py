# --- Environment setup (must run before Paddle imports) ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("GLOG_minloglevel", "2")

# -----------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
import tempfile, json, os

# --- Basic Config ---
DEVICE = os.getenv("DEVICE", "cpu")
OCR_LANG = os.getenv("OCR_LANG", "en")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}

# --- Lazy-load single pipeline ---
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = PPStructureV3(device=DEVICE, lang=OCR_LANG)
    return _pipeline

# --- FastAPI App ---
app = FastAPI(title="PPStructureV3 Minimal API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(files: list[UploadFile] = File(...)):
    results = []

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        # save to tmp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        size = 0
        while chunk := await f.read(1 << 20):
            tmp.write(chunk)
            size += len(chunk)
            if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(413, f"File too large (>{MAX_FILE_SIZE_MB}MB)")
        tmp.close()

        # predict
        try:
            result = get_pipeline().predict(input=tmp.name)
        except Exception as e:
            raise HTTPException(500, f"OCR failed for {f.filename}: {e}")
        finally:
            os.unlink(tmp.name)

        # convert to JSON-safe
        try:
            result_json = json.loads(json.dumps(result, default=str))
        except Exception:
            result_json = str(result)
        results.append({"filename": f.filename, "result": result_json})

    return JSONResponse({"results": results})

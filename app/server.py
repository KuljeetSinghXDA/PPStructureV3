import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
import tempfile, json

# ---- config ----
DEVICE = os.getenv("DEVICE", "cpu")
OCR_LANG = os.getenv("OCR_LANG", "en")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}

# --- Pin latest lightweight PP-OCRv5 mobile models (detection + recognition) ---
# Model *names* below select the mobile/lightweight variants of PP-OCRv5.
# If you pre-download models into your image, set MODEL_DIR_* to point there.
MODEL_NAME_DET = os.getenv("MODEL_NAME_DET", "PP-OCRv5_mobile_det")
MODEL_NAME_REC = os.getenv("MODEL_NAME_REC", "PP-OCRv5_mobile_rec")
MODEL_DIR_DET = os.getenv("MODEL_DIR_DET", None)   # e.g. "/opt/paddle_models/det"
MODEL_DIR_REC = os.getenv("MODEL_DIR_REC", None)   # e.g. "/opt/paddle_models/rec"

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        # instantiate PPStructureV3 with explicit model names (mobile/lightweight)
        # pass local model_dir arguments if you predownloaded model files.
        _pipeline = PPStructureV3(
            device=DEVICE,
            lang=OCR_LANG,
            # lightweight detection & recognition: PP-OCRv5 mobile variants
            text_detection_model_name=MODEL_NAME_DET,
            text_recognition_model_name=MODEL_NAME_REC,
            # only include model_dir args when you have the files locally
            text_detection_model_dir=MODEL_DIR_DET or None,
            text_recognition_model_dir=MODEL_DIR_REC or None,
            # keep other options default for minimal server
        )
    return _pipeline

app = FastAPI(title="PPStructureV3 Minimal API - PP-OCRv5 mobile (pinned)")

@app.on_event("startup")
async def preload():
    # preload to avoid first-request cold start
    _ = get_pipeline()

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

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        size = 0
        while chunk := await f.read(1 << 20):
            tmp.write(chunk)
            size += len(chunk)
            if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                tmp.close(); os.unlink(tmp.name)
                raise HTTPException(413, f"File too large (>{MAX_FILE_SIZE_MB}MB)")
        tmp.close()

        try:
            result = get_pipeline().predict(input=tmp.name)
        except Exception as e:
            raise HTTPException(500, f"OCR failed for {f.filename}: {e}")
        finally:
            os.unlink(tmp.name)

        try:
            result_json = json.loads(json.dumps(result, default=str))
        except Exception:
            result_json = str(result)
        results.append({"filename": f.filename, "result": result_json})

    return JSONResponse({"results": results})

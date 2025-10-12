import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
import tempfile, json

DEVICE = os.getenv("DEVICE", "cpu")
OCR_LANG = os.getenv("OCR_LANG", "en")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}

# Optional toggles via env if needed later
USE_CHART = os.getenv("USE_CHART", "false").lower() == "true"
USE_FORMULA = os.getenv("USE_FORMULA", "false").lower() == "true"
USE_REGION = os.getenv("USE_REGION", "false").lower() == "true"  # heavy if true

# Pick a tiny recognition model by language
if OCR_LANG == "en":
    # ultra-light English text recognition model
    TEXT_REC_MODEL = "en_PP-OCRv4_mobile_rec"
else:
    # multilingual mobile model (larger than en-only, but lighter than server)
    TEXT_REC_MODEL = "PP-OCRv5_mobile_rec"

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = PPStructureV3(
            device=DEVICE,
            lang=OCR_LANG,

            # Disable heavy extras by default
            use_chart_recognition=USE_CHART,          # PP-Chart2Table ~1.4GB if True
            use_formula_recognition=USE_FORMULA,      # prefer off or use small model if needed
            use_region_detection=USE_REGION,          # PP-DocBlockLayout is heavy

            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,

            # Use lightweight suite
            layout_detection_model_name="PP-DocLayout-S",     # small PicoDet-based layout
            text_detection_model_name="PP-OCRv5_mobile_det",  # mobile text detector
            text_recognition_model_name=TEXT_REC_MODEL,       # tiny English or v5 mobile multi-lang

            # Keep table classification tiny
            table_classification_model_name="PP-LCNet_x1_0_table_cls",

            # Use lightweight table structure models instead of SLANeXt
            wired_table_structure_recognition_model_name="SLANet_plus",
            wireless_table_structure_recognition_model_name="SLANet_plus",

            # Prefer CPU-friendly settings
            enable_mkldnn=True if DEVICE.startswith("cpu") else False,
            cpu_threads=4 if DEVICE.startswith("cpu") else 8,

            # Keep detector input modest to limit memory
            text_det_limit_side_len=960,
            text_det_limit_type="max",
        )
    return _pipeline

app = FastAPI(title="PPStructureV3 Minimal API")

@app.on_event("startup")
async def preload():
    _ = get_pipeline()  # preload lightweight models at startup

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
            # Use end-to-end table structure recognition to avoid RT-DETR cell detectors
            result = get_pipeline().predict(
                input=tmp.name,
                use_e2e_wired_table_rec_model=True,
                use_e2e_wireless_table_rec_model=True,
            )
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

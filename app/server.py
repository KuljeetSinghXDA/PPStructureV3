# server.py

import os
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ppstructurev3")

# ------------------------------------------------------------------------------
# Core configuration
# ------------------------------------------------------------------------------
ENABLE_HPI = False
ENABLE_MKLDNN = True

DEVICE = os.getenv("DEVICE", "cpu")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))

# Optional accuracy boosters (leave as-is unless you want to toggle them)
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles
USE_TABLE_RECOGNITION = False
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Model overrides (adjust if you need different models)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# ------------------------------------------------------------------------------
# Detection/recognition parameters (set to None to defer to library defaults)
# These will only be passed to PPStructureV3 if not None.
# ------------------------------------------------------------------------------
LAYOUT_THRESHOLD: Optional[float] = None
TEXT_DET_THRESH: Optional[float] = None
TEXT_DET_BOX_THRESH: Optional[float] = None
TEXT_DET_UNCLIP_RATIO: Optional[float] = None
TEXT_DET_LIMIT_SIDE_LEN: Optional[int] = None
TEXT_DET_LIMIT_TYPE: Optional[str] = None  # "max" or "min"
TEXT_REC_SCORE_THRESH: Optional[float] = None
TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None

# ------------------------------------------------------------------------------
# Apply CPU thread/env hints
# ------------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
if ENABLE_MKLDNN:
    os.environ.setdefault("FLAGS_use_mkldnn", "1")

# ------------------------------------------------------------------------------
# Lazy import after env prepared
# ------------------------------------------------------------------------------
from paddleocr import PPStructureV3  # noqa: E402

# ------------------------------------------------------------------------------
# FastAPI app with lifespan to initialize pipeline once
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log the configuration that will be used
    logger.info(
        "PPStructureV3 config: device=%s, cpu_threads=%s, mkldnn=%s, "
        "layout_model=%s, text_det_model=%s, text_rec_model=%s",
        DEVICE, CPU_THREADS, ENABLE_MKLDNN,
        LAYOUT_DETECTION_MODEL_NAME, TEXT_DETECTION_MODEL_NAME, TEXT_RECOGNITION_MODEL_NAME,
    )
    logger.info(
        "Params: layout_threshold=%s, text_det_thresh=%s, text_det_box_thresh=%s, "
        "text_det_unclip_ratio=%s, text_det_limit_side_len=%s, text_det_limit_type=%s, "
        "text_rec_score_thresh=%s, text_recognition_batch_size=%s",
        LAYOUT_THRESHOLD, TEXT_DET_THRESH, TEXT_DET_BOX_THRESH,
        TEXT_DET_UNCLIP_RATIO, TEXT_DET_LIMIT_SIDE_LEN, TEXT_DET_LIMIT_TYPE,
        TEXT_REC_SCORE_THRESH, TEXT_RECOGNITION_BATCH_SIZE,
    )

    # Build kwargs for PPStructureV3, only including non-None overrides
    base_kwargs = dict(
        device=DEVICE,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        # The following are conditionally included below
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
    )
    pp_kwargs = {k: v for k, v in base_kwargs.items() if v is not None or k in {
        "device",
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_textline_orientation",
        "use_table_recognition",
        "use_formula_recognition",
        "use_chart_recognition",
        "layout_detection_model_name",
        "text_detection_model_name",
        "text_recognition_model_name",
    }}

    pipeline = PPStructureV3(**pp_kwargs)
    app.state.pipeline = pipeline
    yield
    # No special teardown required


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    cfg = {
        "device": DEVICE,
        "cpu_threads": CPU_THREADS,
        "enable_mkldnn": ENABLE_MKLDNN,
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,
        "layout_threshold": LAYOUT_THRESHOLD,
        "text_det_thresh": TEXT_DET_THRESH,
        "text_det_box_thresh": TEXT_DET_BOX_THRESH,
        "text_det_unclip_ratio": TEXT_DET_UNCLIP_RATIO,
        "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
        "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
        "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
        "text_recognition_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
        "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": USE_DOC_UNWARPING,
        "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        "use_table_recognition": USE_TABLE_RECOGNITION,
        "use_formula_recognition": USE_FORMULA_RECOGNITION,
        "use_chart_recognition": USE_CHART_RECOGNITION,
    }
    return JSONResponse({"status": "ok", "config": cfg})


# ------------------------------------------------------------------------------
# OCR parse endpoint
# ------------------------------------------------------------------------------
@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith(("image/", "application/pdf")):
        raise HTTPException(status_code=400, detail="Only image/* or application/pdf is supported")

    tmpdir = tempfile.mkdtemp(prefix="ppstructv3_")
    try:
        in_path = Path(tmpdir) / file.filename
        with open(in_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        def run_predict():
            return app.state.pipeline.predict(str(in_path))

        results = await run_in_threadpool(run_predict)

        # Try to serialize results gracefully
        serializable = []
        for res in results:
            # Many PaddleOCR result objects have save_to_json/print helpers;
            # here we attempt to access a dict-like representation if available.
            item = {}
            if hasattr(res, "to_dict"):
                item = res.to_dict()
            elif hasattr(res, "as_dict"):
                item = res.as_dict()
            elif hasattr(res, "__dict__"):
                # Fallback best-effort; may contain non-serializable fields
                item = {k: v for k, v in res.__dict__.items() if isinstance(v, (str, int, float, list, dict, bool, type(None)))}
            else:
                item = {"str": str(res)}
            serializable.append(item)

        return JSONResponse({"count": len(serializable), "results": serializable})
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Root
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return PlainTextResponse("PPStructureV3 service is running")

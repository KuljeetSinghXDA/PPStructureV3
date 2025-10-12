# --- Environment must be set before any Paddle/NumPy/OpenBLAS import ---
import os

# Threading caps to avoid nested parallelism and dueling thread pools
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.getenv("OPENBLAS_NUM_THREADS", "1"))

# Default: disable MKLDNN on ARM unless explicitly enabled via env
# (Will be mirrored into the pipeline init below)
os.environ.setdefault("FLAGS_use_mkldnn", "0")

# Optional noise reduction and GC tuning
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_eager_delete_tensor_gb", "0.0")

# -----------------------------------------------------------------------

import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from paddleocr import PPStructureV3  # import after envs are applied


# ================= Core Configuration =================

DEVICE = os.getenv("DEVICE", "cpu")
OCR_LANG = os.getenv("OCR_LANG", "en")

# Start conservative; scale via env after stability
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))

# Allow enabling later from Dokploy; default off for ARM stability
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "false").lower() == "true"
ENABLE_HPI = os.getenv("ENABLE_HPI", "false").lower() == "true"

# Optional modules
USE_DOC_ORIENTATION_CLASSIFY = os.getenv("USE_DOC_ORIENTATION_CLASSIFY", "false").lower() == "true"
USE_DOC_UNWARPING = os.getenv("USE_DOC_UNWARPING", "false").lower() == "true"
USE_TEXTLINE_ORIENTATION = os.getenv("USE_TEXTLINE_ORIENTATION", "false").lower() == "true"
USE_TABLE_RECOGNITION = os.getenv("USE_TABLE_RECOGNITION", "true").lower() == "true"
USE_FORMULA_RECOGNITION = os.getenv("USE_FORMULA_RECOGNITION", "false").lower() == "true"
USE_CHART_RECOGNITION = os.getenv("USE_CHART_RECOGNITION", "false").lower() == "true"

# Model names (None -> auto)
LAYOUT_DETECTION_MODEL_NAME = os.getenv("LAYOUT_DETECTION_MODEL_NAME") or None
TEXT_DETECTION_MODEL_NAME = os.getenv("TEXT_DETECTION_MODEL_NAME") or None
TEXT_RECOGNITION_MODEL_NAME = os.getenv("TEXT_RECOGNITION_MODEL_NAME") or None
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = os.getenv("WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME") or None
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = os.getenv("WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME") or None
TABLE_CLASSIFICATION_MODEL_NAME = os.getenv("TABLE_CLASSIFICATION_MODEL_NAME") or None
FORMULA_RECOGNITION_MODEL_NAME = os.getenv("FORMULA_RECOGNITION_MODEL_NAME") or None
CHART_RECOGNITION_MODEL_NAME = os.getenv("CHART_RECOGNITION_MODEL_NAME") or None

# Thresholds / limits
LAYOUT_THRESHOLD = float(os.getenv("LAYOUT_THRESHOLD", "0.5"))
TEXT_DET_THRESH = float(os.getenv("TEXT_DET_THRESH", "0.3"))
TEXT_DET_BOX_THRESH = float(os.getenv("TEXT_DET_BOX_THRESH", "0.6"))
TEXT_DET_UNCLIP_RATIO = float(os.getenv("TEXT_DET_UNCLIP_RATIO", "2.0"))
TEXT_DET_LIMIT_SIDE_LEN = int(os.getenv("TEXT_DET_LIMIT_SIDE_LEN", "960"))
TEXT_DET_LIMIT_TYPE = os.getenv("TEXT_DET_LIMIT_TYPE", "max")
TEXT_REC_SCORE_THRESH = float(os.getenv("TEXT_REC_SCORE_THRESH", "0.0"))
TEXT_RECOGNITION_BATCH_SIZE = int(os.getenv("TEXT_RECOGNITION_BATCH_SIZE", "1"))

ALLOWED_EXTENSIONS = set(
    ext.strip().lower() for ext in os.getenv("ALLOWED_EXTENSIONS", ".pdf,.jpg,.jpeg,.png,.bmp").split(",")
)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Bounded parallelism for predict on shared pipeline
# 1 == fully serialized; raise to >1 cautiously after stability
MAX_PARALLEL_PREDICT = int(os.getenv("MAX_PARALLEL_PREDICT", "1"))

# ================= Singleton Pipeline + Bounded Concurrency =================

_pp = None
_pp_lock = threading.Lock()
_predict_sem = threading.Semaphore(MAX_PARALLEL_PREDICT)


def get_pipeline():
    global _pp
    if _pp is None:
        with _pp_lock:
            if _pp is None:
                _pp = PPStructureV3(
                    device=DEVICE,
                    enable_mkldnn=ENABLE_MKLDNN,
                    enable_hpi=ENABLE_HPI,
                    cpu_threads=CPU_THREADS,
                    lang=OCR_LANG,
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
    return _pp


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = get_pipeline()
    yield


app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse")
async def parse_endpoint(
    files: List[UploadFile] = File(...),
    output_format: Optional[Literal["json", "markdown"]] = Query(default="json"),
):
    ofmt = (output_format or "json").lower()
    if ofmt not in ("json", "markdown"):
        ofmt = "json"

    results = []
    tmp_paths = []

    try:
        for f in files:
            if not f.filename:
                raise HTTPException(status_code=400, detail="Missing filename")

            suffix = Path(f.filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )

            fd, tmp_path = tempfile.mkstemp(prefix="ppsv3_", suffix=suffix, dir="/tmp")
            tmp_paths.append(tmp_path)
            size = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = await f.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    size += len(chunk)
                    out.write(chunk)
                    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_FILE_SIZE_MB}MB)")
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            pp = get_pipeline()

            def _predict(path: str):
                with _predict_sem:
                    return pp.predict(input=path)

            try:
                result = await run_in_threadpool(_predict, tmp_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR processing failed for {f.filename}: {str(e)}")

            if ofmt == "json":
                outdir = tempfile.mkdtemp(prefix="ppsv3_json_")
                try:
                    if hasattr(result, "save_to_json"):
                        result.save_to_json(save_path=outdir)
                        docs = []
                        for name in sorted(os.listdir(outdir)):
                            if name.endswith(".json"):
                                with open(os.path.join(outdir, name), "r", encoding="utf-8") as fh:
                                    docs.append(json.load(fh))
                        results.append({"filename": f.filename, "documents": docs or [vars(result)]})
                    else:
                        results.append({"filename": f.filename, "documents": [vars(result)]})
                finally:
                    shutil.rmtree(outdir, ignore_errors=True)
            else:
                md_body = None
                if hasattr(result, "markdown"):
                    md_body = result.markdown
                elif hasattr(result, "to_markdown"):
                    md_body = result.to_markdown()
                else:
                    outdir = tempfile.mkdtemp(prefix="ppsv3_md_")
                    try:
                        if hasattr(result, "save_to_markdown"):
                            result.save_to_markdown(save_path=outdir)
                            parts = []
                            for name in sorted(os.listdir(outdir)):
                                if name.endswith(".md"):
                                    with open(os.path.join(outdir, name), "r", encoding="utf-8") as fh:
                                        parts.append(fh.read())
                            md_body = "\n\n".join(parts) if parts else str(result)
                        else:
                            md_body = str(result)
                    finally:
                        shutil.rmtree(outdir, ignore_errors=True)
                results.append({"filename": f.filename, "documents_markdown": [md_body]})

        if ofmt == "json":
            return JSONResponse({"results": results})

        body = "\n\n".join(f"# {item['filename']}\n\n" + "\n\n".join(item["documents_markdown"]) for item in results)
        return PlainTextResponse(body, media_type="text/markdown")

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

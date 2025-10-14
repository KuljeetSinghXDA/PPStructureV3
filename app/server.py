# server.py

import os
import json
import tempfile
import shutil
import logging
import glob
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ppstructurev3")

# ---------------- Core configuration ----------------
ENABLE_HPI = False
ENABLE_MKLDNN = True

DEVICE = os.getenv("DEVICE", "cpu")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))

USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

USE_TABLE_RECOGNITION = False
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False

# Models (adjust if you like; leaving as-is is fine)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# ---------------- Params set to None to use library defaults ----------------
LAYOUT_THRESHOLD: Optional[float] = None
TEXT_DET_THRESH: Optional[float] = None
TEXT_DET_BOX_THRESH: Optional[float] = None
TEXT_DET_UNCLIP_RATIO: Optional[float] = None
TEXT_DET_LIMIT_SIDE_LEN: Optional[int] = None
TEXT_DET_LIMIT_TYPE: Optional[str] = None
TEXT_REC_SCORE_THRESH: Optional[float] = None
TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None

# Threading hints
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
if ENABLE_MKLDNN:
    os.environ.setdefault("FLAGS_use_mkldnn", "1")

from paddleocr import PPStructureV3  # noqa: E402

def _safe_get(obj, names):
    for n in names:
        try:
            v = getattr(obj, n)
            return v
        except Exception:
            continue
    return None

def _extract_effective_from_modules(pipeline):
    """
    Try multiple known attribute patterns across PaddleOCR v3 pipeline modules.
    Falls back to parsing YAMLs found in downloaded model dirs.
    """
    effective = {
        "layout_threshold": None,
        "text_det_thresh": None,
        "text_det_box_thresh": None,
        "text_det_unclip_ratio": None,
        "text_det_limit_side_len": None,
        "text_det_limit_type": None,
        "text_rec_score_thresh": None,
        "text_recognition_batch_size": None,
    }

    # Direct attributes the wrapper might expose
    for k in list(effective.keys()):
        effective[k] = getattr(pipeline, k, None)

    # Heuristic: find detection/recognition submodules and their post-processors
    det_mod = _safe_get(pipeline, ["text_detector", "_text_detector", "detector", "ocr_det", "det"])
    rec_mod = _safe_get(pipeline, ["text_recognizer", "_text_recognizer", "recognizer", "ocr_rec", "rec"])

    # Detection post-process (DBPostProcess typically exposes thresh/box_thresh/unclip_ratio)
    det_pp = None
    if det_mod is not None:
        det_pp = _safe_get(det_mod, ["postprocess_op", "_postprocess_op", "post_process", "postprocess"])
        # Preprocess params such as limit_side_len/limit_type may be on the model or transforms
        # Try common attribute names:
        effective["text_det_limit_side_len"] = effective["text_det_limit_side_len"] or _safe_get(det_mod, ["limit_side_len", "det_limit_side_len"])
        effective["text_det_limit_type"] = effective["text_det_limit_type"] or _safe_get(det_mod, ["limit_type", "det_limit_type"])

    if det_pp is not None:
        # Common keys used by DBPostProcess
        effective["text_det_thresh"] = effective["text_det_thresh"] or getattr(det_pp, "thresh", None)
        effective["text_det_box_thresh"] = effective["text_det_box_thresh"] or getattr(det_pp, "box_thresh", None)
        effective["text_det_unclip_ratio"] = effective["text_det_unclip_ratio"] or getattr(det_pp, "unclip_ratio", None)

    # Recognition batch size (often rec_batch_num or batch_size)
    if rec_mod is not None:
        effective["text_recognition_batch_size"] = effective["text_recognition_batch_size"] or _safe_get(rec_mod, ["rec_batch_num", "batch_size"])
        # Some wrappers keep drop_score on post-process
        rec_pp = _safe_get(rec_mod, ["postprocess_op", "_postprocess_op", "post_process", "postprocess"])
        if rec_pp is not None:
            effective["text_rec_score_thresh"] = effective["text_rec_score_thresh"] or _safe_get(rec_pp, ["drop_score", "score_thresh"])

    # Layout threshold sometimes lives on the layout head or filter
    layout_mod = _safe_get(pipeline, ["layout_detector", "_layout_detector", "layout", "doc_layout"])
    if layout_mod is not None:
        effective["layout_threshold"] = effective["layout_threshold"] or _safe_get(layout_mod, ["score_thresh", "layout_threshold", "conf_threshold"])

    # If still missing, try YAMLs from known cache roots
    def read_yaml_defaults():
        try:
            import yaml  # pyyaml is a dependency of PaddleOCR in most installs
        except Exception:
            return

        # Candidate roots
        roots = []
        # PaddleX official models dir (common for PP-StructureV3 downloads)
        roots.append(Path.home() / ".paddlex" / "official_models")
        # PaddleOCR cache dir (depending on versions)
        roots.append(Path.home() / ".paddleocr")
        # General cache
        roots.append(Path(os.getenv("PPOCR_CACHE_DIR", Path.home() / ".cache")))

        # Scan for yaml files that likely belong to det/rec/layout
        yamls = []
        for r in roots:
            if r.exists():
                yamls.extend(Path(r).rglob("*.yml"))
                yamls.extend(Path(r).rglob("*.yaml"))

        def pick(keys, dic):
            for k in keys:
                if isinstance(dic, dict) and k in dic:
                    return dic[k]
            return None

        for y in yamls:
            try:
                data = yaml.safe_load(open(y, "r", encoding="utf-8"))
            except Exception:
                continue

            # DBPostProcess style
            post = pick(["PostProcess", "postprocess", "post_process"], data) or data
            if isinstance(post, dict):
                db = pick(["DBPostProcess", "DB", "DBPostprocess"], post) or post
                if isinstance(db, dict):
                    effective["text_det_thresh"] = effective["text_det_thresh"] or pick(["thresh", "det_db_thresh"], db)
                    effective["text_det_box_thresh"] = effective["text_det_box_thresh"] or pick(["box_thresh", "det_db_box_thresh"], db)
                    effective["text_det_unclip_ratio"] = effective["text_det_unclip_ratio"] or pick(["unclip_ratio", "det_db_unclip_ratio"], db)

            # PreProcess for limit_side_len/type
            pre = pick(["PreProcess", "preprocess", "Transforms", "transforms"], data)
            if isinstance(pre, list):
                for step in pre:
                    if isinstance(step, dict):
                        if any(k in step for k in ["DetResizeForTest", "DBResize", "Resize"]):
                            vals = list(step.values())[0]
                            if isinstance(vals, dict):
                                effective["text_det_limit_side_len"] = effective["text_det_limit_side_len"] or pick(["limit_side_len", "short_size", "max_size"], vals)
                                effective["text_det_limit_type"] = effective["text_det_limit_type"] or pick(["limit_type"], vals)

            # Rec drop_score
            rec_post = post
            if isinstance(rec_post, dict):
                crnn = pick(["CTCLabelDecode", "SARLabelDecode", "NRTRLabelDecode", "RECPostProcess"], rec_post) or rec_post
                if isinstance(crnn, dict):
                    effective["text_rec_score_thresh"] = effective["text_rec_score_thresh"] or pick(["drop_score", "score_thresh"], crnn)

    read_yaml_defaults()
    return effective

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "PPStructureV3 config: device=%s, cpu_threads=%s, mkldnn=%s, layout_model=%s, text_det_model=%s, text_rec_model=%s",
        DEVICE, CPU_THREADS, ENABLE_MKLDNN,
        LAYOUT_DETECTION_MODEL_NAME, TEXT_DETECTION_MODEL_NAME, TEXT_RECOGNITION_MODEL_NAME,
    )
    logger.info(
        "Params: layout_threshold=%s, text_det_thresh=%s, text_det_box_thresh=%s, text_det_unclip_ratio=%s, "
        "text_det_limit_side_len=%s, text_det_limit_type=%s, text_rec_score_thresh=%s, text_recognition_batch_size=%s",
        LAYOUT_THRESHOLD, TEXT_DET_THRESH, TEXT_DET_BOX_THRESH, TEXT_DET_UNCLIP_RATIO,
        TEXT_DET_LIMIT_SIDE_LEN, TEXT_DET_LIMIT_TYPE, TEXT_REC_SCORE_THRESH, TEXT_RECOGNITION_BATCH_SIZE,
    )

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

        # Conditional overrides (None means "do not pass", so library defaults apply)
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
    )
    keep = {
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
    }
    pp_kwargs = {k: v for k, v in base_kwargs.items() if (v is not None) or (k in keep)}

    pipeline = PPStructureV3(**pp_kwargs)
    app.state.pipeline = pipeline

    # Log effective values resolved from the created modules/YAMLs
    effective = _extract_effective_from_modules(pipeline)
    logger.info(
        "Effective params resolved: layout_threshold=%s, text_det_thresh=%s, text_det_box_thresh=%s, "
        "text_det_unclip_ratio=%s, text_det_limit_side_len=%s, text_det_limit_type=%s, "
        "text_rec_score_thresh=%s, text_recognition_batch_size=%s",
        effective.get("layout_threshold"),
        effective.get("text_det_thresh"),
        effective.get("text_det_box_thresh"),
        effective.get("text_det_unclip_ratio"),
        effective.get("text_det_limit_side_len"),
        effective.get("text_det_limit_type"),
        effective.get("text_rec_score_thresh"),
        effective.get("text_recognition_batch_size"),
    )

    app.state.effective = effective
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    # Report both requested (possibly None) and effective values
    cfg = {
        "device": DEVICE,
        "cpu_threads": CPU_THREADS,
        "enable_mkldnn": ENABLE_MKLDNN,
        "layout_detection_model_name": LAYOUT_DETECTION_MODEL_NAME,
        "text_detection_model_name": TEXT_DETECTION_MODEL_NAME,
        "text_recognition_model_name": TEXT_RECOGNITION_MODEL_NAME,
        "requested": {
            "layout_threshold": LAYOUT_THRESHOLD,
            "text_det_thresh": TEXT_DET_THRESH,
            "text_det_box_thresh": TEXT_DET_BOX_THRESH,
            "text_det_unclip_ratio": TEXT_DET_UNCLIP_RATIO,
            "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
            "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
            "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
            "text_recognition_batch_size": TEXT_RECOGNITION_BATCH_SIZE,
        },
        "effective": getattr(app.state, "effective", {}),
    }
    return JSONResponse({"status": "ok", "config": cfg})

@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith(("image/", "application/pdf")):
        raise HTTPException(status_code=400, detail="Only image/* or application/pdf is supported")

    tmpdir = tempfile.mkdtemp(prefix="ppstructv3_")
    try:
        in_path = Path(tmpdir) / (file.filename or "input.bin")
        with open(in_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        def run_predict():
            return app.state.pipeline.predict(str(in_path))

        results = await run_in_threadpool(run_predict)

        serializable = []
        for res in results:
            if hasattr(res, "to_dict"):
                serializable.append(res.to_dict())
            elif hasattr(res, "as_dict"):
                serializable.append(res.as_dict())
            elif hasattr(res, "__dict__"):
                serializable.append({k: v for k, v in res.__dict__.items() if isinstance(v, (str, int, float, list, dict, bool, type(None)))})
            else:
                serializable.append({"str": str(res)})

        return JSONResponse({"count": len(serializable), "results": serializable})
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.get("/")
def root():
    return PlainTextResponse("PPStructureV3 service is running")

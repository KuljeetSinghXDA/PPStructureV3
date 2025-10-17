import os
import io
import uuid
import base64
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from PIL import Image

from paddleocr import PPStructureV3

# =========================
# Defaults tuned for lab reports (documented parameters only)
# =========================
DEVICE = os.getenv("PPOCR_DEVICE", "cpu")
LANG = os.getenv("PPOCR_LANG", "en")

USE_DOC_ORIENTATION_CLASSIFY = True
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = True
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False
USE_REGION_DETECTION = False

# Layout/text settings (official parameters)
LAYOUT_THRESHOLD = None
LAYOUT_NMS = None
LAYOUT_UNCLIP_RATIO = None
LAYOUT_MERGE_BBOXES_MODE = None

TEXT_DET_LIMIT_SIDE_LEN = 1536
TEXT_DET_LIMIT_TYPE = None
TEXT_DET_THRESH = 0.30
TEXT_DET_BOX_THRESH = 0.40
TEXT_DET_UNCLIP_RATIO = 2.0

TEXT_REC_SCORE_THRESH = 0.60
TEXT_RECOGNITION_BATCH_SIZE = 8
TEXTLINE_ORIENTATION_BATCH_SIZE = 8

# Table sub-pipeline names (documented)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"

# Optional name/dir fields are left None to use official defaults/auto-download
REGION_DETECTION_MODEL_NAME = None
TEXT_DETECTION_MODEL_NAME = None
TEXT_RECOGNITION_MODEL_NAME = None
WIRED_TABLE_CELLS_DET_MODEL_NAME = None
WIRELESS_TABLE_CELLS_DET_MODEL_NAME = None
TABLE_ORIENTATION_CLASSIFY_MODEL_NAME = None
FORMULA_RECOGNITION_MODEL_NAME = None
SEAL_TEXT_DETECTION_MODEL_NAME = None
SEAL_TEXT_RECOGNITION_MODEL_NAME = None
CHART_RECOGNITION_MODEL_NAME = None
DOC_ORIENTATION_CLASSIFY_MODEL_NAME = None
DOC_UNWARPING_MODEL_NAME = None
TEXTLINE_ORIENTATION_MODEL_NAME = None

LAYOUT_DETECTION_MODEL_DIR = None
REGION_DETECTION_MODEL_DIR = None
TEXT_DETECTION_MODEL_DIR = None
TEXT_RECOGNITION_MODEL_DIR = None
TABLE_CLASSIFICATION_MODEL_DIR = None
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = None
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR = None
WIRED_TABLE_CELLS_DET_MODEL_DIR = None
WIRELESS_TABLE_CELLS_DET_MODEL_DIR = None
TABLE_ORIENTATION_CLASSIFY_MODEL_DIR = None
FORMULA_RECOGNITION_MODEL_DIR = None
DOC_ORIENTATION_CLASSIFY_MODEL_DIR = None
DOC_UNWARPING_MODEL_DIR = None
TEXTLINE_ORIENTATION_MODEL_DIR = None
SEAL_TEXT_DETECTION_MODEL_DIR = None
SEAL_TEXT_RECOGNITION_MODEL_DIR = None
CHART_RECOGNITION_MODEL_DIR = None

# =========================
# Request/Response models (official schema)
# =========================
class LayoutParsingRequest(BaseModel):
    # Required
    file: str = Field(..., description="URL of an image/PDF or Base64-encoded content")
    # Optional, 0=PDF, 1=Image; if None, server can infer (we try to infer; if ambiguous we require fileType)
    fileType: Optional[int] = Field(None, description="0=PDF, 1=Image")

    # Predict-time toggles (camelCase per official docs)
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useTextlineOrientation: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useTableRecognition: Optional[bool] = None
    useFormulaRecognition: Optional[bool] = None
    useChartRecognition: Optional[bool] = None
    useRegionDetection: Optional[bool] = None

    # Thresholds and sizes
    layoutThreshold: Optional[Union[float, Dict[str, float]]] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[Union[float, List[float], Dict[str, float]]] = None
    layoutMergeBboxesMode: Optional[Union[str, Dict[str, str]]] = None
    textDetLimitSideLen: Optional[int] = None
    textDetLimitType: Optional[str] = None
    textDetThresh: Optional[float] = None
    textDetBoxThresh: Optional[float] = None
    textDetUnclipRatio: Optional[float] = None
    textRecScoreThresh: Optional[float] = None

    # Seal thresholds
    sealDetLimitSideLen: Optional[int] = None
    sealDetLimitType: Optional[str] = None
    sealDetThresh: Optional[float] = None
    sealDetBoxThresh: Optional[float] = None
    sealDetUnclipRatio: Optional[float] = None
    sealRecScoreThresh: Optional[float] = None

    # Table predict behavior (documented)
    useWiredTableCellsTransToHtml: Optional[bool] = None
    useWirelessTableCellsTransToHtml: Optional[bool] = None
    useTableOrientationClassify: Optional[bool] = None
    useOcrResultsWithTableCells: Optional[bool] = None
    useE2eWiredTableRecModel: Optional[bool] = None
    useE2eWirelessTableRecModel: Optional[bool] = None

    # Whether to include output images (see docs)
    visualize: Optional[bool] = None

    @validator("fileType")
    def _check_filetype(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("fileType must be 0 (PDF) or 1 (Image)")
        return v


# =========================
# Helpers
# =========================
def _is_probably_base64(s: str) -> bool:
    # Heuristic: data URI or long chunk of base64 characters
    if s.startswith("data:"):
        return True
    if len(s) > 128 and all(c.isalnum() or c in "+/=\n\r" for c in s[:256]):
        return True
    return False

def _save_base64_to_tmp(b64: str, suffix: str) -> str:
    if b64.startswith("data:"):
        # data:image/png;base64,....
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64, validate=False)
    td = tempfile.mkdtemp(prefix="ppsv3_")
    fp = os.path.join(td, f"upload{suffix}")
    with open(fp, "wb") as f:
        f.write(raw)
    return fp

def _pil_to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    # Use JPEG for outputImages as per docs; for markdown images, PNG/JPEG both acceptable
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _prune_input_path_and_page_index(d: Any) -> Any:
    # Remove 'input_path' and 'page_index' keys anywhere in nested dicts
    if isinstance(d, dict):
        return {
            k: _prune_input_path_and_page_index(v)
            for k, v in d.items()
            if k not in ("input_path", "page_index")
        }
    if isinstance(d, list):
        return [_prune_input_path_and_page_index(v) for v in d]
    return d

def _build_markdown_obj(md: Dict[str, Any]) -> Dict[str, Any]:
    # Official Result.markdown fields: text, images, isStart, isEnd
    # PPStructureV3.markdown returns dict with keys markdown_texts, markdown_images, page_continuation_flags (per docs)
    text = md["markdown_texts"]  # strict (no fallbacks)
    imgs: Dict[str, Image.Image] = md["markdown_images"]
    flags: Tuple[bool, bool] = md["page_continuation_flags"]
    images_b64: Dict[str, str] = {k: _pil_to_base64_png(v) for k, v in imgs.items()}
    return {
        "text": text,
        "images": images_b64,
        "isStart": bool(flags[0]),
        "isEnd": bool(flags[1]),
    }

# =========================
# App lifecycle
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a single pipeline instance with documented parameters only
    app.state.pipeline = PPStructureV3(
        # language and device
        lang=LANG,
        device=DEVICE,

        # sub-pipelines
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
        use_seal_recognition=USE_SEAL_RECOGNITION,
        use_region_detection=USE_REGION_DETECTION,

        # models (names/dirs)
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        layout_detection_model_dir=LAYOUT_DETECTION_MODEL_DIR,
        region_detection_model_name=REGION_DETECTION_MODEL_NAME,
        region_detection_model_dir=REGION_DETECTION_MODEL_DIR,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_detection_model_dir=TEXT_DETECTION_MODEL_DIR,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        text_recognition_model_dir=TEXT_RECOGNITION_MODEL_DIR,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        table_classification_model_dir=TABLE_CLASSIFICATION_MODEL_DIR,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_dir=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_dir=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_DIR,
        wired_table_cells_detection_model_name=WIRED_TABLE_CELLS_DET_MODEL_NAME,
        wired_table_cells_detection_model_dir=WIRED_TABLE_CELLS_DET_MODEL_DIR,
        wireless_table_cells_detection_model_name=WIRELESS_TABLE_CELLS_DET_MODEL_NAME,
        wireless_table_cells_detection_model_dir=WIRELESS_TABLE_CELLS_DET_MODEL_DIR,
        table_orientation_classify_model_name=TABLE_ORIENTATION_CLASSIFY_MODEL_NAME,
        table_orientation_classify_model_dir=TABLE_ORIENTATION_CLASSIFY_MODEL_DIR,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        formula_recognition_model_dir=FORMULA_RECOGNITION_MODEL_DIR,
        doc_orientation_classify_model_name=DOC_ORIENTATION_CLASSIFY_MODEL_NAME,
        doc_orientation_classify_model_dir=DOC_ORIENTATION_CLASSIFY_MODEL_DIR,
        doc_unwarping_model_name=DOC_UNWARPING_MODEL_NAME,
        doc_unwarping_model_dir=DOC_UNWARPING_MODEL_DIR,
        textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL_NAME,
        textline_orientation_model_dir=TEXTLINE_ORIENTATION_MODEL_DIR,
        seal_text_detection_model_name=SEAL_TEXT_DETECTION_MODEL_NAME,
        seal_text_detection_model_dir=SEAL_TEXT_DETECTION_MODEL_DIR,
        seal_text_recognition_model_name=SEAL_TEXT_RECOGNITION_MODEL_NAME,
        seal_text_recognition_model_dir=SEAL_TEXT_RECOGNITION_MODEL_DIR,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        chart_recognition_model_dir=CHART_RECOGNITION_MODEL_DIR,

        # thresholds/batches
        layout_threshold=LAYOUT_THRESHOLD,
        layout_nms=LAYOUT_NMS,
        layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
        layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        textline_orientation_batch_size=TEXTLINE_ORIENTATION_BATCH_SIZE,
    )
    yield


app = FastAPI(title="PP-StructureV3 Service (Official Contract)", version="3.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Official endpoint
# =========================
@app.post("/layout-parsing")
def layout_parsing(req: LayoutParsingRequest):
    """
    Officially documented API:
    - POST /layout-parsing
    - JSON body: {file, fileType, ...predict-time overrides}
    - Response: {logId, errorCode, errorMsg, result: {layoutParsingResults, dataInfo}}
    """
    log_id = str(uuid.uuid4())

    # Materialize input
    tmp_dir = None
    input_arg: Union[str, Path]

    try:
        if _is_probably_base64(req.file):
            # Require fileType to avoid ambiguity; if missing, default based on magic
            suffix = ".pdf" if req.fileType == 0 else ".png"
            tmp_path = _save_base64_to_tmp(req.file, suffix)
            tmp_dir = str(Path(tmp_path).parent)
            input_arg = tmp_path
        else:
            # URL or local path
            input_arg = req.file

        # Map camelCase request keys to predict kwargs (snake_case)
        predict_kwargs: Dict[str, Any] = {}

        def set_if(name_in_req: str, name_in_pred: str):
            v = getattr(req, name_in_req)
            if v is not None:
                predict_kwargs[name_in_pred] = v

        # Toggles
        set_if("useDocOrientationClassify", "use_doc_orientation_classify")
        set_if("useDocUnwarping", "use_doc_unwarping")
        set_if("useTextlineOrientation", "use_textline_orientation")
        set_if("useSealRecognition", "use_seal_recognition")
        set_if("useTableRecognition", "use_table_recognition")
        set_if("useFormulaRecognition", "use_formula_recognition")
        set_if("useChartRecognition", "use_chart_recognition")
        set_if("useRegionDetection", "use_region_detection")

        # Thresholds / sizes
        set_if("layoutThreshold", "layout_threshold")
        set_if("layoutNms", "layout_nms")
        set_if("layoutUnclipRatio", "layout_unclip_ratio")
        set_if("layoutMergeBboxesMode", "layout_merge_bboxes_mode")
        set_if("textDetLimitSideLen", "text_det_limit_side_len")
        set_if("textDetLimitType", "text_det_limit_type")
        set_if("textDetThresh", "text_det_thresh")
        set_if("textDetBoxThresh", "text_det_box_thresh")
        set_if("textDetUnclipRatio", "text_det_unclip_ratio")
        set_if("textRecScoreThresh", "text_rec_score_thresh")

        # Seal thresholds
        set_if("sealDetLimitSideLen", "seal_det_limit_side_len")
        set_if("sealDetLimitType", "seal_det_limit_type")
        set_if("sealDetThresh", "seal_det_thresh")
        set_if("sealDetBoxThresh", "seal_det_box_thresh")
        set_if("sealDetUnclipRatio", "seal_det_unclip_ratio")
        set_if("sealRecScoreThresh", "seal_rec_score_thresh")

        # Table behavior (predict-time)
        set_if("useWiredTableCellsTransToHtml", "use_wired_table_cells_trans_to_html")
        set_if("useWirelessTableCellsTransToHtml", "use_wireless_table_cells_trans_to_html")
        set_if("useTableOrientationClassify", "use_table_orientation_classify")
        set_if("useOcrResultsWithTableCells", "use_ocr_results_with_table_cells")
        set_if("useE2eWiredTableRecModel", "use_e2e_wired_table_rec_model")
        set_if("useE2eWirelessTableRecModel", "use_e2e_wireless_table_rec_model")

        # Run inference
        pipeline: PPStructureV3 = app.state.pipeline
        outputs = pipeline.predict(input=input_arg, **predict_kwargs)

        # Build response
        results: List[Dict[str, Any]] = []
        for res in outputs:
            # Strictly use documented attributes
            res_json = res.json
            if not isinstance(res_json, dict) or "res" not in res_json:
                raise HTTPException(status_code=500, detail="Unexpected result structure from pipeline")

            pruned = _prune_input_path_and_page_index(res_json["res"])

            md = res.markdown  # dict with markdown_texts, markdown_images, page_continuation_flags
            if not isinstance(md, dict):
                raise HTTPException(status_code=500, detail="Missing markdown in result")

            markdown_obj = _build_markdown_obj(md)

            output_images_obj: Optional[Dict[str, str]] = None
            if req.visualize is True or req.visualize is None:
                # Return images by default if visualize is omitted (per docs behavior)
                imgs = res.img  # dict of PIL images
                if isinstance(imgs, dict) and imgs:
                    output_images_obj = {k: _pil_to_base64_jpeg(v) for k, v in imgs.items()}

            results.append(
                {
                    "prunedResult": pruned,
                    "markdown": markdown_obj,
                    "outputImages": output_images_obj,
                }
            )

        response_payload = {
            "logId": log_id,
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": results,
                "dataInfo": {
                    "fileType": req.fileType,
                    "input": "base64" if _is_probably_base64(req.file) else req.file,
                },
            },
        }
        return JSONResponse(response_payload)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {
                "logId": log_id,
                "errorCode": 500,
                "errorMsg": f"Inference failed: {type(e).__name__}: {str(e)}",
            },
            status_code=500,
        )
    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

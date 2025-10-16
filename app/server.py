import os
import io
import tempfile
import threading
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ================= Core Configuration (Pinned Values) =================
DEVICE = "cpu"
CPU_THREADS = 4

# Optional accuracy boosters
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Subpipeline toggles
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = False
USE_CHART_RECOGNITION = False
USE_SEAL_RECOGNITION = False  # newly surfaced toggle

# Model overrides
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters
LAYOUT_THRESHOLD: Optional[Union[float, Dict[int, float]]] = None
TEXT_DET_THRESH: Optional[float] = None
TEXT_DET_BOX_THRESH: Optional[float] = None
TEXT_DET_UNCLIP_RATIO: Optional[float] = None
TEXT_DET_LIMIT_SIDE_LEN: Optional[int] = None
TEXT_DET_LIMIT_TYPE: Optional[str] = None
TEXT_REC_SCORE_THRESH: Optional[float] = None
TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= Utilities =================
def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _human_limit_mb() -> str:
    return f"{MAX_FILE_SIZE_MB} MB"

def _parse_float_or_json(val: Optional[str]) -> Optional[Union[float, Dict[int, Any], Tuple[float, float]]]:
    """Accept a number like '0.5', a list/tuple like '[1.2,2.0]' or '1.2,2.0', or a dict like '{"0":0.5}'."""
    if val is None:
        return None
    s = val.strip()
    # try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list) and len(parsed) == 2 and all(isinstance(x, (int, float)) for x in parsed):
            return (float(parsed[0]), float(parsed[1]))
        if isinstance(parsed, dict):
            # JSON keys will be strings; convert numeric keys to int when possible
            out: Dict[int, Any] = {}
            for k, v in parsed.items():
                try:
                    ik = int(k)
                except Exception:
                    continue
                out[ik] = v
            return out
        if isinstance(parsed, (int, float)):
            return float(parsed)
    except Exception:
        pass
    # try simple "a,b" tuple
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except Exception:
                pass
    # fallback float
    try:
        return float(s)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid JSON/number tuple: {val}")

def _stream_copy_enforcing_limit(src_file: io.BufferedReader, dst_path: Path, max_mb: int) -> None:
    max_bytes = max_mb * 1024 * 1024
    total = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as out:
        while True:
            chunk = src_file.read(1024 * 1024)  # 1 MiB
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large. Max allowed is {_human_limit_mb()}.")
            out.write(chunk)

def _build_predict_kwargs(
    *,
    use_doc_orientation_classify: Optional[bool],
    use_doc_unwarping: Optional[bool],
    use_textline_orientation: Optional[bool],
    use_seal_recognition: Optional[bool],
    use_table_recognition: Optional[bool],
    use_formula_recognition: Optional[bool],
    use_chart_recognition: Optional[bool],
    use_region_detection: Optional[bool],
    layout_threshold: Optional[Union[float, Dict[int, float]]],
    layout_nms: Optional[bool],
    layout_unclip_ratio: Optional[Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]]],
    layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]],
    text_det_limit_side_len: Optional[int],
    text_det_limit_type: Optional[Literal["min", "max"]],
    text_det_thresh: Optional[float],
    text_det_box_thresh: Optional[float],
    text_det_unclip_ratio: Optional[float],
    text_rec_score_thresh: Optional[float],
    seal_det_limit_side_len: Optional[int],
    seal_det_limit_type: Optional[Literal["min", "max"]],
    seal_det_thresh: Optional[float],
    seal_det_box_thresh: Optional[float],
    seal_det_unclip_ratio: Optional[float],
    seal_rec_score_thresh: Optional[float],
    use_wired_table_cells_trans_to_html: Optional[bool],
    use_wireless_table_cells_trans_to_html: Optional[bool],
    use_table_orientation_classify: Optional[bool],
    use_ocr_results_with_table_cells: Optional[bool],
    use_e2e_wired_table_rec_model: Optional[bool],
    use_e2e_wireless_table_rec_model: Optional[bool],
) -> Dict[str, Any]:
    # Only include keys that are not None to let predict() override constructor values selectively.
    kv = dict(
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        use_seal_recognition=use_seal_recognition,
        use_table_recognition=use_table_recognition,
        use_formula_recognition=use_formula_recognition,
        use_chart_recognition=use_chart_recognition,
        use_region_detection=use_region_detection,
        layout_threshold=layout_threshold,
        layout_nms=layout_nms,
        layout_unclip_ratio=layout_unclip_ratio,
        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
        text_det_limit_side_len=text_det_limit_side_len,
        text_det_limit_type=text_det_limit_type,
        text_det_thresh=text_det_thresh,
        text_det_box_thresh=text_det_box_thresh,
        text_det_unclip_ratio=text_det_unclip_ratio,
        text_rec_score_thresh=text_rec_score_thresh,
        seal_det_limit_side_len=seal_det_limit_side_len,
        seal_det_limit_type=seal_det_limit_type,
        seal_det_thresh=seal_det_thresh,
        seal_det_box_thresh=seal_det_box_thresh,
        seal_det_unclip_ratio=seal_det_unclip_ratio,
        seal_rec_score_thresh=seal_rec_score_thresh,
        use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
        use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
        use_table_orientation_classify=use_table_orientation_classify,
        use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
        use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
        use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
    )
    return {k: v for k, v in kv.items() if v is not None}

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Constructor parameters (latest doc) include all model names/dirs, thresholds, device and perf knobs.
    # This instance is kept hot and reused across requests; per-request behavior is controlled via predict() overrides.  (Docs: init + predict params)
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
        # Model names
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,
        # Default thresholds / batch sizes
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
        # Subpipeline defaults
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        use_table_recognition=USE_TABLE_RECOGNITION,
        use_formula_recognition=USE_FORMULA_RECOGNITION,
        use_chart_recognition=USE_CHART_RECOGNITION,
        # You can also add: use_region_detection (default True), use_seal_recognition (False)
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= /parse Endpoint =================
@app.post(
    "/parse",
    response_class=JSONResponse,
    summary="Parse PDF or image with PP-StructureV3 and return structured results",
)
async def parse(
    file: UploadFile = File(..., description="PDF or image."),
    # Output control
    return_format: Literal["json", "markdown"] = Query("json", description="json: per-page JSON. markdown: concatenated markdown text (PDF/image)."),
    # Subpipeline toggles (predict overrides)
    use_doc_orientation_classify: Optional[bool] = Query(None, description="Enable document orientation classification."),
    use_doc_unwarping: Optional[bool] = Query(None, description="Enable document unwarping."),
    use_textline_orientation: Optional[bool] = Query(None, description="Enable text line orientation classification."),
    use_seal_recognition: Optional[bool] = Query(None, description="Enable seal text recognition subpipeline."),
    use_table_recognition: Optional[bool] = Query(None, description="Enable table recognition subpipeline."),
    use_formula_recognition: Optional[bool] = Query(None, description="Enable formula recognition subpipeline."),
    use_chart_recognition: Optional[bool] = Query(None, description="Enable chart parsing module."),
    use_region_detection: Optional[bool] = Query(None, description="Enable region detection helper."),
    # Layout overrides
    layout_threshold_q: Optional[str] = Query(None, alias="layout_threshold", description="float or JSON dict like {\"0\":0.4}."),
    layout_nms: Optional[bool] = Query(None),
    layout_unclip_ratio_q: Optional[str] = Query(None, alias="layout_unclip_ratio", description="float, [w,h], or JSON dict like {\"0\":[1.1,2.0]}."),
    layout_merge_bboxes_mode_q: Optional[str] = Query(None, alias="layout_merge_bboxes_mode", description="\"large\"|\"small\"|\"union\" or JSON dict per class."),
    # OCR overrides
    text_det_limit_side_len: Optional[int] = Query(None, ge=1),
    text_det_limit_type: Optional[Literal["min", "max"]] = Query(None),
    text_det_thresh: Optional[float] = Query(None, gt=0),
    text_det_box_thresh: Optional[float] = Query(None, gt=0),
    text_det_unclip_ratio: Optional[float] = Query(None, gt=0),
    text_rec_score_thresh: Optional[float] = Query(None, ge=0),
    # Seal detection/recog overrides
    seal_det_limit_side_len: Optional[int] = Query(None, ge=1),
    seal_det_limit_type: Optional[Literal["min", "max"]] = Query(None),
    seal_det_thresh: Optional[float] = Query(None, gt=0),
    seal_det_box_thresh: Optional[float] = Query(None, gt=0),
    seal_det_unclip_ratio: Optional[float] = Query(None, gt=0),
    seal_rec_score_thresh: Optional[float] = Query(None, ge=0),
    # Table behavior toggles
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None, description="Directly convert wired cell dets to HTML."),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None, description="Directly convert wireless cell dets to HTML."),
    use_table_orientation_classify: Optional[bool] = Query(None, description="Auto-rotate tilted tables (90/180/270)."),
    use_ocr_results_with_table_cells: Optional[bool] = Query(None, description="Re-segment OCR by cell masks to avoid text loss."),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None, description="End-to-end wired table recognition (no cell detector)."),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None, description="End-to-end wireless table recognition (no cell detector)."),
):
    # Validate extension
    if not _ext_ok(file.filename):
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    # Persist to temp file while enforcing size limit
    with tempfile.TemporaryDirectory(prefix="ppsv3_") as td:
        temp_path = Path(td) / Path(file.filename).name
        try:
            # SpooledTemporaryFile gives a file-like object at file.file
            _stream_copy_enforcing_limit(file.file, temp_path, MAX_FILE_SIZE_MB)
        finally:
            try:
                await file.close()
            except Exception:
                pass

        # Parse complex JSON/number parameters
        layout_threshold = _parse_float_or_json(layout_threshold_q) if layout_threshold_q is not None else None
        layout_unclip_ratio = _parse_float_or_json(layout_unclip_ratio_q) if layout_unclip_ratio_q is not None else None
        layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None
        if layout_merge_bboxes_mode_q is not None:
            # Try dict first, else pass through as string
            try:
                parsed = json.loads(layout_merge_bboxes_mode_q)
                if isinstance(parsed, dict):
                    d2: Dict[int, str] = {}
                    for k, v in parsed.items():
                        try:
                            d2[int(k)] = str(v)
                        except Exception:
                            continue
                    layout_merge_bboxes_mode = d2
                else:
                    layout_merge_bboxes_mode = str(layout_merge_bboxes_mode_q)
            except Exception:
                layout_merge_bboxes_mode = str(layout_merge_bboxes_mode_q)

        predict_kwargs = _build_predict_kwargs(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_region_detection=use_region_detection,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,  # float | (w,h) | dict
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,  # str | dict
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            seal_det_limit_side_len=seal_det_limit_side_len,
            seal_det_limit_type=seal_det_limit_type,
            seal_det_thresh=seal_det_thresh,
            seal_det_box_thresh=seal_det_box_thresh,
            seal_det_unclip_ratio=seal_det_unclip_ratio,
            seal_rec_score_thresh=seal_rec_score_thresh,
            use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
            use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
            use_table_orientation_classify=use_table_orientation_classify,
            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
            use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
            use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
        )

        # Run inference (guarded by semaphore, compute in threadpool)
        async with _semaphore_ctx(app.state.predict_sem):
            output = await run_in_threadpool(lambda: app.state.pipeline.predict(input=str(temp_path), **predict_kwargs))

        # Format response
        if return_format == "markdown":
            # Build one markdown string for images or PDFs
            md_list = [res.markdown for res in output]
            merged = app.state.pipeline.concatenate_markdown_pages(md_list)
            # merged is a string; images are not serialized here
            return JSONResponse(
                content={
                    "input_filename": Path(file.filename).name,
                    "format": "markdown",
                    "markdown": merged,
                    "pages": len(output),
                }
            )

        # default: JSON per page
        pages_json: List[Dict[str, Any]] = []
        for res in output:
            # 'res.json' is a dict consistent with save_to_json(); numpy arrays are converted to lists by the pipeline
            pages_json.append(res.json)

        return JSONResponse(
            content={
                "input_filename": Path(file.filename).name,
                "format": "json",
                "pages": pages_json,
                "num_pages": len(pages_json),
            }
        )

# Small helper context manager for semaphore
from contextlib import asynccontextmanager as _acm

@_acm
async def _semaphore_ctx(sem: threading.Semaphore):
    sem.acquire()
    try:
        yield
    finally:
        sem.release()

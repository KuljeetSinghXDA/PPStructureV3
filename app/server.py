import os
import tempfile
import threading
import json
import base64
import re
import shutil
from pathlib import Path
from typing import Optional, List, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# Toggle high-performance inference (uses MKL-DNN) and set CPU threads
ENABLE_HPI = False
ENABLE_MKLDNN = True

from paddleocr import PPStructureV3

# ============== Core Configuration (Default Pipeline Settings) ==============
DEVICE = "cpu"
CPU_THREADS = 4

# Optional accuracy boosters / preprocessing
USE_DOC_ORIENTATION_CLASSIFY = True    # default True in pipeline
USE_DOC_UNWARPING = True              # default True in pipeline (text image rectification)
USE_TEXTLINE_ORIENTATION = True       # default True in pipeline

# Subpipeline toggles
USE_TABLE_RECOGNITION = True          # enable table structure recognition
USE_FORMULA_RECOGNITION = False       # disabled by default (enable for documents with formulas)
USE_CHART_RECOGNITION = False         # disabled by default (enable for documents with charts)
USE_SEAL_RECOGNITION = True           # enable seal (stamp) text recognition

# Model overrides (using default models or specified ones)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"
# (Seal recognition uses default models provided by PaddleOCR; override not explicitly set here)

# Detection/Recognition parameters (default None means use pipeline defaults)
LAYOUT_THRESHOLD = None            # float: layout detection confidence threshold
LAYOUT_NMS = None                  # bool: whether to apply NMS on layout detection results
LAYOUT_UNCLIP_RATIO = None         # float: expansion ratio for layout detection boxes
LAYOUT_MERGE_BBOXES_MODE = None    # str: layout box merging mode (if any)
TEXT_DET_THRESH = None             # float: text detection threshold
TEXT_DET_BOX_THRESH = None         # float: text detection box threshold
TEXT_DET_UNCLIP_RATIO = None       # float: text detection unclip ratio
TEXT_DET_LIMIT_SIDE_LEN = None     # int: limit of image side length for text detection
TEXT_DET_LIMIT_TYPE = None         # str: "min" or "max", which side to limit
TEXT_REC_SCORE_THRESH = None       # float: text recognition score threshold
TEXT_RECOGNITION_BATCH_SIZE = None # int: batch size for text recognition
SEAL_DET_THRESH = None             # float: seal text detection pixel threshold
SEAL_DET_BOX_THRESH = None         # float: seal text detection box threshold
SEAL_DET_UNCLIP_RATIO = None       # float: seal text detection unclip ratio
SEAL_DET_LIMIT_SIDE_LEN = None     # int: limit of image side length for seal detection
SEAL_DET_LIMIT_TYPE = None         # str: "min" or "max"
SEAL_REC_SCORE_THRESH = None       # float: seal text recognition score threshold

# I/O constraints
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1  # limit parallel /parse requests

# ============== App Initialization with Pipeline ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the PP-StructureV3 pipeline with configured parameters
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,
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
        use_seal_recognition=USE_SEAL_RECOGNITION
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield
    # Cleanup (if any) when application shuts down
    # (No specific cleanup needed for pipeline object in this case)

app = FastAPI(title="PP-StructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown"] = Query("json", description="Response format: 'json' (structured data) or 'markdown' (Markdown text)"),
    use_doc_orientation_classify: Optional[bool] = Query(None, description="Override use of document orientation classification"),
    use_doc_unwarping: Optional[bool] = Query(None, description="Override use of text image unwarping (rectification)"),
    use_textline_orientation: Optional[bool] = Query(None, description="Override use of text-line orientation classification"),
    use_seal_recognition: Optional[bool] = Query(None, description="Override use of seal (stamp) text recognition sub-pipeline"),
    use_table_recognition: Optional[bool] = Query(None, description="Override use of table recognition sub-pipeline"),
    use_formula_recognition: Optional[bool] = Query(None, description="Override use of formula recognition sub-pipeline"),
    use_chart_recognition: Optional[bool] = Query(None, description="Override use of chart recognition sub-pipeline"),
    use_region_detection: Optional[bool] = Query(None, description="Override use of layout region detection (general layout analysis)"),
    use_table_orientation_classify: Optional[bool] = Query(None, description="Override use of table orientation classification (if table images might be rotated)"),
    use_ocr_results_with_table_cells: Optional[bool] = Query(None, description="When table recognition is disabled, use general OCR results for table cells"),
    use_e2e_wired_table_rec_model: Optional[bool] = Query(None, description="Use end-to-end model for wired (grid) table recognition"),
    use_e2e_wireless_table_rec_model: Optional[bool] = Query(None, description="Use end-to-end model for wireless (borderless) table recognition"),
    use_wired_table_cells_trans_to_html: Optional[bool] = Query(None, description="Post-process wired table cells to HTML in Markdown output"),
    use_wireless_table_cells_trans_to_html: Optional[bool] = Query(None, description="Post-process wireless table cells to HTML in Markdown output"),
    layout_threshold: Optional[float] = Query(None, description="Layout detection confidence threshold"),
    layout_nms: Optional[bool] = Query(None, description="Apply NMS for layout detection results"),
    layout_unclip_ratio: Optional[float] = Query(None, description="Expansion ratio for layout detection boxes"),
    layout_merge_bboxes_mode: Optional[str] = Query(None, description="Layout bounding-box merging mode"),
    text_det_limit_type: Optional[str] = Query(None, description="Text detection image size limit type ('min' or 'max')"),
    text_det_limit_side_len: Optional[int] = Query(None, description="Limit value for image side length in text detection"),
    text_det_thresh: Optional[float] = Query(None, description="Text detection threshold"),
    text_det_box_thresh: Optional[float] = Query(None, description="Text detection box threshold"),
    text_det_unclip_ratio: Optional[float] = Query(None, description="Text detection unclip ratio"),
    text_rec_score_thresh: Optional[float] = Query(None, description="Text recognition confidence threshold"),
    seal_det_limit_type: Optional[str] = Query(None, description="Seal detection image size limit type ('min' or 'max')"),
    seal_det_limit_side_len: Optional[int] = Query(None, description="Limit value for image side length in seal detection"),
    seal_det_thresh: Optional[float] = Query(None, description="Seal text detection pixel threshold"),
    seal_det_box_thresh: Optional[float] = Query(None, description="Seal text detection box threshold"),
    seal_det_unclip_ratio: Optional[float] = Query(None, description="Seal text detection unclip ratio"),
    seal_rec_score_thresh: Optional[float] = Query(None, description="Seal text recognition confidence threshold")
):
    """
    Parse the uploaded document (image or PDF) and return its structured content.
    Supports output in JSON (structured data with coordinates) or Markdown (document content in Markdown format).
    """
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file_extension}")

    # Acquire semaphore to limit parallel predictions
    acquired = app.state.predict_sem.acquire(blocking=False)
    if not acquired:
        # Too many concurrent requests
        raise HTTPException(status_code=429, detail="Too many concurrent requests. Please try again later.")

    input_temp_file = None
    output_temp_dir = None
    try:
        # Save uploaded file to a temporary file on disk
        input_suffix = ".pdf" if file_extension == ".pdf" else file_extension  # ensure proper suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp:
            input_temp_file = tmp.name
            size = 0
            # Read and write in chunks to avoid large memory usage
            for chunk in iter(lambda: file.file.read(8192), b""):
                size += len(chunk)
                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large (>{MAX_FILE_SIZE_MB} MB).")
                tmp.write(chunk)
        # Ensure file pointer is at end, then close (NamedTemporaryFile already closed on exiting 'with')

        # Prepare arguments for pipeline.predict
        predict_kwargs = {}
        # Only include overrides that are not None (so None uses the pipeline's default setting)
        if use_doc_orientation_classify is not None:
            predict_kwargs["use_doc_orientation_classify"] = use_doc_orientation_classify
        if use_doc_unwarping is not None:
            predict_kwargs["use_doc_unwarping"] = use_doc_unwarping
        if use_textline_orientation is not None:
            predict_kwargs["use_textline_orientation"] = use_textline_orientation
        if use_seal_recognition is not None:
            predict_kwargs["use_seal_recognition"] = use_seal_recognition
        if use_table_recognition is not None:
            predict_kwargs["use_table_recognition"] = use_table_recognition
        if use_formula_recognition is not None:
            predict_kwargs["use_formula_recognition"] = use_formula_recognition
        if use_chart_recognition is not None:
            predict_kwargs["use_chart_recognition"] = use_chart_recognition
        if use_region_detection is not None:
            # Note: "use_region_detection" corresponds to whether to use layout analysis
            predict_kwargs["use_region_detection"] = use_region_detection
        if use_table_orientation_classify is not None:
            predict_kwargs["use_table_orientation_classify"] = use_table_orientation_classify
        if use_ocr_results_with_table_cells is not None:
            predict_kwargs["use_ocr_results_with_table_cells"] = use_ocr_results_with_table_cells
        if use_e2e_wired_table_rec_model is not None:
            predict_kwargs["use_e2e_wired_table_rec_model"] = use_e2e_wired_table_rec_model
        if use_e2e_wireless_table_rec_model is not None:
            predict_kwargs["use_e2e_wireless_table_rec_model"] = use_e2e_wireless_table_rec_model
        if use_wired_table_cells_trans_to_html is not None:
            predict_kwargs["use_wired_table_cells_trans_to_html"] = use_wired_table_cells_trans_to_html
        if use_wireless_table_cells_trans_to_html is not None:
            predict_kwargs["use_wireless_table_cells_trans_to_html"] = use_wireless_table_cells_trans_to_html
        if layout_threshold is not None:
            predict_kwargs["layout_threshold"] = layout_threshold
        if layout_nms is not None:
            predict_kwargs["layout_nms"] = layout_nms
        if layout_unclip_ratio is not None:
            predict_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
        if layout_merge_bboxes_mode is not None:
            predict_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
        if text_det_limit_type is not None:
            predict_kwargs["text_det_limit_type"] = text_det_limit_type
        if text_det_limit_side_len is not None:
            predict_kwargs["text_det_limit_side_len"] = text_det_limit_side_len
        if text_det_thresh is not None:
            predict_kwargs["text_det_thresh"] = text_det_thresh
        if text_det_box_thresh is not None:
            predict_kwargs["text_det_box_thresh"] = text_det_box_thresh
        if text_det_unclip_ratio is not None:
            predict_kwargs["text_det_unclip_ratio"] = text_det_unclip_ratio
        if text_rec_score_thresh is not None:
            predict_kwargs["text_rec_score_thresh"] = text_rec_score_thresh
        if seal_det_limit_type is not None:
            predict_kwargs["seal_det_limit_type"] = seal_det_limit_type
        if seal_det_limit_side_len is not None:
            predict_kwargs["seal_det_limit_side_len"] = seal_det_limit_side_len
        if seal_det_thresh is not None:
            predict_kwargs["seal_det_thresh"] = seal_det_thresh
        if seal_det_box_thresh is not None:
            predict_kwargs["seal_det_box_thresh"] = seal_det_box_thresh
        if seal_det_unclip_ratio is not None:
            predict_kwargs["seal_det_unclip_ratio"] = seal_det_unclip_ratio
        if seal_rec_score_thresh is not None:
            predict_kwargs["seal_rec_score_thresh"] = seal_rec_score_thresh

        # Run the prediction in a thread (to avoid blocking the event loop)
        try:
            results: List = await run_in_threadpool(
                app.state.pipeline.predict,
                input_temp_file,
                **predict_kwargs
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Document parsing failed: {str(e)}")

        # Prepare output(s)
        if output_format == "markdown":
            # Build a combined markdown content for all pages
            combined_md = ""
            output_temp_dir = tempfile.mkdtemp()
            for idx, res in enumerate(results, start=1):
                page_dir = Path(output_temp_dir) / f"page_{idx}"
                page_dir.mkdir(parents=True, exist_ok=True)
                # Save markdown for this page
                res.save_to_markdown(save_path=str(page_dir))
                # Read the generated markdown file
                md_files = list(page_dir.glob("*.md"))
                if not md_files:
                    continue  # no markdown output for this page (unlikely)
                md_path = md_files[0]
                md_text = md_path.read_text(encoding="utf-8")
                # Embed any images referenced in the markdown as base64
                def replace_image(match: re.Match) -> str:
                    alt_text = match.group(1)
                    img_path = match.group(2)
                    # Resolve image file path (relative to page_dir)
                    img_file = (page_dir / img_path) if not Path(img_path).is_absolute() else Path(img_path)
                    if not img_file.exists():
                        return match.group(0)  # leave unchanged if file not found
                    img_data = img_file.read_bytes()
                    # Determine MIME type from extension
                    ext = img_file.suffix.lower()
                    mime_type = "png"
                    if ext in [".jpg", ".jpeg"]:
                        mime_type = "jpeg"
                    elif ext == ".bmp":
                        mime_type = "bmp"
                    # Base64 encode the image
                    b64_str = base64.b64encode(img_data).decode("ascii")
                    return f"![{alt_text}](data:image/{mime_type};base64,{b64_str})"
                # Replace all image links in the markdown text
                md_text = re.sub(r'!\[(.*?)\]\((.+?)\)', replace_image, md_text)
                # Append to combined markdown, with page break indication if multiple pages
                if len(results) > 1:
                    combined_md += f"<!-- Page {idx} -->\n"
                combined_md += md_text.strip() + "\n\n"
            # Return combined markdown as plain text response (with Markdown media type)
            return PlainTextResponse(content=combined_md, media_type="text/markdown")

        else:
            # Prepare JSON output with structured data and markdown for each page
            output_temp_dir = tempfile.mkdtemp()
            layout_parsing_results = []
            for idx, res in enumerate(results, start=1):
                page_dir = Path(output_temp_dir) / f"page_{idx}"
                page_dir.mkdir(parents=True, exist_ok=True)
                # Save structured JSON and markdown for this page
                res.save_to_json(save_path=str(page_dir))
                res.save_to_markdown(save_path=str(page_dir))
                # Load JSON content
                json_files = list(page_dir.glob("*.json"))
                page_json = {}
                if json_files:
                    # Each saved JSON represents the structured result for one page
                    with open(json_files[0], "r", encoding="utf-8") as jf:
                        try:
                            page_json = json.load(jf)
                        except json.JSONDecodeError:
                            page_json = {}  # If parsing fails, leave empty
                # Load Markdown text
                md_files = list(page_dir.glob("*.md"))
                md_text = ""
                if md_files:
                    md_text = md_files[0].read_text(encoding="utf-8")
                    # Embed images in the markdown text (base64) for completeness
                    def replace_image(match: re.Match) -> str:
                        alt_text = match.group(1)
                        img_path = match.group(2)
                        img_file = (page_dir / img_path) if not Path(img_path).is_absolute() else Path(img_path)
                        if not img_file.exists():
                            return match.group(0)
                        img_data = img_file.read_bytes()
                        ext = img_file.suffix.lower()
                        mime_type = "png"
                        if ext in [".jpg", ".jpeg"]:
                            mime_type = "jpeg"
                        elif ext == ".bmp":
                            mime_type = "bmp"
                        b64_str = base64.b64encode(img_data).decode("ascii")
                        return f"![{alt_text}](data:image/{mime_type};base64,{b64_str})"
                    md_text = re.sub(r'!\[(.*?)\]\((.+?)\)', replace_image, md_text)
                # Assemble page result object
                page_result = {
                    "markdown": {"text": md_text},
                    "prunedResult": page_json
                }
                layout_parsing_results.append(page_result)
            # Prepare dataInfo about input
            file_type_code = 0 if file_extension == ".pdf" else 1
            data_info = {
                "fileType": file_type_code,
                "fileName": file.filename,
                "pageCount": len(results)
            }
            response_body = {
                "result": {
                    "layoutParsingResults": layout_parsing_results,
                    "dataInfo": data_info
                }
            }
            return JSONResponse(content=response_body)
    finally:
        # Release the semaphore for the next request
        app.state.predict_sem.release()
        # Clean up temporary files and directories
        try:
            if input_temp_file:
                os.remove(input_temp_file)
        except Exception:
            pass
        try:
            if output_temp_dir:
                shutil.rmtree(output_temp_dir, ignore_errors=True)
        except Exception:
            pass

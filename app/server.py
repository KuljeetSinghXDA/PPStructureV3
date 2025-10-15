import os
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

# Model overrides
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-M"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_plus"
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
FORMULA_RECOGNITION_MODEL_NAME = "PP-FormulaNet_plus-S"
CHART_RECOGNITION_MODEL_NAME = "PP-Chart2Table"

# Detection/recognition parameters
LAYOUT_THRESHOLD = None
TEXT_DET_THRESH = None
TEXT_DET_BOX_THRESH = None
TEXT_DET_UNCLIP_RATIO = None
TEXT_DET_LIMIT_SIDE_LEN = None
TEXT_DET_LIMIT_TYPE = None
TEXT_REC_SCORE_THRESH = None
TEXT_RECOGNITION_BATCH_SIZE = None


# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# ================= App & Lifespan =================
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    )
    app.state.predict_sem = threading.Semaphore(value=MAX_PARALLEL_PREDICT)
    yield

app = FastAPI(title="PPStructureV3 /parse API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="PDF or image to parse"),
    output: Literal["json", "markdown", "both"] = Query("json", description="Response content type"),
    combine_pdf_pages: bool = Query(
        True,
        description="If input is PDF and output includes markdown, combine pages into one Markdown string",
    ),
):
    """
    Parse uploaded documents using PPStructureV3 pipeline.
    
    Returns structured JSON and/or Markdown representation of the document.
    For PDFs, can optionally concatenate all pages into a single Markdown string.
    """
    # Validate filename and extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Create temporary directory for this request
    tmp_dir = tempfile.mkdtemp(prefix="ppsv3_")
    dst_path = Path(tmp_dir) / Path(file.filename).name
    max_bytes = int(MAX_FILE_SIZE_MB * 1024 * 1024)

    total_bytes = 0
    try:
        # Stream file to disk with size validation
        with open(dst_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB",
                    )
                out.write(chunk)
        
        await file.close()

        # Execute inference with semaphore-guarded threadpool
        sem = app.state.predict_sem

        async def _predict(path_str: str):
            """Run blocking predict call in threadpool with concurrency control"""
            await run_in_threadpool(sem.acquire)
            try:
                return await run_in_threadpool(app.state.pipeline.predict, path_str)
            finally:
                sem.release()

        # Get results (list of result objects, one per page for PDFs)
        results = await _predict(str(dst_path))
        
        # Convert generator to list if needed
        if not isinstance(results, list):
            results = list(results)

        # Build response payload
        is_pdf = ext == ".pdf"
        response_data = {
            "filename": Path(file.filename).name,
            "extension": ext,
            "size_bytes": total_bytes,
            "num_pages": len(results),
            "output_format": output,
        }

        # Extract JSON data from result objects
        if output in ("json", "both"):
            try:
                # Access the json attribute of each result object
                pages_json = []
                for res in results:
                    pages_json.append(res.json)
                response_data["pages_json"] = pages_json
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to extract JSON from results: {str(e)}"
                )

        # Extract Markdown data from result objects
        if output in ("markdown", "both"):
            try:
                # Each result.markdown is a dict with 'markdown_texts' key
                markdown_dicts = [res.markdown for res in results]
                
                if is_pdf and combine_pdf_pages and len(markdown_dicts) > 1:
                    # Attempt to concatenate pages using pipeline method
                    try:
                        # Try standard method first
                        try:
                            combined_md = app.state.pipeline.concatenate_markdown_pages(markdown_dicts)
                        except AttributeError:
                            # Fallback for known bug in some versions
                            combined_md = app.state.pipeline.paddlex_pipeline.concatenate_markdown_pages(markdown_dicts)
                        
                        response_data["markdown"] = combined_md
                    except Exception as concat_error:
                        # Manual fallback: extract markdown_texts and join with page breaks
                        markdown_texts = []
                        for idx, md_dict in enumerate(markdown_dicts):
                            if isinstance(md_dict, dict) and "markdown_texts" in md_dict:
                                markdown_texts.append(md_dict["markdown_texts"])
                            elif isinstance(md_dict, str):
                                markdown_texts.append(md_dict)
                        
                        response_data["markdown"] = "\n\n---\n\n".join(markdown_texts)
                        response_data["concatenation_note"] = "Manual concatenation used due to API limitation"
                else:
                    # Return per-page markdown (list of dicts or extracted texts)
                    if combine_pdf_pages and len(markdown_dicts) == 1:
                        # Single page: return just the text
                        md = markdown_dicts[0]
                        if isinstance(md, dict) and "markdown_texts" in md:
                            response_data["markdown"] = md["markdown_texts"]
                        else:
                            response_data["markdown"] = md
                    else:
                        # Multiple pages without combination: return structured dicts
                        response_data["markdown_pages"] = markdown_dicts
                        
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to extract Markdown from results: {str(e)}"
                )

        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
    finally:
        # Always cleanup temp directory
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

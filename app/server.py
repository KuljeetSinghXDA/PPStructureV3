# requirements (runtime):
#   - fastapi
#   - uvicorn
#   - paddleocr >= 3.0.0
#   - paddlepaddle (cpu or gpu build)
# Notes:
#   - PPStructureV3 APIs: predict(), res.save_to_json(), res.save_to_markdown(), res.markdown, concatenate_markdown_pages()
#   - SLANet_plus end-to-end switches are passed at predict time when table recognition is enabled.

import os
import io
import json
import time
import shutil
import zipfile
import tempfile
import threading
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# PaddleOCR 3.x PP-StructureV3
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

# Detection/recognition parameters (None uses pipeline defaults)
LAYOUT_THRESHOLD: Optional[float] = None
TEXT_DET_THRESH: Optional[float] = None
TEXT_DET_BOX_THRESH: Optional[float] = None
TEXT_DET_UNCLIP_RATIO: Optional[float] = None
TEXT_DET_LIMIT_SIDE_LEN: Optional[int] = None
TEXT_DET_LIMIT_TYPE: Optional[str] = None
TEXT_REC_SCORE_THRESH: Optional[float] = None
TEXT_RECOGNITION_BATCH_SIZE: Optional[int] = None

# Inference backend flags
ENABLE_HPI = False
ENABLE_MKLDNN = True

# I/O and service limits
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 50
MAX_PARALLEL_PREDICT = 1

# Output base
OUTPUT_BASE_DIR = Path("outputs")  # persistent outputs per request
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

def _validate_file(tmp_path: Path):
    ext = tmp_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large: {size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB")

def _unique_output_dir(stem: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = OUTPUT_BASE_DIR / f"{stem}-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _zip_directory(src_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))

def _build_predict_kwargs() -> dict:
    # Enable end-to-end table recognition switches when SLANet_plus is used
    use_e2e = False
    if USE_TABLE_RECOGNITION:
        if str(WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME).lower().startswith("slanet") or \
           str(WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME).lower().startswith("slanet"):
            use_e2e = True
    return {
        # end-to-end table recognition flags (only affect when SLANet_plus models are active)
        "use_e2e_wired_table_rec_model": use_e2e,
        "use_e2e_wireless_table_rec_model": use_e2e,
    }

def _collect_markdown_and_images(res) -> dict:
    # Each result has a .markdown property that contains text and image map
    md_info = getattr(res, "markdown", None)
    if not isinstance(md_info, dict):
        return {"markdown_texts": "", "markdown_images": {}}
    md_texts = md_info.get("markdown_texts") or md_info.get("markdown", "") or ""
    md_images = md_info.get("markdown_images", {}) or {}
    return {"markdown_texts": md_texts, "markdown_images": md_images}

def _concatenate_pages(pipeline: PPStructureV3, md_list: List[dict]) -> str:
    # Try the public API first; fallback to paddlex_pipeline for older builds
    if hasattr(pipeline, "concatenate_markdown_pages"):
        return pipeline.concatenate_markdown_pages(md_list)
    if hasattr(pipeline, "paddlex_pipeline") and hasattr(pipeline.paddlex_pipeline, "concatenate_markdown_pages"):
        return pipeline.paddlex_pipeline.concatenate_markdown_pages(md_list)
    # Fallback: manual join
    return "\n\n".join([item.get("markdown_texts", "") for item in md_list])

def _parse_with_pipeline(pipeline: PPStructureV3, input_path: Path, out_dir: Path) -> dict:
    predict_kwargs = _build_predict_kwargs()
    outputs = pipeline.predict(input=str(input_path), **predict_kwargs)
    is_pdf = input_path.suffix.lower() == ".pdf"

    page_summaries = []
    markdown_pages_info = []

    # Save per-page artifacts
    for i, res in enumerate(outputs):
        # Save per-page JSON and Markdown
        res.save_to_json(save_path=str(out_dir))
        res.save_to_markdown(save_path=str(out_dir))

        # Collect markdown text and images for concatenation
        page_md = _collect_markdown_and_images(res)
        markdown_pages_info.append(page_md)

        # Derive filenames produced by save_to_*:
        # By default, the pipeline uses the input stem (plus page index for PDFs) for filenames.
        # To make paths deterministic, infer by scanning directory for the most recent files that match page index.
        # Simplify by computing conventional names:
        stem = input_path.stem
        page_tag = f"_p{i+1}" if is_pdf else ""
        json_file = out_dir / f"{stem}{page_tag}.json"
        md_file = out_dir / f"{stem}{page_tag}.md"
        # If files do not exist under the conventional name, fall back to listing:
        if not json_file.exists():
            candidates = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            json_file = candidates[0] if candidates else None
        if not md_file.exists():
            candidates = sorted(out_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            # Avoid picking the combined md if already present
            md_file = next((p for p in candidates if "combined" not in p.stem), candidates[0] if candidates else None)

        page_summaries.append({
            "page_index": i + 1,
            "json_file": str(json_file) if json_file else None,
            "markdown_file": str(md_file) if md_file else None,
        })

    combined_markdown_file = None
    combined_markdown_text = None

    # For PDFs (or multi-image batches), create combined Markdown
    if len(markdown_pages_info) > 1 or is_pdf:
        combined_markdown_text = _concatenate_pages(pipeline, markdown_pages_info)
        combined_markdown_file = out_dir / f"{input_path.stem}_combined.md"
        combined_markdown_file.write_text(combined_markdown_text, encoding="utf-8")

        # Persist page-level markdown images
        for page in markdown_pages_info:
            md_imgs = page.get("markdown_images") or {}
            for rel_path, pil_img in md_imgs.items():
                img_path = out_dir / rel_path
                img_path.parent.mkdir(parents=True, exist_ok=True)
                pil_img.save(img_path)

    # Zip all artifacts for convenient download
    zip_path = out_dir.with_suffix(".zip")
    _zip_directory(out_dir, zip_path)

    return {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "zip_file": str(zip_path),
        "pages": page_summaries,
        "combined_markdown_file": str(combined_markdown_file) if combined_markdown_file else None,
        "combined_markdown_text": combined_markdown_text,
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PPStructureV3(
        device=DEVICE,
        enable_mkldnn=ENABLE_MKLDNN,
        enable_hpi=ENABLE_HPI,
        cpu_threads=CPU_THREADS,

        # Model overrides
        layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
        text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
        formula_recognition_model_name=FORMULA_RECOGNITION_MODEL_NAME,
        chart_recognition_model_name=CHART_RECOGNITION_MODEL_NAME,

        # Thresholds and batch sizes (None -> defaults)
        layout_threshold=LAYOUT_THRESHOLD,
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,

        # Sub-pipeline toggles
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
    file: UploadFile = File(..., description="PDF or image"),
    combine_pdf: bool = Query(True, description="If PDF with multiple pages, also emit combined Markdown"),
):
    # Persist upload to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        suffix = Path(file.filename).suffix
        in_path = tmpdir / f"input{suffix}"
        # Stream to disk and measure size
        size = 0
        with in_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                out.write(chunk)
        # Validate size/ext
        _validate_file(in_path)

        # Create a persistent output dir for artifacts
        out_dir = _unique_output_dir(Path(file.filename).stem)

        # Limit concurrent predicts
        async with _SemaphoreAsyncCtx(app.state.predict_sem):
            # Run parsing in a thread to avoid blocking event loop
            result = await run_in_threadpool(_parse_with_pipeline, app.state.pipeline, in_path, out_dir)

        # Optionally drop combined markdown for non-PDF or if disabled
        if not combine_pdf and result.get("combined_markdown_file"):
            # Remove combined artifacts if not requested
            cmf = Path(result["combined_markdown_file"])
            if cmf.exists():
                cmf.unlink()
            result["combined_markdown_file"] = None
            result["combined_markdown_text"] = None

        return JSONResponse(content=result)

class _SemaphoreAsyncCtx:
    def __init__(self, sem: threading.Semaphore):
        self.sem = sem
    async def __aenter__(self):
        loop = threading.get_ident()
        # Acquire in a threadpool to avoid blocking
        await run_in_threadpool(self.sem.acquire)
        return self
    async def __aexit__(self, exc_type, exc, tb):
        self.sem.release()
        return False

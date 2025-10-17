import io
import os
import shutil
import tempfile
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Native PP-StructureV3 pipeline (PaddleOCR 3.x)
# Docs: predict(), save_to_json(), save_to_markdown(), model-name params
# https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html
from paddleocr import PPStructureV3  # type: ignore  # installed at runtime


app = FastAPI(
    title="PP-StructureV3 (CPU, arm64) Service",
    version="1.0.0",
    description="FastAPI wrapper around PaddleOCR PP-StructureV3 for JSON + Markdown outputs."
)

class ParseOptions(BaseModel):
    # Core toggles
    return_json: bool = True
    return_markdown: bool = True

    # Save results to a persistent directory (mounted volume) if desired
    save_dir: Optional[str] = Field(
        default=None,
        description="If set, save native JSON/Markdown files into this directory."
    )

    # Performance/device
    device: str = Field(default="cpu", description="Device string, e.g. 'cpu'.")

    # Medical lab reports are largely English with tables; we set English rec by default.
    lang: str = Field(default="en", description="OCR language for recognition model selection.")

    # Requested model overrides (defaults set to your spec)
    layout_detection_model_name: Optional[str] = Field(default="PP-DocLayout-L")
    text_detection_model_name: Optional[str] = Field(default="PP-OCRv5_mobile_det")
    text_recognition_model_name: Optional[str] = Field(default="en_PP-OCRv5_mobile_rec")

    # Keep other modules at their defaults; expose toggles so you can turn them on if needed
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False

    # Optional thresholds/tunables for tricky docs (left None = pipeline defaults)
    layout_threshold: Optional[float] = None
    text_det_limit_side_len: Optional[int] = None


# Cache pipelines by a small config key to avoid reloading weights for every request
# This keeps "as per native implementation" while supporting runtime overrides safely.
from functools import lru_cache

def _pipeline_key(opts: ParseOptions) -> tuple:
    return (
        opts.device,
        opts.lang,
        opts.layout_detection_model_name or "",
        opts.text_detection_model_name or "",
        opts.text_recognition_model_name or "",
        opts.use_doc_orientation_classify,
        opts.use_doc_unwarping,
        opts.use_textline_orientation,
        opts.layout_threshold,
        opts.text_det_limit_side_len,
    )

@lru_cache(maxsize=4)
def get_pipeline_cached(key: tuple) -> PPStructureV3:
    # Unpack back into a PPStructureV3 instance
    (
        device,
        lang,
        layout_name,
        det_name,
        rec_name,
        use_doc_ori,
        use_unwarp,
        use_textline_ori,
        layout_thresh,
        det_limit_len,
    ) = key

    kwargs: Dict[str, Any] = dict(
        device=device,
        lang=lang,
        use_doc_orientation_classify=use_doc_ori,
        use_doc_unwarping=use_unwarp,
        use_textline_orientation=use_textline_ori,
        layout_detection_model_name=layout_name or None,
        text_detection_model_name=det_name or None,
        text_recognition_model_name=rec_name or None,
    )
    if layout_thresh is not None:
        kwargs["layout_threshold"] = float(layout_thresh)
    if det_limit_len is not None:
        kwargs["text_det_limit_side_len"] = int(det_limit_len)

    return PPStructureV3(**kwargs)

def get_pipeline(opts: ParseOptions) -> PPStructureV3:
    return get_pipeline_cached(_pipeline_key(opts))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/parse")
async def parse(
    files: List[UploadFile] = File(..., description="One or more images or PDFs."),
    # Options can be provided as JSON body or query params
    options: ParseOptions = Body(default=ParseOptions())
):
    # Prepare an overall response container
    response_payload: Dict[str, Any] = {
        "engine": "PP-StructureV3",
        "device": options.device,
        "results": []
    }

    # Ensure save_dir exists if provided
    if options.save_dir:
        os.makedirs(options.save_dir, exist_ok=True)

    # Initialize pipeline (cached by config)
    pipeline = get_pipeline(options)

    # Per-request temp root
    with tempfile.TemporaryDirectory(prefix="ppstructv3_") as req_tmpdir:
        for uf in files:
            # Per-file working dir to isolate outputs
            file_base_tmp = tempfile.mkdtemp(prefix="file_", dir=req_tmpdir)
            # Keep original filename for extension handling
            original_name = uf.filename or "upload"
            _, ext = os.path.splitext(original_name)
            tmp_path = os.path.join(file_base_tmp, f"input{ext or ''}")

            # Persist upload to disk for pipeline consumption (PDFs must be paths)
            content = await uf.read()
            with open(tmp_path, "wb") as f:
                f.write(content)

            # Native pipeline predict
            # Docs: output is a list; for PDFs, each page is one result element
            # We will save native JSON/Markdown and then re-load those to return.
            preds = pipeline.predict(input=tmp_path)  # type: ignore  # API per docs

            # Where to save native outputs for this file
            native_out_dir = os.path.join(file_base_tmp, "native")
            os.makedirs(native_out_dir, exist_ok=True)

            page_json: List[Dict[str, Any]] = []
            page_markdown: List[str] = []

            # Save page-wise results using native helpers
            for res in preds:
                if options.return_json:
                    res.save_to_json(save_path=native_out_dir)  # per docs
                if options.return_markdown:
                    res.save_to_markdown(save_path=native_out_dir)  # per docs

            # Collect files that were saved
            if options.return_json:
                for name in sorted(os.listdir(native_out_dir)):
                    if name.lower().endswith(".json"):
                        with open(os.path.join(native_out_dir, name), "r", encoding="utf-8") as jf:
                            try:
                                page_json.append(json.load(jf))
                            except Exception:
                                # As a fallback, return raw text if parse fails
                                page_json.append({"raw_json": jf.read()})

            if options.return_markdown:
                for name in sorted(os.listdir(native_out_dir)):
                    if name.lower().endswith(".md"):
                        with open(os.path.join(native_out_dir, name), "r", encoding="utf-8") as mf:
                            page_markdown.append(mf.read())

            # Optionally persist native files to a user-provided directory
            persisted_dir = None
            if options.save_dir:
                persisted_dir = os.path.join(options.save_dir, os.path.splitext(os.path.basename(original_name))[0])
                os.makedirs(persisted_dir, exist_ok=True)
                for name in os.listdir(native_out_dir):
                    shutil.copy2(os.path.join(native_out_dir, name), os.path.join(persisted_dir, name))

            response_payload["results"].append({
                "filename": original_name,
                "pages": len(preds),
                "json": page_json if options.return_json else None,
                "markdown": "\n\n".join(page_markdown) if options.return_markdown else None,
                "markdown_pages": page_markdown if options.return_markdown else None,
                "saved_to": persisted_dir
            })

    return JSONResponse(response_payload)


# Basic index
@app.get("/")
def index():
    return {
        "name": "PP-StructureV3 (FastAPI)",
        "endpoints": {
            "POST /parse": "Upload multiple files (images or PDFs) and get JSON + Markdown",
            "GET /healthz": "Health check"
        },
        "tips": [
            "For best accuracy on English medical lab reports, keep lang='en'.",
            "If you need higher accuracy (slower), consider switching to server models via request options: "
            "text_recognition_model_name='en_PP-OCRv5_server_rec', text_detection_model_name='PP-OCRv5_server_det'.",
        ],
        "docs": {
            "pp-structurev3": "See official usage for predict/save_to_json/save_to_markdown.",
            "installation": "See official PaddlePaddle/PaddleOCR installation notes."
        }
    }

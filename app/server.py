from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Literal, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import os, tempfile, threading, json
from paddleocr import PPStructureV3  # PP-StructureV3 pipeline within PaddleOCR [web:22]

# Runtime config (Dokploy Environment)
OCR_LANG = os.getenv("OCR_LANG", "en")                 # English only [web:22]
CPU_THREADS = int(os.getenv("CPU_THREADS", "1"))       # Minimal intra-op threads for stability [web:22]
# Default MKLDNN OFF; can enable later via env if stable on this ARM build
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "false").lower() == "true"  # [web:22]

# Cap OMP/BLAS threads before model init
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))        # [web:22]
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))   # [web:22]

# Lighter layout model default to reduce conv load; still PP-StructureV3-compatible
LAYOUT_MODEL_NAME = os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayout-L")                 # was PP-DocLayout_plus-L [web:22]
WIRED_TABLE_STRUCT_MODEL_NAME = os.getenv("WIRED_TABLE_STRUCT_MODEL_NAME", "SLANeXt_wired")  # [web:22]
TEXT_DET_MODEL_NAME = os.getenv("TEXT_DET_MODEL_NAME", "PP-OCRv5_server_det")       # [web:22]
TEXT_REC_MODEL_NAME = os.getenv("TEXT_REC_MODEL_NAME", "PP-OCRv5_server_rec")       # [web:22]

# Lazy, thread-safe pipeline with startup pre-warm
_pp = None
_pp_lock = threading.Lock()

def get_pipeline():
    global _pp
    if _pp is None:
        with _pp_lock:
            if _pp is None:
                _pp = PPStructureV3(
                    device="cpu",
                    enable_mkldnn=ENABLE_MKLDNN,
                    cpu_threads=CPU_THREADS,
                    lang=OCR_LANG,
                    layout_detection_model_name=LAYOUT_MODEL_NAME,
                    wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCT_MODEL_NAME,
                    text_detection_model_name=TEXT_DET_MODEL_NAME,
                    text_recognition_model_name=TEXT_REC_MODEL_NAME,
                    # Disable modules near prior crash sites
                    use_doc_orientation_classify=False,
                    use_textline_orientation=False,
                    use_doc_unwarping=False,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                )  # [web:22]
    return _pp  # [web:22]

# Robust invoker to cover __call__/predict/process/infer across PaddleOCR releases
def run_pps_v3(pipeline, input_path: str):
    if callable(pipeline):
        return pipeline(input_path)          # [web:22]
    if hasattr(pipeline, "predict"):
        return pipeline.predict(input_path)  # [web:22]
    if hasattr(pipeline, "process"):
        return pipeline.process(input_path)  # [web:22]
    if hasattr(pipeline, "infer"):
        return pipeline.infer(input_path)    # [web:22]
    raise RuntimeError("Unsupported PPStructureV3 API surface")  # [web:22]

def to_markdown(result) -> str:
    pages = result if isinstance(result, list) else [result]  # [web:22]
    lines = ["# Document"]  # [web:22]
    for page_idx, page in enumerate(pages, start=1):
        lines.append(f"\n## Page {page_idx}")  # [web:22]
        items = page if isinstance(page, list) else page.get("res", []) or page.get("result", [])  # [web:22]
        for item in items:
            itype = item.get("type") or item.get("block_type")  # [web:22]
            if itype in {"text", "paragraph", "title"}:
                text = item.get("text") or item.get("res", {}).get("text") or ""  # [web:22]
                if text:
                    lines.append(text.strip())  # [web:22]
            elif itype in {"table"}:
                res = item.get("res") or {}  # [web:22]
                grid = res.get("table", None) or res.get("cells", None)  # [web:22]
                if isinstance(grid, list) and grid and isinstance(grid[0], list):
                    header = [str(c) for c in grid[0]]  # [web:22]
                    lines.append("| " + " | ".join(header) + " |")  # [web:22]
                    lines.append("| " + " | ".join(["---"] * len(header)) + " |")  # [web:22]
                    for row in grid[1:]:
                        lines.append("| " + " | ".join(str(c) for c in row) + " |")  # [web:22]
                elif "html" in res:
                    lines.append(res["html"])  # [web:22]
                else:
                    lines.append("```
                    lines.append(json.dumps(item, ensure_ascii=False))  #[1]
                    lines.append("```")  # [web:22]
            else:
                lines.append("```
                lines.append(json.dumps(item, ensure_ascii=False))  #[1]
                lines.append("```")  # [web:22]
    return "\n".join(lines)  # [web:22]

@asynccontextmanager
async def lifespan(app):
    _ = get_pipeline()  # Pre-warm so models are created before first request [web:22]
    yield  # [web:22]

app = FastAPI(lifespan=lifespan)  # [web:22]

@app.get("/health")
def health():
    return {"status": "ok"}  # [web:22]

@app.post("/parse")
async def parse_doc(
    file: UploadFile = File(...),
    output_format: Optional[Literal["json", "markdown"]] = Query(default="json")
):
    suffix = Path(file.filename or "").suffix.lower()  # [web:22]
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}  # [web:22]
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}"
        )  # [web:22]
    fd, tmp_path = tempfile.mkstemp(prefix="ppsv3_", suffix=suffix, dir="/tmp")  # [web:22]
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)  # 1 MiB chunk [web:22]
                if not chunk:
                    break  # [web:22]
                f.write(chunk)  # [web:22]
        pp = get_pipeline()                 # Ensure pipeline exists (warm or hot) [web:22]
        result = run_pps_v3(pp, tmp_path)  # No save_path; returns structured result [web:22]
        if output_format == "markdown":
            md = to_markdown(result)  # [web:22]
            return PlainTextResponse(md, media_type="text/markdown")  # [web:22]
        return JSONResponse(result)  # [web:22]
    finally:
        try:
            os.unlink(tmp_path)  # Always delete inputs after inference [web:22]
        except FileNotFoundError:
            pass  # [web:22]

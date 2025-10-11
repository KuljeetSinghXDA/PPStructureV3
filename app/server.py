from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Literal, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import os, tempfile, threading, json
from paddleocr import PPStructureV3  # PP-StructureV3 pipeline within PaddleOCR

# Runtime config (Dokploy Environment)
OCR_LANG = os.getenv("OCR_LANG", "en")                 # English-only OCR [docs recommend switching off the CN-EN default for pure English]
CPU_THREADS = int(os.getenv("CPU_THREADS", "1"))       # Minimal intra-op threads for ARM64 stability
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "false").lower() == "true"  # Keep off unless verified stable

# Cap OMP/BLAS threads before model init
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))

# Auto-select all models by NOT specifying names (let pipeline defaults apply)
LAYOUT_MODEL_NAME = None
WIRED_TABLE_STRUCT_MODEL_NAME = None
TEXT_DET_MODEL_NAME = None
TEXT_REC_MODEL_NAME = None

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
                    lang=OCR_LANG,  # honored only when no model names are specified
                    layout_detection_model_name=LAYOUT_MODEL_NAME,
                    wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCT_MODEL_NAME,
                    text_detection_model_name=TEXT_DET_MODEL_NAME,
                    text_recognition_model_name=TEXT_REC_MODEL_NAME,
                    # Keep optional modules off for stability on current ARM64 runtime
                    use_doc_orientation_classify=False,
                    use_textline_orientation=False,
                    use_doc_unwarping=False,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                )
    return _pp

def run_pps_v3(pipeline, input_path: str):
    if callable(pipeline):
        return pipeline(input_path)
    if hasattr(pipeline, "predict"):
        return pipeline.predict(input_path)
    if hasattr(pipeline, "process"):
        return pipeline.process(input_path)
    if hasattr(pipeline, "infer"):
        return pipeline.infer(input_path)
    raise RuntimeError("Unsupported PPStructureV3 API surface")

FENCE = "`" * 3

def to_markdown(result) -> str:
    pages = result if isinstance(result, list) else [result]
    lines = ["# Document"]
    for page_idx, page in enumerate(pages, start=1):
        lines.append(f"\n## Page {page_idx}")
        items = page if isinstance(page, list) else page.get("res", []) or page.get("result", [])
        for item in items:
            itype = item.get("type") or item.get("block_type")
            if itype in {"text", "paragraph", "title"}:
                text = item.get("text") or item.get("res", {}).get("text") or ""
                if text:
                    lines.append(text.strip())
            elif itype in {"table"}:
                res = item.get("res") or {}
                grid = res.get("table", None) or res.get("cells", None)
                if isinstance(grid, list) and grid and isinstance(grid[0], list):
                    header = [str(c) for c in grid[0]]
                    lines.append("| " + " | ".join(header) + " |")
                    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                    for row in grid[1:]:
                        lines.append("| " + " | ".join(str(c) for c in row) + " |")
                elif "html" in res:
                    lines.append(res["html"])
                else:
                    lines.append(FENCE + "json")
                    lines.append(json.dumps(item, ensure_ascii=False))
                    lines.append(FENCE)
            else:
                lines.append(FENCE + "json")
                lines.append(json.dumps(item, ensure_ascii=False))
                lines.append(FENCE)
    return "\n".join(lines)

@asynccontextmanager
async def lifespan(app):
    _ = get_pipeline()  # pre-warm models at startup so defaults are fetched before first request
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse_doc(
    file: UploadFile = File(...),
    output_format: Optional[Literal["json", "markdown"]] = Query(default="json")
):
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}"
        )
    fd, tmp_path = tempfile.mkstemp(prefix="ppsv3_", suffix=suffix, dir="/tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        pp = get_pipeline()
        result = run_pps_v3(pp, tmp_path)
        if output_format == "markdown":
            md = to_markdown(result)
            return PlainTextResponse(md, media_type="text/markdown")
        return JSONResponse(result)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

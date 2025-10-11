from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import List, Literal, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import os, tempfile, threading, json
from paddleocr import PPStructureV3

# Runtime config
OCR_LANG = os.getenv("OCR_LANG", "en")
CPU_THREADS = int(os.getenv("CPU_THREADS", "1"))
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "false").lower() == "true"

# Cap native threads before model init
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))

# Auto-select all models by NOT specifying names
LAYOUT_MODEL_NAME = None
WIRED_TABLE_STRUCT_MODEL_NAME = None
TEXT_DET_MODEL_NAME = None
TEXT_REC_MODEL_NAME = None

# Lazy singleton with pre-warm at startup
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
                    # Disable optional modules for stability
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
async def lifespan(app: FastAPI):
    _ = get_pipeline()
    yield

app = FastAPI(lifespan=lifespan)

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
            allowed = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
            if suffix not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}"
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
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            pp = get_pipeline()
            res = run_pps_v3(pp, tmp_path)

            if ofmt == "markdown":
                md = to_markdown(res)
                results.append({"filename": f.filename, "markdown": md})
            else:
                results.append({"filename": f.filename, "json": res})

        if ofmt == "markdown":
            body = "\n\n".join(f"# {r['filename']}\n\n{r['markdown']}" for r in results)
            return PlainTextResponse(body, media_type="text/markdown")
        return JSONResponse({"results": results})
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

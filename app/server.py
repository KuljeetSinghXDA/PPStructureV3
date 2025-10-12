from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import List, Optional, Literal
from pathlib import Path
from contextlib import asynccontextmanager
import os, tempfile, threading, json, glob, shutil
from paddleocr import PPStructureV3

# Helper to read env without hardcoding defaults
def getenv_required(name: str) -> str:
    val = os.getenv(name)
    if val is None or (isinstance(val, str) and val.strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

def getenv_optional_bool(name: str):
    v = os.getenv(name)
    if v is None:
        return None
    return v.lower() in ("1","true","yes","on")

def getenv_optional_int(name: str):
    v = os.getenv(name)
    if v is None:
        return None
    return int(v)

# Required runtime env
OCR_LANG = getenv_required("OCR_LANG")
DEFAULT_OUTPUT_FORMAT = getenv_required("DEFAULT_OUTPUT_FORMAT").lower()
ALLOWED_SUFFIXES = [s.strip().lower() for s in getenv_required("ALLOWED_SUFFIXES").split(",") if s.strip()]

# Optional runtime env
ENABLE_MKLDNN = getenv_optional_bool("ENABLE_MKLDNN")
CPU_THREADS = getenv_optional_int("CPU_THREADS")

# Single shared pipeline (thread-safe)
_pp = None
_pp_lock = threading.Lock()

def get_pipeline():
    global _pp
    if _pp is None:
        with _pp_lock:
            if _pp is None:
                kwargs = {"device": "cpu", "lang": OCR_LANG}
                if ENABLE_MKLDNN is not None:
                    kwargs["enable_mkldnn"] = ENABLE_MKLDNN
                if CPU_THREADS is not None:
                    kwargs["cpu_threads"] = CPU_THREADS
                _pp = PPStructureV3(**kwargs)
    return _pp

def run_pps_v3(pipeline, input_path: str):
    if hasattr(pipeline, "predict"):
        return pipeline.predict(input=input_path)
    if callable(pipeline):
        return pipeline(input_path)
    if hasattr(pipeline, "process"):
        return pipeline.process(input_path)
    if hasattr(pipeline, "infer"):
        return pipeline.infer(input_path)
    raise RuntimeError("Unsupported PPStructureV3 API surface")

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
    output_format: Optional[Literal["json", "markdown"]] = Query(None),
):
    ofmt = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
    if ofmt not in ("json","markdown"):
        raise HTTPException(status_code=400, detail="output_format must be 'json' or 'markdown'")

    results = []
    tmpdirs = []
    try:
        for f in files:
            if not f.filename:
                raise HTTPException(status_code=400, detail="Missing filename")
            suffix = Path(f.filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
            # Save to a temp file with original suffix so PPStructureV3 can auto-handle PDFs/images
            tmpdir = tempfile.mkdtemp(prefix="pps3_")
            tmpdirs.append(tmpdir)
            tmp_path = os.path.join(tmpdir, f"upload{suffix}")
            size = 0
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = await f.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    size += len(chunk)
                    out.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            pp = get_pipeline()
            output = run_pps_v3(pp, tmp_path)

            if ofmt == "json":
                # Follow docs pattern: save_to_json then read back for API response
                for res in output:
                    res.save_to_json(save_path=tmpdir)
                json_files = sorted(glob.glob(os.path.join(tmpdir, "*.json")))
                json_docs = []
                for jf in json_files:
                    try:
                        with open(jf, "r", encoding="utf-8") as fh:
                            json_docs.append(json.load(fh))
                    except Exception:
                        continue
                results.append({"filename": f.filename, "documents": json_docs})
            else:
                for res in output:
                    res.save_to_markdown(save_path=tmpdir)
                md_files = sorted(glob.glob(os.path.join(tmpdir, "*.md")))
                md_docs = []
                for mf in md_files:
                    try:
                        with open(mf, "r", encoding="utf-8") as fh:
                            md_docs.append(fh.read())
                    except Exception:
                        continue
                results.append({"filename": f.filename, "documents_markdown": md_docs})

        if ofmt == "json":
            return JSONResponse({"results": results})
        body = "\n\n".join(f"# {r['filename']}\n\n" + "\n\n".join(r["documents_markdown"]) for r in results)
        return PlainTextResponse(body, media_type="text/markdown")
    finally:
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

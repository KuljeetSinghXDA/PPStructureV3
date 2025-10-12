# --- Environment must be set before any Paddle/NumPy/OpenBLAS import ---
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

from paddleocr import PPStructureV3  # import after envs are applied


_pp = None
_pp_lock = threading.Lock()


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
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
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
                    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_FILE_SIZE_MB}MB)")
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            pp = get_pipeline()

            def _predict(path: str):
                with _predict_sem:
                    return pp.predict(input=path)

            try:
                result = await run_in_threadpool(_predict, tmp_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR processing failed for {f.filename}: {str(e)}")

            if ofmt == "json":
                outdir = tempfile.mkdtemp(prefix="ppsv3_json_")
                try:
                    if hasattr(result, "save_to_json"):
                        result.save_to_json(save_path=outdir)
                        docs = []
                        for name in sorted(os.listdir(outdir)):
                            if name.endswith(".json"):
                                with open(os.path.join(outdir, name), "r", encoding="utf-8") as fh:
                                    docs.append(json.load(fh))
                        results.append({"filename": f.filename, "documents": docs or [vars(result)]})
                    else:
                        results.append({"filename": f.filename, "documents": [vars(result)]})
                finally:
                    shutil.rmtree(outdir, ignore_errors=True)
            else:
                md_body = None
                if hasattr(result, "markdown"):
                    md_body = result.markdown
                elif hasattr(result, "to_markdown"):
                    md_body = result.to_markdown()
                else:
                    outdir = tempfile.mkdtemp(prefix="ppsv3_md_")
                    try:
                        if hasattr(result, "save_to_markdown"):
                            result.save_to_markdown(save_path=outdir)
                            parts = []
                            for name in sorted(os.listdir(outdir)):
                                if name.endswith(".md"):
                                    with open(os.path.join(outdir, name), "r", encoding="utf-8") as fh:
                                        parts.append(fh.read())
                            md_body = "\n\n".join(parts) if parts else str(result)
                        else:
                            md_body = str(result)
                    finally:
                        shutil.rmtree(outdir, ignore_errors=True)
                results.append({"filename": f.filename, "documents_markdown": [md_body]})

        if ofmt == "json":
            return JSONResponse({"results": results})

        body = "\n\n".join(f"# {item['filename']}\n\n" + "\n\n".join(item["documents_markdown"]) for item in results)
        return PlainTextResponse(body, media_type="text/markdown")

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

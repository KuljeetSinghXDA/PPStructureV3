from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import List, Literal, Optional
from pathlib import Path
import os, tempfile

router = APIRouter()

@router.post("/parse")
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
            # Basic validation
            if not f.filename:
                raise HTTPException(status_code=400, detail="Missing filename")
            suffix = Path(f.filename).suffix.lower()
            allowed = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
            if suffix not in allowed:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}")

            # Write to a temp file with the original suffix to let the pipeline detect type
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

            # Run the pipeline with a version-safe wrapper
            res = run_pps_v3(app.state.struct, tmp_path)

            if ofmt == "markdown":
                md = to_markdown(res)
                results.append({"filename": f.filename, "markdown": md})
            else:
                results.append({"filename": f.filename, "json": res})

        if ofmt == "markdown":
            body = "\n\n".join(f"# {r['filename']}\n\n{r['markdown']}" for r in results)
            return PlainTextResponse(body, media_type="text/markdown")  # ensure markdown content-type
        return JSONResponse({"results": results})
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
import tempfile, json

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = PPStructureV3(device=DEVICE, lang=OCR_LANG)
    return _pipeline

app = FastAPI(title="PPStructureV3 Minimal API")

@app.on_event("startup")
async def preload():
    _ = get_pipeline()  # preload PaddleOCR models at startup

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(files: list[UploadFile] = File(...)):
    results = []

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        size = 0
        while chunk := await f.read(1 << 20):
            tmp.write(chunk)
            size += len(chunk)
            if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                tmp.close(); os.unlink(tmp.name)
                raise HTTPException(413, f"File too large (>{MAX_FILE_SIZE_MB}MB)")
        tmp.close()

        try:
            result = get_pipeline().predict(input=tmp.name)
        except Exception as e:
            raise HTTPException(500, f"OCR failed for {f.filename}: {e}")
        finally:
            os.unlink(tmp.name)

        try:
            result_json = json.loads(json.dumps(result, default=str))
        except Exception:
            result_json = str(result)
        results.append({"filename": f.filename, "result": result_json})

    return JSONResponse({"results": results})

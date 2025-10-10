from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import os, io, tempfile, fitz, requests

# PaddleOCR pipeline (PP-StructureV3)
from paddleocr import PPStructureV3

PORT = int(os.getenv("PORT", "8080"))
OCR_LANG = os.getenv("OCR_LANG", "en")
USE_DOC_ORIENTATION_CLASSIFY = os.getenv("USE_DOC_ORIENTATION_CLASSIFY", "false").lower() == "true"
USE_DOC_UNWARPING = os.getenv("USE_DOC_UNWARPING", "false").lower() == "true"
USE_TEXTLINE_ORIENTATION = os.getenv("USE_TEXTLINE_ORIENTATION", "false").lower() == "true"

ENABLE_TABLE = os.getenv("ENABLE_TABLE", "true").lower() == "true"
ENABLE_CHART = os.getenv("ENABLE_CHART", "false").lower() == "true"
ENABLE_FORMULA = os.getenv("ENABLE_FORMULA", "false").lower() == "true"
ENABLE_SEAL = os.getenv("ENABLE_SEAL", "false").lower() == "true"

LAYOUT_MODEL = os.getenv("LAYOUT_MODEL", "PP-DocLayout_plus-L")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")
TABLE_MODEL = os.getenv("TABLE_MODEL", "PP-TableMagic-L")
CHART_MODEL = os.getenv("CHART_MODEL", "PP-Chart2Table-L")
FORMULA_MODEL = os.getenv("FORMULA_MODEL", "PP-FormulaNet_plus-L")

MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "40"))

app = FastAPI(title="PP-StructureV3 Extractor", version="1.0.0")

def bytes_limit_ok(nbytes: int) -> bool:
    return (nbytes / (1024 * 1024)) <= MAX_FILE_MB

def load_pdf_to_images(pdf_bytes: bytes) -> List[io.BytesIO]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = io.BytesIO(pix.tobytes("png"))
            images.append(img)
    return images

def init_pipeline():
    # Defaults include PP-DocLayout_plus-L and PP-OCRv5; language set to English
    # Optional toggles for orientation/unwarping/textline orientation
    pipeline = PPStructureV3(
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        lang=OCR_LANG,
        ocr_version=OCR_VERSION,
        device="cpu"
    )
    return pipeline

def run_ppstructurev3_on_pages(pipeline, images: List[io.BytesIO]):
    all_pages = []
    for img in images:
        img.seek(0)
        res = pipeline.predict(img)
        # Convert results to JSON-serializable dicts
        page_items = []
        for r in res:
            if hasattr(r, "to_dict"):
                page_items.append(r.to_dict())
            else:
                try:
                    page_items.append(dict(r))
                except Exception:
                    page_items.append({"repr": repr(r)})
        all_pages.append(page_items)
    return all_pages

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract")
async def extract(pdf_url: Optional[str] = Query(default=None), file: Optional[UploadFile] = File(default=None)):
    if not pdf_url and not file:
        raise HTTPException(status_code=400, detail="Provide pdf_url query param or upload a PDF file")
    if pdf_url:
        r = requests.get(pdf_url, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch PDF")
        if not bytes_limit_ok(len(r.content)):
            raise HTTPException(status_code=413, detail="PDF too large")
        pdf_bytes = r.content
    else:
        content = await file.read()
        if not bytes_limit_ok(len(content)):
            raise HTTPException(status_code=413, detail="PDF too large")
        pdf_bytes = content

    images = load_pdf_to_images(pdf_bytes)
    if not images:
        raise HTTPException(status_code=400, detail="Empty PDF or no renderable pages")

    pipeline = init_pipeline()
    structured = run_ppstructurev3_on_pages(pipeline, images)

    # Basic text aggregation from structured items
    text_pages = []
    for page in structured:
        lines = []
        for item in page:
            txt = item.get("text") or item.get("res", {}).get("text")
            if txt:
                if isinstance(txt, list):
                    lines.extend([str(t) for t in txt])
                else:
                    lines.append(str(txt))
        text_pages.append("\n".join(lines))

    return JSONResponse(
        {
            "pages": structured,
            "text": text_pages,
            "meta": {
                "pages": len(structured),
                "lang": OCR_LANG,
                "layout_model": LAYOUT_MODEL,
                "ocr_version": OCR_VERSION,
                "table_model": TABLE_MODEL,
                "chart_model": CHART_MODEL if ENABLE_CHART else None,
                "formula_model": FORMULA_MODEL if ENABLE_FORMULA else None,
            },
        }
    )

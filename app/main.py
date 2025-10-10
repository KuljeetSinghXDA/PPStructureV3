from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tempfile import TemporaryDirectory
import fitz  # PyMuPDF
import os
from typing import List

# Core pipeline
from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 (arm64) - Lab Reports")

# Defaults optimized for English lab reports (tables), CPU-only
# Flagship models pinned by name; heavy optional modules disabled by default
PIPELINE = PPStructureV3(
    device="cpu",
    layout_detection_model_name="PP-DocLayout_plus-L",
    text_recognition_model_name="PP-OCRv5_server_rec",
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    enable_table=True,
    enable_formula=False,
    enable_chart=False,
    enable_seal=False,
)

def pdf_to_images(pdf_bytes: bytes, dpi: int = 220) -> List[str]:
    paths = []
    with TemporaryDirectory() as td:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = os.path.join(td, f"page_{i+1:04d}.png")
            pix.save(out)
            paths.append(out)
        # Copy to a persistent tmp for pipeline predict (PPStructureV3 streams files)
        # Here we just return paths inside temp; the pipeline consumes them immediately
        return list(paths)

def extract_text_from_results(results) -> dict:
    pages = []
    full_text_lines = []
    for page in results:
        page_text_lines = []
        for block in page.res:
            if block["type"] in ("text", "title", "paragraph_title"):
                txt = block.get("text", "").strip()
                if txt:
                    page_text_lines.append(txt)
            if block["type"] == "table":
                # Prefer markdown_table if provided; else fall back to text
                md = block.get("markdown", "").strip()
                if md:
                    page_text_lines.append(md)
                else:
                    ttxt = block.get("text", "").strip()
                    if ttxt:
                        page_text_lines.append(ttxt)
        page_text = "\n".join(page_text_lines).strip()
        pages.append({"page_index": page.page_index, "text": page_text})
        if page_text:
            full_text_lines.append(page_text)
    return {
        "pages": pages,
        "text": "\n\n".join(full_text_lines).strip()
    }

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "pdf"):
        raise HTTPException(status_code=415, detail="Only PDF is supported")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    # Convert PDF to images and run pipeline
    img_paths = pdf_to_images(data, dpi=220)
    if not img_paths:
        raise HTTPException(status_code=400, detail="Could not rasterize PDF")
    results = PIPELINE.predict(img_paths)
    payload = extract_text_from_results(results)
    return JSONResponse(payload)

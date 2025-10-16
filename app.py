import os
import json
import tempfile
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 API", description="Document parsing with PP-StructureV3 for medical lab reports")

# All configurations are centralized here for easy adjustment
# Models as specified: PP-DocLayout-L for layout, PP-OCRv5_mobile_det for detection, en_PP-OCRv5_mobile_rec for recognition
LAYOUT_MODEL_NAME = "PP-DocLayout-L"
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# Enable all features for full native PP-StructureV3 implementation
USE_TABLE_RECOGNITION = True
USE_FORMULA_RECOGNITION = True
USE_DOC_ORIENTATION_CLASSIFY = True  # For rotated/scanned medical reports
USE_DOC_UNWARPING = True  # For distorted scans common in medical docs
USE_TEXTLINE_ORIENTATION = True  # For vertical text in tables/labels
USE_SEAL_RECOGNITION = False  # Default; enable if seals are present in reports

# Parameters tuned for best accuracy on medical lab reports (dense text/tables, small fonts)
LAYOUT_THRESHOLD = 0.4  # Lower than default (0.5) for higher recall on complex layouts
TEXT_DET_THRESH = 0.3  # Default; pixel threshold for sensitive text detection
TEXT_DET_BOX_THRESH = 0.5  # Slightly lower than default (0.6) for more precise boxes in dense areas
TEXT_REC_SCORE_THRESH = 0.7  # Higher than default (0.0) for precision on medical terms
TEXT_DET_LIMIT_SIDE_LEN = 1280  # Higher than default (960) for high-res scans
TEXT_DET_UNCLIP_RATIO = 1.5  # Default; expands regions for full words
LAYOUT_UNCLIP_RATIO = 1.5  # Default
LAYOUT_NMS = True  # Default non-max suppression
DEVICE = "cpu"  # For ARM64 CPU
ENABLE_HPI = False  # Disable for CPU accuracy over speed
PRECISION = "fp32"  # Full precision for accuracy

# Initialize the pipeline with all configs (models download automatically on first run)
pipeline = PPStructureV3(
    layout_detection_model_name=LAYOUT_MODEL_NAME,
    text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
    text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
    use_table_recognition=USE_TABLE_RECOGNITION,
    use_formula_recognition=USE_FORMULA_RECOGNITION,
    use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
    use_doc_unwarping=USE_DOC_UNWARPING,
    use_textline_orientation=USE_TEXTLINE_ORIENTATION,
    use_seal_recognition=USE_SEAL_RECOGNITION,
    layout_threshold=LAYOUT_THRESHOLD,
    text_det_thresh=TEXT_DET_THRESH,
    text_det_box_thresh=TEXT_DET_BOX_THRESH,
    text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
    text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
    text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
    layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
    layout_nms=LAYOUT_NMS,
    device=DEVICE,
    enable_hpi=ENABLE_HPI,
    precision=PRECISION,
    lang="en",  # English for medical reports
    ocr_version="PP-OCRv5"  # Matches specified models
)

@app.post("/parse")
async def parse_documents(files: List[UploadFile] = File(..., description="Multiple image/PDF files for parsing")):
    """
    Endpoint to parse multiple input files (images or PDFs) using PP-StructureV3.
    Outputs JSON (structured data per page) and Markdown (formatted document) per file.
    Supports batch processing; handles multi-page PDFs by concatenating Markdown.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    temp_files = []  # Track temp files for cleanup

    try:
        for file in files:
            if not file.content_type.startswith(('image/', 'application/pdf')):
                raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}: {file.content_type}")

            # Save uploaded file to temp path
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
                temp_files.append(tmp_path)

            # Run PP-StructureV3 pipeline (handles images as single-page, PDFs as multi-page)
            output = pipeline.predict(input=tmp_path)

            if not output:
                results.append({
                    "filename": file.filename,
                    "error": "No output generated",
                    "pages_json": [],
                    "markdown": ""
                })
                continue

            # For multi-page (PDFs), collect per-page JSON and concatenate Markdown
            page_jsons = []
            markdown_infos = []
            for res in output:
                # Convert numpy arrays to lists for JSON serialization
                page_json = res.json
                # Recursively convert any remaining numpy (e.g., boxes, polys) to lists
                def convert_numpy(obj):
                    if hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
                        return [convert_numpy(item) for item in obj]
                    return obj
                page_json = json.loads(json.dumps(convert_numpy(page_json), ensure_ascii=False))
                page_jsons.append(page_json)

                markdown_infos.append(res.markdown)

            # Concatenate Markdown for multi-page documents
            concatenated_markdown = pipeline.concatenate_markdown_pages(markdown_infos)

            results.append({
                "filename": file.filename,
                "pages_json": page_jsons,  # List of JSON dicts (one per page)
                "markdown": concatenated_markdown  # Full concatenated Markdown string
            })

        return {"results": results}

    finally:
        # Cleanup temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

@app.get("/")
async def root():
    return {"message": "PP-StructureV3 API is running. Use POST /parse for document parsing."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

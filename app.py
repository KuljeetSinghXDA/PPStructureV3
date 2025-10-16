from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
from typing import List
import tempfile
import os
import json
from pathlib import Path

# ============================================
# CONFIGURATION SECTION
# All models, parameters, and settings are configurable here.
# Follows native PP-StructureV3 implementation with specified models.
# Optimized for best accuracy on English medical lab reports (e.g., tables with numerical data, text regions).
# ============================================

# Core Models (as specified: PP-DocLayout-L for layout, PP-OCRv5_mobile_det for detection, en_PP-OCRv5_mobile_rec for recognition)
LAYOUT_DETECTION_MODEL_NAME = "PP-DocLayout-L"  # High-precision layout (90.4% mAP, 23 classes incl. tables)
TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"  # Mobile det for efficiency on ARM64 CPU
TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"  # English-specific rec for medical terms/numbers

# Language and OCR Version
LANG = "en"  # English for medical lab reports
OCR_VERSION = "PP-OCRv5"  # Matches specified models

# Device (CPU for ARM64)
DEVICE = "cpu"

# Pipeline Features (all enabled as per native PP-StructureV3 for full document analysis)
USE_TABLE_RECOGNITION = True  # Critical for medical tables
USE_REGION_DETECTION = True  # Detect blocks (headers, tables)
USE_DOC_ORIENTATION_CLASSIFY = True  # Correct orientation
USE_FORMULA_RECOGNITION = False  # Disable if not needed for labs (default off for efficiency)
USE_SEAL_RECOGNITION = False  # Disable for non-stamped docs
ENABLE_HPI = True  # High-performance inference (MKLDNN on CPU)

# Layout Parameters (tuned for accuracy on structured reports)
LAYOUT_THRESHOLD = 0.4  # Lower for detecting subtle tables (default 0.5)
LAYOUT_NMS = True  # Non-max suppression
LAYOUT_UNCLIP_RATIO = 1.2  # Expand boxes for better table coverage

# Text Detection Parameters (for dense medical text/tables)
TEXT_DET_LIMIT_SIDE_LEN = 960
TEXT_DET_THRESH = 0.2  # Lower for small/dense text in reports
TEXT_DET_BOX_THRESH = 0.5  # Adjusted for precision
TEXT_DET_UNCLIP_RATIO = 2.0

# Text Recognition Parameters
TEXT_REC_SCORE_THRESH = 0.7  # Higher to filter low-conf medical terms

# Table-Specific Parameters (for lab report tables: wired/wireless)
USE_WIRED_TABLE_CELLS_TRANS_TO_HTML = True
USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML = True
USE_OCR_RESULTS_WITH_TABLE_CELLS = True  # Re-OCR cells for complete data
USE_TABLE_ORIENTATION_CLASSIFY = True  # Handle rotated tables
TABLE_CLASSIFICATION_MODEL_NAME = "PP-LCNet_x1_0_table_cls"  # Default high-accuracy classifier
WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANeXt_wired"  # Default for bordered tables (common in labs)
WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = "SLANet_wireless"  # Default for borderless

# Other Defaults (as per native: no custom paths, auto-download models)
PADDLEX_CONFIG = None  # Set to YAML path if custom fine-tuned config
VISUALIZE = False  # Disable for production (saves memory on ARM64)
MAX_NUM_INPUT_IMGS = None  # Process all pages in PDFs

# ============================================
# FASTAPI APP SETUP
# ============================================

app = FastAPI(title="PP-StructureV3 API", description="Document parsing with PP-StructureV3 for medical lab reports")

# Initialize PP-StructureV3 pipeline with all configs
pipeline = PPStructureV3(
    lang=LANG,
    ocr_version=OCR_VERSION,
    layout_detection_model_name=LAYOUT_DETECTION_MODEL_NAME,
    text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
    text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
    table_classification_model_name=TABLE_CLASSIFICATION_MODEL_NAME,
    wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
    wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME,
    use_table_recognition=USE_TABLE_RECOGNITION,
    use_region_detection=USE_REGION_DETECTION,
    use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
    device=DEVICE,
    enable_hpi=ENABLE_HPI,
    paddlex_config=PADDLEX_CONFIG,
    # Layout params
    layout_threshold=LAYOUT_THRESHOLD,
    layout_nms=LAYOUT_NMS,
    layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
    # Text det params
    text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
    text_det_thresh=TEXT_DET_THRESH,
    text_det_box_thresh=TEXT_DET_BOX_THRESH,
    text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
    # Text rec params
    text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
    # Table params
    use_wired_table_cells_trans_to_html=USE_WIRED_TABLE_CELLS_TRANS_TO_HTML,
    use_wireless_table_cells_trans_to_html=USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML,
    use_ocr_results_with_table_cells=USE_OCR_RESULTS_WITH_TABLE_CELLS,
    use_table_orientation_classify=USE_TABLE_ORIENTATION_CLASSIFY,
    visualize=VISUALIZE,
    max_num_input_imgs=MAX_NUM_INPUT_IMGS,
)

@app.post("/parse")
async def parse_documents(files: List[UploadFile] = File(..., description="Multiple image/PDF files for parsing")):
    """
    Endpoint to parse multiple documents using PP-StructureV3.
    - Supports images (PNG/JPG) and PDFs (multi-page).
    - Outputs JSON (structured layout/tables/OCR) and Markdown (readable format) per file.
    - Configurable via variables in this file only.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for file in files:
        if not file.content_type.startswith(('image/', 'application/pdf')):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

        # Read and save temp file (PPStructureV3 requires file path)
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        try:
            # Run PP-StructureV3 prediction (handles multi-page PDFs as list of results)
            output = pipeline.predict(
                input=tmp_path,
                use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
                use_table_recognition=USE_TABLE_RECOGNITION,
                use_region_detection=USE_REGION_DETECTION,
                layout_threshold=LAYOUT_THRESHOLD,
                layout_nms=LAYOUT_NMS,
                layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
                text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
                text_det_thresh=TEXT_DET_THRESH,
                text_det_box_thresh=TEXT_DET_BOX_THRESH,
                text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
                use_wired_table_cells_trans_to_html=USE_WIRED_TABLE_CELLS_TRANS_TO_HTML,
                use_wireless_table_cells_trans_to_html=USE_WIRELESS_TABLE_CELLS_TRANS_TO_HTML,
                use_ocr_results_with_table_cells=USE_OCR_RESULTS_WITH_TABLE_CELLS,
                use_table_orientation_classify=USE_TABLE_ORIENTATION_CLASSIFY,
                visualize=VISUALIZE,
            )

            # Collect JSON and Markdown per page/file
            page_jsons = []
            page_markdowns = []
            for res in output:
                # JSON: Full structured output (layout, tables, OCR)
                page_jsons.append(res.json)

                # Markdown: Readable format (text, tables as HTML/MD)
                if hasattr(res, 'markdown') and res.markdown:
                    md_text = res.markdown.get('markdown_texts', [''])[0] if res.markdown.get('markdown_texts') else ''
                    page_markdowns.append(md_text)
                else:
                    page_markdowns.append('')  # Empty if no MD

            # Concatenate Markdown pages with separators for multi-page docs
            full_markdown = '\n\n---\n\n'.join(page_markdowns)

            results.append({
                "filename": file.filename,
                "pages_count": len(output),
                "json": page_jsons,  # List of JSON dicts (one per page)
                "markdown": full_markdown  # Concatenated MD string
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    return JSONResponse(content={"results": results})

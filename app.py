from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 API", description="API for document structure analysis using PaddleOCR PP-StructureV3")

# Initialize the PP-StructureV3 pipeline with the specified models for medical lab reports.
# This configuration is optimized for English text in structured documents like lab reports.
# All settings are configurable here as per the requirement.
ocr_pipeline = PPStructureV3(
    # Layout Detection: Use the high-precision model as requested.
    layout_detection_model_name="PP-DocLayout-L",
    # Text Detection: Use the mobile, efficient model as requested.
    text_detection_model_name="PP-OCRv5_mobile_det",
    # Text Recognition: Use the English-optimized mobile model as requested for best accuracy on English lab reports.
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    # Keep other modules at their default settings for a full-featured pipeline.
    use_doc_orientation_classify=True,   # Correct document orientation
    use_doc_unwarping=False,             # Disabled by default for speed, can be enabled if needed
    use_textline_orientation=True,       # Correct text line orientation
    use_seal_recognition=True,           # Detect and recognize seals
    use_table_recognition=True,          # Full table recognition pipeline
    use_formula_recognition=True,        # Recognize mathematical formulas
    use_chart_recognition=False,         # Chart parsing is very heavy; disabled by default
    use_region_detection=True,           # Use region detection for better layout
    device="cpu",                        # Ensure CPU usage
    enable_hpi=False,                    # High-Performance Inference is often GPU-focused; keep off for CPU
    enable_mkldnn=True,                  # Enable MKL-DNN for CPU acceleration if available
    cpu_threads=4,                       # Adjust based on your CPU; 4 is a reasonable default
)

@app.post("/parse", summary="Parse Document Images/PDFs")
async def parse_document(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Endpoint to parse one or multiple document images or PDFs.
    Returns a JSON object containing the structured results, including JSON and Markdown for each file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_results = {}

    for file in files:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            try:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

                # Run the PP-StructureV3 pipeline on the temporary file
                output = ocr_pipeline.predict(input=tmp_file_path)

                # Aggregate results for this file
                file_results = []
                for page_result in output:
                    # Get the full JSON result
                    json_result = page_result.json
                    # Get the Markdown result as a string
                    markdown_result = page_result.markdown.get("text", "")

                    file_results.append({
                        "json": json_result,
                        "markdown": markdown_result
                    })

                all_results[file.filename] = file_results

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

    return JSONResponse(content=all_results)

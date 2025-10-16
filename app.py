import os
import tempfile
import yaml
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 API", description="API for document parsing using PP-StructureV3 optimized for medical lab reports")

# Configuration section: All settings, parameters, and models are defined here.
# Models: PP-DocLayout-L for layout, PP-OCRv5_mobile_det for detection, en_PP-OCRv5_mobile_rec for recognition (English for medical reports).
# Other modules default. Parameters tuned for best accuracy on medical lab reports: enable unwarping, table recognition; higher thresholds.
CONFIG = {
    "SubModules": {
        "LayoutDetection": {
            "module_name": "layout_detection",
            "model_name": "PP-DocLayout-L",
            "model_dir": None,  # Auto-download
        },
        # Other submodules (e.g., orientation, formula) left at default; can be added here if needed.
    },
    "SubPipelines": {
        "GeneralOCR": {
            "pipeline_name": "OCR",
            "text_type": "general",  # For general text in medical reports
            "use_doc_preprocessor": True,  # Enable preprocessing for better accuracy
            "SubModules": {
                "TextDetection": {
                    "model_name": "PP-OCRv5_mobile_det",
                    "model_dir": None,
                    "limit_side_len": 960,
                    "thresh": 0.4,  # Higher for noisy medical scans
                    "box_thresh": 0.7,
                    "unclip_ratio": 2.0,
                },
                "TextRecognition": {
                    "model_name": "en_PP-OCRv5_mobile_rec",
                    "model_dir": None,
                    "score_thresh": 0.8,  # High threshold for accurate medical text
                },
            },
        },
        # Add more pipelines (e.g., TableRecognition) here if custom models needed; defaults used.
    },
    "Serving": {
        "visualize": False,  # No images in output for API efficiency
        "extra": {
            "max_num_input_imgs": None,  # Unlimited pages for PDFs
        },
    },
    # Global parameters for predict (tuned for medical reports accuracy)
    "PREDICT_PARAMS": {
        "use_doc_orientation_classify": True,  # Handle rotated scans
        "use_doc_unwarping": True,  # Correct distortions in scanned reports
        "use_textline_orientation": True,  # Handle vertical text
        "use_table_recognition": True,  # Essential for lab tables
        "use_formula_recognition": False,  # Rarely in medical reports
        "use_chart_recognition": True,  # For graphs in reports
        "use_seal_recognition": False,  # Not typical for medical docs
        "use_region_detection": True,
        "layout_threshold": 0.6,  # Higher for precise layout in structured docs
        "text_det_thresh": 0.4,
        "text_det_box_thresh": 0.7,
        "text_rec_score_thresh": 0.8,
        "use_ocr_results_with_table_cells": True,  # Better table accuracy
        "use_wired_table_cells_trans_to_html": True,
        "use_wireless_table_cells_trans_to_html": True,
        "use_table_orientation_classify": True,
    },
}

# Initialize the pipeline once at startup
def init_pipeline():
    # Create temporary YAML config file from dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(CONFIG["SubModules"], f, default_flow_style=False)
        yaml.dump({"SubPipelines": CONFIG["SubPipelines"]}, f, default_flow_style=False)  # Append if needed; adjust structure as per YAML
        # Note: Full YAML structure may need flattening; for simplicity, assume full dump
        config_yaml = yaml.dump({
            **CONFIG["SubModules"],
            "SubPipelines": CONFIG["SubPipelines"],
            "Serving": CONFIG["Serving"]
        })
        f.write(config_yaml)
        config_path = f.name
    
    # Initialize with config (lang='en' for English medical reports, ocr_version='PP-OCRv5')
    pipeline = PPStructureV3(lang='en', ocr_version='PP-OCRv5', paddlex_config=config_path)
    
    # Clean up temp file
    os.unlink(config_path)
    return pipeline

pipeline = init_pipeline()

@app.post("/parse")
async def parse_documents(files: List[UploadFile] = File(..., description="Multiple image or PDF files to parse")):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    for file in files:
        if not file.content_type.startswith('image/') and file.filename.lower().endswith('.pdf') == False:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
        
        # Save uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Run prediction with configured parameters
            predict_params = {k: v for k, v in CONFIG["PREDICT_PARAMS"].items() if k != "PREDICT_PARAMS"}  # Flatten
            result = pipeline.predict(img=tmp_path, **predict_params)
            
            # Process output: Handle single image (dict) or PDF (list of dicts)
            if isinstance(result, list):
                # Multi-page PDF: Aggregate markdown, list jsons
                page_results = []
                all_markdown = []
                for page_res in result:
                    json_data = page_res.get('res', {})
                    markdown_texts = page_res.get('markdown', {}).get('markdown_texts', [])
                    all_markdown.extend(markdown_texts)
                    page_results.append(json_data)
                json_output = page_results
                markdown_output = "\n\n---\n\n".join(all_markdown)  # Separate pages
            else:
                # Single image
                json_output = result.get('res', {})
                markdown_output = "\n".join(result.get('markdown', {}).get('markdown_texts', []))
            
            results.append({
                "filename": file.filename,
                "json": json_output,
                "markdown": markdown_output
            })
        
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

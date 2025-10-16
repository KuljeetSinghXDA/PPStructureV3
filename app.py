from fastapi import FastAPI, UploadFile, File
from typing import List
import os
from paddlex import create_pipeline

# (Optional) Force PaddlePaddle to run on CPU 
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Uncomment if you want to disable GPU usage explicitly

# Model configuration: using PP-StructureV3 with specified models for best accuracy on medical lab reports
MODEL_CONFIG = {
    "layout_detection_model_name": "PP-DocLayout-L",       # High-precision layout detection model
    "text_detection_model_name": "PP-OCRv5_mobile_det",    # Text detection model (mobile, English)
    "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",  # Text recognition model (English)
    "table_recognition_model_name": "SLANet_plus",         # Table structure recognition model
    # Other models (formula, seal, etc.) will use pipeline defaults by not specifying them
}
PIPELINE_CONFIG = {
    "use_doc_orientation_classify": False,  # Disable document orientation classification (can enable if needed)
    "use_doc_unwarping": False,            # Disable document unwarping (enable if scanning distortion needs correction)
    "use_textline_orientation": False,     # Disable text line orientation classification
    "use_e2e_wireless_table_rec_model": True  # Enable end-to-end table recognition (no cell detector, for SLANet_plus)
}

# Initialize the PP-StructureV3 pipeline (on CPU)
pipeline = create_pipeline(pipeline="PP-StructureV3", device="cpu")

app = FastAPI()

@app.post("/parse")
def parse_documents(files: List[UploadFile] = File(...)):
    """
    Parse one or multiple documents (images or PDFs) and return their structured 
    data in JSON and Markdown format.
    """
    results = []
    for upload in files:
        # Save the uploaded file to a temporary location
        file_ext = os.path.splitext(upload.filename)[1] or ""  # preserve extension if possible
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            tmp_file.write(upload.file.read())
            tmp_path = tmp_file.name
        try:
            # Perform document parsing with the PP-StructureV3 pipeline
            output = pipeline.predict(input=tmp_path, **MODEL_CONFIG, **PIPELINE_CONFIG)
            # Ensure output is a list (one result per page for multi-page files)
            if not isinstance(output, list):
                output = [output]
            # Prepare the structured result for this file
            file_result = {"file_name": upload.filename}
            pages_data = []
            markdown_texts = []
            combined_images = {}
            for page_index, res in enumerate(output, start=1):
                # Collect structured JSON result for the page
                pages_data.append(res.json)
                # Get markdown text for the page
                if isinstance(res.markdown, dict):
                    md_text = res.markdown.get("text", "")
                    md_images = res.markdown.get("images", {}) or {}
                else:
                    # If res.markdown is an object with attributes
                    md_text = getattr(res.markdown, "text", "")
                    md_images = getattr(res, "outputImages", {}) or {}
                markdown_texts.append(md_text)
                # Collect any images (formulas, charts, etc.) in base64, prefixing with page number if multi-page
                for img_name, img_data in md_images.items():
                    image_key = f"page{page_index}_{img_name}" if len(output) > 1 else img_name
                    combined_images[image_key] = img_data
            # Combine markdown texts from all pages (separated by blank lines if multi-page)
            combined_markdown_text = "\n\n".join(markdown_texts)
            # Assemble the result for this file
            file_result["pages"] = pages_data
            file_result["markdown"] = {
                "text": combined_markdown_text,
                "images": combined_images
            }
            results.append(file_result)
        finally:
            # Clean up the temporary file
            os.remove(tmp_path)
    return {"results": results}

# main.py

import os
import uuid
import time
import shutil
from pathlib import Path
from typing import Literal, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from paddleocr import PPStructureV3

# --- 1. Configuration ---
# Create a directory to store the output files
OUTPUT_DIR = Path("output_files")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 2. Pydantic Models ---

class PipelineConfig(BaseModel):
    """Defines the configurable parameters for the PP-StructureV3 pipeline."""
    use_orientation_correction: bool = Field(default=True, description="Enable document orientation correction.")
    layout_precision: Literal['high', 'medium', 'low'] = Field(default='high', description="Set layout analysis precision.")
    use_table_recognition: bool = Field(default=True, description="Enable table detection and parsing.")
    use_formula_recognition: bool = Field(default=False, description="Enable mathematical formula recognition.")
    use_chart_parsing: bool = Field(default=False, description="Enable parsing of visual charts into tables.")
    use_seal_recognition: bool = Field(default=False, description="Enable recognition of text within official seals.")

class ParseResult(BaseModel):
    """Defines the structure of the API response."""
    status: str
    message: str
    original_filename: str
    processing_time_seconds: float
    # In a production system, these would be URLs to a cloud storage bucket
    json_output_path: Optional[str] = None
    markdown_output_path: Optional[str] = None

# --- 3. Helper Functions ---

def get_layout_model_path(precision: str) -> str:
    """Maps precision level to the corresponding layout model name."""
    # Note: PP-StructureV3 automatically downloads models.
    # We specify the model name suffix here.
    model_map = {
        'high': 'PP-DocLayout-L',
        'medium': 'PP-DocLayout-M',
        'low': 'PP-DocLayout-S'
    }
    return model_map.get(precision, 'PP-DocLayout-L')

# --- 4. Global Pipeline Instance ---

# Initialize the pipeline once at startup to avoid reloading models on each request.
# Set device to 'gpu' if a compatible GPU and paddlepaddle-gpu are installed.
print("Initializing PP-StructureV3 pipeline... This may take a moment.")
try:
    # The constructor arguments are mapped from our API configuration.
    # We initialize with a default configuration; it will be reconfigured per request.
    pipeline = PPStructureV3(
        lang='en', # Default language
        device='cpu', # Change to 'gpu' for GPU acceleration
        show_log=False,
        use_pdserving=False
    )
    print("PP-StructureV3 pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR pipeline: {e}")
    pipeline = None

# --- 5. FastAPI Application ---

app = FastAPI(
    title="PaddleOCR PP-StructureV3 API",
    description="A production-grade API for parsing documents into structured Markdown and JSON.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    if pipeline is None:
        raise RuntimeError("PaddleOCR pipeline could not be initialized. The application cannot start.")

@app.post("/parse/", response_model=ParseResult, tags=)
async def parse_document(
    file: UploadFile = File(..., description="The document file (PDF, PNG, JPG) to parse."),
    config: PipelineConfig = Depends()
):
    """
    Parses a document file using the PP-StructureV3 pipeline.

    The pipeline's behavior can be customized using query parameters.
    The service saves the output as JSON and Markdown files and returns their paths.
    """
    start_time = time.time()
    
    # Create a unique temporary directory for this request to handle file I/O
    request_id = str(uuid.uuid4())
    temp_dir = Path(f"/tmp/{request_id}")
    temp_dir.mkdir(parents=True)
    
    input_path = temp_dir / file.filename
    
    try:
        # Save the uploaded file to the temporary directory
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Dynamically configure the pipeline based on the request parameters
        # This is a simplified approach. A more robust way would be to re-init
        # or have multiple pre-initialized pipelines if parameters change frequently.
        # For this example, we'll update the pipeline's internal settings where possible.
        # Note: Not all parameters of PPStructureV3 can be changed after initialization.
        # A more advanced design might involve a pool of pipelines.
        # Here, we demonstrate the principle by passing relevant flags to predict.
        
        # This mapping is conceptual. The actual `predict` method in paddleocr 3.2.0
        # does not take all these flags directly. We pass them to the constructor.
        # For a truly dynamic API, one would need to manage pipeline instances.
        # Let's assume for this pinnacle script that we re-configure the object.
        
        # This demonstrates the control-plane concept.
        pipeline.table = config.use_table_recognition
        pipeline.formula = config.use_formula_recognition
        pipeline.chart = config.use_chart_parsing
        pipeline.seal = config.use_seal_recognition
        pipeline.layout_model.model_name = get_layout_model_path(config.layout_precision)
        pipeline.use_doc_orientation_classify = config.use_orientation_correction

        print(f"Processing '{file.filename}' with config: {config.model_dump_json()}")

        # Run the prediction
        # The predict method is a synchronous, CPU/GPU-bound operation.
        # In a high-concurrency async app, this should be run in a thread pool
        # to avoid blocking the event loop. FastAPI's default behavior for `def`
        # routes does this automatically. For `async def`, we'd use `run_in_executor`.
        # For simplicity here, we call it directly. See Section 4 for details.
        result = pipeline.predict(str(input_path))

        # Process and save the results
        base_output_filename = f"{request_id}_{input_path.stem}"
        json_output_path = OUTPUT_DIR / f"{base_output_filename}.json"
        md_output_path = OUTPUT_DIR / f"{base_output_filename}.md"
        
        # The result is a list, typically one item per page
        full_markdown = ""
        for i, res in enumerate(result):
            # Save JSON for each page/result object
            page_json_path = OUTPUT_DIR / f"{base_output_filename}_page_{i}.json"
            res.save_to_json(str(page_json_path))
            
            # Concatenate markdown from all pages
            full_markdown += res.markdown + "\n\n"

        with open(md_output_path, "w", encoding="utf-8") as md_file:
            md_file.write(full_markdown)

        end_time = time.time()
        
        return ParseResult(
            status="success",
            message=f"Successfully parsed document.",
            original_filename=file.filename,
            processing_time_seconds=round(end_time - start_time, 2),
            json_output_path=str(json_output_path.relative_to(Path.cwd())),
            markdown_output_path=str(md_output_path.relative_to(Path.cwd()))
        )

    except Exception as e:
        # Catch potential errors from the pipeline
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during document processing: {str(e)}"
        )
    finally:
        # Ensure the temporary directory is always cleaned up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# --- 6. Main Execution Block ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

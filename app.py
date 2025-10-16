from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import json
import tempfile
import uuid
from datetime import datetime

try:
    from paddleocr import PPStructureV3
except ImportError:
    raise ImportError("PaddleOCR is not installed. Please install it with: pip install paddleocr[all]")

from models_config import get_structurev3_kwargs, get_pipeline_kwargs

# Initialize FastAPI app
app = FastAPI(
    title="PP-StructureV3 Document Parser",
    description="Document parsing API using PaddleOCR PP-StructureV3 for medical lab reports",
    version="3.2.0"
)

# Global variable for PP-StructureV3 model
structure_engine = None

class DocumentParser:
    def __init__(self):
        self.engine = None
        self.is_initialized = False
    
    def initialize_engine(self):
    """Initialize PP-StructureV3 engine with configuration"""
    try:
        print("Initializing PP-StructureV3 engine...")
        
        # Get configuration kwargs - USING CORRECTED VERSION
        init_kwargs = get_structurev3_kwargs()
        
        # Print configuration for debugging
        print("Configuration being used:")
        for key, value in init_kwargs.items():
            print(f"  {key}: {value}")
        
        # Initialize PP-StructureV3 engine
        self.engine = PPStructureV3(**init_kwargs)
        self.is_initialized = True
        
        print("PP-StructureV3 engine initialized successfully")
        
    except Exception as e:
        print(f"Error initializing PP-StructureV3 engine: {str(e)}")
        raise e
    
    def parse_document(self, image_path: str) -> Dict[str, Any]:
        """Parse a single document using PP-StructureV3"""
        if not self.is_initialized or self.engine is None:
            raise RuntimeError("PP-StructureV3 engine not initialized")
        
        try:
            # Get pipeline execution kwargs
            pipeline_kwargs = get_pipeline_kwargs()
            
            print(f"Parsing document: {image_path}")
            print(f"Pipeline kwargs: {pipeline_kwargs}")
            
            # Process document with PP-StructureV3
            result = self.engine(image_path, **pipeline_kwargs)
            
            # Format response
            return self._format_response(result, image_path)
            
        except Exception as e:
            print(f"Error parsing document {image_path}: {str(e)}")
            raise e
    
    def _format_response(self, result: List, image_path: str) -> Dict[str, Any]:
        """Format PP-StructureV3 result into structured response"""
        if not result:
            return {
                "status": "error",
                "message": "No content extracted from document",
                "file": os.path.basename(image_path)
            }
        
        # Extract layout information
        layouts = []
        extracted_text = []
        tables = []
        formulas = []
        
        for item in result:
            # Layout information
            if 'layout' in item:
                layouts.append({
                    "label": item['layout']['label'],
                    "bbox": item['layout']['bbox'],
                    "score": float(item['layout']['score'])
                })
            
            # Extracted text
            if 'text' in item and 'bbox' in item:
                extracted_text.append({
                    "text": item['text'],
                    "bbox": item['bbox'],
                    "confidence": float(item.get('confidence', 0.0)),
                    "text_region": item.get('text_region', [])
                })
            
            # Tables
            if 'table' in item:
                tables.append({
                    "html": item['table']['html'],
                    "bbox": item['table']['bbox'],
                    "score": float(item['table']['score'])
                })
            
            # Formulas
            if 'formula' in item:
                formulas.append({
                    "latex": item['formula']['latex'],
                    "bbox": item['formula']['bbox'],
                    "score": float(item['formula']['score'])
                })
        
        # Generate Markdown content
        markdown_content = self._generate_markdown(extracted_text, tables, formulas)
        
        return {
            "status": "success",
            "file": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "layout_regions": len(layouts),
                "text_regions": len(extracted_text),
                "tables": len(tables),
                "formulas": len(formulas)
            },
            "content": {
                "layouts": layouts,
                "text": extracted_text,
                "tables": tables,
                "formulas": formulas
            },
            "markdown": markdown_content
        }
    
    def _generate_markdown(self, text_regions: List, tables: List, formulas: List) -> str:
        """Generate Markdown content from extracted elements"""
        markdown_lines = []
        
        # Add text content
        if text_regions:
            markdown_lines.append("# Document Content\n")
            for i, text_region in enumerate(text_regions, 1):
                markdown_lines.append(f"{text_region['text']}\n")
        
        # Add tables
        if tables:
            markdown_lines.append("\n# Tables\n")
            for i, table in enumerate(tables, 1):
                markdown_lines.append(f"## Table {i}\n")
                # For simplicity, we include the HTML table
                # In production, you might want to convert HTML table to Markdown
                markdown_lines.append(f"```html\n{table['html']}\n```\n\n")
        
        # Add formulas
        if formulas:
            markdown_lines.append("\n# Formulas\n")
            for i, formula in enumerate(formulas, 1):
                markdown_lines.append(f"## Formula {i}\n")
                markdown_lines.append(f"LaTeX: ${formula['latex']}$\n\n")
        
        return "\n".join(markdown_lines) if markdown_lines else "No content extracted"

# Global document parser instance
document_parser = DocumentParser()

@app.on_event("startup")
async def startup_event():
    """Initialize PP-StructureV3 engine on startup"""
    try:
        document_parser.initialize_engine()
    except Exception as e:
        print(f"Failed to initialize PP-StructureV3 engine: {e}")
        # Don't raise here to allow the app to start, but parsing will fail

@app.get("/")
async def root():
    return {
        "message": "PP-StructureV3 Document Parser API",
        "version": "3.2.0",
        "status": "ready" if document_parser.is_initialized else "initializing"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if document_parser.is_initialized else "unhealthy",
        "engine_initialized": document_parser.is_initialized,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/parse")
async def parse_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple document files to parse")
):
    """
    Parse multiple documents using PP-StructureV3
    Returns JSON with extracted content and Markdown
    """
    if not document_parser.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="PP-StructureV3 engine is not initialized. Please try again later."
        )
    
    if not files:
        raise HTTPException(
            status_code=400, 
            detail="No files provided"
        )
    
    results = []
    temp_files = []
    
    try:
        for file in files:
            # Validate file type
            if not file.content_type.startswith('image/') and file.content_type != 'application/pdf':
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.content_type}. Only images and PDFs are supported."
                )
            
            # Create temporary file
            file_extension = os.path.splitext(file.filename)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_files.append(temp_file.name)
            
            # Write uploaded file to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            # Parse document
            result = document_parser.parse_document(temp_file.name)
            results.append(result)
        
        # Cleanup temporary files in background
        background_tasks.add_task(cleanup_temp_files, temp_files)
        
        return JSONResponse(content={
            "status": "success",
            "processed_files": len(results),
            "results": results
        })
    
    except Exception as e:
        # Cleanup on error
        cleanup_temp_files(temp_files)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )

@app.post("/parse-single")
async def parse_single_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Single document file to parse")
):
    """
    Parse a single document using PP-StructureV3
    """
    if not document_parser.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="PP-StructureV3 engine is not initialized"
        )
    
    temp_file = None
    try:
        # Validate file type
        if not file.content_type.startswith('image/') and file.content_type != 'application/pdf':
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Create temporary file
        file_extension = os.path.splitext(file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file_path = temp_file.name
        
        # Write uploaded file to temporary file
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Parse document
        result = document_parser.parse_document(temp_file_path)
        
        # Cleanup in background
        background_tasks.add_task(cleanup_temp_files, [temp_file_path])
        
        return JSONResponse(content=result)
    
    except Exception as e:
        if temp_file and os.path.exists(temp_file.name):
            cleanup_temp_files([temp_file.name])
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up temporary file {file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

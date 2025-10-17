FROM --platform=linux/arm64/v8 python:3.13-slim

# Install system dependencies for PDF handling (poppler), OpenCV, and image libs
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pinned PaddlePaddle (CPU, ARM64 via custom index)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install upgraded PaddleOCR with doc-parser extras for PP-StructureV3
RUN pip install --no-cache-dir "paddleocr[doc-parser]==3.3.0"

# Install ONNX dependencies for export and inference
RUN pip install --no-cache-dir paddle2onnx onnxruntime paddlex

# Install API dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart beautifulsoup4 lxml

# Create models export script
RUN mkdir -p /models && \
    cat > /export_models.py << 'EOF'
from paddleocr import PPStructureV3
import os

# Tuned pipeline config (same as app.py)
pipeline = PPStructureV3(
    layout_detection_model_name='PP-DocLayout-L',
    text_detection_model_name='PP-OCRv5_mobile_det',
    text_recognition_model_name='en_PP-OCRv5_mobile_rec',
    text_det_limit_side_len=1280,
    text_det_limit_type='min',
    text_det_thresh=0.2,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=2.5,
    layout_threshold=0.4,
    layout_unclip_ratio=1.5,
    use_e2e_wireless_table_rec_model=True,
    use_ocr_results_with_table_cells=True,
)

# Export sub-models to ONNX (layout, det, rec, table, etc.) using PaddleX plugin
os.system('paddlex --install paddle2onnx')
for model_name in ['layout', 'det', 'rec', 'table_cls', 'table_rec', 'formula']:  # Key sub-models
    os.system(f'paddlex --paddle2onnx --paddle_model_dir ~/.paddlex/official_models/{model_name} --onnx_model_dir /models/{model_name} --opset_version 17')
print("ONNX models exported to /models/")
EOF

# Run export on build (downloads and converts models)
RUN python /export_models.py

# Create the FastAPI app script
RUN cat > /app.py << 'EOF'
import os
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
from bs4 import BeautifulSoup

app = FastAPI(title="PP-StructureV3 API", version="3.3.0")

# Tuned pipeline for medical lab reports: small fonts, dense tables/layouts, with ONNX backend
pipeline = PPStructureV3(
    # Specified models
    layout_detection_model_name='PP-DocLayout-L',
    text_detection_model_name='PP-OCRv5_mobile_det',
    text_recognition_model_name='en_PP-OCRv5_mobile_rec',
    # ONNX loading (sub-models from /models/)
    layout_model_dir='/models/layout',
    det_model_dir='/models/det',
    rec_model_dir='/models/rec',
    # Assuming table/formula similarly; adjust if needed
    table_model_dir='/models/table_rec',  # For e2e table
    # Backend flag (if supported; fallback to native if not)
    backend='onnxruntime',  # Enables ONNX Runtime for acceleration
    # Tuning for small fonts (lower thresholds, higher resolution/expansion)
    text_det_limit_side_len=1280,
    text_det_limit_type='min',
    text_det_thresh=0.2,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=2.5,
    # Tuning for dense layouts/tables
    layout_threshold=0.4,
    layout_unclip_ratio=1.5,
    use_e2e_wireless_table_rec_model=True,
    use_ocr_results_with_table_cells=True,
    # Defaults for others (table classification, formula, etc.)
)

def html_to_md(html: str) -> str:
    if not html:
        return "No table content available."
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return "Unable to parse table HTML."
    md_rows = []
    header_done = False
    for tr in table.find_all('tr'):
        cells = [th.get_text(strip=True) if tr.find('th') else td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
        if cells:
            row = '| ' + ' | '.join(cells) + ' |'
            md_rows.append(row)
            if not header_done and any('th' in str(cell) for cell in tr.find_all(['th', 'td'])):
                header_done = True
                sep = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                md_rows.insert(1, sep)
    return '\n'.join(md_rows) if md_rows else "Empty table."

@app.post("/parse")
async def parse_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    results = []
    for file in files:
        content_type = file.content_type or ''
        if not (content_type.startswith('image/') or content_type == 'application/pdf'):
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}: {content_type}")
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        try:
            # Run PP-StructureV3 (handles PDF pages natively, now with ONNX acceleration)
            page_results = pipeline.predict(tmp_path)
            results.append({
                "filename": file.filename,
                "pages": page_results  # List of dicts, one per page
            })
        finally:
            os.unlink(tmp_path)
    # Generate Markdown rendering
    md_content = ""
    for file_res in results:
        md_content += f"# {file_res['filename']}\n\n"
        for page_idx, page_data in enumerate(file_res['pages']):
            md_content += f"## Page {page_idx + 1}\n\n"
            # Extract and sort elements by approximate position (avg y then x)
            elements = []
            for item in page_data:
                bbox = item.get('bbox', [[0,0],[0,0],[0,0],[0,0]])
                avg_y = sum(pt[1] for pt in bbox) / 4
                avg_x = sum(pt[0] for pt in bbox) / 4
                typ = item.get('type', 'unknown')
                conf = item.get('confidence', 0.0)
                if typ == 'table':
                    html = item.get('res', {}).get('html', '')
                    table_md = html_to_md(html)
                    elements.append((avg_y, avg_x, f"**Table** (conf: {conf:.2f})\n\n{table_md}\n\n"))
                elif typ in ['text', 'title', 'list']:
                    res = item.get('res', [])
                    if isinstance(res, list):
                        texts = [t[1] for t in res if len(t) > 1]
                        text_content = ' '.join(texts)
                    else:
                        text_content = str(res)
                    elements.append((avg_y, avg_x, f"{text_content}\n"))
                else:  # figure, formula, etc.
                    res = item.get('res', '')
                    res_str = f"${res}$" if typ == 'formula' else str(res)
                    elements.append((avg_y, avg_x, f"**{typ.upper()}** (conf: {conf:.2f}): {res_str}\n"))
            # Sort and render
            elements.sort(key=lambda e: (e[0], e[1]))
            for _, _, content in elements:
                md_content += content
            md_content += "---\n\n"
    return {
        "json": results,
        "markdown": md_content.strip()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

WORKDIR /
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

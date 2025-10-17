FROM --platform=linux/arm64/v8 python:3.13-slim

# Install system dependencies for PDF handling (poppler), OpenCV, image libs, and build tools
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pinned PaddlePaddle (CPU, ARM64 via custom index)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# Install upgraded PaddleOCR with doc-parser extras for PP-StructureV3
RUN pip install --no-cache-dir "paddleocr[doc-parser]==3.3.0"

# Install ONNX dependencies for export and inference (wheels available for py3.12 aarch64)
RUN pip install --no-cache-dir paddle2onnx onnxruntime paddlex

# Install API dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart beautifulsoup4 lxml

# Create models export script
RUN mkdir -p /models && \
    cat > /export_models.py << 'EOF'
import os
import glob
from paddleocr import PPStructureV3

# Download dummy image to trigger model downloads
os.system('wget -O /tmp/dummy.jpg https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs_en/11.jpg')

# Tuned pipeline config (downloads models on predict)
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

# Run predict to download all models
pipeline.predict('/tmp/dummy.jpg')
os.unlink('/tmp/dummy.jpg')

# Dynamically find and export to ONNX: Key sub-models
home = os.path.expanduser('~')
exported = []

# Detection model (en_PP-OCRv5_mobile_det_infer)
det_dirs = glob.glob(f"{home}/.paddleocr/whl/det_en/*_det_infer")
if det_dirs:
    det_dir = det_dirs[0]
    os.system(f'paddle2onnx --model_dir {det_dir} --model_filename inference.pdmodel --params_filename inference.pdiparams --save_dir /models/det --opset_version 17')
    exported.append('det')

# Recognition model (en_PP-OCRv5_mobile_rec_infer)
rec_dirs = glob.glob(f"{home}/.paddleocr/whl/rec_en/*_rec_infer")
if rec_dirs:
    rec_dir = rec_dirs[0]
    os.system(f'paddle2onnx --model_dir {rec_dir} --model_filename inference.pdmodel --params_filename inference.pdiparams --save_dir /models/rec --opset_version 17')
    exported.append('rec')

# Layout model (PP-DocLayout-L_infer)
layout_dirs = glob.glob(f"{home}/.paddlex/official_models/layout/*_infer")
if layout_dirs:
    layout_dir = layout_dirs[0]
    # Adjust filename for PaddleX models
    os.system(f'paddle2onnx --model_dir {layout_dir} --model_filename model.pdmodel --params_filename model.pdiparams --save_dir /models/layout --opset_version 17')
    exported.append('layout')

# Note: Table/formula defaults not exported; add glob if needed

print(f"ONNX export completed for: {', '.join(exported)}")
EOF

# Run export on build (downloads ~500MB models and converts)
RUN python /export_models.py && rm /export_models.py

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

# Tuned pipeline with ONNX loading for acceleration (falls back if dirs missing)
pipeline = PPStructureV3(
    # Specified models with ONNX dirs (auto-uses .onnx if present)
    layout_detection_model_name='PP-DocLayout-L',
    layout_model_dir='/models/layout' if os.path.exists('/models/layout') else None,
    text_detection_model_name='PP-OCRv5_mobile_det',
    det_model_dir='/models/det' if os.path.exists('/models/det') else None,
    text_recognition_model_name='en_PP-OCRv5_mobile_rec',
    rec_model_dir='/models/rec' if os.path.exists('/models/rec') else None,
    # CPU optimizations (MKLDNN for native; ONNX auto-detected)
    enable_mkldnn=True,
    cpu_threads=4,  # Utilize all Ampere cores
    # Tuning for small fonts/dense tables
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
            # Run PP-StructureV3 (with ONNX/MKLDNN acceleration)
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

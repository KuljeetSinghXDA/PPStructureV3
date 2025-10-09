"""
Comprehensive API Guide for PaddleOCR PP-OCRv5 and PP-StructureV3
Optimized for lab reports, medical documents, and complex document parsing
"""

GUIDE_CONTENT = """
# PaddleOCR & PP-StructureV3 API Guide

## Overview
This service provides two powerful endpoints for text extraction and document parsing:
- `/ocr` - Fast text recognition using PP-OCRv5 (server-grade detection + recognition)
- `/structure` - Advanced document parsing with PP-StructureV3 (layout, tables, formulas, reading order)

---

## Endpoint 1: `/ocr` - Text Recognition

### Description
Extracts text from images using PP-OCRv5 server models with high accuracy for English text.
Best for: Simple text extraction, receipts, labels, single-page documents.

### Usage
**POST** `/ocr`
- **Input**: Multipart form with one or more image files
- **Output**: JSON with detected text, confidence scores, and bounding boxes

### Example Request (curl)
```
curl -X POST "http://your-domain.com/ocr" \\
-F "files=@lab_report.jpg" \\
-F "files=@prescription.png"
```

### Example Response
```
{
  "results": [
    {
      "filename": "lab_report.jpg",
      "texts": ["Patient Name: John Doe", "Hemoglobin: 14.5 g/dL", "WBC: 7200/μL"],
      "scores": [0.98, 0.96, 0.97],
      "boxes": [
        [100, 50, 300, 80],
        [100, 100, 300, 130],
        [100, 150, 300, 180]
      ]
    }
  ]
}
```

### Current Configuration
- **Detection Model**: PP-OCRv5_server_det (high-accuracy layout detection)
- **Recognition Model**: PP-OCRv5_server_rec (server-grade English recognition)
- **Language**: English (en)
- **Device**: CPU with optimized threading (OMP_NUM_THREADS=1)
- **Orientation Classification**: Disabled by default (enable via USE_DOC_ORI=true)
- **Text Unwarp**: Disabled by default (enable via USE_UNWARP=true)

### When to Use
- Quick text extraction from photos or scans
- Lab reports with clear text layout
- Receipts, invoices, forms with standard text
- Documents without complex tables or formulas

---

## Endpoint 2: `/structure` - Document Parsing

### Description
Comprehensive document analysis using PP-StructureV3, including layout detection, table extraction, 
formula recognition, reading order recovery, and output to JSON or Markdown.

Best for: Lab reports with tables, scientific papers, invoices, forms, multi-column documents.

### Usage
**POST** `/structure?output_format=json` or `/structure?output_format=markdown`
- **Input**: Multipart form with one or more image files
- **Query Parameter**: `output_format` (optional) - `json` or `markdown` (default: json from .env)
- **Output**: JSON structure or Markdown text with parsed document elements

### Example Request (curl) - JSON Output
```
curl -X POST "http://your-domain.com/structure?output_format=json" \\
-F "files=@blood_test.pdf" \\
-F "files=@chemistry_panel.jpg"
```

### Example Response (JSON)
```
{
  "results": [
    {
      "filename": "blood_test.pdf",
      "documents": [
        {
          "layout": [
            {"type": "title", "bbox": [50, 20, 500, 60], "text": "Complete Blood Count"},
            {"type": "table", "bbox": [50, 80, 500, 300], "html": "<table>...</table>"},
            {"type": "text", "bbox": [50, 320, 500, 360], "text": "Reference ranges..."}
          ],
          "tables": [
            {
              "cells": [
                {"row": 0, "col": 0, "text": "Test Name"},
                {"row": 0, "col": 1, "text": "Result"},
                {"row": 0, "col": 2, "text": "Reference Range"},
                {"row": 1, "col": 0, "text": "Hemoglobin"},
                {"row": 1, "col": 1, "text": "14.5 g/dL"},
                {"row": 1, "col": 2, "text": "13.5-17.5 g/dL"}
              ],
              "html": "<table><tr><td>Test Name</td><td>Result</td>...</tr></table>"
            }
          ],
          "reading_order": [0, 1, 2]
        }
      ]
    }
  ]
}
```

### Example Request (curl) - Markdown Output
```
curl -X POST "http://your-domain.com/structure?output_format=markdown" \\
-F "files=@lab_report.jpg"
```

### Example Response (Markdown)
```
# lab_report.jpg

# Complete Blood Count

| Test Name | Result | Reference Range |
| :-- | :-- | :-- |
| Hemoglobin | 14.5 g/dL | 13.5-17.5 g/dL |
| WBC | 7200/μL | 4500-11000/μL |
| RBC | 5.2 M/μL | 4.5-5.9 M/μL |

Reference ranges may vary by laboratory and patient demographics.
```

### PP-StructureV3 Capabilities

#### 1. Layout Detection
Identifies document regions:
- **Title** - Document headers and section titles
- **Text** - Paragraphs and body text
- **Table** - Tabular data (see Table Recognition)
- **Figure** - Images, charts, diagrams
- **Formula** - Mathematical expressions (LaTeX output)
- **Caption** - Image/table captions
- **Header/Footer** - Page headers and footers
- **List** - Bulleted and numbered lists

#### 2. Table Recognition
Advanced table structure extraction:
- **Cell-level detection** with row/column indices
- **Spanning cells** (rowspan/colspan) handled correctly
- **HTML output** for direct rendering
- **Borderless tables** supported
- **Complex nested tables** (medical reports often have these)

Example table from lab report:
```
{
  "table": {
    "cells": [...],
    "html": "<table><tr><td>Parameter</td><td>Value</td><td>Unit</td><td>Range</td></tr>...</table>"
  }
}
```

#### 3. Formula Recognition
Converts mathematical formulas to LaTeX:
- **Inline formulas** (e.g., x^2 + y^2 = r^2)
- **Display formulas** (complex equations)
- **Chemical formulas** (common in lab reports: H2O, NaCl, etc.)

Output format: LaTeX source code for direct rendering

#### 4. Reading Order Recovery
Multi-column and complex layout reading order:
- Determines logical reading sequence
- Handles multi-column layouts (common in scientific papers)
- Preserves semantic flow across pages

#### 5. Chart Understanding (Experimental)
Extracts data from charts and graphs (if present in lab reports)

### When to Use PP-StructureV3
- **Lab reports with tables** (blood panels, chemistry results, urinalysis)
- **Medical documents** with structured data
- **Invoices and forms** with complex layouts
- **Scientific papers** with formulas and multi-column text
- **Reports requiring Markdown export** for downstream LLM processing

---

## Environment Configuration

### Current Settings (from .env)
```
# OCR Pipeline
OCR_LANG=en                          # Language (en, ch, fr, etc.)
OCR_VERSION=PP-OCRv5                 # Pipeline version
TEXT_DET_MODEL=PP-OCRv5_server_det   # Server-grade detection
TEXT_REC_MODEL=PP-OCRv5_server_rec   # Server-grade recognition

# Performance Tuning
OMP_NUM_THREADS=1                    # OpenMP threads (1 avoids oversubscription)
OPENBLAS_NUM_THREADS=1               # BLAS threads
CPU_THREADS=auto                     # Paddle threads (auto = 2-8 based on CPU count)

# Model Source
PADDLE_PDX_MODEL_SOURCE=BOS          # Model download mirror (BOS or HF)

# Structure Pipeline
STRUCT_DEFAULT_FORMAT=json           # Default output: json or markdown

# Preprocessing (shared by OCR and Structure)
USE_DOC_ORI=false                    # Document orientation classification (90/180/270°)
USE_UNWARP=false                     # Document image unwarping (curved/skewed docs)
USE_TEXTLINE_ORI=false               # Text-line orientation (180° rotated text)
```

### Advanced Preprocessing Options

#### Document Orientation Classification (`USE_DOC_ORI=true`)
Automatically detects and corrects:
- 90° rotation (portrait/landscape)
- 180° upside-down documents
- 270° rotation

**When to enable**: Scanned documents with unknown orientation, batch processing with mixed orientations

**Cost**: Adds ~50ms per image

#### Document Unwarping (`USE_UNWARP=true`)
Corrects curved or skewed document images:
- Photos of documents taken at an angle
- Curved pages from bound books or folders
- Fisheye lens distortion

**When to enable**: Mobile phone captures, book scans

**Cost**: Adds ~100-200ms per image

#### Text-line Orientation (`USE_TEXTLINE_ORI=true`)
Detects and corrects 180° rotated text at line level (angle classifier for det boxes)

**When to enable**: Mixed orientation text in the same image (rare)

**Cost**: Minimal (~10ms per text line)

---

## Use Cases & Examples

### 1. Blood Test Lab Report
**Scenario**: Extract patient results from Complete Blood Count (CBC) report with table

**Endpoint**: `/structure?output_format=json`

**Input**: Blood test PDF or JPG with structured table

**Output**: JSON with:
- Patient info (extracted from title/text regions)
- Table with test names, results, units, reference ranges
- Reading order preserved

**Why Structure**: Tables require cell-level extraction; OCR alone returns unstructured text

---

### 2. Prescription or Rx Form
**Scenario**: Extract doctor's prescription details

**Endpoint**: `/ocr` (if simple text) or `/structure` (if complex form with sections)

**Input**: Prescription image

**Output**: 
- Patient name, date, doctor signature
- Medication names, dosages, instructions
- Pharmacy details

**Tips**: 
- Enable `USE_DOC_ORI=true` if prescriptions may be scanned at different angles
- Use `/structure` if prescription includes tables (medication schedule)

---

### 3. Chemistry Panel or Urinalysis Report
**Scenario**: Multi-page lab report with multiple tables and reference notes

**Endpoint**: `/structure?output_format=markdown`

**Input**: Multi-page PDF or multiple images

**Output**: Markdown with:
- Section titles (Chemistry, Hematology, etc.)
- Tables converted to Markdown format
- Footnotes and reference ranges

**Why Markdown**: Easier to feed into LLMs or documentation systems

---

### 4. Scientific Paper with Formulas
**Scenario**: Research paper with mathematical equations and tables

**Endpoint**: `/structure?output_format=json`

**Output**: 
- Layout structure (title, abstract, sections)
- Formulas in LaTeX format
- Tables with cell-level data
- Reading order for multi-column layouts

---

### 5. Invoice or Receipt with Line Items
**Scenario**: Medical billing invoice with itemized charges

**Endpoint**: `/structure` (for table extraction) or `/ocr` (for simple receipts)

**Output**:
- Header (invoice number, date, patient)
- Line items table (description, quantity, unit price, total)
- Footer (subtotal, tax, grand total)

---

## API Design Best Practices

### Batch Processing
Both endpoints support multiple files in a single request:
```
curl -X POST "http://your-domain.com/ocr" \\
-F "files=@report1.jpg" \\
-F "files=@report2.jpg" \\
-F "files=@report3.jpg"
```

### Error Handling
- Missing or corrupt images return empty results for that file
- Check `results` array length matches input file count
- Structure endpoint may return partial results if specific modules fail

### Performance Tips
1. **Use /ocr for simple text extraction** - Faster than /structure (avg 200-500ms vs 1-2s per page)
2. **Disable unused preprocessing** - Keep USE_DOC_ORI/USE_UNWARP false unless needed
3. **Batch requests** - Send multiple files in one POST instead of sequential requests
4. **Image quality** - Higher resolution improves accuracy but increases processing time
5. **Optimal image size** - 1000-2000px width is ideal; downscale ultra-high-res images

### Output Format Selection (Structure)
- **JSON** (`output_format=json`): Best for programmatic access, structured data extraction, database insertion
- **Markdown** (`output_format=markdown`): Best for LLM input, human-readable output, documentation generation

---

## Health Check

### GET `/healthz`
Returns service status and configuration:
```
{
  "status": "ok",
  "lang": "en",
  "ocr_version": "PP-OCRv5",
  "det_model": "PP-OCRv5_server_det",
  "rec_model": "PP-OCRv5_server_rec",
  "det_cached": true,
  "rec_cached": true,
  "struct_default_format": "json"
}
```

Use this to verify:
- Service is running
- Models are downloaded and cached
- Configuration matches expectations

---

## Model Information

### PP-OCRv5 (General OCR)
- **Architecture**: PP-LCNetV3 (detection) + SVTRv2 (recognition)
- **Languages**: English (current), supports 80+ languages
- **Accuracy**: Server models (used here) achieve 96%+ accuracy on English printed text
- **Speed**: ~200-500ms per page on CPU (Ampere A1)

### PP-StructureV3 (Document Parsing)
- **Modules**: 7 specialized pipelines (layout, table, formula, chart, reading order, seal, conversion)
- **Layout Detection**: PP-DocLayout with 10-class segmentation
- **Table Recognition**: SLANet (structure) + PP-OCRv5 (cell text)
- **Formula Recognition**: PP-FormulaNet (LaTeX output)
- **Output Formats**: JSON, Markdown, HTML (tables)
- **Speed**: ~1-3s per page depending on complexity

---

## Limitations & Known Issues

### Current Limitations
1. **Language**: English only in current configuration (change OCR_LANG to support others)
2. **Handwriting**: Limited support; server models optimized for printed text
3. **Low quality images**: Accuracy degrades below ~300 DPI or with heavy noise
4. **Very large files**: Processing time scales with image size; consider downsizing >4000px images
5. **Multi-page PDFs**: Send as individual images (PDF rendering not built-in)

### Known Issues
1. **Mobile fallback warning**: "lang/ocr_version will be ignored when model names/dirs are set" - This is expected and not an error
2. **ccache warning**: Build-time notice; does not affect runtime
3. **Structure on simple images**: Overkill for plain text; use /ocr for faster results

---

## Advanced: Model Customization

### Switching Languages
Edit `.env`:
```
OCR_LANG=ch  # Chinese
# or
OCR_LANG=fr  # French
# etc.
```
Requires model re-download on first use.

### Using Mobile Models (faster, slightly lower accuracy)
Edit `.env`:
```
TEXT_DET_MODEL=PP-OCRv5_mobile_det
TEXT_REC_MODEL=en_PP-OCRv5_mobile_rec
```
Speed: ~100-200ms per page (vs 200-500ms for server models)

### Enabling Orientation/Unwarp Globally
Edit `.env`:
```
USE_DOC_ORI=true
USE_UNWARP=true
```
Restart service for changes to take effect.

---

## Support & Documentation

### Official Resources
- PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
- PP-OCRv5 Docs: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/
- PP-StructureV3 Docs: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/PP-StructureV3.html
- PaddleX Docs: https://paddlepaddle.github.io/PaddleX/

### Service Info
- **Deployment**: Dokploy on Ampere A1 (ARM64 CPU)
- **Container Port**: 8000
- **Health Check**: GET /healthz
- **API Docs**: GET /docs (FastAPI auto-generated)

---

## Quick Reference

| Task | Endpoint | Format | Speed | Use Case |
|------|----------|--------|-------|----------|
| Simple text extraction | `/ocr` | JSON | Fast (200-500ms) | Receipts, labels, forms |
| Lab report with table | `/structure?output_format=json` | JSON | Medium (1-2s) | Blood tests, chemistry panels |
| Document for LLM | `/structure?output_format=markdown` | Markdown | Medium (1-2s) | RAG pipelines, Q&A |
| Scientific paper | `/structure` | JSON/MD | Slow (2-3s) | Formulas, multi-column, tables |
| Batch processing | `/ocr` or `/structure` | JSON | Linear scaling | Multiple reports at once |

---

## Example Integration (Python)

```
import requests

# Simple OCR
files = {'files': open('lab_report.jpg', 'rb')}
response = requests.post('http://your-domain.com/ocr', files=files)
print(response.json())

# Structure with Markdown
files = {'files': open('blood_test.pdf', 'rb')}
response = requests.post(
    'http://your-domain.com/structure',
    files=files,
    params={'output_format': 'markdown'}
)
print(response.text)

# Batch OCR
files = [
    ('files', open('report1.jpg', 'rb')),
    ('files', open('report2.jpg', 'rb')),
]
response = requests.post('http://your-domain.com/ocr', files=files)
results = response.json()['results']
```

---

## Conclusion

This API provides production-ready OCR and document parsing for lab reports and medical documents:
- **Fast text extraction** via `/ocr` (PP-OCRv5 server models)
- **Complex document parsing** via `/structure` (PP-StructureV3)
- **Flexible output** (JSON for databases, Markdown for LLMs)
- **Optimized for English** lab reports with tables, formulas, and structured data

For questions or custom configurations, refer to the official PaddleOCR documentation or contact your system administrator.
"""

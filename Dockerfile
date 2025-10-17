FROM --platform=linux/arm64/v8 python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    USE_CHART_RECOGNITION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ && \
    python -m pip install "paddleocr[doc-parser]==3.3.0" fastapi "uvicorn[standard]" python-multipart pymupdf

RUN cat > /app.py << 'EOF'
import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Precise monkey-patch on the module-level function used by chart VLM loader.
try:
    from paddlex.inference.models.common.vlm.transformers import model_utils as _mu

    # Patch module-level from_pretrained
    if hasattr(_mu, "from_pretrained"):
        _orig_mu_fp = _mu.from_pretrained
        def _patched_mu_from_pretrained(*args, **kwargs):
            kwargs.setdefault("transpose_weight_keys", None)
            kwargs.setdefault("key_mapping", None)
            return _orig_mu_fp(*args, **kwargs)
        _mu.from_pretrained = _patched_mu_from_pretrained

    # Patch classmethod PreTrainedModel.from_pretrained as added safety
    if hasattr(_mu, "PreTrainedModel") and hasattr(_mu.PreTrainedModel, "from_pretrained"):
        _orig_cls_fp = _mu.PreTrainedModel.from_pretrained
        _orig_cls_fp_func = _orig_cls_fp.__func__ if isinstance(_orig_cls_fp, classmethod) else _orig_cls_fp
        def _patched_cls_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault("transpose_weight_keys", None)
            if "key_mapping" not in kwargs:
                km = getattr(cls, "_checkpoint_conversion_mapping", None)
                kwargs["key_mapping"] = km
            return _orig_cls_fp_func(cls, *args, **kwargs)
        _mu.PreTrainedModel.from_pretrained = classmethod(_patched_cls_from_pretrained)

    # Patch specific PPChart2TableInference wrapper if present
    try:
        from paddlex.inference.models.doc_vlm.predictor import PPChart2TableInference as _ChartInf
        if hasattr(_ChartInf, "from_pretrained"):
            _orig_ci = _ChartInf.from_pretrained
            _orig_ci_func = _orig_ci.__func__ if isinstance(_orig_ci, classmethod) else _orig_ci
            def _patched_chart_ci(cls, *args, **kwargs):
                kwargs.setdefault("transpose_weight_keys", None)
                kwargs.setdefault("key_mapping", None)
                return _orig_ci_func(cls, *args, **kwargs)
            _ChartInf.from_pretrained = classmethod(_patched_chart_ci)
    except Exception:
        pass
except Exception:
    # If internal symbol paths change later, use USE_CHART_RECOGNITION=0 as a fallback.
    pass

from paddleocr import PPStructureV3  # import after patch is applied

app = FastAPI(title="PP-StructureV3 API (ARM64, charts patched)", version="3.3.0")

_use_chart = os.getenv("USE_CHART_RECOGNITION", "1") not in ("0", "false", "False")

pipeline = PPStructureV3(
    layout_detection_model_name="PP-DocLayout-L",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_chart_recognition=_use_chart,   # toggle if needed
    cpu_threads=4,
    # Accuracy-oriented tuning for dense medical reports
    text_det_limit_side_len=1920,
    text_det_limit_type="max",
    text_det_thresh=0.20,
    text_det_box_thresh=0.30,
    text_det_unclip_ratio=2.5
)

def predict_collect(path: Path) -> Dict[str, Any]:
    results = pipeline.predict(
        str(path),
        use_e2e_wireless_table_rec_model=True,
        use_ocr_results_with_table_cells=True
    )
    pages = []
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        json_files, md_files = [], []
        for res in results:
            res.save_to_json(save_path=str(out_dir))
            res.save_to_markdown(save_path=str(out_dir))
        for p in sorted(out_dir.glob("*.json")):
            json_files.append(p)
        for p in sorted(out_dir.glob("*.md")):
            md_files.append(p)
        for i in range(max(len(json_files), len(md_files))):
            page_json = {}
            page_md = ""
            if i < len(json_files):
                try:
                    page_json = json.loads(json_files[i].read_text(encoding="utf-8"))
                except Exception:
                    page_json = {}
            if i < len(md_files):
                page_md = md_files[i].read_text(encoding="utf-8")
            pages.append({"page_index": i, "json": page_json, "markdown": page_md})
    return {"pages": pages}

@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    outputs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for uf in files:
            suffix = Path(uf.filename or "").suffix or ".bin"
            target = tmpdir / (Path(uf.filename or f"upload{len(outputs)}{suffix}").name)
            with target.open("wb") as w:
                shutil.copyfileobj(uf.file, w)
            file_res = predict_collect(target)
            outputs.append({"filename": uf.filename, **file_res})
    return JSONResponse({"files": outputs})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

WORKDIR /
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

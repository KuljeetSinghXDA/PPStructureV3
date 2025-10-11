from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os, tempfile
from paddleocr import PPStructureV3  # pipeline usage documented for CPU + lang + tuning [web:23]

# Env from Dokploy UI (English + CPU tuning)
OCR_LANG = os.getenv("OCR_LANG", "en")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "true").lower() == "true"

# Flagship model names (latest weights fetched automatically)
LAYOUT_MODEL_NAME = os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayout_plus-L")
WIRED_TABLE_STRUCT_MODEL_NAME = os.getenv("WIRED_TABLE_STRUCT_MODEL_NAME", "SLANeXt_wired")
TEXT_DET_MODEL_NAME = os.getenv("TEXT_DET_MODEL_NAME", "PP-OCRv5_server_det")
TEXT_REC_MODEL_NAME = os.getenv("TEXT_REC_MODEL_NAME", "PP-OCRv5_server_rec")

pp = PPStructureV3(
    device="cpu",
    enable_mkldnn=ENABLE_MKLDNN,
    cpu_threads=CPU_THREADS,
    lang=OCR_LANG,
    layout_detection_model_name=LAYOUT_MODEL_NAME,
    wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCT_MODEL_NAME,
    text_detection_model_name=TEXT_DET_MODEL_NAME,
    text_recognition_model_name=TEXT_REC_MODEL_NAME,
    use_doc_orientation_classify=True,
    use_textline_orientation=False,
    use_doc_unwarping=False,
    use_formula_recognition=False,
    use_chart_recognition=False,
)

# Robust call wrapper for PP-StructureV3 across releases
def run_pps_v3(pipeline, input_path: str):
    # Newer builds may offer .predict() or .process(); some offer __call__ [web:23]
    if callable(pipeline):
        return pipeline(input_path)
    if hasattr(pipeline, "predict"):
        return pipeline.predict(input_path)
    if hasattr(pipeline, "process"):
        return pipeline.process(input_path)
    if hasattr(pipeline, "infer"):
        return pipeline.infer(input_path)
    raise RuntimeError("Unsupported PPStructureV3 API: no callable/predict/process/infer found")

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse_doc(file: UploadFile = File(...)):
    # Ephemeral temp file to avoid disk usage; omit save_path so nothing is written [web:23]
    fd, tmp_path = tempfile.mkstemp(prefix="ppsv3_", dir="/tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
        result = run_pps_v3(pp, tmp_path)
        return JSONResponse(result)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

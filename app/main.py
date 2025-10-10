from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from paddleocr import PPStructureV3
import os
import io
from PIL import Image

def env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")

def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except:
        return default

# Threading and MKL-DNN controls for CPU
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", os.getenv("CPU_THREADS", "4"))
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", os.getenv("CPU_THREADS", "4"))

# Build pipeline with flagship models; disable non-essential modules for lab reports
pipeline = PPStructureV3(
    # Device and CPU tuning
    device="cpu",
    enable_mkldnn=env_bool("ENABLE_MKLDNN", True),
    cpu_threads=env_int("CPU_THREADS", 4),

    # Module toggles
    use_doc_orientation_classify=env_bool("USE_DOC_ORIENTATION_CLASSIFY", False),
    use_doc_unwarping=env_bool("USE_DOC_UNWARPING", False),
    use_textline_orientation=env_bool("USE_TEXTLINE_ORIENTATION", False),
    use_region_detection=env_bool("USE_REGION_DETECTION", True),
    use_table_recognition=env_bool("USE_TABLE_RECOGNITION", True),
    use_formula_recognition=env_bool("USE_FORMULA_RECOGNITION", False),
    use_chart_recognition=env_bool("USE_CHART_RECOGNITION", False),
    use_seal_recognition=env_bool("USE_SEAL_RECOGNITION", False),

    # Flagship model names
    layout_detection_model_name=os.getenv("LAYOUT_DETECTION_MODEL_NAME", "PP-DocLayout_plus-L"),
    region_detection_model_name=os.getenv("REGION_DETECTION_MODEL_NAME", "PP-DocBlockLayout"),
    text_detection_model_name=os.getenv("TEXT_DETECTION_MODEL_NAME", "PP-OCRv5_server_det"),
    text_recognition_model_name=os.getenv("TEXT_RECOGNITION_MODEL_NAME", "PP-OCRv5_server_rec"),
    table_orientation_classify_model_name=os.getenv("TABLE_ORIENTATION_CLASSIFY_MODEL_NAME", "PP-LCNet_x1_0_table_cls"),
    wired_table_structure_recognition_model_name=os.getenv("WIRED_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME", "SLANeXt_wired"),
    wireless_table_structure_recognition_model_name=os.getenv("WIRELESS_TABLE_STRUCTURE_RECOGNITION_MODEL_NAME", "SLANeXt_wireless"),
    wired_table_cells_detection_model_name=os.getenv("WIRED_TABLE_CELLS_DETECTION_MODEL_NAME", "RT-DETR-L_wired_table_cell_det"),
    wireless_table_cells_detection_model_name=os.getenv("WIRELESS_TABLE_CELLS_DETECTION_MODEL_NAME", "RT-DETR-L_wireless_table_cell_det"),
)

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse")
async def parse(file: UploadFile = File(...), return_markdown: bool = Form(True)):
    content = await file.read()
    # Convert PDF pages outside scope; PP-StructureV3 accepts images directly
    img = Image.open(io.BytesIO(content)).convert("RGB")
    results = pipeline.predict(input=img)
    out = []
    for res in results:
        item = res.to_dict()
        if return_markdown:
            item["markdown"] = res.to_markdown()
        out.append(item)
    return JSONResponse(content={"results": out})

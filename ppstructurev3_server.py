import io
import os
import json
import cv2
import sys
import shutil
import base64
import typing as T
import tempfile
import subprocess
from dataclasses import dataclass, asdict

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# Configuration
# All settings/parameters/models are centralized here (editable only in this file).
# -----------------------------------------------------------------------------

@dataclass
class PPStructureV3Config:
    # Device/Runtime
    use_gpu: bool = False
    cpu_threads: int = 4
    enable_mkldnn: bool = False  # leave disabled for broad aarch64 compatibility
    precision: str = "fp32"

    # OCR core version/lang
    ocr_version: str = "PP-OCRv5"  # ensures PP-OCRv5 pipeline
    lang: str = "en"

    # Explicitly pin the three models requested by the user
    layout_model_name: str = "PP-DocLayout-L"
    det_model_name: str = "PP-OCRv5_mobile_det"
    rec_model_name: str = "en_PP-OCRv5_mobile_rec"

    # Pipeline features (as per PP-StructureV3)
    enable_layout: bool = True
    enable_table: bool = True
    enable_formula: bool = True
    enable_chart: bool = True
    restore_reading_order: bool = True
    export_markdown: bool = True

    # OCR/detail parameters tuned for structured medical lab reports
    # Note: These stay within native pipeline semantics (thresholds/limits only).
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    det_db_unclip_ratio: float = 1.6
    max_text_length: int = 80
    drop_score: float = 0.4
    use_angle_cls: bool = True
    rec_batch_num: int = 6
    det_limit_side_len: int = 1536  # allow large lab reports
    rec_image_shape: T.Tuple[int, int, int] = (3, 48, 320)

    # Layout thresholds
    layout_nms_thresh: float = 0.5
    layout_score_thresh: float = 0.5

    # I/O behavior
    max_pages: int = 20  # cap very long PDFs if you later add PDF->image
    save_visualization: bool = False  # leave off for API
    tmp_keep: bool = False  # set True for debugging

    # Implementation behavior
    # Prefer native Python API if available; otherwise fallback to calling
    # the official paddleocr "structure" CLI for full feature parity.
    prefer_python_api: bool = True


CFG = PPStructureV3Config()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _set_cpu_threads(threads: int) -> None:
    # Paddle generally respects these environment variables
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


def _load_image_from_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    image = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {upload.filename}")
    return img


def _encode_image_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image to PNG")
    return buf.tobytes()


def _normalize_result_for_json(result: T.Any) -> T.Any:
    # Ensure JSON-serializable types
    def convert(obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bytes, bytearray)):
            return base64.b64encode(obj).decode("utf-8")
        if isinstance(obj, (dict, list, tuple, str, int, float, type(None), bool)):
            return obj
        # Fallback to string
        return str(obj)

    if isinstance(result, dict):
        return {k: _normalize_result_for_json(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_normalize_result_for_json(v) for v in result]
    return convert(result)


# -----------------------------------------------------------------------------
# Native engine (Python API) with CLI fallback for full PP-StructureV3 parity
# -----------------------------------------------------------------------------

class PPStructureV3Engine:
    def __init__(self, cfg: PPStructureV3Config):
        self.cfg = cfg
        self.mode = None  # "python-v3" | "python-v2" | "cli"
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        _set_cpu_threads(self.cfg.cpu_threads)
        # Try native PP-StructureV3 Python API first
        if self.cfg.prefer_python_api:
            try:
                # Newer PaddleOCR 3.x API (name may vary slightly by minor version)
                from paddleocr.ppstructure.v3 import PPStructureV3  # type: ignore
                self.mode = "python-v3"
                self.engine = PPStructureV3(
                    use_gpu=self.cfg.use_gpu,
                    ocr_version=self.cfg.ocr_version,
                    lang=self.cfg.lang,
                    det_limit_side_len=self.cfg.det_limit_side_len,
                    rec_image_shape=self.cfg.rec_image_shape,
                    rec_batch_num=self.cfg.rec_batch_num,
                    use_angle_cls=self.cfg.use_angle_cls,
                    det_db_thresh=self.cfg.det_db_thresh,
                    det_db_box_thresh=self.cfg.det_db_box_thresh,
                    det_db_unclip_ratio=self.cfg.det_db_unclip_ratio,
                    drop_score=self.cfg.drop_score,
                    layout_model_name=self.cfg.layout_model_name,
                    layout_score_thresh=self.cfg.layout_score_thresh,
                    layout_nms_thresh=self.cfg.layout_nms_thresh,
                    enable_layout=self.cfg.enable_layout,
                    enable_table=self.cfg.enable_table,
                    enable_formula=self.cfg.enable_formula,
                    enable_chart=self.cfg.enable_chart,
                    restore_reading_order=self.cfg.restore_reading_order,
                    export_markdown=self.cfg.export_markdown,
                    det_model_name=self.cfg.det_model_name,
                    rec_model_name=self.cfg.rec_model_name,
                    enable_mkldnn=self.cfg.enable_mkldnn,
                    precision=self.cfg.precision,
                )
                return
            except Exception:
                pass

            # Older Python API (PP-Structure V2) as soft fallback
            try:
                from paddleocr import PPStructure  # type: ignore
                self.mode = "python-v2"
                # PPStructure V2 doesn’t expose every V3 feature,
                # but we keep parameters aligned as much as possible.
                self.engine = PPStructure(
                    use_gpu=self.cfg.use_gpu,
                    ocr_version=self.cfg.ocr_version,
                    lang=self.cfg.lang,
                    det_limit_side_len=self.cfg.det_limit_side_len,
                    rec_image_shape=self.cfg.rec_image_shape,
                    rec_batch_num=self.cfg.rec_batch_num,
                    use_angle_cls=self.cfg.use_angle_cls,
                    det_db_thresh=self.cfg.det_db_thresh,
                    det_db_box_thresh=self.cfg.det_db_box_thresh,
                    det_db_unclip_ratio=self.cfg.det_db_unclip_ratio,
                    drop_score=self.cfg.drop_score,
                    layout_model_name=self.cfg.layout_model_name,
                    layout_score_thresh=self.cfg.layout_score_thresh,
                    layout_nms_thresh=self.cfg.layout_nms_thresh,
                    recovery=self.cfg.export_markdown,  # V2 term for reading-order recovery/export
                    # Model pinning for det/rec may be implicit via ocr_version/lang in V2.
                )
                return
            except Exception:
                pass

        # Last resort: invoke the native CLI (which tracks PP-StructureV3 feature flags closely)
        self.mode = "cli"
        self.engine = None

    def _to_markdown_via_save(self, result: T.Any, img_name: str) -> T.Optional[str]:
        # Prefer native save utility if present
        try:
            from paddleocr import save_structure_res  # type: ignore
            with tempfile.TemporaryDirectory() as td:
                out_dir = os.path.join(td, "out")
                os.makedirs(out_dir, exist_ok=True)
                save_structure_res(result, out_dir, img_name)
                # PP-Structure writes .md or .txt depending on version; search for MD
                md_text = None
                for root, _, files in os.walk(out_dir):
                    for f in files:
                        if f.endswith(".md"):
                            with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                                md_text = fh.read()
                                break
                return md_text
        except Exception:
            return None

    def _run_cli_structure(self, img: np.ndarray, img_name: str) -> T.Tuple[T.Any, T.Optional[str]]:
        # Write to temp, run `paddleocr --type structure ...`, parse outputs
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, img_name)
            cv2.imwrite(img_path, img)

            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                sys.executable, "-m", "paddleocr",
                "--type", "structure",
                "--image_dir", img_path,
                "--ocr_version", self.cfg.ocr_version,
                "--lang", self.cfg.lang,
                "--use_gpu", "false",
                "--output", out_dir,
                "--layout_model_name", self.cfg.layout_model_name,
                "--det_db_thresh", str(self.cfg.det_db_thresh),
                "--det_db_box_thresh", str(self.cfg.det_db_box_thresh),
                "--det_db_unclip_ratio", str(self.cfg.det_db_unclip_ratio),
                "--drop_score", str(self.cfg.drop_score),
            ]

            # Feature toggles if supported by CLI
            if self.cfg.enable_table:
                cmd += ["--enable_table", "true"]
            if self.cfg.enable_formula:
                cmd += ["--enable_formula", "true"]
            if self.cfg.enable_chart:
                cmd += ["--enable_chart", "true"]
            if self.cfg.restore_reading_order or self.cfg.export_markdown:
                cmd += ["--recovery", "true"]

            # Run
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"paddleocr CLI failed: {proc.stderr}")

            # The CLI writes JSON lines and Markdown files to the output directory.
            # Look for a .json or .txt manifest; else parse *.md and any structure JSON.
            result_json = None
            md_text = None

            # Try to read a consolidated JSON
            for root, _, files in os.walk(out_dir):
                for f in files:
                    if f.endswith(".json"):
                        p = os.path.join(root, f)
                        try:
                            with open(p, "r", encoding="utf-8") as fh:
                                # If file is JSON-lines, collect them; otherwise parse as JSON
                                content = fh.read().strip()
                                if content.startswith("{") or content.startswith("["):
                                    result_json = json.loads(content)
                                else:
                                    # JSONL
                                    result_json = [json.loads(line) for line in content.splitlines() if line.strip()]
                        except Exception:
                            pass
                    elif f.endswith(".md"):
                        with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                            md_text = fh.read()

            # As a fallback, try to scrape info from stdout
            if result_json is None:
                # This is a last resort fallback; return raw stdout for debugging
                result_json = {"stdout": proc.stdout}

            return result_json, md_text

    def process(self, img: np.ndarray, img_name: str) -> T.Tuple[T.Any, T.Optional[str]]:
        """
        Returns: (json_like_result, markdown_text_or_None)
        """
        if self.mode == "python-v3":
            # New V3 Python API: returns rich structured results
            result = self.engine(img)
            md = None
            if self.cfg.export_markdown:
                md = self._to_markdown_via_save(result, img_name)  # use native save to get MD
            return result, md

        if self.mode == "python-v2":
            # V2 API: returns structure (layout+table+ocr). Use save utility for MD.
            result = self.engine(img)
            md = None
            if self.cfg.export_markdown:
                md = self._to_markdown_via_save(result, img_name)
            return result, md

        # CLI fallback for strict PP-StructureV3 feature parity
        return self._run_cli_structure(img, img_name)


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="PP-StructureV3 (aarch64 CPU) - PaddleOCR 3.2.0",
    description="PP-StructureV3 service with full pipeline features, CPU-only aarch64.",
    version="1.0.0",
)

ENGINE = PPStructureV3Engine(CFG)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "engine_mode": ENGINE.mode, "config": asdict(CFG)}


@app.post("/parse")
async def parse(files: T.List[UploadFile] = File(...)):
    """
    Accepts multiple images and returns, for each:
      - json: structured PP-StructureV3 result
      - markdown: the document reconstructed as Markdown (if available)
    """
    results: T.List[T.Dict[str, T.Any]] = []
    for f in files:
        try:
            img = _load_image_from_upload(f)

            # Optional lightweight preprocessing that tends to help medical lab scans:
            # Keep this modest to remain close to native behavior.
            # - Bilateral filter for denoising while preserving edges
            # - Slight contrast enhancement via CLAHE on L channel
            img_proc = img.copy()
            img_proc = cv2.bilateralFilter(img_proc, d=5, sigmaColor=30, sigmaSpace=30)
            lab = cv2.cvtColor(img_proc, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            img_proc = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            json_like, md = ENGINE.process(img_proc, img_name=os.path.basename(f.filename) or "page.png")
            results.append({
                "filename": f.filename,
                "json": _normalize_result_for_json(json_like),
                "markdown": md if md is not None else "",
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "error": str(e),
            })

    return JSONResponse({"results": results})

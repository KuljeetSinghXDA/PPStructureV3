from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # OpenCV will be provided by paddleocr extras

# Prefer PPStructureV3, fallback to PPStructure when needed for compatibility
try:
    from paddleocr import PPStructureV3 as _PPStructClass  # type: ignore
except Exception:
    from paddleocr import PPStructure as _PPStructClass  # type: ignore


class PPStructureEngine:
    """
    Thin wrapper around PaddleOCR's PP-Structure pipeline.
    Config is aligned with the 3.x pipeline options; unknown arguments are ignored by fallback.
    """

    def __init__(
        self,
        device: str = "cpu",
        enable_mkldnn: bool = True,
        enable_hpi: bool = False,
        cpu_threads: int = 4,

        # Optional accuracy modules
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,

        # Subpipelines
        use_table_recognition: bool = True,
        use_formula_recognition: bool = False,
        use_chart_recognition: bool = False,

        # Model overrides (names follow 3.x docs)
        layout_detection_model_name: Optional[str] = None,
        text_detection_model_name: Optional[str] = None,
        text_recognition_model_name: Optional[str] = None,
        wired_table_structure_recognition_model_name: Optional[str] = None,
        wireless_table_structure_recognition_model_name: Optional[str] = None,
        table_classification_model_name: Optional[str] = None,
        formula_recognition_model_name: Optional[str] = None,
        chart_recognition_model_name: Optional[str] = None,

        # Thresholds and batch sizes
        layout_threshold: Optional[float] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_rec_score_thresh: Optional[float] = None,
        text_recognition_batch_size: Optional[int] = None,
    ) -> None:
        kwargs: Dict[str, Any] = dict(
            device=device,
            enable_mkldnn=enable_mkldnn,
            enable_hpi=enable_hpi,
            cpu_threads=cpu_threads,

            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,

            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,

            layout_detection_model_name=layout_detection_model_name,
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,
            wired_table_structure_recognition_model_name=wired_table_structure_recognition_model_name,
            wireless_table_structure_recognition_model_name=wireless_table_structure_recognition_model_name,
            table_classification_model_name=table_classification_model_name,
            formula_recognition_model_name=formula_recognition_model_name,
            chart_recognition_model_name=chart_recognition_model_name,

            layout_threshold=layout_threshold,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_rec_score_thresh=text_rec_score_thresh,
            text_recognition_batch_size=text_recognition_batch_size,
        )

        # Drop None-valued keys to avoid surprising overrides
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Remove any deprecated arguments if present
        kwargs.pop("show_log", None)

        # Try to initialize the engine
        self._engine = None
        last_err = None
        try:
            self._engine = _PPStructClass(**kwargs)
        except Exception as e:
            last_err = e
            # Fallback: try minimal arguments
            minimal = dict(device=device)
            if "enable_mkldnn" in kwargs:
                minimal["enable_mkldnn"] = kwargs["enable_mkldnn"]
            try:
                self._engine = _PPStructClass(**minimal)
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize PP-Structure engine: {e2}") from last_err

        self._lock = threading.Lock()

    def infer_image(self, pil_image) -> List[dict]:
        """
        Run structure parsing on a single PIL image (RGB).
        Returns the raw list of items from PP-Structure.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required but not found. Ensure paddleocr extras installed.")
        np_img = np.array(pil_image)
        if np_img.ndim == 3 and np_img.shape[2] == 3:
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            bgr = np_img

        # PP-Structure engines are not always thread-safe; guard inference
        with self._lock:
            result = self._engine(bgr)
        return result

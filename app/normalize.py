from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _bbox_to_xywh(bbox: List[float]) -> List[float]:
    # Paddle returns [x1, y1, x2, y2]; normalize to [x, y, w, h]
    if not bbox or len(bbox) != 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(max(0, x2 - x1)), float(max(0, y2 - y1))]


def normalize_ppstructure_result(
    raw_items: List[Dict[str, Any]],
    page_index: int,
    page_width: int,
    page_height: int,
) -> Dict[str, Any]:
    """
    Produce a normalized page schema while preserving raw engine results.
    Normalized element schema:
    - type: paragraph|title|table|figure|equation|chart|stamp|header|footer|caption|unknown
    - bbox: [x,y,w,h]
    - text: str (for text-like blocks)
    - lines: [ {text, bbox?} ] (when available)
    - table: { html, markdown?, cells? }
    - equation: { latex }
    - chart: { html?, markdown?, cells? }
    - confidence: float? (best-effort)
    """
    elements: List[Dict[str, Any]] = []

    for item in raw_items or []:
        itype = item.get("type") or item.get("layout", {}).get("type") or "unknown"
        itype = str(itype).lower()
        bbox = item.get("bbox") or item.get("bbox3") or item.get("box") or []
        xywh = _bbox_to_xywh(bbox)

        res = item.get("res", {})
        element: Dict[str, Any] = {
            "type": itype,
            "bbox": xywh,
            "confidence": None,
        }

        # Textual blocks (text, title, list, caption, header, footer)
        if itype in ("text", "title", "paragraph", "list", "caption", "header", "footer"):
            # Paddle often returns res as a list of lines: [{'text': '...'}, ...]
            text = ""
            lines = []
            if isinstance(res, list):
                for line in res:
                    t = line.get("text") if isinstance(line, dict) else str(line)
                    if t:
                        lines.append({"text": t})
                text = "\n".join([l["text"] for l in lines])
            elif isinstance(res, dict):
                # Could be {'text': '...'} or {'lines': [...]}
                if "lines" in res and isinstance(res["lines"], list):
                    for line in res["lines"]:
                        t = line.get("text") if isinstance(line, dict) else str(line)
                        if t:
                            lines.append({"text": t})
                    text = "\n".join([l["text"] for l in lines])
                else:
                    t = res.get("text")
                    if isinstance(t, str):
                        text = t
            element["type"] = "title" if itype == "title" else "paragraph"
            element["text"] = text
            if lines:
                element["lines"] = lines

        # Tables (wired/wireless)
        elif itype in ("table", "wired_table", "wireless_table"):
            table_data: Dict[str, Any] = {}
            if isinstance(res, dict):
                if "html" in res:
                    table_data["html"] = res.get("html")
                if "cells" in res:
                    table_data["cells"] = res.get("cells")
                # Some variants expose 'structure' or 'cell_bbox'
                for k in ("structure", "cell_bbox", "cell_box"):
                    if k in res:
                        table_data[k] = res.get(k)
            element["type"] = "table"
            element["table"] = table_data

        # Equations / Formulas
        elif itype in ("equation", "formula", "latex"):
            latex = None
            if isinstance(res, dict):
                latex = res.get("latex") or res.get("text")
            element["type"] = "equation"
            element["equation"] = {"latex": latex} if latex else {}

        # Charts
        elif itype in ("chart", "chart2table"):
            chart_data: Dict[str, Any] = {}
            if isinstance(res, dict):
                # If a table reconstruction is provided
                if "html" in res:
                    chart_data["html"] = res.get("html")
                if "cells" in res:
                    chart_data["cells"] = res.get("cells")
            element["type"] = "chart"
            element["chart"] = chart_data

        # Figures / images / stamps
        elif itype in ("figure", "image", "stamp", "seal"):
            element["type"] = "figure" if itype in ("figure", "image") else "stamp"
            # The PP-Structure result may include an extracted image path under 'img'
            if "img" in item:
                element["image"] = {"path": item["img"]}

        else:
            element["type"] = itype

        elements.append(element)

    return {
        "page_index": page_index,
        "page_width": page_width,
        "page_height": page_height,
        "elements": elements,
        "raw": raw_items,  # preserve raw engine output for completeness
    }

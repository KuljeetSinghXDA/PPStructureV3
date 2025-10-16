from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import pypdfium2 as pdfium
from PIL import Image


def parse_page_range(page_range: str, total_pages: int) -> List[int]:
    """
    Acceptable values:
    - "all"
    - "1-3"
    - "1,2,5"
    - "1-3,6,8-10"
    Page numbers are 1-based in the API. Returns 0-based sorted unique indices.
    """
    page_range = (page_range or "all").strip().lower()
    if page_range in ("all", "*"):
        return list(range(total_pages))
    pages: set[int] = set()
    parts = [p.strip() for p in page_range.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            start = max(1, int(a))
            end = min(total_pages, int(b))
            if start <= end:
                for x in range(start, end + 1):
                    pages.add(x - 1)
        else:
            x = int(part)
            if 1 <= x <= total_pages:
                pages.add(x - 1)
    return sorted(pages)


def render_pdf_to_images(
    pdf_path: str,
    dpi: int = 180,
    page_range: str = "all",
    max_pages: Optional[int] = None,
) -> List[Tuple[Image.Image, int, int, int]]:
    """
    Returns a list of tuples: (PIL.Image RGB, page_index (0-based), width_px, height_px)
    """
    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)
    idxs = parse_page_range(page_range, total_pages)
    if max_pages is not None and max_pages > 0:
        idxs = idxs[:max_pages]

    scale = dpi / 72.0  # PDF points are 72 DPI
    out: List[Tuple[Image.Image, int, int, int]] = []
    for pno in idxs:
        page = pdf.get_page(pno)
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        w, h = pil.size
        out.append((pil, pno, w, h))
        page.close()
    pdf.close()
    return out

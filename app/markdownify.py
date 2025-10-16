from __future__ import annotations

from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

def _html_table_to_markdown(html: str) -> str:
    """
    Convert an HTML table to GitHub-Flavored Markdown.
    Best-effort flattening of rowspan/colspan (cells beyond first are padded as empty).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return html.strip()

    # Build a grid of cells
    grid: List[List[str]] = []
    max_cols = 0
    for row in table.find_all("tr"):
        row_cells: List[str] = []
        for cell in row.find_all(["td", "th"]):
            txt = cell.get_text(separator=" ", strip=True)
            colspan = int(cell.get("colspan", "1") or "1")
            rowspan = int(cell.get("rowspan", "1") or "1")
            # Append cell, then pad for colspan > 1
            row_cells.append(txt)
            for _ in range(colspan - 1):
                row_cells.append("")
            # Rowspan can't be represented in Markdown; we ignore and flatten
        max_cols = max(max_cols, len(row_cells))
        grid.append(row_cells)

    # Normalize column counts
    for r in grid:
        if len(r) < max_cols:
            r += [""] * (max_cols - len(r))

    lines: List[str] = []
    if not grid:
        return ""

    # Header: if the first row seems like a header (th), otherwise synthesize a header
    header_cells = grid[0]
    header = "| " + " | ".join(c or " " for c in header_cells) + " |"
    sep = "| " + " | ".join(["---"] * max_cols) + " |"
    lines.append(header)
    lines.append(sep)
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def pages_to_markdown(
    pages: List[Dict[str, Any]],
    table_format: str = "markdown",  # 'markdown' or 'html'
    include_page_headings: bool = True,
) -> str:
    """
    Convert normalized pages into a single Markdown document.
    """
    out_lines: List[str] = []
    for i, page in enumerate(pages, start=1):
        if include_page_headings:
            out_lines.append(f"## Page {i}")
            out_lines.append("")

        for el in page.get("elements", []):
            etype = el.get("type")
            if etype == "title" and el.get("text"):
                out_lines.append(f"# {el['text']}")
                out_lines.append("")
            elif etype in ("paragraph", "header", "footer", "caption") and el.get("text"):
                out_lines.append(el["text"])
                out_lines.append("")
            elif etype == "equation" and el.get("equation", {}).get("latex"):
                latex = el["equation"]["latex"]
                out_lines.append("$$")
                out_lines.append(latex)
                out_lines.append("$$")
                out_lines.append("")
            elif etype == "table":
                table = el.get("table", {})
                html = table.get("html")
                if html and table_format == "markdown":
                    out_lines.append(_html_table_to_markdown(html))
                    out_lines.append("")
                elif html and table_format == "html":
                    out_lines.append(html.strip())
                    out_lines.append("")
            elif etype == "chart":
                chart = el.get("chart", {})
                html = chart.get("html")
                if html and table_format == "markdown":
                    out_lines.append(_html_table_to_markdown(html))
                    out_lines.append("")
                elif html and table_format == "html":
                    out_lines.append(html.strip())
                    out_lines.append("")
            elif etype in ("figure", "stamp"):
                # Placeholder; user can post-process images via raw/path
                out_lines.append(f"<!-- {etype} at bbox {el.get('bbox')} -->")
                out_lines.append("")
            else:
                # Fallback: if text exists, print it; otherwise comment
                if el.get("text"):
                    out_lines.append(el["text"])
                    out_lines.append("")
                else:
                    out_lines.append(f"<!-- {etype} element -->")
                    out_lines.append("")

    return "\n".join(out_lines).strip()markdownify.py

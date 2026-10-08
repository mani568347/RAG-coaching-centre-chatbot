"""PDF parsing for the CodeCraft Academy knowledge base.

Docling is the main parser: it extracts text, keeps the document structure
(headings vs body vs tables), detects section titles and preserves page
numbers for every element.

pdfplumber is used specifically on pages that contain ruled/fee tables so
that structured rows (and exact characters such as the rupee sign) are
recovered faithfully; if Docling's table export mangles such characters,
the pdfplumber rows replace it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem

from src.config import DOCUMENTS_DIR


@dataclass
class Section:
    """A logical section of one page: heading + ordered lines.

    Lines prefixed with 'TABLE_ROW|' are structured table rows recovered by
    pdfplumber (cells joined with ' | ').
    """

    heading: str
    page: int
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines)


def _clean(text: str) -> str:
    text = text.replace("\u25a0", "₹")  # font-mangled rupee sign seen in some extracts
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _repair_cells(cells: list[str]) -> list[str]:
    """Fix two known extraction artefacts inside one table row:

    1. Rupee glyph: this academy PDF maps the ₹ character to 'n'
       ('n48,000' -> '₹48,000').
    2. Column bleed: a cell like 'g6 Months' whose leading letter is the end
       of the previous cell's truncated word ('...Learnin').
    """
    cells = [_clean(c) if isinstance(c, str) else "" for c in cells]
    # currency glyph first, so a leading 'n'+digits is no longer mistaken
    # for a bleed letter
    cells = [re.sub(r"^n(?=[\d,]+)", "₹", c) for c in cells]
    for i in range(1, len(cells)):
        if cells[i] and cells[i - 1] and cells[i - 1][-1].isalpha():
            m = re.match(r"^([a-z])(\d.*)$", cells[i])
            if m:
                cells[i - 1] += m.group(1)
                cells[i] = m.group(2)
    return cells


def _document_type(filename: str) -> str:
    lowered = filename.lower()
    if "fee" in lowered:
        return "fees"
    if "syllabus" in lowered:
        return "syllabus"
    if "timing" in lowered or "batch" in lowered:
        return "timings"
    if "faculty" in lowered:
        return "faculty"
    if "admission" in lowered or "facilit" in lowered:
        return "admissions_facilities"
    return "general"


def _docling_items(pdf_path: Path):
    """Yield (kind, page_number, text) for every Docling item.

    kind is one of: title, heading, paragraph, table_caption, list, other.
    Page numbers come from Docling's provenance data (1-based).
    """
    opts = PdfPipelineOptions()
    opts.do_ocr = True          # support scanned pages where applicable
    opts.table_structure_options.do_cell_matching = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
        }
    )
    result = converter.convert(str(pdf_path))
    doc = result.document
    if doc is None:
        raise RuntimeError(f"Docling could not parse {pdf_path.name}")

    for item, _level in doc.iterate_items():
        pages = [p.page_no for p in getattr(item, "prov", []) if getattr(p, "page_no", None)]
        page = int(pages[0]) if pages else 1
        class_name = type(item).__name__
        if class_name == "SectionHeaderItem":
            text = _clean(item.text) if hasattr(item, "text") else ""
            yield ("heading", page, text)
        elif class_name == "TitleItem":
            yield ("title", page, _clean(item.text))
        elif class_name == "TableItem":
            yield ("table", page, item)
        elif class_name == "ListItem":
            yield ("list", page, _clean(item.text))
        else:
            text = _clean(getattr(item, "text", ""))
            if text:
                yield ("paragraph", page, text)


def _pdfplumber_pages(pdf_path: Path):
    """Return per-page {text_lines, tables} using pdfplumber."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            text = page.extract_text() or ""
            pages.append(
                {
                    "lines": [_clean(l) for l in text.splitlines() if l.strip()],
                    "tables": [
                        [_repair_cells(row) for row in table if any(row)]
                        for table in tables
                    ],
                }
            )
    return pages


def parse_pdf(pdf_path: Path) -> list[Section]:
    """Parse one PDF into page-aware, heading-aware sections."""
    pages_info = _pdfplumber_pages(pdf_path)

    # Flatten pdfplumber structure per page: heading-like lines and tables
    page_tables: dict[int, list[list[list[str]]]] = {
        i + 1: info["tables"] for i, info in enumerate(pages_info)
    }

    sections: list[Section] = []
    current: Section | None = None
    seen_table_pages: set[int] = set()

    def flush():
        nonlocal current
        if current and current.lines:
            sections.append(current)
        current = None

    for kind, page, payload in _docling_items(pdf_path):
        if kind == "heading":
            # A heading on the same page that repeats the current heading
            # (Docling sometimes re-emits the title) should not split.
            if current and current.page == page and current.heading == payload:
                continue
            flush()
            current = Section(heading=payload, page=page)
        elif kind == "title":
            flush()
            current = Section(heading=payload, page=page)
        elif kind == "table":
            if current is None:
                current = Section(heading=pdf_path.stem.replace("_", " "), page=page)
            rows = page_tables.get(page) or []
            if rows:
                # Use pdfplumber's structured rows for ruled/fee tables:
                # exact cell text (₹ amounts), header first.
                for row in rows[0]:
                    if any(row):
                        current.lines.append("TABLE_ROW|" + " | ".join(row))
                if len(rows) > 1:
                    for extra in rows[1:]:
                        header = " | ".join(c for c in extra[0] if c)
                        if header:
                            current.lines.append(f"Sub-table header: {header}")
                        for row in extra[1:]:
                            if any(row):
                                current.lines.append("TABLE_ROW|" + " | ".join(row))
                seen_table_pages.add(page)
            else:
                md = payload.export_to_markdown() if isinstance(payload, TableItem) else str(payload)
                for line in _clean(md).splitlines():
                    if line.strip("| -") == "":
                        continue
                    cells = _repair_cells(line.strip("|").split("|"))
                    current.lines.append("TABLE_ROW|" + " | ".join(cells))
                seen_table_pages.add(page)
        elif kind in ("paragraph", "list"):
            if current is None:
                current = Section(heading=pdf_path.stem.replace("_", " "), page=page)
            if kind == "list":
                current.lines.append(f"- {payload}")
            else:
                for line in payload.splitlines():
                    if line.strip():
                        current.lines.append(line.strip())

    flush()

    # Attach pdfplumber table rows that Docling missed entirely for a page
    for page, rows in page_tables.items():
        if not rows:
            continue
        if page in seen_table_pages:
            continue
        already = any(l.startswith("TABLE_ROW|") for s in sections if s.page == page for l in s.lines)
        if already:
            continue
        target = next((s for s in sections if s.page == page and not s.lines), None)
        if target is None:
            target = Section(heading=f"Table on page {page}", page=page)
            sections.append(target)
        for row in rows[0]:
            if any(row):
                target.lines.append("TABLE_ROW|" + " | ".join(row))

    # Drop empty sections (Docling title-only pages etc.)
    sections = [s for s in sections if s.lines]
    if not sections:
        raise RuntimeError(f"No readable content extracted from {pdf_path.name}")
    return sections


def parse_all(documents_dir: Path = DOCUMENTS_DIR) -> dict[str, list[Section]]:
    """Parse every PDF in the documents directory. Returns {filename: sections}."""
    pdfs = sorted(documents_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF documents found in {documents_dir}")
    return {p.name: parse_pdf(p) for p in pdfs}

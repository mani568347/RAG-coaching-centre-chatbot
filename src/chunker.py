"""Chunking: ~200 tokens, 0 overlap, section-aware, with contextual prefixes.

Each chunk gets a short context line describing its origin (document, year,
section, course/category when available) prepended before embedding — the
contextual retrieval step. Facts are never invented in the prefix; the
course/category is only added when it literally appears in the chunk.
"""
from dataclasses import dataclass
import re

from src.config import CHUNK_CHARS
from src.pdf_parser import Section, _document_type

CATEGORY_HINTS = [
    "Artificial Intelligence & Machine Learning",
    "Full Stack Web Development",
    "Data Science & Analytics",
    "Cloud & DevOps Fundamentals",
    "Python Programming",
    "AI & ML",
    "Data Science",
    "Full Stack",
    "Cloud & DevOps",
    "Python",
    "Java",
    "Morning",
    "Evening",
    "Weekend",
    "Project Lab",
    "Doubt",
]


@dataclass
class Chunk:
    chunk_id: str
    text: str            # context line + chunk body (what gets embedded)
    body: str            # raw body without the context line
    source: str
    page: int
    section: str
    document_type: str
    token_estimate: int

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "document_type": self.document_type,
            "chunk_id": self.chunk_id,
        }


def _doc_year(filename: str) -> str:
    m = re.search(r"(20\d{2})", filename)
    return m.group(1) if m else ""


def _clean_title(raw: str) -> str:
    return re.sub(r"\s+—\s+.*$", "", raw).strip()


def _detect_category(body: str) -> str | None:
    lowered = body.lower()
    for hint in CATEGORY_HINTS:
        if hint.lower() in lowered:
            return hint
    return None


def _context_line(source: str, section_heading: str, body: str) -> str:
    year = _doc_year(source)
    doc_name = _clean_title(source.replace("_", " ").replace(".pdf", ""))
    doc_name = re.sub(r"\s*20\d{2}\s*$", "", doc_name).strip()
    label = f"{doc_name} ({year})" if year else doc_name
    parts = [f"Context: from the {label} document"]
    if section_heading and section_heading != doc_name:
        parts.append(f", section '{section_heading}'")
    category = _detect_category(body)
    if category:
        parts.append(f"; course/topic in this chunk: {category}")
    return "".join(parts) + "."


def _split_lines(lines: list[str], max_chars: int) -> list[str]:
    """Greedily group prose lines into blocks under max_chars (0 overlap)."""
    blocks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in lines:
        if buf and buf_len + len(line) + 1 > max_chars:
            blocks.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def _table_blocks(header: str, rows: list[str], max_chars: int) -> list[str]:
    """Group table rows into blocks; rows are never split, header repeats."""
    blocks: list[str] = []
    current = [header]
    cur_len = len(header)
    for row in rows:
        if cur_len + len(row) + 1 > max_chars and len(current) > 1:
            blocks.append("\n".join(current))
            current = [header, row]
            cur_len = len(header) + len(row) + 1
        else:
            current.append(row)
            cur_len += len(row) + 1
    blocks.append("\n".join(current))
    return blocks


def _section_blocks(section: Section) -> list[str]:
    """Split one section into body blocks in document order.

    Prose lines are grouped by length; contiguous TABLE_ROW runs become
    header+row blocks so tables stay structured.
    """
    blocks: list[str] = []

    prose_buf: list[str] = []
    table_buf: list[str] = []

    def flush_prose():
        nonlocal prose_buf
        if prose_buf:
            blocks.extend(_split_lines(prose_buf, CHUNK_CHARS))
            prose_buf = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            blocks.extend(_table_blocks(table_buf[0], table_buf[1:], CHUNK_CHARS))
            table_buf = []

    for line in section.lines:
        if line.startswith("TABLE_ROW|"):
            flush_prose()
            table_buf.append(line[len("TABLE_ROW|"):].strip())
        else:
            flush_table()
            prose_buf.append(line)
    flush_prose()
    flush_table()

    # Prepend the section heading to the first block so every chunk of a
    # section is anchored; merge very short trailing blocks.
    if section.heading and blocks:
        blocks[0] = f"{section.heading}\n{blocks[0]}"
    merged: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if merged and len(b) < 60 and not b.startswith("TABLE") and len(merged[-1]) + len(b) < CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n" + b
        else:
            merged.append(b)
    return merged


def chunk_sections(filename: str, sections: list[Section]) -> list[Chunk]:
    doc_type = _document_type(filename)
    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        for body in _section_blocks(section):
            ctx = _context_line(filename, section.heading, body)
            text = f"{ctx}\n\n{body}"
            chunk_id = f"{filename[:-4]}_p{section.page}_c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    body=body,
                    source=filename,
                    page=section.page,
                    section=section.heading or filename,
                    document_type=doc_type,
                    token_estimate=max(1, round(len(text) / 4)),
                )
            )
            idx += 1
    return chunks


def chunk_document_map(doc_map: dict[str, list[Section]]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for filename, sections in doc_map.items():
        all_chunks.extend(chunk_sections(filename, sections))
    return all_chunks

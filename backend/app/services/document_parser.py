"""Extract text and source locations from supported uploaded files."""

from dataclasses import dataclass
import csv
import io
from importlib import import_module
from pathlib import Path
from typing import BinaryIO


class DocumentParseError(ValueError):
    """A file is unsupported, corrupt, or contains no readable text."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    media_type: str
    pages: tuple[ParsedPage, ...]
    source_url: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


def parse_document(source: str | Path | BinaryIO | bytes, filename: str | None = None,
                   content_type: str | None = None) -> ParsedDocument:
    """Parse PDF, delimited data, or text while retaining source numbering."""
    inferred_name = filename or (Path(source).name if isinstance(source, (str, Path)) else "upload")
    suffix = Path(inferred_name).suffix.lower()
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, (str, Path)):
        try:
            data = Path(source).read_bytes()
        except OSError as exc:
            raise DocumentParseError(f"Cannot read {inferred_name}: {exc}") from exc
    else:
        try:
            data = source.read()
        except OSError as exc:
            raise DocumentParseError(f"Cannot read {inferred_name}: {exc}") from exc
    if not isinstance(data, bytes):
        raise DocumentParseError("Uploaded file must provide bytes")

    if suffix == ".pdf" or content_type == "application/pdf" or data.startswith(b"%PDF-"):
        try:
            fitz = import_module("fitz")
            with fitz.open(stream=data, filetype="pdf") as pdf:
                pages = tuple(ParsedPage(i + 1, page.get_text("text").strip())
                              for i, page in enumerate(pdf))
        except ImportError as exc:
            raise DocumentParseError(
                "PDF support requires PyMuPDF; install it with 'pip install pymupdf'"
            ) from exc
        except Exception as exc:
            raise DocumentParseError(f"Unreadable PDF {inferred_name}: {exc}") from exc
        if not any(page.text for page in pages):
            raise DocumentParseError(f"PDF {inferred_name} contains no extractable text")
        return ParsedDocument(inferred_name, "application/pdf", pages)

    is_csv = suffix == ".csv" or content_type in {"text/csv", "application/csv"}
    is_tsv = suffix in {".tsv", ".tab"} or content_type == "text/tab-separated-values"
    if is_csv or is_tsv:
        text = _decode_text(data, inferred_name)
        delimiter = "\t" if is_tsv else _detect_delimiter(text)
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        except csv.Error as exc:
            raise DocumentParseError(f"Could not read delimited file {inferred_name}: {exc}") from exc
        rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            raise DocumentParseError(f"Delimited file {inferred_name} is empty")
        headers = rows[0]
        rendered = [f"Columns: {' | '.join(headers)}"]
        for row_number, row in enumerate(rows[1:], start=1):
            values = [f"{headers[index] if index < len(headers) and headers[index] else f'Column {index + 1}'}: {value}"
                      for index, value in enumerate(row)]
            rendered.append(f"Row {row_number}: " + " | ".join(values))
        if len(rows) == 1:
            rendered.append("No data rows")
        media_type = "text/tab-separated-values" if delimiter == "\t" else "text/csv"
        return ParsedDocument(inferred_name, media_type,
                              (ParsedPage(1, "\n".join(rendered)),))

    if (suffix in {".txt", ".text", ".md", ".markdown"}
            or content_type in {"text/plain", "text/markdown"}):
        text = _decode_text(data, inferred_name).strip()
        if not text:
            raise DocumentParseError(f"Text file {inferred_name} is empty")
        # Treat plain text as a single source page; line and character offsets remain useful.
        return ParsedDocument(inferred_name, "text/plain", (ParsedPage(1, text),))

    raise DocumentParseError(
        f"Unsupported file type for {inferred_name}; expected PDF, CSV/TSV, or a text/Markdown file"
    )


def _decode_text(data: bytes, filename: str) -> str:
    """Decode common text exports, including BOM-marked UTF variants and Windows text."""
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encodings = ("utf-32",)
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError(f"Text file {filename} could not be decoded")


def _detect_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","

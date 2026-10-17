"""Extract text and source locations from supported uploaded files."""

from dataclasses import dataclass
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
    """Parse PDF or plain UTF-8 text, retaining one-based page/source numbering."""
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

    if suffix in {".txt", ".text"} or content_type in {"text/plain", "text/markdown"}:
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentParseError(f"Text file {inferred_name} is not valid UTF-8") from exc
        if not text:
            raise DocumentParseError(f"Text file {inferred_name} is empty")
        # Treat plain text as a single source page; line and character offsets remain useful.
        return ParsedDocument(inferred_name, "text/plain", (ParsedPage(1, text),))

    raise DocumentParseError(f"Unsupported file type for {inferred_name}; expected PDF or plain text")

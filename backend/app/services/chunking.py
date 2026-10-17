"""Split parsed pages into citation-ready overlapping chunks."""

from dataclasses import dataclass
from uuid import uuid4

from app.services.document_parser import ParsedDocument


@dataclass(frozen=True)
class TextChunk:
    id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    source_url: str | None = None


def chunk_document(document: ParsedDocument, document_id: str, *, chunk_size: int = 1200,
                   overlap: int = 180) -> list[TextChunk]:
    """Chunk each page independently; offsets refer to ParsedDocument.text."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    chunks: list[TextChunk] = []
    document_offset = 0
    for page_index, page in enumerate(document.pages):
        text = page.text
        cursor = 0
        while cursor < len(text):
            limit = min(cursor + chunk_size, len(text))
            end = limit
            if limit < len(text):
                # Prefer ending at a nearby word boundary without making tiny chunks.
                boundary = text.rfind(" ", cursor + chunk_size // 2, limit)
                if boundary > cursor:
                    end = boundary
            content = text[cursor:end].strip()
            leading_trim = len(text[cursor:end]) - len(text[cursor:end].lstrip())
            if content:
                start_pos = document_offset + cursor + leading_trim
                chunks.append(TextChunk(str(uuid4()), str(document_id), document.filename,
                                        page.page_number, content, start_pos,
                                        start_pos + len(content), document.source_url))
            if end >= len(text):
                break
            cursor = max(cursor + 1, end - overlap)
        document_offset += len(text)
        if page_index < len(document.pages) - 1:
            document_offset += 2  # ParsedDocument.text joins pages with two newlines.
    return chunks

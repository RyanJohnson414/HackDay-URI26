import pytest

from app.services.chunking import chunk_document
from app.services.document_parser import DocumentParseError, ParsedDocument, ParsedPage, parse_document


def test_plain_text_parser_preserves_page_text_and_media_type():
    parsed = parse_document(b"\xef\xbb\xbf First line\nSecond line ", filename="notes.txt")

    assert parsed.filename == "notes.txt"
    assert parsed.media_type == "text/plain"
    assert len(parsed.pages) == 1
    assert parsed.pages[0].page_number == 1
    assert parsed.text == "First line\nSecond line"


@pytest.mark.parametrize(
    ("data", "filename", "message"),
    [
        (b"", "empty.txt", "is empty"),
        (b"<html></html>", "page.html", "Unsupported file type"),
    ],
)
def test_parser_rejects_invalid_or_unsupported_input(data, filename, message):
    with pytest.raises(DocumentParseError, match=message):
        parse_document(data, filename=filename)


def test_chunking_respects_offsets_page_numbers_and_overlap():
    document = ParsedDocument(
        filename="source.txt",
        media_type="text/plain",
        pages=(ParsedPage(1, "one two three four five six"), ParsedPage(2, "seven eight nine")),
    )

    chunks = chunk_document(document, "doc", chunk_size=12, overlap=3)

    assert len(chunks) > 2
    assert all(chunk.document_id == "doc" for chunk in chunks)
    assert [chunk.page_number for chunk in chunks][-1] == 2
    for chunk in chunks:
        assert chunk.text
        assert chunk.start_char < chunk.end_char
        assert chunk.end_char - chunk.start_char == len(chunk.text)


def test_parser_decodes_windows_text_and_utf16_bom():
    windows_text = parse_document(bytes((99, 97, 102, 233)), filename="notes.txt")
    utf16_text = parse_document("hello".encode("utf-16"), filename="notes.txt")

    assert windows_text.text == "caf" + chr(233)
    assert utf16_text.text == "hello"


def test_csv_parser_preserves_headers_and_rows():
    parsed = parse_document(
        b"Name,Score\nAda,0.91\nLin,0.82\n",
        filename="scores.csv",
        content_type="text/csv",
    )

    assert parsed.media_type == "text/csv"
    assert "Name: Ada | Score: 0.91" in parsed.text
    assert "Name: Lin | Score: 0.82" in parsed.text


def test_tsv_parser_uses_tab_delimiter():
    parsed = parse_document(b"Name\tScore\nAda\t0.91", filename="scores.tsv")

    assert parsed.media_type == "text/tab-separated-values"
    assert "Name: Ada | Score: 0.91" in parsed.text


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunking_rejects_invalid_size_or_overlap(chunk_size, overlap):
    document = ParsedDocument("source.txt", "text/plain", (ParsedPage(1, "text"),))

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(document, "doc", chunk_size=chunk_size, overlap=overlap)

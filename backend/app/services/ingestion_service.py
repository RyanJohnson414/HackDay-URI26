"""Orchestrate parse -> chunk -> embed -> discover -> persist -> topic grouping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import Chunk, Connection as ConnectionRow, Document
from app.services.chunking import TextChunk, chunk_document
from app.services.connection_engine import ConnectionEngine
from app.services.document_parser import DocumentParseError, ParsedDocument, ParsedPage, parse_document
from app.services.embeddings import EmbeddedChunk, EmbeddingService
from app.services.topic_service import rebuild_topics
from app.services.webpage_reader import WebPageReadError, fetch_webpage


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    chunk_count: int
    connection_count: int
    topic_count: int
    related_topic_count: int
    warnings: tuple[str, ...] = ()


class IngestionService:
    """Coordinate services and database writes without changing their algorithms."""

    def __init__(self, *, embedder: EmbeddingService | None = None,
                 connection_engine: ConnectionEngine | None = None):
        self.embedder = embedder or EmbeddingService()
        self.connection_engine = connection_engine or ConnectionEngine()

    def ingest_upload(self, session: Session, *, filename: str, content_type: str,
                      data: bytes) -> IngestionResult:
        document = Document(id=str(uuid4()), filename=filename, title=Path(filename).stem[:512],
                            domain="Uncategorized", tags=[], media_type=content_type,
                            status="processing")
        session.add(document)
        session.commit()
        try:
            parsed = parse_document(data, filename=filename, content_type=content_type)
        except DocumentParseError as exc:
            self._mark_failed(session, document, str(exc))
            raise
        return self._complete(session, document, parsed)

    def ingest_url(self, session: Session, url: str) -> IngestionResult:
        document = Document(id=str(uuid4()), filename=url, title=url[:512], domain="Uncategorized",
                            tags=[], media_type="text/html", source_url=url, status="processing")
        session.add(document)
        session.commit()
        try:
            page = fetch_webpage(url)
        except WebPageReadError as exc:
            self._mark_failed(session, document, str(exc))
            raise
        document.filename = page.title or page.url
        document.source_url = page.url
        document.media_type = page.document.media_type
        document.title = (page.title or page.url)[:512]
        return self._complete(session, document, page.document)

    def ingest_note(self, session: Session, *, title: str, domain: str, content: str,
                    tags: list[str]) -> IngestionResult:
        filename = f"{title.strip()}.txt"[:512]
        document = Document(id=str(uuid4()), filename=filename, title=title.strip()[:512],
                            domain=domain.strip()[:256], tags=list(dict.fromkeys(tags)),
                            media_type="text/plain", status="processing")
        session.add(document)
        session.commit()
        parsed = ParsedDocument(filename=filename, media_type="text/plain",
                                pages=(ParsedPage(1, content.strip()),))
        return self._complete(session, document, parsed)

    def _complete(self, session: Session, document: Document,
                  parsed: ParsedDocument) -> IngestionResult:
        warnings: list[str] = []
        try:
            chunks = chunk_document(parsed, document.id)
            if not chunks:
                raise ValueError("Document produced no text chunks")
            embedded = self.embedder.embed_chunks(chunks)
            document.filename = parsed.filename or document.filename
            document.media_type = parsed.media_type
            document.source_url = parsed.source_url or document.source_url
            for item in embedded:
                chunk = item.chunk
                session.add(Chunk(id=chunk.id, document_id=chunk.document_id,
                                  page_number=chunk.page_number, text=chunk.text,
                                  start_char=chunk.start_char, end_char=chunk.end_char,
                                  embedding=list(item.vector), embedding_model=item.model))
            session.flush()
            session.commit()
        except Exception as exc:
            session.rollback()
            self._mark_failed(session, document, str(exc))
            raise

        connection_count = 0
        # Re-evaluate connections after storing chunks. Existing rows are upserted by
        # unordered chunk pair, so each upload can refresh scores without duplicate edges.
        try:
            all_chunks = session.query(Chunk).order_by(Chunk.id).all()
            documents = {row.id: row for row in session.query(Document).all()}
            embedded_corpus = []
            for row in all_chunks:
                source_document = documents[row.document_id]
                text_chunk = TextChunk(
                    row.id, row.document_id, source_document.filename, row.page_number, row.text,
                    row.start_char, row.end_char, source_document.source_url,
                )
                embedded_corpus.append(EmbeddedChunk(text_chunk, tuple(row.embedding), row.embedding_model))
            connections = self.connection_engine.find_connections(embedded_corpus)
            current_keys = set()
            existing = session.query(ConnectionRow).all()
            existing_by_pair = {tuple(sorted((edge.source_chunk_id, edge.target_chunk_id))): edge
                                for edge in existing}
            for edge in connections:
                key = tuple(sorted((edge.source.chunk_id, edge.target.chunk_id)))
                if key in current_keys:
                    continue
                current_keys.add(key)
                row = existing_by_pair.get(key)
                if row is None:
                    row = ConnectionRow(id=edge.id, source_chunk_id=edge.source.chunk_id,
                                        target_chunk_id=edge.target.chunk_id)
                    session.add(row)
                row.similarity = edge.similarity
                row.strength = edge.strength
                row.weight = edge.weight
                row.title = edge.title
                row.explanation = edge.explanation
                row.shared_concept = edge.shared_concept
                connection_count += 1
            session.commit()
        except Exception as exc:
            session.rollback()
            warnings.append(f"Connection discovery failed: {exc}")

        try:
            topic_count, related_topic_count = rebuild_topics(session)
        except Exception as exc:
            session.rollback()
            topic_count, related_topic_count = 0, 0
            warnings.append(f"Topic grouping failed: {exc}")

        document.status = "complete" if not warnings else "partial"
        document.error = "\n".join(warnings) or None
        session.commit()
        return IngestionResult(document, len(embedded), connection_count,
                               topic_count, related_topic_count, tuple(warnings))

    @staticmethod
    def _mark_failed(session: Session, document: Document, error: str) -> None:
        session.rollback()
        stored = session.get(Document, document.id)
        if stored is not None:
            stored.status = "failed"
            stored.error = error
            session.commit()

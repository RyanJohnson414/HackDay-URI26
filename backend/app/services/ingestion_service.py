"""Orchestrate parse -> chunk -> embed -> discover -> persist -> topic grouping."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import Chunk, Connection as ConnectionRow, Document, Feedback
from app.services.chunking import TextChunk, chunk_document
from app.services.connection_engine import ConnectionEngine
from app.services.document_parser import DocumentParseError, ParsedDocument, ParsedPage, parse_document
from app.services.embeddings import EmbeddedChunk, EmbeddingService
from app.services.gemini_client import GeminiClient
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
                 connection_engine: ConnectionEngine | None = None,
                 gemini_client: GeminiClient | None = None):
        self.gemini = gemini_client or GeminiClient()
        self.embedder = embedder or EmbeddingService()
        self.connection_engine = connection_engine or ConnectionEngine(gemini_client=self.gemini)

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
        try:
            parsed = parse_document(page.text.encode("utf-8"), filename=page.title or page.url,
                                    content_type="text/plain")
        except DocumentParseError as exc:
            self._mark_failed(session, document, str(exc))
            raise
        parsed = replace(parsed, media_type=page.document.media_type, source_url=page.url)
        document.filename = page.title or page.url
        document.source_url = page.url
        document.title = (page.title or page.url)[:512]
        return self._complete(session, document, parsed)

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
        return self._complete(session, document, parsed, preserve_metadata=True)

    def _complete(self, session: Session, document: Document,
                  parsed: ParsedDocument, *, preserve_metadata: bool = False) -> IngestionResult:
        warnings: list[str] = []
        try:
            analysis = self.gemini.analyze_document(
                parsed.filename, [(page.page_number, page.text) for page in parsed.pages]
            )
            if not preserve_metadata:
                document.title = analysis.title[:512]
                document.domain = analysis.domain[:256]
            document.tags = list(dict.fromkeys([
                *(tag.strip() for tag in (document.tags or []) if tag.strip()),
                *(tag.strip() for tag in analysis.tags if tag.strip()),
            ]))[:20]
            document.sections = [section.model_dump() for section in analysis.sections]
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
        # Compare this upload against the corpus, including nearest neighbors that
        # fall below the embedding gate, and upsert model-confirmed edges.
        try:
            all_chunks = session.query(Chunk).order_by(Chunk.id).all()
            documents = {row.id: row for row in session.query(Document).all()}
            current_model = embedded[0].model
            current_dimensions = len(embedded[0].vector)
            if any(len(item.vector) != current_dimensions or item.model != current_model for item in embedded):
                raise ValueError("New document chunks must use one embedding model and vector dimension")

            stale_chunks = [
                row for row in all_chunks
                if row.embedding_model != current_model or len(row.embedding) != current_dimensions
            ]
            if stale_chunks:
                stale_text_chunks = [
                    TextChunk(
                        row.id, row.document_id, documents[row.document_id].filename, row.page_number,
                        row.text, row.start_char, row.end_char, documents[row.document_id].source_url,
                    )
                    for row in stale_chunks
                ]
                refreshed_embeddings = self.embedder.embed_chunks(stale_text_chunks)
                if len(refreshed_embeddings) != len(stale_chunks) or any(
                    item.model != current_model or len(item.vector) != current_dimensions
                    for item in refreshed_embeddings
                ):
                    raise ValueError("Could not refresh older chunks to the active embedding model")
                for row, item in zip(stale_chunks, refreshed_embeddings):
                    row.embedding = list(item.vector)
                    row.embedding_model = item.model
                session.commit()

            embedded_corpus = []
            for row in all_chunks:
                source_document = documents[row.document_id]
                text_chunk = TextChunk(
                    row.id, row.document_id, source_document.filename, row.page_number, row.text,
                    row.start_char, row.end_char, source_document.source_url,
                )
                embedded_corpus.append(EmbeddedChunk(text_chunk, tuple(row.embedding), row.embedding_model))
            contexts = {document.id: _document_context(document) for document in documents.values()}
            connections = self.connection_engine.find_connections(
                embedded_corpus,
                # Recheck the full corpus so the broader candidate and confidence
                # thresholds also apply to already-stored source pairs.
                focus_document_ids=set(documents),
                document_contexts=contexts,
                on_candidate_error=lambda error: warnings.append(
                    f"Connection candidate skipped: {error}"
                ),
            )
            current_keys = set()
            new_document_chunk_ids = {item.chunk.id for item in embedded}
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
                row.discipline_scope = _discipline_scope(
                    documents[edge.source.document_id].domain,
                    documents[edge.target.document_id].domain,
                )
                row.semantic_distance = _semantic_distance(edge.weight)
                if edge.source.chunk_id in new_document_chunk_ids or edge.target.chunk_id in new_document_chunk_ids:
                    connection_count += 1
            focused_chunk_ids = {item.chunk.id for item in embedded_corpus}
            stale_edge_ids = [
                edge.id for edge in existing
                if (
                    edge.source_chunk_id in focused_chunk_ids
                    or edge.target_chunk_id in focused_chunk_ids
                )
                and tuple(sorted((edge.source_chunk_id, edge.target_chunk_id))) not in current_keys
            ]
            if stale_edge_ids:
                session.query(Feedback).filter(
                    Feedback.connection_id.in_(stale_edge_ids)
                ).delete(synchronize_session=False)
                session.query(ConnectionRow).filter(
                    ConnectionRow.id.in_(stale_edge_ids)
                ).delete(synchronize_session=False)
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


def _document_context(document: Document) -> str:
    fields = [
        f"Document title: {document.title}",
        f"Domain: {document.domain}",
        f"Tags: {', '.join(document.tags or [])}",
    ]
    sections = [
        f"{section.get('title', '')}: {section.get('summary', '')}"
        for section in (document.sections or [])
        if section.get("title") or section.get("summary")
    ]
    if sections:
        fields.append("Analyzed sections: " + " | ".join(sections[:10]))
    return "\n".join(fields)


def _discipline_scope(source_domain: str, target_domain: str) -> str:
    source = source_domain.strip().casefold()
    target = target_domain.strip().casefold()
    if not source or not target or "uncategorized" in {source, target}:
        return "unknown"
    return "within_discipline" if source == target else "cross_discipline"


def _semantic_distance(weight: float) -> str:
    if weight >= 0.80:
        return "local"
    if weight >= 0.60:
        return "related"
    return "distant"

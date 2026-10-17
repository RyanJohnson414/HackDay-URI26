"""Load plain-text context-check sources through the normal ingestion pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import select

from app.core.config import BACKEND_DIR
from app.db.models import Chunk, Connection, Document, Feedback
from app.db.session import SessionLocal
from app.services.ingestion_service import IngestionService
from app.services.topic_service import rebuild_topics

logger = logging.getLogger(__name__)
CONTEXT_SOURCE_DIR = BACKEND_DIR / "context_check_sources"


@dataclass(frozen=True)
class StartupSourceReport:
    loaded: int
    already_loaded: int
    failed: tuple[str, ...]


def load_context_sources(
    *,
    session_factory: sessionmaker[Session] = SessionLocal,
    source_dir: Path = CONTEXT_SOURCE_DIR,
    ingestion_service_factory: Callable[[], IngestionService] = IngestionService,
) -> StartupSourceReport:
    """Ingest fixture text on startup; never supply labels, vectors, or graph edges."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Context-check source directory does not exist: {source_dir}")
    sources = sorted(source_dir.glob("*.txt"))
    if not sources:
        raise ValueError(f"No .txt context-check sources found in {source_dir}")

    service = ingestion_service_factory()
    loaded = 0
    already_loaded = 0
    failed: list[str] = []
    with session_factory() as session:
        for source in sources:
            previous = session.query(Document).filter(Document.filename == source.name).all()
            completed = [document for document in previous if document.status == "complete"]
            incomplete = [document for document in previous if document.status != "complete"]
            if completed:
                source_text = source.read_text(encoding="utf-8-sig").strip()
                unchanged = all(_stored_text_matches(session, document, source_text)
                                for document in completed)
                if unchanged:
                    if incomplete:
                        _remove_source(session, incomplete)
                    already_loaded += 1
                    continue
                _remove_source(session, previous)
            elif incomplete:
                _remove_source(session, incomplete)

            try:
                result = service.ingest_upload(
                    session,
                    filename=source.name,
                    content_type="text/plain",
                    data=source.read_bytes(),
                )
                if result.document.status == "complete":
                    loaded += 1
                else:
                    failed.append(source.name)
                    logger.error(
                        "Startup context source %s was only partially ingested: %s",
                        source.name, result.document.error,
                    )
            except Exception:
                session.rollback()
                failed.append(source.name)
                logger.exception("Could not ingest startup context source %s", source.name)

    if failed:
        logger.error("Startup context sources failed ingestion: %s", ", ".join(failed))
    return StartupSourceReport(loaded, already_loaded, tuple(failed))


def _stored_text_matches(session: Session, document: Document, expected: str) -> bool:
    chunks = session.scalars(
        select(Chunk).where(Chunk.document_id == document.id)
        .order_by(Chunk.page_number, Chunk.start_char)
    ).all()
    if not chunks:
        return False
    covered = bytearray(len(expected))
    for chunk in chunks:
        start, end = chunk.start_char, chunk.end_char
        if start < 0 or end > len(expected) or expected[start:end] != chunk.text:
            return False
        for index in range(start, end):
            if not expected[index].isspace():
                covered[index] = 1
    return all(found or character.isspace() for found, character in zip(covered, expected))


def _remove_source(session: Session, documents: list[Document]) -> None:
    document_ids = [document.id for document in documents]
    chunk_ids = [
        chunk_id for (chunk_id,) in session.query(Chunk.id)
        .filter(Chunk.document_id.in_(document_ids)).all()
    ]
    if chunk_ids:
        connection_ids = [
            connection_id for (connection_id,) in session.query(Connection.id).filter(
                (Connection.source_chunk_id.in_(chunk_ids))
                | (Connection.target_chunk_id.in_(chunk_ids))
            ).all()
        ]
        if connection_ids:
            session.query(Feedback).filter(Feedback.connection_id.in_(connection_ids)).delete(
                synchronize_session=False
            )
            session.query(Connection).filter(Connection.id.in_(connection_ids)).delete(
                synchronize_session=False
            )
        session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).delete(synchronize_session=False)
    session.query(Document).filter(Document.id.in_(document_ids)).delete(synchronize_session=False)
    rebuild_topics(session)

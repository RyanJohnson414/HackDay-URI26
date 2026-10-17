from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.db.session import get_db
from app.schemas.document import DocumentResponse, IngestionResponse, NoteCreate, URLDocumentRequest
from app.services.document_parser import DocumentParseError
from app.services.embeddings import EmbeddingError
from app.services.ingestion_service import IngestionService
from app.services.webpage_reader import WebPageReadError

router = APIRouter(prefix="/documents", tags=["documents"])


def _result_response(result) -> IngestionResponse:
    document = DocumentResponse(
        id=result.document.id, filename=result.document.filename,
        title=result.document.title, domain=result.document.domain, tags=result.document.tags,
        media_type=result.document.media_type, source_url=result.document.source_url,
        status=result.document.status, error=result.document.error,
        created_at=result.document.created_at, chunk_count=result.chunk_count,
    )
    return IngestionResponse(document=document, connection_count=result.connection_count,
                             topic_count=result.topic_count,
                             related_topic_count=result.related_topic_count,
                             warnings=list(result.warnings))


@router.post("/upload", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...), session: Session = Depends(get_db)):
    data = file.file.read()
    try:
        result = IngestionService().ingest_upload(
            session, filename=file.filename or "upload", content_type=file.content_type or "",
            data=data,
        )
        return _result_response(result)
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/url", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def ingest_url(payload: URLDocumentRequest, session: Session = Depends(get_db)):
    try:
        result = IngestionService().ingest_url(session, str(payload.url))
        return _result_response(result)
    except WebPageReadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/note", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def create_note(payload: NoteCreate, session: Session = Depends(get_db)):
    try:
        result = IngestionService().ingest_note(
            session, title=payload.title, domain=payload.domain,
            content=payload.content, tags=payload.tags,
        )
        return _result_response(result)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentResponse])
def list_documents(session: Session = Depends(get_db)):
    counts = dict(session.query(Chunk.document_id, func.count(Chunk.id))
                  .group_by(Chunk.document_id).all())
    documents = session.query(Document).order_by(Document.created_at.desc()).all()
    return [DocumentResponse(
        id=document.id, filename=document.filename, media_type=document.media_type,
        title=document.title, domain=document.domain, tags=document.tags,
        source_url=document.source_url, status=document.status, error=document.error,
        created_at=document.created_at, chunk_count=counts.get(document.id, 0),
    ) for document in documents]

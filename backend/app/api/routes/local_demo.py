"""File-backed local demo endpoints used by the website source forms."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.document import IngestionResponse, NoteCreate, URLDocumentRequest
from app.schemas.graph import GraphResponse, TopicSummaryResponse
from app.services.document_parser import DocumentParseError
from app.services.embeddings import EmbeddingError
from app.services.gemini_client import GeminiClientError
from app.services.local_demo_store import get_local_demo_store
from app.services.webpage_reader import WebPageReadError

router = APIRouter(prefix="/local", tags=["local demo"])


@router.get("/graph", response_model=GraphResponse)
def get_local_graph():
    return get_local_demo_store().graph()


@router.post("/documents/upload", response_model=IngestionResponse,
             status_code=status.HTTP_201_CREATED)
def upload_local_document(file: UploadFile = File(...)):
    try:
        return get_local_demo_store().ingest_upload(
            filename=file.filename or "upload", content_type=file.content_type or "",
            data=file.file.read(),
        )
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (EmbeddingError, GeminiClientError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/documents/url", response_model=IngestionResponse,
             status_code=status.HTTP_201_CREATED)
def ingest_local_url(payload: URLDocumentRequest):
    try:
        return get_local_demo_store().ingest_url(str(payload.url))
    except (WebPageReadError, DocumentParseError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (EmbeddingError, GeminiClientError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/documents/note", response_model=IngestionResponse,
             status_code=status.HTTP_201_CREATED)
def create_local_note(payload: NoteCreate):
    try:
        return get_local_demo_store().ingest_note(
            title=payload.title, domain=payload.domain, content=payload.content,
            tags=payload.tags,
        )
    except (EmbeddingError, GeminiClientError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/graph/topics/{topic_id}/summary", response_model=TopicSummaryResponse)
def summarize_local_topic(topic_id: str):
    try:
        return get_local_demo_store().summarize_topic(topic_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeminiClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Chunk, Connection, Document, Topic, TopicRelation
from app.db.session import get_db
from app.schemas.document import DocumentResponse
from app.schemas.graph import GraphChunk, GraphConnection, GraphResponse, GraphTopic, TopicLink, TopicSummaryResponse
from app.services.embeddings import cosine_similarity
from app.services.gemini_client import GeminiClient, GeminiClientError

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/topics/{topic_id}/summary", response_model=TopicSummaryResponse)
def summarize_topic(topic_id: str, session: Session = Depends(get_db)):
    topic = session.get(Topic, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic was not found")
    chunks = (session.query(Chunk).filter(Chunk.topic_id == topic_id)
              .order_by(Chunk.document_id, Chunk.page_number, Chunk.start_char).all())
    if not chunks:
        raise HTTPException(status_code=422, detail="This topic has no source passages to summarize")
    documents = {doc.id: doc for doc in session.query(Document).filter(
        Document.id.in_({chunk.document_id for chunk in chunks})).all()}
    passages = [
        f"Source: {documents[chunk.document_id].title or documents[chunk.document_id].filename}; "
        f"page {chunk.page_number}\n{chunk.text[:1800]}"
        for chunk in chunks if chunk.document_id in documents
    ][:20]
    try:
        result = GeminiClient().summarize_topic(topic.label, topic.keywords, passages)
    except GeminiClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return TopicSummaryResponse(topic_id=topic.id, subject=topic.label, summary=result.summary,
                                key_points=result.key_points,
                                source_count=len({chunk.document_id for chunk in chunks}))


@router.get("", response_model=GraphResponse)
def get_graph(session: Session = Depends(get_db)):
    documents = session.query(Document).order_by(Document.created_at).all()
    document_by_id = {document.id: document for document in documents}
    chunks = session.query(Chunk).order_by(Chunk.document_id, Chunk.page_number, Chunk.start_char).all()
    topics = session.query(Topic).order_by(Topic.label).all()
    topic_by_id = {topic.id: topic for topic in topics}
    connections = session.query(Connection).order_by(Connection.weight.desc()).all()
    topic_members: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.topic_id:
            topic_members.setdefault(chunk.topic_id, []).append(chunk.id)
    return GraphResponse(
        documents=[DocumentResponse(
            id=document.id, filename=document.filename, media_type=document.media_type,
            title=document.title, domain=document.domain, tags=document.tags,
            source_url=document.source_url, status=document.status, error=document.error,
            created_at=document.created_at,
            chunk_count=sum(chunk.document_id == document.id for chunk in chunks),
        ) for document in documents],
        chunks=[GraphChunk(
            id=chunk.id, document_id=chunk.document_id, topic_id=chunk.topic_id,
            page_number=chunk.page_number, text=chunk.text, start_char=chunk.start_char,
            end_char=chunk.end_char, source_url=document_by_id[chunk.document_id].source_url,
            centroid_distance=(1 - cosine_similarity(chunk.embedding, topic_by_id[chunk.topic_id].centroid)
                               if chunk.topic_id in topic_by_id else None),
        ) for chunk in chunks],
        connections=[GraphConnection(
            id=edge.id, source_chunk_id=edge.source_chunk_id,
            target_chunk_id=edge.target_chunk_id, similarity=edge.similarity,
            strength=edge.strength, weight=edge.weight, title=edge.title,
            explanation=edge.explanation, shared_concept=edge.shared_concept,
        ) for edge in connections],
        topics=[GraphTopic(id=topic.id, label=topic.label, keywords=topic.keywords,
                           chunk_ids=topic_members.get(topic.id, [])) for topic in topics],
        related_topics=[TopicLink(source_topic_id=link.source_topic_id,
                                  target_topic_id=link.target_topic_id,
                                  similarity=link.similarity)
                        for link in session.query(TopicRelation).all()],
    )

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Chunk, Connection, Feedback
from app.db.session import get_db
from app.schemas.discovery import DiscoveryResponse, FeedbackCreate, FeedbackResponse

router = APIRouter(tags=["discoveries"])


@router.get("/discoveries", response_model=list[DiscoveryResponse])
def list_discoveries(limit: int = 100, session: Session = Depends(get_db)):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    rows = session.query(Connection).order_by(Connection.weight.desc()).limit(limit).all()
    return [DiscoveryResponse(
        id=row.id, source_chunk_id=row.source_chunk_id, target_chunk_id=row.target_chunk_id,
        title=row.title, explanation=row.explanation, shared_concept=row.shared_concept,
        similarity=row.similarity, strength=row.strength, weight=row.weight,
    ) for row in rows]


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def create_feedback(payload: FeedbackCreate, session: Session = Depends(get_db)):
    if session.get(Connection, payload.connection_id) is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    row = Feedback(id=str(uuid4()), connection_id=payload.connection_id,
                   rating=payload.rating, comment=payload.comment)
    session.add(row)
    session.commit()
    session.refresh(row)
    return FeedbackResponse(id=row.id, connection_id=row.connection_id,
                            rating=row.rating, comment=row.comment, created_at=row.created_at)


@router.get("/feedback", response_model=list[FeedbackResponse])
def list_feedback(connection_id: str | None = None, limit: int = 100,
                  session: Session = Depends(get_db)):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    query = session.query(Feedback)
    if connection_id:
        query = query.filter(Feedback.connection_id == connection_id)
    rows = query.order_by(Feedback.created_at.desc()).limit(limit).all()
    return [FeedbackResponse(id=row.id, connection_id=row.connection_id,
                             rating=row.rating, comment=row.comment,
                             created_at=row.created_at) for row in rows]

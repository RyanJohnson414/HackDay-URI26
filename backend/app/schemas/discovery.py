from datetime import datetime

from pydantic import BaseModel, Field


class DiscoveryResponse(BaseModel):
    id: str
    source_chunk_id: str
    target_chunk_id: str
    title: str
    explanation: str
    shared_concept: str
    similarity: float
    strength: float
    weight: float


class FeedbackCreate(BaseModel):
    connection_id: str
    rating: int = Field(ge=1, le=5, description="1 is poor, 5 is useful")
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: str
    connection_id: str
    rating: int
    comment: str | None
    created_at: datetime

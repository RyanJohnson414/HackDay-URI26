from pydantic import BaseModel

from app.schemas.document import DocumentResponse


class GraphChunk(BaseModel):
    id: str
    document_id: str
    topic_id: str | None
    page_number: int
    text: str
    start_char: int
    end_char: int
    source_url: str | None
    centroid_distance: float | None


class GraphConnection(BaseModel):
    id: str
    source_chunk_id: str
    target_chunk_id: str
    similarity: float
    strength: float
    weight: float
    title: str
    explanation: str
    shared_concept: str


class GraphTopic(BaseModel):
    id: str
    label: str
    keywords: list[str]
    chunk_ids: list[str]


class TopicLink(BaseModel):
    source_topic_id: str
    target_topic_id: str
    similarity: float


class TopicSummaryResponse(BaseModel):
    topic_id: str
    subject: str
    summary: str
    key_points: list[str]
    source_count: int


class GraphResponse(BaseModel):
    documents: list[DocumentResponse]
    chunks: list[GraphChunk]
    connections: list[GraphConnection]
    topics: list[GraphTopic]
    related_topics: list[TopicLink]

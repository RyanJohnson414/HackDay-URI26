from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class DocumentSectionResponse(BaseModel):
    title: str
    summary: str
    page_start: int
    page_end: int


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    title: str
    domain: str
    tags: list[str]
    sections: list[DocumentSectionResponse] = Field(default_factory=list)
    media_type: str
    source_url: str | None
    status: str
    error: str | None
    created_at: datetime
    chunk_count: int = 0


class URLDocumentRequest(BaseModel):
    url: HttpUrl


class NoteCreate(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    domain: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=200_000)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("title", "domain", "content")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, values: list[str]) -> list[str]:
        cleaned = [tag.strip() for tag in values if tag.strip()]
        if any(len(tag) > 80 for tag in cleaned):
            raise ValueError("Tags must be 80 characters or fewer")
        return list(dict.fromkeys(cleaned))


class IngestionResponse(BaseModel):
    document: DocumentResponse
    connection_count: int
    topic_count: int
    related_topic_count: int
    warnings: list[str] = Field(default_factory=list)

"""Embedding generation and vector helpers (independent of the explanation model)."""

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from app.core.config import get_settings
from app.services.chunking import TextChunk


class EmbeddingError(RuntimeError):
    """Embedding configuration or provider failure."""


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: TextChunk
    vector: tuple[float, ...]
    model: str


class EmbeddingService:
    def __init__(self, *, client=None, model: str | None = None):
        self.model = model or get_settings().embedding_model
        self._client = client

    def _get_client(self):
        if self._client is None:
            settings = get_settings()
            if not settings.gemini_api_key:
                raise EmbeddingError("GEMINI_API_KEY is required to create embeddings")
            try:
                from google import genai
                self._client = genai.Client(api_key=settings.gemini_api_key)
            except Exception as exc:
                raise EmbeddingError(f"Could not initialize embedding client: {exc}") from exc
        return self._client

    def embed_text(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        try:
            dimensions = get_settings().embedding_dimensions
            response = self._get_client().models.embed_content(
                model=self.model,
                contents=text,
                config={"output_dimensionality": dimensions},
            )
            values = response.embeddings[0].values
            vector = tuple(float(value) for value in values)
            if not vector or not all(map(lambda value: value == value and abs(value) != float("inf"), vector)):
                raise ValueError("provider returned an empty or non-finite vector")
            return vector
        except Exception as exc:
            if isinstance(exc, EmbeddingError):
                raise
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(chunk, self.embed_text(chunk.text), self.model) for chunk in chunks]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must have the same non-zero dimension")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

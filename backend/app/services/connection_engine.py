"""Rank cross-document chunk pairs and validate them with the configured Gemma model."""

from dataclasses import dataclass
from uuid import uuid4

from app.services.embeddings import EmbeddedChunk, cosine_similarity
from app.services.gemini_client import ConnectionAssessment, GeminiClient


class ConnectionEngineError(RuntimeError):
    """Candidate generation or evaluation failed."""


@dataclass(frozen=True)
class ChunkReference:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    start_char: int
    end_char: int
    source_url: str | None = None


@dataclass(frozen=True)
class Connection:
    id: str
    source: ChunkReference
    target: ChunkReference
    similarity: float
    strength: float
    weight: float
    title: str
    explanation: str
    shared_concept: str


@dataclass(frozen=True)
class _Candidate:
    left: EmbeddedChunk
    right: EmbeddedChunk
    similarity: float


def _reference(item: EmbeddedChunk) -> ChunkReference:
    chunk = item.chunk
    return ChunkReference(chunk.id, chunk.document_id, chunk.document_name,
                          chunk.page_number, chunk.start_char, chunk.end_char, chunk.source_url)


class ConnectionEngine:
    MIN_CONNECTION_STRENGTH = 0.75

    def __init__(self, *, gemini_client: GeminiClient | None = None):
        self.gemini = gemini_client or GeminiClient()

    def find_connections(self, embedded_chunks: list[EmbeddedChunk], *,
                         similarity_threshold: float = 0.35,
                         max_candidates: int = 100,
                         max_connections: int = 50) -> list[Connection]:
        """Return evaluated cross-document edges, each grounded in two real chunk IDs."""
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if max_candidates < 0 or max_connections < 0:
            raise ValueError("candidate and connection limits cannot be negative")
        candidates: list[_Candidate] = []
        for index, left in enumerate(embedded_chunks):
            for right in embedded_chunks[index + 1:]:
                if left.chunk.document_id == right.chunk.document_id:
                    continue
                try:
                    similarity = cosine_similarity(left.vector, right.vector)
                except ValueError as exc:
                    raise ConnectionEngineError(f"Invalid vectors for chunks {left.chunk.id} and {right.chunk.id}: {exc}") from exc
                # Negative and weak semantic matches are not sent to the language model.
                if similarity >= similarity_threshold:
                    candidates.append(_Candidate(left, right, similarity))
        candidates.sort(key=lambda item: item.similarity, reverse=True)

        results: list[Connection] = []
        for candidate in candidates[:max_candidates]:
            try:
                assessment: ConnectionAssessment = self.gemini.assess_connection(
                    candidate.left.chunk.text, candidate.right.chunk.text)
            except Exception as exc:
                raise ConnectionEngineError(
                    f"Could not assess chunks {candidate.left.chunk.id} and {candidate.right.chunk.id}: {exc}"
                ) from exc
            # Keep only model-confirmed high-confidence connections. The threshold is
            # strict so a score of exactly 0.75 does not count as "above 0.75".
            if not assessment.relevant or assessment.strength <= self.MIN_CONNECTION_STRENGTH:
                continue
            # Blend semantic retrieval and model judgment so neither alone sets the edge weight.
            weight = max(0.0, min(1.0, 0.4 * candidate.similarity + 0.6 * assessment.strength))
            results.append(Connection(str(uuid4()), _reference(candidate.left), _reference(candidate.right),
                                      candidate.similarity, assessment.strength, weight,
                                      assessment.title, assessment.explanation, assessment.shared_concept))
        results.sort(key=lambda item: item.weight, reverse=True)
        return results[:max_connections]

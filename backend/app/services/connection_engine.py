"""Rank cross-document chunk pairs and validate them with the configured Gemma model."""

from dataclasses import dataclass
import re
from collections.abc import Callable
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
    retrieval_score: float


def _reference(item: EmbeddedChunk) -> ChunkReference:
    chunk = item.chunk
    return ChunkReference(chunk.id, chunk.document_id, chunk.document_name,
                          chunk.page_number, chunk.start_char, chunk.end_char, chunk.source_url)


class ConnectionEngine:
    MIN_CONNECTION_STRENGTH = 0.50

    def __init__(self, *, gemini_client: GeminiClient | None = None):
        self.gemini = gemini_client or GeminiClient()

    def find_connections(self, embedded_chunks: list[EmbeddedChunk], *,
                         similarity_threshold: float = 0.01,
                         max_candidates: int | None = None,
                         max_connections: int | None = None,
                         focus_document_ids: set[str] | None = None,
                         document_contexts: dict[str, str] | None = None,
                         on_candidate_error: Callable[[str], None] | None = None) -> list[Connection]:
        """Return model-validated cross-document edges grounded in real chunk IDs.

        Focused documents also receive a small nearest-neighbor review below the
        embedding threshold so Gemma can confirm relationships vectors miss.
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if ((max_candidates is not None and max_candidates < 0)
                or (max_connections is not None and max_connections < 0)):
            raise ValueError("candidate and connection limits cannot be negative")
        candidates: list[_Candidate] = []
        focused_candidates: dict[str, list[_Candidate]] = {}
        for index, left in enumerate(embedded_chunks):
            for right in embedded_chunks[index + 1:]:
                if left.chunk.document_id == right.chunk.document_id:
                    continue
                is_focused_pair = bool(focus_document_ids) and (
                    left.chunk.document_id in focus_document_ids
                    or right.chunk.document_id in focus_document_ids
                )
                if focus_document_ids and not is_focused_pair:
                    continue
                try:
                    similarity = cosine_similarity(left.vector, right.vector)
                    retrieval_score = similarity
                except ValueError:
                    # Old and new embeddings may have different dimensions after a
                    # model/configuration change; Gemma can still assess their text.
                    similarity = 0.0
                    retrieval_score = _lexical_similarity(left.chunk.text, right.chunk.text)

                candidate = _Candidate(left, right, similarity, retrieval_score)
                if similarity >= similarity_threshold and (not focus_document_ids or is_focused_pair):
                    candidates.append(candidate)
                if is_focused_pair:
                    focused = left if left.chunk.document_id in focus_document_ids else right
                    focused_candidates.setdefault(focused.chunk.id, []).append(candidate)

        if focus_document_ids:
            selected = {
                (item.left.chunk.id, item.right.chunk.id): item
                for item in candidates
            }
            for neighbors in focused_candidates.values():
                neighbors.sort(key=lambda item: (
                    item.retrieval_score, item.left.chunk.id, item.right.chunk.id
                ), reverse=True)
                for item in neighbors[:5]:
                    selected[(item.left.chunk.id, item.right.chunk.id)] = item
            candidates = list(selected.values())

        candidates.sort(key=lambda item: (
            item.retrieval_score, item.left.chunk.id, item.right.chunk.id
        ), reverse=True)

        results: list[Connection] = []
        candidates_to_assess = candidates if max_candidates is None else candidates[:max_candidates]
        for candidate in candidates_to_assess:
            try:
                assessment: ConnectionAssessment = self.gemini.assess_connection(
                    _with_context(candidate.left, document_contexts),
                    _with_context(candidate.right, document_contexts),
                )
            except Exception as exc:
                message = (
                    f"Could not assess chunks {candidate.left.chunk.id} "
                    f"and {candidate.right.chunk.id}: {exc}"
                )
                if on_candidate_error is None:
                    raise ConnectionEngineError(message) from exc
                on_candidate_error(message)
                continue
            # Discard generic/weak matches; the model score is calibrated to 0..1.
            if not assessment.relevant or assessment.strength < self.MIN_CONNECTION_STRENGTH:
                continue
            weight = max(0.0, min(1.0, 0.35 * candidate.similarity + 0.65 * assessment.strength))
            explanation = assessment.explanation
            if assessment.evidence_a:
                explanation += f"\nSource A evidence: \"{assessment.evidence_a}\""
            if assessment.evidence_b:
                explanation += f"\nSource B evidence: \"{assessment.evidence_b}\""
            results.append(Connection(str(uuid4()), _reference(candidate.left), _reference(candidate.right),
                                      candidate.similarity, assessment.strength, weight,
                                      assessment.title, explanation, assessment.shared_concept))
        results.sort(key=lambda item: item.weight, reverse=True)
        return results if max_connections is None else results[:max_connections]


def _lexical_similarity(left: str, right: str) -> float:
    """Rank incomparable vectors by shared terms so relevant text reaches Gemma."""
    left_terms = set(re.findall(r"[a-z0-9]{3,}", left.casefold()))
    right_terms = set(re.findall(r"[a-z0-9]{3,}", right.casefold()))
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 0.0


def _with_context(item: EmbeddedChunk, contexts: dict[str, str] | None) -> str:
    passage = item.chunk.text
    context = contexts.get(item.chunk.document_id, "").strip() if contexts else ""
    if not context:
        return passage
    return f"{context}\nSOURCE PASSAGE:\n{passage}"

"""Focused tests for real, cross-document connection discovery."""

from collections.abc import Sequence

import pytest

from app.services.chunking import TextChunk
from app.services.connection_engine import ConnectionEngine, ConnectionEngineError
from app.services.embeddings import EmbeddedChunk
from app.services.gemini_client import ConnectionAssessment


def embedded(chunk_id: str, document_id: str, vector: Sequence[float], *, text: str | None = None):
    chunk = TextChunk(
        id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.txt",
        page_number=2,
        text=text or f"Passage {chunk_id}",
        start_char=10,
        end_char=25,
        source_url=None,
    )
    return EmbeddedChunk(chunk=chunk, vector=tuple(vector), model="test-embedding")


def assessment(*, relevant: bool = True, strength: float = 0.8, title: str = "Shared pattern"):
    return ConnectionAssessment(
        relevant=relevant,
        strength=strength,
        title=title,
        explanation="Both passages describe the same useful underlying pattern.",
        shared_concept="feedback loops",
    )


class FakeGemini:
    def __init__(self, results=(), *, error: Exception | None = None):
        self.results = list(results)
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def assess_connection(self, text_a: str, text_b: str):
        self.calls.append((text_a, text_b))
        if self.error:
            raise self.error
        return self.results.pop(0)


def test_ignores_same_document_and_below_threshold_pairs():
    gemini = FakeGemini([assessment()])
    engine = ConnectionEngine(gemini_client=gemini)
    chunks = [
        embedded("a1", "doc-a", [1, 0]),
        embedded("a2", "doc-a", [0.8, 0.6]),
        embedded("b1", "doc-b", [0, 1]),
    ]

    result = engine.find_connections(chunks, similarity_threshold=0.7)

    assert result == []
    assert gemini.calls == []


def test_connection_has_real_cross_document_references_and_blended_weight():
    gemini = FakeGemini([assessment(strength=0.8)])
    engine = ConnectionEngine(gemini_client=gemini)
    left = embedded("chunk-a", "doc-a", [1, 0])
    right = embedded("chunk-b", "doc-b", [0.8, 0.6])

    result = engine.find_connections([left, right], similarity_threshold=0.7)

    assert len(result) == 1
    connection = result[0]
    assert connection.source.chunk_id == "chunk-a"
    assert connection.target.chunk_id == "chunk-b"
    assert connection.source.document_id != connection.target.document_id
    assert (connection.source.page_number, connection.source.start_char, connection.source.end_char) == (2, 10, 25)
    assert connection.similarity == pytest.approx(0.8)
    assert connection.weight == pytest.approx(0.4 * 0.8 + 0.6 * 0.8)


def test_connection_requires_gemini_strength_strictly_above_point_seventy_five():
    gemini = FakeGemini([assessment(strength=0.75)])
    engine = ConnectionEngine(gemini_client=gemini)

    result = engine.find_connections(
        [embedded("a", "doc-a", [1, 0]), embedded("b", "doc-b", [0.8, 0.6])],
        similarity_threshold=0.7,
    )

    assert result == []
    assert len(gemini.calls) == 1


def test_irrelevant_model_assessments_are_not_returned():
    gemini = FakeGemini([assessment(relevant=False, strength=0.0)])
    engine = ConnectionEngine(gemini_client=gemini)

    result = engine.find_connections(
        [embedded("a", "doc-a", [1, 0]), embedded("b", "doc-b", [0.8, 0.6])],
        similarity_threshold=0.7,
    )

    assert result == []
    assert len(gemini.calls) == 1


def test_candidate_and_result_limits_are_respected():
    gemini = FakeGemini([assessment(strength=0.7), assessment(strength=0.9)])
    engine = ConnectionEngine(gemini_client=gemini)
    chunks = [
        embedded("a", "doc-a", [1, 0]),
        embedded("b", "doc-b", [0.9, 0.435889894]),
        embedded("c", "doc-c", [0.8, -0.6]),
    ]

    result = engine.find_connections(chunks, similarity_threshold=0.7, max_candidates=2, max_connections=1)

    assert len(gemini.calls) == 2
    assert len(result) == 1
    assert result[0].target.chunk_id == "c"
    assert result[0].weight == pytest.approx(0.4 * 0.8 + 0.6 * 0.9)


def test_default_limits_allow_more_than_forty_candidates_and_twenty_connections():
    pair_count = 55
    gemini = FakeGemini([assessment() for _ in range(pair_count)])
    engine = ConnectionEngine(gemini_client=gemini)
    chunks = [
        embedded(f"chunk-{index}", f"doc-{index}", [1, 0])
        for index in range(11)
    ]

    result = engine.find_connections(chunks)

    assert len(gemini.calls) == pair_count
    assert len(result) == 50


def test_invalid_vector_dimensions_are_reported_as_engine_errors():
    engine = ConnectionEngine(gemini_client=FakeGemini())

    with pytest.raises(ConnectionEngineError, match="Invalid vectors"):
        engine.find_connections(
            [embedded("a", "doc-a", [1, 0]), embedded("b", "doc-b", [1, 0, 0])]
        )


def test_model_failures_are_reported_with_candidate_context():
    engine = ConnectionEngine(gemini_client=FakeGemini(error=RuntimeError("provider unavailable")))

    with pytest.raises(ConnectionEngineError, match="chunk-a.*chunk-b.*provider unavailable"):
        engine.find_connections(
            [embedded("chunk-a", "doc-a", [1, 0]), embedded("chunk-b", "doc-b", [0.8, 0.6])],
            similarity_threshold=0.7,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_similarity_threshold_must_be_in_unit_interval(threshold: float):
    engine = ConnectionEngine(gemini_client=FakeGemini())

    with pytest.raises(ValueError, match="similarity_threshold"):
        engine.find_connections([], similarity_threshold=threshold)

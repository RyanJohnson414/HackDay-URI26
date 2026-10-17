"""Group related chunks into topics and link semantically similar topic groups."""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter

from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, Topic, TopicRelation
from app.services.embeddings import cosine_similarity

_TOPIC_NAMESPACE = uuid.UUID("d8977a4f-4369-4883-bf76-6548f4af8458")
_STOP_WORDS = set("""about after again against all also among an and any are as at be because been
before being between both but by can could did do does each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself may me more most
must my myself no nor not of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves use used using based research study studies result results
shows show may might one two also new paper document data""".split())


def _cosine(left: list[float], right: list[float]) -> float:
    try:
        return cosine_similarity(left, right)
    except ValueError as exc:
        raise ValueError(f"Topic vectors must have a common non-zero dimension: {exc}") from exc


def _terms(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text)
            if word.lower() not in _STOP_WORDS]


def _topic_name(chunks: list[Chunk], corpus_frequency: Counter[str], corpus_size: int,
                source_title: str | None) -> tuple[str, list[str]]:
    frequencies = Counter(word for chunk in chunks for word in _terms(chunk.text))
    ranked = sorted(frequencies, key=lambda word: (
        frequencies[word] * math.log((corpus_size + 1) / (corpus_frequency[word] + 1)), word
    ), reverse=True)
    keywords = ranked[:3]
    label = source_title or " · ".join(word.capitalize() for word in keywords) or "General topic"
    return label[:256], keywords


def _unique_topic_label(label: str, used_labels: set[str]) -> str:
    base = label.strip() or "General topic"
    candidate = base
    suffix = 2
    while candidate.casefold() in used_labels:
        marker = f" ({suffix})"
        candidate = f"{base[:256 - len(marker)]}{marker}"
        suffix += 1
    used_labels.add(candidate.casefold())
    return candidate


def rebuild_topics(session: Session, *, topic_similarity: float = 0.72,
                   related_similarity: float = 0.58) -> tuple[int, int]:
    """Recompute primary topic groups and cross-links from all persisted chunk vectors.

    Chunks join a topic when connected by pairwise cosine similarity at or above
    ``topic_similarity``. Separate topics are linked when their centroid similarity
    reaches ``related_similarity``.
    """
    if not 0 <= topic_similarity <= 1 or not -1 <= related_similarity <= 1:
        raise ValueError("Topic similarity thresholds are outside their valid ranges")
    chunks = session.query(Chunk).order_by(Chunk.id).all()
    if not chunks:
        session.query(TopicRelation).delete(synchronize_session=False)
        session.query(Topic).delete(synchronize_session=False)
        session.commit()
        return 0, 0
    if any(not chunk.embedding for chunk in chunks):
        raise ValueError("All persisted chunks must have non-empty embeddings for topic grouping")
    # Connected components give a lightweight, deterministic grouping for the MVP corpus size.
    parents = list(range(len(chunks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parents[root_r] = root_l

    for left in range(len(chunks)):
        for right in range(left + 1, len(chunks)):
            if len(chunks[left].embedding) != len(chunks[right].embedding):
                continue
            if _cosine(chunks[left].embedding, chunks[right].embedding) >= topic_similarity:
                union(left, right)
    groups: dict[int, list[Chunk]] = {}
    for index, chunk in enumerate(chunks):
        groups.setdefault(find(index), []).append(chunk)

    all_term_counts = [Counter(set(_terms(chunk.text))) for chunk in chunks]
    corpus_frequency: Counter[str] = Counter()
    for counts in all_term_counts:
        corpus_frequency.update(counts)

    documents = {document.id: document for document in session.query(Document).all()}
    session.query(TopicRelation).delete(synchronize_session=False)
    for chunk in chunks:
        chunk.topic_id = None
    session.flush()
    session.query(Topic).delete(synchronize_session=False)
    topic_rows: list[Topic] = []
    chunk_assignments: list[tuple[Chunk, str]] = []
    used_labels: set[str] = set()
    for group in groups.values():
        members = sorted(group, key=lambda chunk: chunk.id)
        dimension = len(members[0].embedding)
        centroid = [sum(chunk.embedding[index] for chunk in members) / len(members)
                    for index in range(dimension)]
        topic_id = str(uuid.uuid5(_TOPIC_NAMESPACE, ":".join(chunk.id for chunk in members)))
        member_documents = {chunk.document_id for chunk in members}
        source_title = None
        if len(member_documents) == 1:
            source_title = documents.get(next(iter(member_documents)), None)
            source_title = source_title.title if source_title else None
        label, keywords = _topic_name(members, corpus_frequency, len(chunks), source_title)
        label = _unique_topic_label(label, used_labels)
        topic = Topic(id=topic_id, label=label, keywords=keywords, centroid=centroid)
        topic_rows.append(topic)
        session.add(topic)
        for chunk in members:
            chunk_assignments.append((chunk, topic_id))
    session.flush()
    for chunk, topic_id in chunk_assignments:
        chunk.topic_id = topic_id
    session.flush()

    relation_count = 0
    for index, left in enumerate(topic_rows):
        for right in topic_rows[index + 1:]:
            if len(left.centroid) != len(right.centroid):
                continue
            similarity = _cosine(left.centroid, right.centroid)
            if similarity >= related_similarity:
                session.add(TopicRelation(source_topic_id=left.id, target_topic_id=right.id,
                                          similarity=similarity))
                relation_count += 1
    session.commit()
    return len(topic_rows), relation_count

import pytest
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, Topic, TopicRelation
from app.services.topic_service import rebuild_topics


def add_document(session: Session, document_id: str, filename: str) -> None:
    session.add(Document(
        id=document_id, filename=filename, title=filename, domain="Test",
        tags=[], media_type="text/plain", status="complete",
    ))


def add_chunk(session: Session, chunk_id: str, document_id: str, vector: list[float]) -> None:
    session.add(Chunk(
        id=chunk_id, document_id=document_id, page_number=1, text=f"Chunk {chunk_id}",
        start_char=0, end_char=10, embedding=vector, embedding_model="test",
    ))


def test_rebuild_groups_similar_chunks_and_links_related_topics(empty_schema):
    engine = empty_schema
    with Session(engine) as session:
        add_document(session, "doc-a", "biology.txt")
        add_document(session, "doc-b", "networks.txt")
        add_document(session, "doc-c", "astronomy.txt")
        session.flush()
        add_chunk(session, "chunk-a", "doc-a", [1.0, 0.0])
        add_chunk(session, "chunk-b", "doc-b", [0.8, 0.6])
        add_chunk(session, "chunk-c", "doc-c", [0.0, 1.0])
        session.commit()

        topic_count, relation_count = rebuild_topics(session, related_similarity=0.3)

        chunks = session.query(Chunk).order_by(Chunk.id).all()
        topics = session.query(Topic).all()
        relations = session.query(TopicRelation).all()
        assert (topic_count, relation_count) == (2, 1)
        assert chunks[0].topic_id == chunks[1].topic_id
        assert chunks[2].topic_id != chunks[0].topic_id
        assert len(topics) == 2
        assert relations[0].similarity == pytest.approx(0.316227766)


def test_rebuild_clears_topics_when_no_chunks(empty_schema):
    with Session(empty_schema) as session:
        session.add(Topic(id="orphan", label="Old", keywords=[], centroid=[1.0]))
        session.commit()

        assert rebuild_topics(session) == (0, 0)
        assert session.query(Topic).count() == 0
        assert session.query(TopicRelation).count() == 0


def test_rebuild_keeps_topics_for_incompatible_embedding_dimensions(empty_schema):
    with Session(empty_schema) as session:
        add_document(session, "doc-a", "a.txt")
        add_document(session, "doc-b", "b.txt")
        session.flush()
        add_chunk(session, "chunk-a", "doc-a", [1.0, 0.0])
        add_chunk(session, "chunk-b", "doc-b", [1.0, 0.0, 0.0])
        session.commit()

        assert rebuild_topics(session) == (2, 0)
        topics = session.query(Topic).order_by(Topic.label).all()
        assert {topic.label for topic in topics} == {"a.txt", "b.txt"}
        assert all(chunk.topic_id for chunk in session.query(Chunk).all())


def test_topic_labels_are_unique_for_separate_groups(empty_schema):
    with Session(empty_schema) as session:
        add_document(session, "doc-a", "Same title")
        add_document(session, "doc-b", "Same title")
        session.flush()
        add_chunk(session, "chunk-a", "doc-a", [1.0, 0.0])
        add_chunk(session, "chunk-b", "doc-b", [0.0, 1.0])
        session.commit()

        assert rebuild_topics(session) == (2, 0)
        labels = {topic.label for topic in session.query(Topic).all()}
        assert labels == {"Same title", "Same title (2)"}

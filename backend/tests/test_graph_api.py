from app.db.models import Chunk, Connection, Document, Topic, TopicRelation


def test_empty_graph_endpoint_returns_complete_response(api_client):
    client, _ = api_client

    response = client.get("/api/graph")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [],
        "chunks": [],
        "connections": [],
        "topics": [],
        "related_topics": [],
    }


def test_graph_endpoint_returns_database_documents_chunks_and_edges(api_client):
    client, session_factory = api_client
    with session_factory() as session:
        document_a = Document(
            id="doc-a", filename="biology.txt", title="Biology", domain="Science",
            tags=["ecology"], media_type="text/plain", status="complete",
        )
        document_b = Document(
            id="doc-b", filename="networks.txt", title="Networks", domain="Computing",
            tags=["routing"], media_type="text/plain", status="complete",
        )
        topic_a = Topic(id="topic-a", label="Ecology", keywords=["colony"], centroid=[1.0, 0.0])
        topic_b = Topic(id="topic-b", label="Routing", keywords=["local"], centroid=[0.6, 0.8])
        chunk_a = Chunk(
            id="chunk-a", document_id="doc-a", page_number=1, text="A source passage",
            start_char=0, end_char=16, embedding=[1.0, 0.0], embedding_model="test",
            topic_id="topic-a",
        )
        chunk_b = Chunk(
            id="chunk-b", document_id="doc-b", page_number=2, text="Another passage",
            start_char=10, end_char=25, embedding=[0.6, 0.8], embedding_model="test",
            topic_id="topic-b",
        )
        session.add_all([document_a, document_b, topic_a, topic_b])
        session.flush()
        session.add_all([chunk_a, chunk_b])
        session.flush()
        session.add(Connection(
            id="edge-a-b", source_chunk_id="chunk-a", target_chunk_id="chunk-b",
            similarity=0.6, strength=0.9, weight=0.78, title="Shared local adaptation",
            explanation="Both passages explain systems that adapt to nearby conditions.",
            shared_concept="local adaptation", discipline_scope="cross_discipline",
            semantic_distance="related",
        ))
        session.add(TopicRelation(source_topic_id="topic-a", target_topic_id="topic-b", similarity=0.6))
        session.commit()

    response = client.get("/api/graph")
    body = response.json()

    assert response.status_code == 200
    assert {document["id"] for document in body["documents"]} == {"doc-a", "doc-b"}
    assert {chunk["id"] for chunk in body["chunks"]} == {"chunk-a", "chunk-b"}
    assert body["documents"][0]["chunk_count"] == 1
    assert body["connections"] == [{
        "id": "edge-a-b", "source_chunk_id": "chunk-a", "target_chunk_id": "chunk-b",
        "similarity": 0.6, "strength": 0.9, "weight": 0.78,
        "title": "Shared local adaptation",
        "explanation": "Both passages explain systems that adapt to nearby conditions.",
        "shared_concept": "local adaptation",
        "discipline_scope": "cross_discipline",
        "semantic_distance": "related",
    }]
    assert body["related_topics"] == [{
        "source_topic_id": "topic-a", "target_topic_id": "topic-b", "similarity": 0.6,
    }]


def test_invalid_note_is_rejected_without_database_write(api_client):
    client, session_factory = api_client

    response = client.post("/api/documents/note", json={
        "title": "  ", "domain": "Science", "content": "A valid note", "tags": [],
    })

    assert response.status_code == 422
    with session_factory() as session:
        assert session.query(Document).count() == 0

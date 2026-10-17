"""Create a small, clearly labeled starter graph in a new local database."""

from sqlalchemy.orm import Session

from app.db.models import Chunk, Connection, Document, Topic, TopicRelation
from app.db.session import SessionLocal


def seed_example_data() -> bool:
    """Seed the empty application database once; return whether rows were inserted."""
    with SessionLocal() as session:
        if session.query(Document.id).first() is not None:
            return False

        biology_id = "starter-example-biology"
        network_id = "starter-example-network"
        biology_topic_id = "starter-example-topic-biology"
        network_topic_id = "starter-example-topic-networks"
        biology_chunk_id = "starter-example-chunk-biology"
        network_chunk_id = "starter-example-chunk-networks"

        biology_text = (
            "Ant colonies coordinate through local signals. Individual ants respond to "
            "nearby chemical trails, allowing the colony to adapt without a central commander."
        )
        network_text = (
            "Mesh networks route traffic through local decisions. Each router reacts to "
            "nearby congestion, allowing the network to adapt without a central controller."
        )
        session.add_all([
            Document(
                id=biology_id,
                filename="Starter example - ant colony coordination.txt",
                title="Ant colony coordination",
                domain="Biology",
                tags=["starter-example", "collective behavior", "local signals"],
                sections=[{"title": "Local coordination", "summary":
                           "How nearby signals coordinate colony behavior.",
                           "page_start": 1, "page_end": 1}],
                media_type="text/plain", status="complete",
            ),
            Document(
                id=network_id,
                filename="Starter example - mesh networks.txt",
                title="Adaptive mesh networks",
                domain="Computer networks",
                tags=["starter-example", "mesh networks", "adaptive routing"],
                sections=[{"title": "Adaptive routing", "summary":
                           "How routers adapt to local traffic conditions.",
                           "page_start": 1, "page_end": 1}],
                media_type="text/plain", status="complete",
            ),
            Topic(id=biology_topic_id, label="Collective behavior",
                  keywords=["colony", "signals", "coordination"], centroid=[1.0, 0.0]),
            Topic(id=network_topic_id, label="Adaptive routing",
                  keywords=["network", "routing", "congestion"], centroid=[0.8, 0.6]),
        ])
        session.flush()
        session.add_all([
            Chunk(id=biology_chunk_id, document_id=biology_id, page_number=1,
                  text=biology_text, start_char=0, end_char=len(biology_text),
                  embedding=[1.0, 0.0], embedding_model="starter-example-vector",
                  topic_id=biology_topic_id),
            Chunk(id=network_chunk_id, document_id=network_id, page_number=1,
                  text=network_text, start_char=0, end_char=len(network_text),
                  embedding=[0.8, 0.6], embedding_model="starter-example-vector",
                  topic_id=network_topic_id),
        ])
        session.flush()
        session.add_all([
            Connection(
                id="starter-example-connection",
                source_chunk_id=biology_chunk_id,
                target_chunk_id=network_chunk_id,
                similarity=0.8,
                strength=0.9,
                weight=0.86,
                title="Coordination without central control",
                explanation=("Both passages describe systems that coordinate through local "
                             "responses to nearby information."),
                shared_concept="Local coordination",
            ),
            TopicRelation(source_topic_id=biology_topic_id, target_topic_id=network_topic_id,
                          similarity=0.8),
        ])
        session.commit()
        return True

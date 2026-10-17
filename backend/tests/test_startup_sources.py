from pathlib import Path
from types import SimpleNamespace

from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.db.models import Base
from app.services.startup_sources import CONTEXT_SOURCE_DIR, load_context_sources


class RecordingIngestionService:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.uploads: list[tuple[str, bytes]] = []

    def ingest_upload(self, session, *, filename: str, content_type: str, data: bytes):
        assert content_type == "text/plain"
        self.uploads.append((filename, data))
        document = Document(
            id=f"fixture-{filename}", filename=filename, title=Path(filename).stem,
            domain="Uncategorized", tags=[], sections=[], media_type=content_type,
            status="complete",
        )
        session.add(document)
        text = data.decode("utf-8-sig").strip()
        session.add(Chunk(
            id=f"fixture-chunk-{filename}", document_id=document.id, page_number=1,
            text=text, start_char=0, end_char=len(text), embedding=[1.0],
            embedding_model="test-embedding",
        ))
        session.commit()
        return SimpleNamespace(document=document)


def test_bundled_context_sources_cover_shared_topics_and_cross_domain_bridge():
    sources = {source.name: source.read_text(encoding="utf-8").casefold()
               for source in CONTEXT_SOURCE_DIR.glob("*.txt")}

    assert set(sources) == {
        "adaptive_traffic_signals.txt",
        "artificial_neural_networks.txt",
        "ant_foraging.txt",
        "bread_fermentation_and_proofing.txt",
        "cake_leavening_and_recipe_steps.txt",
        "distributed_robot_swarming.txt",
        "fish_schooling.txt",
        "fish_schooling_and_neural_networks.txt",
        "networked_behavior_comparison.txt",
        "neural_network_and_brain_circuits.txt",
        "neuron_population_activity.txt",
        "recipe_instructions_and_baking_results.txt",
        "memory_consolidation.txt",
        "neural_network_learning.txt",
        "synaptic_plasticity.txt",
    }
    assert "local" in sources["ant_foraging.txt"] and "feedback" in sources["ant_foraging.txt"]
    assert "local" in sources["fish_schooling.txt"] and "coordinated group" in sources["fish_schooling.txt"]
    assert "feedback" in sources["adaptive_traffic_signals.txt"]
    assert "hippocampus" in sources["memory_consolidation.txt"]
    assert "memory consolidation" in sources["neural_network_learning.txt"]
    assert "backpropagation" in sources["networked_behavior_comparison.txt"]
    assert "structural rather than literal" in sources["networked_behavior_comparison.txt"]
    assert "artificial neural network" in sources["artificial_neural_networks.txt"]
    assert "biological neural network" in sources["neural_network_and_brain_circuits.txt"]
    assert "fish schooling" in sources["fish_schooling_and_neural_networks.txt"]
    assert "carbon dioxide" in sources["bread_fermentation_and_proofing.txt"]
    assert "baking powder" in sources["cake_leavening_and_recipe_steps.txt"]
    assert "stopping cue" in sources["recipe_instructions_and_baking_results.txt"]
    assert "nearby cells" in sources["neuron_population_activity.txt"]
    assert "nearby teammates" in sources["distributed_robot_swarming.txt"]
    assert "synaptic" in sources["synaptic_plasticity.txt"]
    assert all(not any(line.startswith(prefix) for line in text.splitlines()
                       for prefix in ("title:", "topic:", "connection:", "similarity:", "strength:"))
               for text in sources.values())


def test_context_sources_are_plain_content_and_ingested_once(empty_schema, tmp_path):
    from sqlalchemy.orm import sessionmaker

    source_dir = tmp_path / "context_check_sources"
    source_dir.mkdir()
    content = {
        "ant_foraging.txt": "Ants reinforce useful pheromone trails through local feedback.",
        "fish_schooling.txt": "Fish coordinate direction by responding to nearby neighbors.",
        "networked_behavior_comparison.txt": "Fish and artificial networks use local inputs to shape collective behavior.",
        "adaptive_traffic_signals.txt": "Traffic signals adjust timing from local queue measurements.",
        "memory_consolidation.txt": "Hippocampal replay helps stabilize memories across distributed brain regions.",
        "neural_network_learning.txt": "Backpropagation updates model weights using prediction error.",
        "neuron_population_activity.txt": "Nearby neurons coordinate firing through changing activity patterns.",
        "distributed_robot_swarming.txt": "Neighboring robots adapt movement from local feedback.",
        "synaptic_plasticity.txt": "Synaptic plasticity changes communication strength between neurons.",
    }
    for filename, text in content.items():
        (source_dir / filename).write_text(text, encoding="utf-8")

    factory = sessionmaker(bind=empty_schema, expire_on_commit=False, class_=Session)
    service = RecordingIngestionService(factory)
    service_factory_calls = []

    def create_service():
        service_factory_calls.append(True)
        return service

    first = load_context_sources(
        session_factory=factory,
        source_dir=source_dir,
        ingestion_service_factory=create_service,
    )
    second = load_context_sources(
        session_factory=factory,
        source_dir=source_dir,
        ingestion_service_factory=create_service,
    )

    assert first.loaded == len(content)
    assert first.already_loaded == 0
    assert first.failed == ()
    assert second.loaded == 0
    assert second.already_loaded == len(content)
    assert len(service.uploads) == len(content)
    assert len(service_factory_calls) == 2
    assert dict(service.uploads) == {
        filename: text.encode("utf-8") for filename, text in content.items()
    }

    changed_text = content["fish_schooling.txt"] + " Schools also respond to local feedback."
    (source_dir / "fish_schooling.txt").write_text(changed_text, encoding="utf-8")
    updated = load_context_sources(
        session_factory=factory,
        source_dir=source_dir,
        ingestion_service_factory=create_service,
    )
    assert updated.loaded == 1
    assert updated.already_loaded == len(content) - 1
    assert dict(service.uploads)["fish_schooling.txt"] == changed_text.encode("utf-8")
    with Session(empty_schema) as session:
        assert session.query(Document).count() == len(content)
        fish_document = session.query(Document).filter_by(filename="fish_schooling.txt").one()
        assert session.query(Chunk).filter_by(document_id=fish_document.id).one().text == changed_text
        assert all(not document.tags and not document.sections
                   for document in session.query(Document).all())


def test_context_sources_retry_incomplete_startup_documents(empty_schema, tmp_path):
    from sqlalchemy.orm import sessionmaker

    source_dir = tmp_path / "context_check_sources"
    source_dir.mkdir()
    (source_dir / "retry.txt").write_text("Source text only.", encoding="utf-8")
    factory = sessionmaker(bind=empty_schema, expire_on_commit=False, class_=Session)
    attempts = 0

    class RetryService:
        def ingest_upload(self, session, *, filename: str, content_type: str, data: bytes):
            nonlocal attempts
            attempts += 1
            document = Document(
                id=f"attempt-{attempts}", filename=filename, title=filename,
                domain="Uncategorized", tags=[], sections=[], media_type=content_type,
                status="failed" if attempts == 1 else "complete",
            )
            session.add(document)
            session.commit()
            if attempts == 1:
                raise RuntimeError("simulated transient provider error")
            return SimpleNamespace(document=document)

    def create_service():
        return RetryService()

    first = load_context_sources(
        session_factory=factory,
        source_dir=source_dir,
        ingestion_service_factory=create_service,
    )
    second = load_context_sources(
        session_factory=factory,
        source_dir=source_dir,
        ingestion_service_factory=create_service,
    )

    assert first.failed == ("retry.txt",)
    assert second.loaded == 1
    assert second.failed == ()
    with Session(empty_schema) as session:
        documents = session.query(Document).all()
        assert len(documents) == 1
        assert documents[0].status == "complete"

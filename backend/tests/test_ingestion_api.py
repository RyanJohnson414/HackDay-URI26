from app.api.routes import documents as document_routes
from app.db.models import Chunk, Connection
from app.services import ingestion_service
from app.services.connection_engine import ConnectionEngine
from app.services.document_parser import ParsedDocument, ParsedPage
from app.services.embeddings import EmbeddedChunk
from app.services.gemini_client import ConnectionAssessment, DocumentAnalysis, DocumentSection
from app.services.ingestion_service import IngestionService
from app.services.webpage_reader import WebPage


class FakeEmbedder:
    def embed_chunks(self, chunks):
        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=(1.0, 0.0) if "biology" in chunk.text.lower() else (0.6, 0.8),
                model="test-embedding",
            )
            for chunk in chunks
        ]


class FakeGemini:
    def analyze_document(self, _filename: str, _pages):
        return DocumentAnalysis(
            title="Generated analysis title",
            domain="Generated topic",
            tags=["automatically-tagged"],
            sections=[DocumentSection(
                title="Generated section", summary="Summary grounded in the source passages.",
                page_start=1, page_end=1,
            )],
        )

    def assess_connection(self, _left: str, _right: str):
        return ConnectionAssessment(
            relevant=True,
            strength=0.9,
            title="Shared adaptive system",
            explanation="Both passages describe systems that adapt to changing local conditions.",
            shared_concept="local adaptation",
        )


def install_fake_ingestion(monkeypatch):
    gemini = FakeGemini()
    monkeypatch.setattr(
        document_routes,
        "IngestionService",
        lambda: IngestionService(
            embedder=FakeEmbedder(),
            connection_engine=ConnectionEngine(gemini_client=gemini),
            gemini_client=gemini,
        ),
    )

def test_note_ingestion_persists_and_exposes_cross_document_connection(api_client, monkeypatch):
    client, session_factory = api_client
    gemini = FakeGemini()
    monkeypatch.setattr(
        document_routes,
        "IngestionService",
        lambda: IngestionService(
            embedder=FakeEmbedder(),
            connection_engine=ConnectionEngine(gemini_client=gemini),
            gemini_client=gemini,
        ),
    )
    notes = [
        {
            "title": "Biology",
            "domain": "Science",
            "content": "Biology explains how a colony adjusts its behavior using local environmental signals.",
            "tags": [" ecology ", "ecology"],
        },
        {
            "title": "Networks",
            "domain": "Computing",
            "content": "Networks route traffic by reacting to current local conditions and nearby congestion.",
            "tags": ["routing"],
        },
    ]

    first = client.post("/api/documents/note", json=notes[0])
    second = client.post("/api/documents/note", json=notes[1])

    assert first.status_code == second.status_code == 201
    assert first.json()["document"]["tags"] == ["ecology", "automatically-tagged"]
    assert first.json()["document"]["sections"][0]["title"] == "Generated section"
    assert second.json()["connection_count"] == 1
    assert second.json()["topic_count"] == 2
    graph = client.get("/api/graph")
    body = graph.json()
    assert len(body["documents"]) == 2
    assert len(body["chunks"]) == 2
    assert len(body["connections"]) == 1
    assert body["connections"][0]["shared_concept"] == "local adaptation"
    assert len(body["related_topics"]) == 1

    with session_factory() as session:
        assert session.query(Chunk).count() == 2
        assert session.query(Connection).count() == 1


def test_file_upload_uses_parser_and_persists_gemini_metadata(api_client, monkeypatch):
    client, _ = api_client
    install_fake_ingestion(monkeypatch)
    note_response = client.post("/api/documents/note", json={
        "title": "Existing source",
        "domain": "Research",
        "content": "Local signals help systems adapt to changing conditions.",
        "tags": [],
    })
    assert note_response.status_code == 201

    response = client.post("/api/documents/upload", files={
        "file": ("field-notes.txt", b"Local signals help systems adapt to changing conditions.", "text/plain"),
    })

    assert response.status_code == 201
    document = response.json()["document"]
    assert document["title"] == "Generated analysis title"
    assert document["domain"] == "Generated topic"
    assert "automatically-tagged" in document["tags"]
    assert document["sections"][0]["title"] == "Generated section"
    assert document["chunk_count"] == 1
    assert response.json()["connection_count"] == 1


def test_url_ingestion_reparses_extracted_text_and_persists_metadata(api_client, monkeypatch):
    client, _ = api_client
    install_fake_ingestion(monkeypatch)
    parsed_page = ParsedDocument(
        filename="Research page", media_type="text/html",
        pages=(ParsedPage(1, "Research on local feedback and adaptive systems."),),
        source_url="https://example.org/research",
    )
    monkeypatch.setattr(ingestion_service, "fetch_webpage", lambda _url: WebPage(
        url="https://example.org/research", title="Research page",
        text=parsed_page.text, document=parsed_page,
    ))

    response = client.post("/api/documents/url", json={"url": "https://example.org/research"})

    assert response.status_code == 201
    document = response.json()["document"]
    assert document["source_url"] == "https://example.org/research"
    assert document["media_type"] == "text/html"
    assert document["sections"][0]["page_start"] == 1
    assert document["chunk_count"] == 1

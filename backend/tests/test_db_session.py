from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

from app.db.models import Base, Document
from app.db.session import _resolve_database_url, init_db
from app.db import seed as example_seed


def test_relative_sqlite_url_resolves_from_backend_directory():
    url = _resolve_database_url("sqlite:///./knowledge_graph.db")

    assert Path(url.removeprefix("sqlite:///")).as_posix().endswith(
        "HackDay-URI26/backend/knowledge_graph.db"
    )


def test_absolute_sqlite_and_memory_urls_are_preserved():
    absolute_url = "sqlite:///C:/data/graph.db"

    resolved = make_url(_resolve_database_url(absolute_url))
    assert resolved.database.replace("\\", "/") == "C:/data/graph.db"
    assert _resolve_database_url("sqlite:///:memory:") == "sqlite:///:memory:"


def test_startup_creates_schema_at_configured_path(test_engine, tmp_path):
    init_db()

    assert (tmp_path / "test_graph.db").is_file()
    assert set(inspect(test_engine).get_table_names()) == set(Document.metadata.tables)
    with test_engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1


def test_startup_migrates_legacy_document_columns(test_engine):
    with test_engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE documents ("
            "id VARCHAR(36) PRIMARY KEY, filename VARCHAR(512) NOT NULL, "
            "media_type VARCHAR(128) NOT NULL, source_url TEXT, status VARCHAR(32) NOT NULL, "
            "error TEXT, created_at DATETIME NOT NULL)"
        )
        connection.exec_driver_sql(
            "INSERT INTO documents (id, filename, media_type, status, created_at) "
            "VALUES ('legacy', 'legacy.txt', 'text/plain', 'complete', CURRENT_TIMESTAMP)"
        )

    init_db()
    columns = {column["name"] for column in inspect(test_engine).get_columns("documents")}

    assert {"title", "domain", "tags", "sections"} <= columns
    with test_engine.connect() as connection:
        row = connection.execute(
            text("SELECT title, domain, tags FROM documents WHERE id = 'legacy'")
        ).one()
    assert row.title == "legacy.txt"
    assert row.domain == "Uncategorized"
    assert row.tags == "[]"


def test_startup_is_idempotent(test_engine):
    init_db()
    init_db()

    assert "connections" in inspect(test_engine).get_table_names()


def test_startup_migrates_and_backfills_connection_classifications(test_engine):
    tables_without_connections = [
        table for table in Base.metadata.sorted_tables if table.name != "connections"
    ]
    Base.metadata.create_all(bind=test_engine, tables=tables_without_connections)
    with test_engine.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE connections (
                id VARCHAR(36) PRIMARY KEY,
                source_chunk_id VARCHAR(36) NOT NULL,
                target_chunk_id VARCHAR(36) NOT NULL,
                similarity FLOAT NOT NULL,
                strength FLOAT NOT NULL,
                weight FLOAT NOT NULL,
                title VARCHAR(256) NOT NULL,
                explanation TEXT NOT NULL,
                shared_concept VARCHAR(256) NOT NULL,
                created_at DATETIME NOT NULL,
                FOREIGN KEY(source_chunk_id) REFERENCES chunks(id),
                FOREIGN KEY(target_chunk_id) REFERENCES chunks(id)
            )
        """)
        connection.exec_driver_sql("""
            INSERT INTO documents (id, filename, title, domain, tags, sections, media_type, status, created_at)
            VALUES
                ('doc-a', 'a.txt', 'A', 'Neuroscience', '[]', '[]', 'text/plain', 'complete', CURRENT_TIMESTAMP),
                ('doc-b', 'b.txt', 'B', 'neuroscience', '[]', '[]', 'text/plain', 'complete', CURRENT_TIMESTAMP),
                ('doc-c', 'c.txt', 'C', 'Computer science', '[]', '[]', 'text/plain', 'complete', CURRENT_TIMESTAMP)
        """)
        connection.exec_driver_sql("""
            INSERT INTO chunks (id, document_id, page_number, text, start_char, end_char, embedding, embedding_model)
            VALUES
                ('chunk-a', 'doc-a', 1, 'A', 0, 1, '[1.0, 0.0]', 'test'),
                ('chunk-b', 'doc-b', 1, 'B', 0, 1, '[1.0, 0.0]', 'test'),
                ('chunk-c', 'doc-c', 1, 'C', 0, 1, '[1.0, 0.0]', 'test')
        """)
        connection.exec_driver_sql("""
            INSERT INTO connections (
                id, source_chunk_id, target_chunk_id, similarity, strength, weight,
                title, explanation, shared_concept, created_at
            ) VALUES
                ('edge-local', 'chunk-a', 'chunk-b', 0.9, 0.9, 0.86,
                 'Local', 'Same discipline relation.', 'neural activity', CURRENT_TIMESTAMP),
                ('edge-cross', 'chunk-a', 'chunk-c', 0.2, 0.7, 0.55,
                 'Cross discipline', 'Cross discipline relation.', 'learning', CURRENT_TIMESTAMP)
        """)

    init_db()

    with test_engine.connect() as connection:
        rows = connection.exec_driver_sql(
            "SELECT id, discipline_scope, semantic_distance FROM connections"
        ).all()
    values = {row[0]: (row[1], row[2]) for row in rows}
    assert values == {
        "edge-local": ("within_discipline", "local"),
        "edge-cross": ("cross_discipline", "distant"),
    }


def test_example_database_seed_is_loaded_once(api_client, monkeypatch):
    _, session_factory = api_client
    monkeypatch.setattr(example_seed, "SessionLocal", session_factory)

    assert example_seed.seed_example_data() is True
    assert example_seed.seed_example_data() is False

    with session_factory() as session:
        documents = session.query(Document).all()
        assert len(documents) == 2
        assert all("starter-example" in document.tags for document in documents)
        assert all(document.sections for document in documents)

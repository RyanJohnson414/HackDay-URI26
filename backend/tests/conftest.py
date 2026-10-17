from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db import session as db_session
from app.db.models import Base
from app import main as main_app
from app.main import app


@pytest.fixture
def test_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_path = tmp_path / "test_graph.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    engine = create_engine(database_url, future=True, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _connection_record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    monkeypatch.setattr(db_session, "DATABASE_URL", database_url)
    monkeypatch.setattr(db_session, "engine", engine)
    yield engine
    engine.dispose()


@pytest.fixture
def api_client(test_engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    monkeypatch.setattr(main_app, "load_context_sources", lambda: None)
    test_session = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False, class_=Session)

    def override_get_db():
        session = test_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[db_session.get_db] = override_get_db
    with TestClient(app) as client:
        yield client, test_session
    app.dependency_overrides.clear()


@pytest.fixture
def empty_schema(test_engine):
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    Base.metadata.drop_all(bind=test_engine)

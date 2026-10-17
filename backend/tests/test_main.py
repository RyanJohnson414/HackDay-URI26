from threading import Event

from fastapi.testclient import TestClient

from app import main as main_app
from app.main import app
from app.services.startup_sources import StartupSourceReport


def test_health_is_available_while_context_sources_load(test_engine, monkeypatch):
    loader_started = Event()
    allow_loader_to_finish = Event()

    def slow_loader():
        loader_started.set()
        if not allow_loader_to_finish.wait(timeout=2):
            raise TimeoutError("Test did not release the startup source loader")
        return StartupSourceReport(loaded=0, already_loaded=0, failed=())

    monkeypatch.setattr(main_app, "load_context_sources", slow_loader)

    with TestClient(app) as client:
        assert loader_started.wait(timeout=1)
        assert client.get("/health").json() == {"status": "ok"}
        allow_loader_to_finish.set()

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_local_graph_do_not_require_database_startup():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get("/api/local/graph")
        assert response.status_code == 200
        assert "documents" in response.json()

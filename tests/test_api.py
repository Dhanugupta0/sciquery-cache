"""API endpoint tests."""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def client():
    """Create a test client. Requires book index to exist."""
    from app.api import app
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestSession:
    def test_create_session(self, client):
        r = client.post("/session")
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0


class TestChat:
    def test_unknown_session(self, client):
        r = client.post("/chat", json={"session_id": "nonexistent", "message": "hello"})
        assert r.status_code == 404

    def test_empty_message(self, client):
        # Create a session first
        sid = client.post("/session").json()["session_id"]
        r = client.post("/chat", json={"session_id": sid, "message": ""})
        assert r.status_code == 422

    def test_response_format(self, client):
        """Response must have reply, citations, cache_hit, latency_ms."""
        sid = client.post("/session").json()["session_id"]
        r = client.post("/chat", json={"session_id": sid, "message": "What is refraction?"})
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert "citations" in data
        assert "cache_hit" in data
        assert "latency_ms" in data
        assert isinstance(data["citations"], list)
        assert isinstance(data["cache_hit"], bool)
        assert isinstance(data["latency_ms"], (int, float))

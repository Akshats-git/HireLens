"""Tests for system endpoints: /, /health."""


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "HireLens API"
    assert "docs" in body


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["models_loaded"] is True
    assert "timestamp" in body

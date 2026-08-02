"""
Tests for /api/v1/health and /api/v1/chat.

REWRITE NOTE: the original test asserted
`response.json() == {"status": "ok"}` for /health. The real route (defined
in app.py, not monitoring.py's separate /health/* family) returns a richer
payload: {"status": "healthy"|"degraded", "database": "ok"|"error",
"rag_ready": bool, "rag_files": int, "async_worker_alive": bool,
"version": str, "timestamp": str}. An exact dict equality check against
{"status": "ok"} would fail against the real shape on every field.

The original chat test posted to a bare `/chat` with `user_id` in the body
and expected unauthenticated 200 — the real route is `/api/v1/chat`,
requires a JWT via @require_auth, and has no user_id field in the request
body at all (the user is derived entirely from the token via
current_uid()).
"""

import uuid
import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def auth_headers(client):
    username = f"chattest_{uuid.uuid4().hex[:10]}"
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": "TestPassword123!"},
    )
    assert resp.status_code == 201, f"Registration failed: {resp.get_json()}"
    token = resp.get_json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint(client):
    """
    /api/v1/health is intentionally unauthenticated (no @require_auth) so
    container orchestrators can probe it without credentials.
    """
    resp = client.get("/api/v1/health")
    assert resp.status_code in (200, 503)  # 503 if DB is genuinely down
    body = resp.get_json()
    assert body["status"] in ("healthy", "degraded")
    assert "database" in body
    assert "version" in body
    assert "timestamp" in body


def test_chat_endpoint_requires_auth(client):
    """No Authorization header at all should be rejected before anything else runs."""
    resp = client.post("/api/v1/chat", json={"message": "Hello"})
    assert resp.status_code == 401


def test_chat_endpoint_with_valid_auth(client, auth_headers):
    """
    A simple message from an authenticated user should return the standard
    success envelope shape: {status, message, data: {response, thread_id,
    sources, is_new_thread, model}, timestamp}.

    NOTE: this test exercises the real model_router fallback chain and
    therefore needs at least one working LLM provider credential
    (GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY) in the test
    environment to actually reach success. Without one, expect a 502
    MODEL_ERROR — which is itself useful signal that no provider is
    configured, not a false pass.
    """
    resp = client.post(
        "/api/v1/chat",
        json={"message": "Say hello in one short sentence.", "use_grounding": False},
        headers=auth_headers,
    )

    if resp.status_code == 502:
        pytest.skip("No LLM provider reachable in this test environment (MODEL_ERROR).")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    data = body["data"]
    assert "response" in data
    assert "thread_id" in data
    assert data["is_new_thread"] is True


def test_chat_endpoint_requires_message_or_file(client, auth_headers):
    """An empty body with neither message nor file should be rejected as a 400."""
    resp = client.post("/api/v1/chat", json={}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "VALIDATION_ERROR"
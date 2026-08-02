"""
Tests for the /api/v1/chat rate limit, enforced via rate_limiter.py's
rate_limit() decorator.

REWRITE NOTE: the original test posted 101 requests expecting a limit of
100, and checked `responses[-1].headers` on an int (status code), which
would have raised AttributeError even if the rest of the test were
otherwise correct — `responses` was a list of status codes, not response
objects, so `.headers` was never a valid thing to call on it.

The real limit, after the rate_limiter.py signature-mismatch bug we found
and fixed (app.py was calling `@rate_limit(limit=40, window=60)` against a
decorator that only accepts `requests_per_minute=`/`requests_per_hour=` —
a TypeError that would have crashed the app at import time), is
`@rate_limit(requests_per_minute=40)` on /api/v1/chat specifically.

This test also requires Redis to be reachable, since rate_limiter.py's
_check_rate_limit fails OPEN (allows all requests) when Redis is down —
if Redis isn't running, this test would pass for the wrong reason (no
limiting happened at all, not "limiting worked correctly"). The fixture
below checks Redis connectivity up front and skips with a clear reason
if it's unavailable, rather than giving a false pass.
"""

import uuid
import pytest

from app import app as flask_app
from redis_client import get_redis_client

CHAT_RPM_LIMIT = 40  # matches @rate_limit(requests_per_minute=40) on /api/v1/chat


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def auth_headers(client):
    username = f"ratetest_{uuid.uuid4().hex[:10]}"
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": "TestPassword123!"},
    )
    assert resp.status_code == 201, f"Registration failed: {resp.get_json()}"
    token = resp.get_json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def require_redis():
    """
    Skip these tests if Redis isn't reachable, rather than silently
    passing for the wrong reason. rate_limiter.py fails open without
    Redis — every request would be allowed, and a test asserting "the
    41st request gets a 429" would never actually observe a 429, not
    because rate limiting is broken, but because it never ran at all.
    """
    redis = get_redis_client()
    if not redis.connected:
        pytest.skip("Redis is not reachable — rate limiting fails open without it.")


def test_rate_limit_blocks_after_threshold(client, auth_headers):
    """
    Sending more than CHAT_RPM_LIMIT requests within the same minute
    should eventually return 429 with a Retry-After header, per
    rate_limiter.py's _build_429_response().
    """
    last_response = None
    for i in range(CHAT_RPM_LIMIT + 5):
        last_response = client.post(
            "/api/v1/chat",
            json={"message": f"test message {i}"},
            headers=auth_headers,
        )
        if last_response.status_code == 429:
            break

    assert last_response.status_code == 429
    assert "Retry-After" in last_response.headers
    body = last_response.get_json()
    assert body["code"] == "RATE_LIMIT_EXCEEDED"


def test_rate_limit_headers_present_on_allowed_request(client, auth_headers):
    """
    Even a single allowed request should carry the X-RateLimit-* headers
    that _add_rate_limit_headers() attaches.
    """
    resp = client.post(
        "/api/v1/chat",
        json={"message": "single test message"},
        headers=auth_headers,
    )
    # This single request should not itself be rate limited.
    assert resp.status_code != 429
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
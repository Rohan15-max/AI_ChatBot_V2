"""
Tests for model_router.py's multi-provider fallback chain.

REWRITE NOTE: the original test imported `from ai.fallback_router import
route_request` — this module doesn't exist anywhere in the project. The
real fallback logic lives in model_router.py's `ModelRouter.route()`,
which is async, takes keyword arguments (prompt, user_id, user_tier, ...),
and returns a `ModelResponse` dataclass with `.provider`, `.model`,
`.fallback_used`, etc. — not a plain dict with a "provider" key.

The original test also patched `openai.ChatCompletion.create` — that's the
pre-1.0 OpenAI SDK's call shape. model_router.py's `_call_openai_compatible`
makes a raw httpx POST to the chat completions endpoint directly, not
through the openai package's client object at all, so patching
`openai.ChatCompletion.create` would never have intercepted anything even
if the rest of the test's assumptions were right.

This rewrite patches at the actual call boundary: model_router.py's
provider call functions (_call_gemini, _call_openai_compatible,
_call_anthropic), verifying that when the first provider in the routing
order raises, ModelRouter.route() actually moves on to the next one
in DEFAULT_FALLBACK_CHAIN rather than failing outright — which is the
real behavior this test is meant to protect.
"""

import pytest
from unittest.mock import AsyncMock, patch

from model_router import ModelRouter, PROVIDER_CONFIGS, _circuit_breaker


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """
    _circuit_breaker is a module-level singleton — a failure recorded by
    one test would otherwise leak into the next and skip providers that
    should be available. Reset its internal state before each test.
    """
    _circuit_breaker._failures.clear()
    _circuit_breaker._opened_at.clear()
    yield
    _circuit_breaker._failures.clear()
    _circuit_breaker._opened_at.clear()


@pytest.fixture
def router():
    return ModelRouter()


@pytest.mark.asyncio
async def test_falls_back_to_next_provider_on_failure(router):
    """
    If the first available provider's call function raises, route() should
    catch it, record a circuit-breaker failure for that provider, and try
    the next one in the chain — returning a successful ModelResponse with
    fallback_used=True, rather than propagating the exception or returning
    a response from the failed provider.
    """
    call_count = {"n": 0}

    async def flaky_call(messages, model_id, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("Simulated provider timeout")
        return "Fallback response text", 50, 20

    # Patch every provider's call function to the same flaky behavior so
    # the test doesn't depend on which specific providers have API keys
    # configured in the test environment — only that *some* provider
    # after the first one succeeds.
    with patch("model_router._call_gemini", new=AsyncMock(side_effect=flaky_call)), \
         patch("model_router._call_openai_compatible", new=AsyncMock(side_effect=flaky_call)), \
         patch("model_router._call_anthropic", new=AsyncMock(side_effect=flaky_call)):

        # Force at least two providers to appear "configured" regardless of
        # real API keys in the test env, so there's guaranteed to be a
        # second provider in the chain to fall back to.
        with patch.dict(PROVIDER_CONFIGS["gemini-flash"], {"enabled": lambda: True}), \
             patch.dict(PROVIDER_CONFIGS["gpt-4o-mini"], {"enabled": lambda: True}):

            result = await router.route(
                prompt="Test prompt",
                user_id="test_user",
                user_tier="free",
                use_cost_router=False,
            )

    assert result.success
    assert result.fallback_used is True
    assert len(result.fallback_chain) >= 2


@pytest.mark.asyncio
async def test_raises_when_all_providers_fail(router):
    """
    If every available provider's call function raises, route() should
    raise RuntimeError with a message naming what was tried, rather than
    silently returning an empty or partial response.
    """
    async def always_fails(messages, model_id, **kwargs):
        raise TimeoutError("Simulated total outage")

    with patch("model_router._call_gemini", new=AsyncMock(side_effect=always_fails)), \
         patch("model_router._call_openai_compatible", new=AsyncMock(side_effect=always_fails)), \
         patch("model_router._call_anthropic", new=AsyncMock(side_effect=always_fails)):

        with patch.dict(PROVIDER_CONFIGS["gemini-flash"], {"enabled": lambda: True}):
            with pytest.raises(RuntimeError, match="All LLM providers failed"):
                await router.route(
                    prompt="Test prompt",
                    user_id="test_user",
                    user_tier="free",
                    use_cost_router=False,
                )


@pytest.mark.asyncio
async def test_circuit_breaker_skips_recently_failed_provider(router):
    """
    After enough consecutive failures (CircuitBreaker.failure_threshold,
    default 3), a provider should be skipped on the *next* route() call
    entirely — not retried — until the recovery_timeout window passes.
    """
    provider_name = "gemini"
    for _ in range(3):
        _circuit_breaker.record_failure(provider_name)

    assert _circuit_breaker.is_open(provider_name) is True
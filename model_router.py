"""
DevMentor AI — Model Router
=============================
Production-grade multi-model LLM router with:

- Async calls with retry + exponential backoff
- Full provider support: Gemini, OpenAI, Anthropic, DeepSeek, Ollama
- Cost-aware routing via CostRouter integration
- Streaming support (token-by-token)
- Conversation history support (multi-turn)
- Real token counting from API responses
- Per-provider timeout and retry configuration
- Structured response with full metadata
- Circuit breaker pattern per provider
- Graceful fallback chain
- Usage logging integration

Usage:
    from model_router import ModelRouter

    router = ModelRouter()

    # Simple call
    result = await router.route(
        prompt="Explain RAG",
        user_id="u123",
        user_tier="pro",
    )
    print(result.response)

    # With conversation history
    result = await router.route(
        prompt="Follow up question",
        user_id="u123",
        messages=[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    )

    # Streaming
    async for chunk in router.stream(prompt, user_id):
        yield chunk
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import get_settings

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()


# ===========================================================================
# Data Classes
# ===========================================================================

class FinishReason(str, Enum):
    STOP      = "stop"
    MAX_TOKENS = "max_tokens"
    ERROR     = "error"
    FILTERED  = "filtered"


@dataclass
class ModelResponse:
    """Structured response from any LLM provider."""
    response: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: int
    cost_usd: float = 0.0
    finish_reason: FinishReason = FinishReason.STOP
    fallback_used: bool = False
    fallback_chain: List[str] = field(default_factory=list)
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.response) and self.finish_reason != FinishReason.ERROR


@dataclass
class Message:
    """A single conversation message."""
    role: str    # user | assistant | system
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ===========================================================================
# Token Estimation
# ===========================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation when API doesn't return counts.
    Rule of thumb: ~4 characters per token for English text.
    """
    return max(1, len(text) // 4)


def build_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """
    Build OpenAI-compatible message list from prompt + history.

    Args:
        prompt:        Current user message
        system_prompt: Optional system instruction
        history:       Prior conversation turns

    Returns:
        List of {role, content} dicts
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": prompt})
    return messages


# ===========================================================================
# Circuit Breaker (per provider)
# ===========================================================================

class CircuitBreaker:
    """
    Simple circuit breaker to stop calling failing providers.

    States:
    - CLOSED: Normal operation
    - OPEN:   Provider is failing, skip it
    - HALF_OPEN: Testing if provider recovered
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}

    def is_open(self, provider: str) -> bool:
        """Return True if circuit is open (provider should be skipped)."""
        if provider not in self._failures:
            return False
        if self._failures[provider] < self.failure_threshold:
            return False
        opened = self._opened_at.get(provider, 0)
        if time.time() - opened > self.recovery_timeout:
            # Try half-open
            self._failures[provider] = self.failure_threshold - 1
            return False
        return True

    def record_success(self, provider: str) -> None:
        """Reset failure count on success."""
        self._failures[provider] = 0
        self._opened_at.pop(provider, None)

    def record_failure(self, provider: str) -> None:
        """Increment failure count."""
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= self.failure_threshold:
            self._opened_at[provider] = time.time()
            logger.warning("Circuit breaker OPEN for provider: %s", provider)


# Global circuit breaker instance
_circuit_breaker = CircuitBreaker()


# ===========================================================================
# Provider Call Functions
# ===========================================================================

async def _call_gemini(
    messages: List[Dict[str, str]],
    model_id: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Tuple[str, int, int]:
    """
    Call Google Gemini API.

    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    import google.generativeai as genai

    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("Gemini API key not configured")

    genai.configure(api_key=api_key.get_secret_value())

    # Convert messages to Gemini format
    system_instruction = None
    gemini_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            gemini_messages.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            gemini_messages.append({"role": "model", "parts": [msg["content"]]})

    model_kwargs = {
        "generation_config": {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
    }
    if system_instruction:
        model_kwargs["system_instruction"] = system_instruction

    model = genai.GenerativeModel(model_id, **model_kwargs)

    loop = asyncio.get_running_loop()

    if gemini_messages and len(gemini_messages) > 1:
        # Multi-turn conversation
        chat = model.start_chat(history=gemini_messages[:-1])
        last_user = gemini_messages[-1]["parts"][0]
        response = await loop.run_in_executor(
            None, lambda: chat.send_message(last_user)
        )
    else:
        # Single turn
        prompt = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(prompt)
        )

    text = response.text

    # Try to get real token counts from usage metadata
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", estimate_tokens(str(messages))) if usage else estimate_tokens(str(messages))
    completion_tokens = getattr(usage, "candidates_token_count", estimate_tokens(text)) if usage else estimate_tokens(text)

    return text, int(prompt_tokens), int(completion_tokens)


async def _call_openai_compatible(
    messages: List[Dict[str, str]],
    model_id: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: int = 60,
) -> Tuple[str, int, int]:
    """
    Call any OpenAI-compatible API (OpenAI, DeepSeek, Ollama, etc.)

    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", estimate_tokens(str(messages)))
    completion_tokens = usage.get("completion_tokens", estimate_tokens(text))

    return text, int(prompt_tokens), int(completion_tokens)


async def _call_anthropic(
    messages: List[Dict[str, str]],
    model_id: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Tuple[str, int, int]:
    """
    Call Anthropic Claude API.

    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    api_key = settings.anthropic_api_key
    if not api_key:
        raise ValueError("Anthropic API key not configured")

    system_prompt = None
    anthropic_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            anthropic_messages.append(msg)

    payload: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": anthropic_messages,
    }
    if system_prompt:
        payload["system"] = system_prompt

    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key.get_secret_value(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    prompt_tokens = usage.get("input_tokens", estimate_tokens(str(messages)))
    completion_tokens = usage.get("output_tokens", estimate_tokens(text))

    return text, int(prompt_tokens), int(completion_tokens)


# ===========================================================================
# Provider Registry
# ===========================================================================

PROVIDER_CONFIGS = {
    "gemini-flash": {
        "provider": "gemini",
        "model_id": settings.gemini_model,
        "fn": _call_gemini,
        "enabled": lambda: bool(settings.gemini_api_key),
    },
    "gemini-pro": {
        "provider": "gemini",
        "model_id": settings.gemini_pro_model,
        "fn": _call_gemini,
        "enabled": lambda: bool(settings.gemini_api_key),
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": settings.openai_model,
        "fn": lambda msgs, model_id, **kw: _call_openai_compatible(
            msgs, model_id,
            api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else "",
            **kw,
        ),
        "enabled": lambda: bool(settings.openai_api_key),
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": settings.openai_pro_model,
        "fn": lambda msgs, model_id, **kw: _call_openai_compatible(
            msgs, model_id,
            api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else "",
            **kw,
        ),
        "enabled": lambda: bool(settings.openai_api_key),
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": settings.anthropic_model,
        "fn": _call_anthropic,
        "enabled": lambda: bool(settings.anthropic_api_key),
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": settings.anthropic_pro_model,
        "fn": _call_anthropic,
        "enabled": lambda: bool(settings.anthropic_api_key),
    },
    "ollama": {
        "provider": "ollama",
        "model_id": settings.ollama_model,
        "fn": lambda msgs, model_id, **kw: _call_openai_compatible(
            msgs, model_id,
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
            **kw,
        ),
        "enabled": lambda: settings.ollama_enabled,
    },
}

# Default fallback chain
DEFAULT_FALLBACK_CHAIN = [
    "gemini-flash",
    "gpt-4o-mini",
    "claude-haiku",
    "gemini-pro",
    "gpt-4o",
    "claude-sonnet",
    "ollama",
]


# ===========================================================================
# Model Router
# ===========================================================================

class ModelRouter:
    """
    Production-grade multi-model LLM router.

    Features:
    - Cost-aware model selection via CostRouter
    - Automatic fallback chain on failures
    - Circuit breaker per provider
    - Full async with retry logic
    - Streaming support
    - Conversation history support
    """

    def __init__(self):
        self._cost_router = None

    def _get_cost_router(self):
        if self._cost_router is None:
            try:
                from ai.cost_router import get_cost_router
                self._cost_router = get_cost_router()
            except Exception:
                pass
        return self._cost_router

    async def route(
        self,
        prompt: str,
        user_id: str = "anonymous",
        user_tier: str = "free",
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        preferred_model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cost_router: bool = True,
    ) -> ModelResponse:
        """
        Route a prompt to the best available LLM.

        Args:
            prompt:          User's message
            user_id:         User identifier for cost tracking
            user_tier:       Subscription tier (free/pro/enterprise)
            system_prompt:   System instruction override
            messages:        Prior conversation history
            preferred_model: Hint for model selection
            max_tokens:      Override max response tokens
            temperature:     Override sampling temperature
            use_cost_router: Whether to use cost-aware routing

        Returns:
            ModelResponse with response text and metadata
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        prompt = prompt.strip()
        start_time = time.time()
        max_tokens = max_tokens or settings.llm_max_tokens
        temperature = temperature if temperature is not None else settings.llm_temperature

        # Build message list
        full_messages = build_messages(prompt, system_prompt, messages)

        # Determine routing order
        if preferred_model and preferred_model in PROVIDER_CONFIGS:
            routing_order = [preferred_model] + [
                m for m in DEFAULT_FALLBACK_CHAIN if m != preferred_model
            ]
        elif use_cost_router:
            routing_order = self._get_cost_routing_order(
                user_id, prompt, user_tier
            )
        else:
            routing_order = DEFAULT_FALLBACK_CHAIN

        # Filter to available, non-circuit-broken providers
        available = [
            m for m in routing_order
            if m in PROVIDER_CONFIGS
            and PROVIDER_CONFIGS[m]["enabled"]()
            and not _circuit_breaker.is_open(PROVIDER_CONFIGS[m]["provider"])
        ]

        if not available:
            raise RuntimeError(
                "No LLM providers available. "
                "Please configure at least one API key in your .env file."
            )

        last_error = None
        tried = []

        for model_name in available:
            config = PROVIDER_CONFIGS[model_name]
            provider = config["provider"]
            model_id = config["model_id"]
            tried.append(model_name)

            try:
                logger.info(
                    "Calling %s/%s for user %s",
                    provider, model_name, user_id,
                )

                text, prompt_tokens, completion_tokens = await self._call_with_retry(
                    fn=config["fn"],
                    messages=full_messages,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                _circuit_breaker.record_success(provider)

                duration_ms = int((time.time() - start_time) * 1000)
                total_tokens = prompt_tokens + completion_tokens

                # Deduct cost
                cost = 0.0
                cost_router = self._get_cost_router()
                if cost_router:
                    try:
                        cost = cost_router.deduct_cost(
                            user_id, model_name,
                            prompt_tokens, completion_tokens,
                            user_tier,
                        )
                    except Exception:
                        pass

                # Log usage
                self._log_usage_async(
                    user_id, model_name, provider,
                    prompt_tokens, completion_tokens, duration_ms, cost,
                )

                logger.info(
                    "LLM response: model=%s tokens=%d duration=%dms cost=$%.5f",
                    model_name, total_tokens, duration_ms, cost,
                )

                return ModelResponse(
                    response=text,
                    model=model_name,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    duration_ms=duration_ms,
                    cost_usd=cost,
                    fallback_used=len(tried) > 1,
                    fallback_chain=tried,
                )

            except Exception as exc:
                logger.warning(
                    "Model %s failed: %s — trying next in chain",
                    model_name, exc,
                )
                _circuit_breaker.record_failure(provider)
                last_error = exc
                continue

        raise RuntimeError(
            f"All LLM providers failed after trying: {tried}. "
            f"Last error: {last_error}"
        )

    async def _call_with_retry(
        self,
        fn,
        messages: List[Dict],
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, int, int]:
        """
        Call an LLM function with exponential backoff retry.
        """
        max_retries = settings.llm_max_retries
        base_delay = settings.llm_retry_delay

        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(
                    fn(messages, model_id, max_tokens=max_tokens, temperature=temperature),
                    timeout=settings.llm_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "LLM timeout on attempt %d/%d (timeout=%ds)",
                    attempt + 1, max_retries, settings.llm_timeout,
                )
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(base_delay * (2 ** attempt))

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {429, 503}:
                    # Rate limited — back off
                    retry_after = int(exc.response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                    logger.warning("Rate limited. Waiting %ds before retry.", retry_after)
                    await asyncio.sleep(retry_after)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise

            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(base_delay * (2 ** attempt))

    async def stream(
        self,
        prompt: str,
        user_id: str = "anonymous",
        user_tier: str = "free",
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from Gemini Flash.

        Yields token chunks as they arrive.
        Falls back to non-streaming if streaming fails.

        Usage:
            async for chunk in router.stream(prompt, user_id):
                yield f"data: {chunk}\\n\\n"
        """
        import google.generativeai as genai

        if not settings.gemini_api_key:
            # Fall back to non-streaming
            result = await self.route(prompt, user_id, user_tier, system_prompt, messages)
            yield result.response
            return

        try:
            genai.configure(api_key=settings.gemini_api_key.get_secret_value())
            model = genai.GenerativeModel(
                settings.gemini_model,
                generation_config={"temperature": settings.llm_temperature},
                system_instruction=system_prompt,
            )

            full_messages = build_messages(prompt, None, messages)
            last_user_msg = next(
                (m["content"] for m in reversed(full_messages) if m["role"] == "user"),
                prompt,
            )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(last_user_msg, stream=True),
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.error("Streaming failed, falling back to non-streaming: %s", exc)
            result = await self.route(prompt, user_id, user_tier, system_prompt, messages)
            yield result.response

    def _get_cost_routing_order(
        self,
        user_id: str,
        prompt: str,
        user_tier: str,
    ) -> List[str]:
        """Get routing order from CostRouter."""
        try:
            cost_router = self._get_cost_router()
            if not cost_router:
                return DEFAULT_FALLBACK_CHAIN

            decision = cost_router.route(
                user_id=user_id,
                query=prompt,
                user_tier=user_tier,
            )

            if decision.rejected:
                logger.warning(
                    "Cost router rejected request for user %s: %s",
                    user_id, decision.rejection_reason,
                )
                return []

            preferred = decision.model
            if preferred and preferred in PROVIDER_CONFIGS:
                return [preferred] + [
                    m for m in DEFAULT_FALLBACK_CHAIN if m != preferred
                ]

        except Exception as exc:
            logger.error("Cost router error: %s", exc)

        return DEFAULT_FALLBACK_CHAIN

    def _log_usage_async(
        self,
        user_id: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
        cost_usd: float,
    ) -> None:
        """Fire-and-forget usage logging."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(
                    self._log_usage(
                        user_id, model, provider,
                        prompt_tokens, completion_tokens,
                        duration_ms, cost_usd,
                    )
                )
        except Exception:
            pass

    async def _log_usage(
        self,
        user_id: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
        cost_usd: float,
    ) -> None:
        """Log usage to analytics system."""
        try:
            from analytics import get_analytics
            analytics = get_analytics()
            analytics.track_llm_usage(
                user_id=user_id,
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
            )
        except Exception as exc:
            logger.debug("Usage logging failed (non-critical): %s", exc)

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of currently available providers and their status."""
        result = []
        for name, config in PROVIDER_CONFIGS.items():
            provider = config["provider"]
            result.append({
                "name": name,
                "provider": provider,
                "model_id": config["model_id"],
                "enabled": config["enabled"](),
                "circuit_open": _circuit_breaker.is_open(provider),
            })
        return result


# ===========================================================================
# Singleton
# ===========================================================================
_router_instance: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """Get or create the global ModelRouter singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter()
    return _router_instance


# Backward compatible singleton
model_router = get_model_router()
"""
DevMentor AI — Cost-Aware Model Router
========================================
Production-grade LLM routing with:

- Query complexity scoring for intelligent model selection
- Per-user budget tracking with Redis persistence
- Per-tier spending limits (free / pro / enterprise)
- Real-time cost calculation with up-to-date pricing
- Monthly budget reset
- Fallback chain when preferred model is unavailable
- Admin override per user
- Detailed routing decisions with reasoning
- Cost analytics and reporting

How it works:
    1. Analyze query complexity (length, keywords, question type)
    2. Check user's remaining budget for this month
    3. Select cheapest model that can handle the complexity
    4. Deduct cost after response is generated
    5. Fall back to cheaper model if budget is low

Usage:
    from ai.cost_router import CostRouter

    router = CostRouter()
    decision = router.route(user_id="u123", query="Explain RAG", estimated_tokens=500)

    if decision.rejected:
        return "Budget exceeded"

    response = call_llm(decision.model, query)
    router.deduct_cost(user_id, decision.model, actual_tokens_used)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
# Model Definitions
# ===========================================================================

class ModelTier(str, Enum):
    CHEAP    = "cheap"      # Fast, low cost — simple queries
    STANDARD = "standard"   # Balanced — most queries
    PREMIUM  = "premium"    # Most capable — complex queries
    LOCAL    = "local"      # Free — self-hosted models


@dataclass
class ModelConfig:
    """Configuration and pricing for a single LLM model."""
    name: str                          # Internal routing name
    provider: str                      # gemini | openai | anthropic | ollama
    api_model_id: str                  # Actual API model string
    tier: ModelTier
    input_cost_per_1k: float           # USD per 1000 input tokens
    output_cost_per_1k: float          # USD per 1000 output tokens
    max_tokens: int                    # Maximum context window
    max_output_tokens: int             # Maximum response tokens
    supports_vision: bool = False
    supports_functions: bool = False
    avg_latency_ms: int = 1000         # Approximate latency
    quality_score: float = 0.5        # 0-1 quality estimate
    enabled: bool = True
    description: str = ""


# Current model catalog with real pricing (as of 2025)
MODEL_CATALOG: Dict[str, ModelConfig] = {
    # -----------------------------------------------------------------------
    # Google Gemini
    # -----------------------------------------------------------------------
    "gemini-flash": ModelConfig(
        name="gemini-flash",
        provider="gemini",
        api_model_id="gemini-3.5-flash",
        tier=ModelTier.CHEAP,
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
        max_tokens=1_000_000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_functions=True,
        avg_latency_ms=800,
        quality_score=0.75,
        description="Fast and cheap — best for simple queries",
    ),
    "gemini-pro": ModelConfig(
        name="gemini-pro",
        provider="gemini",
        api_model_id="gemini-2.5-flash",
        tier=ModelTier.PREMIUM,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
        max_tokens=2_000_000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_functions=True,
        avg_latency_ms=2000,
        quality_score=0.92,
        description="Most capable Gemini — complex reasoning",
    ),

    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        api_model_id="gpt-4o-mini",
        tier=ModelTier.STANDARD,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        max_tokens=128_000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_functions=True,
        avg_latency_ms=1200,
        quality_score=0.82,
        description="Fast, capable, affordable — best all-rounder",
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider="openai",
        api_model_id="gpt-4o",
        tier=ModelTier.PREMIUM,
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
        max_tokens=128_000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_functions=True,
        avg_latency_ms=2500,
        quality_score=0.95,
        description="OpenAI flagship — best for complex tasks",
    ),

    # -----------------------------------------------------------------------
    # Anthropic Claude
    # -----------------------------------------------------------------------
    "claude-haiku": ModelConfig(
        name="claude-haiku",
        provider="anthropic",
        api_model_id="claude-3-haiku-20240307",
        tier=ModelTier.CHEAP,
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        max_tokens=200_000,
        max_output_tokens=4096,
        supports_vision=True,
        supports_functions=True,
        avg_latency_ms=700,
        quality_score=0.78,
        description="Fastest Claude — quick tasks and summarization",
    ),
    "claude-sonnet": ModelConfig(
        name="claude-sonnet",
        provider="anthropic",
        api_model_id="claude-sonnet-4-6",
        tier=ModelTier.STANDARD,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=200_000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_functions=True,
        avg_latency_ms=2000,
        quality_score=0.93,
        description="Best balance of speed and capability",
    ),

    # -----------------------------------------------------------------------
    # Local / Free (Ollama)
    # -----------------------------------------------------------------------
    "ollama-llama3": ModelConfig(
        name="ollama-llama3",
        provider="ollama",
        api_model_id="llama3",
        tier=ModelTier.LOCAL,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_tokens=8192,
        max_output_tokens=4096,
        avg_latency_ms=3000,
        quality_score=0.70,
        description="Free local model — no API cost",
    ),
}

# Per-tier monthly budgets in USD
TIER_MONTHLY_BUDGETS: Dict[str, float] = {
    "free":       2.0,
    "pro":        50.0,
    "enterprise": 500.0,
    "anonymous":  0.1,
}

# Complexity thresholds for model selection
COMPLEXITY_THRESHOLDS = {
    "cheap":    0.3,   # score < 0.3 → use cheap model
    "standard": 0.65,  # score 0.3-0.65 → use standard model
    "premium":  1.0,   # score > 0.65 → use premium model
}


# ===========================================================================
# Routing Decision
# ===========================================================================

@dataclass
class RoutingDecision:
    """Result of the routing algorithm."""
    model: Optional[str]               # Model name to use (None if rejected)
    provider: str                      # Provider name
    api_model_id: str                  # API model string
    rejected: bool = False             # Whether request was rejected
    rejection_reason: Optional[str] = None
    estimated_cost_usd: float = 0.0
    complexity_score: float = 0.0
    complexity_factors: List[str] = field(default_factory=list)
    fallback_used: bool = False        # Whether a fallback model was selected
    budget_remaining_usd: float = 0.0
    reasoning: str = ""


# ===========================================================================
# Redis Budget Store
# ===========================================================================

def _get_redis():
    """Get Redis client, returns None if unavailable."""
    try:
        from redis_client import get_redis_client
        client = get_redis_client()
        client.ping()
        return client
    except Exception:
        return None


def _budget_key(user_id: str) -> str:
    return f"budget:monthly:{user_id}"


def _budget_reset_key(user_id: str) -> str:
    return f"budget:reset:{user_id}"


# ===========================================================================
# Query Complexity Scorer
# ===========================================================================

class ComplexityScorer:
    """
    Scores query complexity from 0.0 (trivial) to 1.0 (very complex).

    Used to route simple queries to cheap models and complex queries
    to premium models — optimizing cost without sacrificing quality.
    """

    # Keywords that indicate complex, reasoning-heavy queries
    COMPLEX_KEYWORDS = {
        "explain", "analyze", "compare", "evaluate", "design", "architect",
        "implement", "debug", "optimize", "refactor", "review", "critique",
        "synthesize", "research", "investigate", "hypothesis", "strategy",
        "tradeoffs", "pros and cons", "difference between", "how does",
        "why does", "what causes", "step by step", "in detail",
    }

    # Keywords that indicate simple, lookup-style queries
    SIMPLE_KEYWORDS = {
        "what is", "define", "list", "name", "when", "who", "where",
        "how many", "yes or no", "true or false", "translate", "summarize",
    }

    # Code-related indicators (need capable model)
    CODE_KEYWORDS = {
        "code", "function", "class", "algorithm", "bug", "error", "exception",
        "implement", "write", "create", "build", "develop", "script",
        "python", "javascript", "typescript", "sql", "api", "database",
    }

    def score(self, query: str, context_length: int = 0) -> Tuple[float, List[str]]:
        """
        Score query complexity.

        Args:
            query:          User's query text
            context_length: Length of conversation context

        Returns:
            Tuple of (score 0.0-1.0, list of factor descriptions)
        """
        if not query:
            return 0.0, []

        query_lower = query.lower()
        words = query_lower.split()
        score = 0.0
        factors = []

        # ---- Query length factor ----
        if len(query) > 500:
            score += 0.2
            factors.append("Long query (>500 chars)")
        elif len(query) > 200:
            score += 0.1
            factors.append("Medium query (>200 chars)")
        elif len(query) < 30:
            score -= 0.1
            factors.append("Very short query (<30 chars)")

        # ---- Word count factor ----
        if len(words) > 50:
            score += 0.15
            factors.append("High word count (>50 words)")

        # ---- Complex keyword detection ----
        complex_matches = [k for k in self.COMPLEX_KEYWORDS if k in query_lower]
        if complex_matches:
            score += min(0.3, len(complex_matches) * 0.1)
            factors.append(f"Complex keywords: {', '.join(complex_matches[:3])}")

        # ---- Simple keyword detection ----
        simple_matches = [k for k in self.SIMPLE_KEYWORDS if k in query_lower]
        if simple_matches and not complex_matches:
            score -= 0.15
            factors.append(f"Simple keywords: {', '.join(simple_matches[:2])}")

        # ---- Code detection ----
        code_matches = [k for k in self.CODE_KEYWORDS if k in query_lower]
        if code_matches:
            score += min(0.25, len(code_matches) * 0.08)
            factors.append(f"Code-related: {', '.join(code_matches[:3])}")

        # ---- Multiple questions ----
        question_count = query.count("?")
        if question_count > 2:
            score += 0.1
            factors.append(f"Multiple questions ({question_count})")

        # ---- Context length factor ----
        if context_length > 10000:
            score += 0.15
            factors.append("Long conversation context")
        elif context_length > 3000:
            score += 0.05
            factors.append("Medium conversation context")

        # ---- Technical indicators ----
        if any(indicator in query_lower for indicator in ["```", "def ", "class ", "SELECT ", "import "]):
            score += 0.2
            factors.append("Contains code snippets")

        # Clamp to 0-1
        score = max(0.0, min(1.0, score))

        return score, factors


# ===========================================================================
# Cost Router
# ===========================================================================

class CostRouter:
    """
    Intelligent cost-aware LLM router.

    Selects the cheapest model capable of handling each query,
    while respecting per-user monthly budgets and tier limits.
    """

    def __init__(self):
        self.scorer = ComplexityScorer()
        self._in_memory_budgets: Dict[str, float] = {}  # Fallback when Redis unavailable

    def route(
        self,
        user_id: str,
        query: str,
        estimated_tokens: int = 500,
        user_tier: str = "free",
        context_length: int = 0,
        require_vision: bool = False,
        require_functions: bool = False,
        force_model: Optional[str] = None,
        preferred_provider: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Select the best model for a query based on complexity and budget.

        Args:
            user_id:            User identifier for budget tracking
            query:              User's query text
            estimated_tokens:   Estimated total tokens (prompt + completion)
            user_tier:          User's subscription tier
            context_length:     Current conversation context length in chars
            require_vision:     Whether the request includes images
            require_functions:  Whether function/tool calling is needed
            force_model:        Override routing and use this specific model
            preferred_provider: Prefer this provider if available

        Returns:
            RoutingDecision with selected model and cost estimate
        """
        # ---- Admin override ----
        if force_model:
            model_config = MODEL_CATALOG.get(force_model)
            if model_config:
                cost = self._estimate_cost(model_config, estimated_tokens)
                return RoutingDecision(
                    model=force_model,
                    provider=model_config.provider,
                    api_model_id=model_config.api_model_id,
                    estimated_cost_usd=cost,
                    reasoning=f"Forced model override: {force_model}",
                    budget_remaining_usd=self._get_budget(user_id, user_tier),
                )

        # ---- Score complexity ----
        complexity, factors = self.scorer.score(query, context_length)

        # ---- Get available models ----
        available = self._get_available_models(
            require_vision=require_vision,
            require_functions=require_functions,
            preferred_provider=preferred_provider,
        )

        if not available:
            return RoutingDecision(
                model=None,
                provider="none",
                api_model_id="none",
                rejected=True,
                rejection_reason="No LLM providers configured",
                complexity_score=complexity,
                complexity_factors=factors,
            )

        # ---- Select model by complexity ----
        selected_model = self._select_by_complexity(complexity, available)

        # ---- Check budget ----
        budget_remaining = self._get_budget(user_id, user_tier)
        estimated_cost = self._estimate_cost(
            MODEL_CATALOG[selected_model], estimated_tokens
        )

        if estimated_cost > budget_remaining:
            # Try to find cheaper model
            cheaper = self._find_cheaper_model(available, budget_remaining, estimated_tokens)
            if cheaper:
                selected_model = cheaper
                estimated_cost = self._estimate_cost(
                    MODEL_CATALOG[selected_model], estimated_tokens
                )
                factors.append(f"Downgraded to {selected_model} due to budget")
                fallback_used = True
            else:
                logger.warning(
                    "Budget exceeded for user %s (remaining=$%.4f, needed=$%.4f)",
                    user_id, budget_remaining, estimated_cost,
                )
                return RoutingDecision(
                    model=None,
                    provider="none",
                    api_model_id="none",
                    rejected=True,
                    rejection_reason=f"Monthly budget exceeded. Remaining: ${budget_remaining:.4f}",
                    complexity_score=complexity,
                    complexity_factors=factors,
                    budget_remaining_usd=budget_remaining,
                )
        else:
            fallback_used = False

        model_config = MODEL_CATALOG[selected_model]

        reasoning = (
            f"Complexity={complexity:.2f} → {model_config.tier.value} tier. "
            f"Est. cost=${estimated_cost:.4f}. "
            f"Budget remaining=${budget_remaining:.4f}."
        )

        logger.info(
            "Routing decision: %s (complexity=%.2f, cost=$%.4f)",
            selected_model, complexity, estimated_cost,
        )

        return RoutingDecision(
            model=selected_model,
            provider=model_config.provider,
            api_model_id=model_config.api_model_id,
            estimated_cost_usd=estimated_cost,
            complexity_score=complexity,
            complexity_factors=factors,
            fallback_used=fallback_used,
            budget_remaining_usd=budget_remaining,
            reasoning=reasoning,
        )

    def deduct_cost(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_tier: str = "free",
    ) -> float:
        """
        Deduct actual token cost from user's monthly budget.

        Call this AFTER receiving the LLM response with actual token counts.

        Args:
            user_id:       User identifier
            model:         Model name that was used
            input_tokens:  Actual input tokens consumed
            output_tokens: Actual output tokens consumed
            user_tier:     User's tier (for budget limit reference)

        Returns:
            Actual cost deducted in USD
        """
        model_config = MODEL_CATALOG.get(model)
        if not model_config:
            logger.warning("Unknown model for cost deduction: %s", model)
            return 0.0

        input_cost = model_config.input_cost_per_1k * input_tokens / 1000
        output_cost = model_config.output_cost_per_1k * output_tokens / 1000
        total_cost = input_cost + output_cost

        self._deduct_budget(user_id, total_cost, user_tier)

        logger.info(
            "Cost deducted: $%.5f for user %s (model=%s, in=%d, out=%d tokens)",
            total_cost, user_id, model, input_tokens, output_tokens,
        )

        return total_cost

    def get_budget_status(self, user_id: str, user_tier: str = "free") -> Dict[str, Any]:
        """
        Get a user's current budget status.

        Returns:
            Dict with remaining, used, limit, reset_date
        """
        limit = TIER_MONTHLY_BUDGETS.get(user_tier, TIER_MONTHLY_BUDGETS["free"])
        remaining = self._get_budget(user_id, user_tier)
        used = limit - remaining

        return {
            "user_id": user_id,
            "tier": user_tier,
            "monthly_limit_usd": limit,
            "used_usd": round(max(0, used), 5),
            "remaining_usd": round(max(0, remaining), 5),
            "usage_percent": round((used / limit * 100) if limit > 0 else 0, 1),
            "is_exhausted": remaining <= 0,
        }

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get pricing and capability info for a model."""
        config = MODEL_CATALOG.get(model_name)
        if not config:
            return None
        return {
            "name": config.name,
            "provider": config.provider,
            "tier": config.tier,
            "input_cost_per_1k": config.input_cost_per_1k,
            "output_cost_per_1k": config.output_cost_per_1k,
            "max_tokens": config.max_tokens,
            "supports_vision": config.supports_vision,
            "supports_functions": config.supports_functions,
            "quality_score": config.quality_score,
            "description": config.description,
        }

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all configured and enabled models."""
        return [
            self.get_model_info(name)
            for name, config in MODEL_CATALOG.items()
            if config.enabled and self._is_provider_configured(config.provider)
        ]

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _get_available_models(
        self,
        require_vision: bool = False,
        require_functions: bool = False,
        preferred_provider: Optional[str] = None,
    ) -> List[str]:
        """Get list of available model names based on config and capabilities."""
        available = []

        for name, config in MODEL_CATALOG.items():
            if not config.enabled:
                continue
            if not self._is_provider_configured(config.provider):
                continue
            if require_vision and not config.supports_vision:
                continue
            if require_functions and not config.supports_functions:
                continue
            available.append(name)

        # Sort preferred provider first
        if preferred_provider:
            available.sort(
                key=lambda m: (0 if MODEL_CATALOG[m].provider == preferred_provider else 1)
            )

        return available

    def _is_provider_configured(self, provider: str) -> bool:
        """Check if a provider has API keys configured."""
        if provider == "gemini":
            return bool(settings.gemini_api_key)
        if provider == "openai":
            return bool(settings.openai_api_key)
        if provider == "anthropic":
            return bool(settings.anthropic_api_key)
        if provider == "ollama":
            return settings.ollama_enabled
        return False

    def _select_by_complexity(self, complexity: float, available: List[str]) -> str:
        """Select appropriate model tier based on complexity score."""
        if complexity < COMPLEXITY_THRESHOLDS["cheap"]:
            target_tier = ModelTier.CHEAP
        elif complexity < COMPLEXITY_THRESHOLDS["standard"]:
            target_tier = ModelTier.STANDARD
        else:
            target_tier = ModelTier.PREMIUM

        # Find best model in target tier
        tier_models = [
            m for m in available
            if MODEL_CATALOG[m].tier == target_tier
        ]

        if tier_models:
            # Among same-tier models, prefer highest quality
            return max(tier_models, key=lambda m: MODEL_CATALOG[m].quality_score)

        # Fallback: find closest available tier
        for fallback_tier in [ModelTier.STANDARD, ModelTier.CHEAP, ModelTier.PREMIUM, ModelTier.LOCAL]:
            fallback_models = [m for m in available if MODEL_CATALOG[m].tier == fallback_tier]
            if fallback_models:
                return max(fallback_models, key=lambda m: MODEL_CATALOG[m].quality_score)

        return available[0]

    def _estimate_cost(self, config: ModelConfig, estimated_tokens: int) -> float:
        """Estimate cost for a request."""
        # Assume 70% input, 30% output token split
        input_tokens = int(estimated_tokens * 0.7)
        output_tokens = int(estimated_tokens * 0.3)
        return (
            config.input_cost_per_1k * input_tokens / 1000 +
            config.output_cost_per_1k * output_tokens / 1000
        )

    def _find_cheaper_model(
        self,
        available: List[str],
        budget: float,
        estimated_tokens: int,
    ) -> Optional[str]:
        """Find the best model that fits within budget."""
        affordable = [
            m for m in available
            if self._estimate_cost(MODEL_CATALOG[m], estimated_tokens) <= budget
        ]

        if not affordable:
            return None

        # Return highest quality affordable model
        return max(affordable, key=lambda m: MODEL_CATALOG[m].quality_score)

    def _get_budget(self, user_id: str, user_tier: str = "free") -> float:
        """Get user's remaining monthly budget."""
        limit = TIER_MONTHLY_BUDGETS.get(user_tier, TIER_MONTHLY_BUDGETS["free"])
        redis = _get_redis()

        if redis:
            try:
                key = _budget_key(user_id)
                value = redis.get(key)
                if value is not None:
                    return float(value)
                # Initialize budget
                self._init_budget(user_id, user_tier)
                return limit
            except Exception as exc:
                logger.error("Redis budget get failed: %s", exc)

        # Fallback to in-memory
        return self._in_memory_budgets.get(user_id, limit)

    def _deduct_budget(self, user_id: str, amount: float, user_tier: str = "free") -> None:
        """Deduct amount from user's budget."""
        redis = _get_redis()

        if redis:
            try:
                key = _budget_key(user_id)
                if not redis.exists(key):
                    self._init_budget(user_id, user_tier)
                redis.incrbyfloat(key, -amount)
                return
            except Exception as exc:
                logger.error("Redis budget deduct failed: %s", exc)

        # Fallback
        current = self._in_memory_budgets.get(
            user_id,
            TIER_MONTHLY_BUDGETS.get(user_tier, 2.0),
        )
        self._in_memory_budgets[user_id] = current - amount

    def _init_budget(self, user_id: str, user_tier: str = "free") -> None:
        """Initialize budget for a new month."""
        limit = TIER_MONTHLY_BUDGETS.get(user_tier, TIER_MONTHLY_BUDGETS["free"])
        redis = _get_redis()

        if redis:
            try:
                key = _budget_key(user_id)
                # Set budget with TTL until end of month
                import calendar
                now = datetime.now(timezone.utc)
                days_in_month = calendar.monthrange(now.year, now.month)[1]
                seconds_remaining = (days_in_month - now.day) * 86400
                redis.setex(key, max(seconds_remaining, 3600), str(limit))
            except Exception as exc:
                logger.error("Redis budget init failed: %s", exc)
        else:
            self._in_memory_budgets[user_id] = limit


# ===========================================================================
# Module-level singleton
# ===========================================================================
_router_instance: Optional[CostRouter] = None


def get_cost_router() -> CostRouter:
    """Get or create the global CostRouter singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = CostRouter()
    return _router_instance


# ===========================================================================
# Backward compatible functions
# ===========================================================================

def route_query(
    user_id: str,
    query: str,
    estimated_tokens: int = 500,
    user_tier: str = "free",
) -> str:
    """
    Simple routing — returns model name string or 'reject'.
    Backward compatible with original CostAwareRouter.route().
    """
    decision = get_cost_router().route(user_id, query, estimated_tokens, user_tier)
    if decision.rejected:
        return "reject"
    return decision.model or "reject"


def deduct_cost(user_id: str, model: str, tokens: int, user_tier: str = "free") -> float:
    """
    Backward compatible cost deduction.
    Assumes 50/50 input/output token split.
    """
    return get_cost_router().deduct_cost(
        user_id, model,
        input_tokens=tokens // 2,
        output_tokens=tokens // 2,
        user_tier=user_tier,
    )
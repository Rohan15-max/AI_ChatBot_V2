"""
DevMentor AI — Agent Tools
============================
Production-grade secure tool registry with:

- Whitelist-only tool execution (no eval, no subprocess, no filesystem)
- Safe AST-based calculator
- Real web search via Serper (Google search results), with DuckDuckGo as
  a no-key fallback if SERPER_API_KEY isn't configured
- Wikipedia lookup
- Current time/date
- Unit converter
- JSON formatter
- Weather lookup
- Token-aware tool output truncation
- Full audit logging per tool call
- OpenAI function-calling format support
- Rate limiting per tool per user (now actually enforced — see note below)

UPGRADE NOTES (this revision):

1. WEB SEARCH: The previous `_web_search` called only DuckDuckGo's Instant
   Answer API, which returns a curated "instant answer" box (Wikipedia-style
   summaries) rather than ranked organic search results. It frequently
   returned nothing useful for anything beyond well-known factual lookups.
   `_web_search` now calls Serper (a real Google search results API) when
   SERPER_API_KEY is configured, returning actual ranked results with
   titles, snippets, and URLs — the same shape of data Gemini's own
   "grounding" feature uses internally. DuckDuckGo remains as a fallback
   for when no Serper key is set, so the tool degrades gracefully rather
   than failing outright, but its limitations are unchanged.

2. RATE LIMITING: `ToolRegistry.__init__` declared a `self._call_counts`
   dict intended for per-tool-per-user throttling, and each ToolDefinition
   has a `rate_limit_per_minute` field — but nothing in `execute()` ever
   read or wrote `_call_counts`, so the declared limits were entirely
   decorative. `execute()` now actually enforces them via a sliding-window
   check before running the tool.

Usage:
    from agent_tools import tool_registry, parse_tool_calls

    result = tool_registry.execute("calculator", {"expression": "2 + 2"}, user_id="u123")
    tool_calls = parse_tool_calls(llm_response)
"""

import ast
import json
import logging
import operator
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx

from config import get_settings

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Tool whitelist from environment
# ---------------------------------------------------------------------------
_DEFAULT_TOOLS = "web_search,calculator,current_time,wikipedia,unit_converter,json_formatter,weather"
ALLOWED_TOOLS = [
    t.strip()
    for t in os.getenv("AGENT_TOOLS_ENABLED", _DEFAULT_TOOLS).split(",")
    if t.strip()
]

# Max output length per tool (prevents context overflow)
MAX_TOOL_OUTPUT_LENGTH = 2000

# Allowed domains for URL fetcher (security boundary)
URL_FETCH_ALLOWLIST = {
    "en.wikipedia.org",
    "api.duckduckgo.com",
    "wttr.in",
    "api.open-meteo.com",
    "google.serper.dev",
}


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class ToolDefinition:
    """Metadata for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]          # JSON Schema for parameters
    fn: Callable
    enabled: bool = True
    requires_network: bool = False
    rate_limit_per_minute: int = 30


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ToolCall:
    """A parsed tool call from LLM output."""
    tool: str
    parameters: Any
    call_id: Optional[str] = None


# ===========================================================================
# Safe Calculator
# ===========================================================================

_SAFE_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}

_SAFE_MATH_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _safe_eval_node(node: ast.AST) -> float:
    """
    Recursively evaluate an AST node using only whitelisted operators.
    No exec, no eval, no imports — completely safe.
    """
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants allowed, got: {type(node.value).__name__}")
        return node.value

    elif isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if not op_fn:
            raise ValueError(f"Operator '{type(node.op).__name__}' is not allowed")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large (max 100)")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero")
        return op_fn(left, right)

    elif isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if not op_fn:
            raise ValueError(f"Unary operator '{type(node.op).__name__}' is not allowed")
        return op_fn(_safe_eval_node(node.operand))

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Function calls are restricted")
        func_name = node.func.id
        if func_name not in _SAFE_MATH_FUNCS:
            raise ValueError(f"Function '{func_name}' is not allowed. Allowed: {list(_SAFE_MATH_FUNCS)}")
        args = [_safe_eval_node(a) for a in node.args]
        return _SAFE_MATH_FUNCS[func_name](*args)

    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def _calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    No eval() — uses Python AST parsing with operator whitelist.
    Supports: +, -, *, /, //, %, **, abs(), round(), min(), max()

    Args:
        expression: Math expression string e.g. "2 + 2 * 10"

    Returns:
        Result string e.g. "22" or error message
    """
    if not expression or not expression.strip():
        return "Error: Empty expression"

    expression = expression.strip()

    if len(expression) > 500:
        return "Error: Expression too long (max 500 characters)"

    forbidden = ["import", "exec", "eval", "__", "open", "os.", "sys."]
    if any(f in expression for f in forbidden):
        return "Error: Forbidden expression"

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree.body)

        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        elif isinstance(result, float):
            return f"{result:.10g}"
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as exc:
        return f"Error: {exc}"
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except Exception as exc:
        logger.warning("Calculator error for expression '%s': %s", expression[:50], exc)
        return f"Error: {exc}"


# ===========================================================================
# Web Search — Serper (real results) with DuckDuckGo fallback (no key needed)
# ===========================================================================

def _format_serper_results(data: dict, num_results: int) -> str:
    """Render Serper's JSON response into the same plain-text shape other tools use."""
    results = []

    # "Answer box" — Serper's equivalent of a direct/instant answer, when present
    answer_box = data.get("answerBox")
    if answer_box:
        answer_text = answer_box.get("answer") or answer_box.get("snippet")
        if answer_text:
            results.append(f"**Direct answer:** {answer_text}")

    # "Knowledge graph" — entity facts (people, places, companies), when present
    kg = data.get("knowledgeGraph")
    if kg and kg.get("description"):
        title = kg.get("title", "")
        results.append(f"**{title}:** {kg['description']}")

    # Organic results — the actual ranked search results
    organic = data.get("organic", [])[:num_results]
    for i, item in enumerate(organic, 1):
        title = item.get("title", "Untitled")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        entry = f"{i}. **{title}**"
        if snippet:
            entry += f"\n   {snippet}"
        if link:
            entry += f"\n   {link}"
        results.append(entry)

    if not results:
        return ""
    return "\n\n".join(results)


def _web_search_serper(query: str, num_results: int, api_key: str) -> Optional[str]:
    """
    Search via Serper (Google search results API). Returns None on any
    failure so the caller can fall back to DuckDuckGo rather than
    surfacing a raw error to the model.
    """
    try:
        with httpx.Client(timeout=10) as client:
            response = client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": min(max(num_results, 1), 10)},
            )
            response.raise_for_status()
            data = response.json()
        formatted = _format_serper_results(data, num_results)
        return formatted or None
    except httpx.HTTPStatusError as exc:
        logger.warning("Serper search failed (HTTP %s): %s", exc.response.status_code, query[:80])
        return None
    except Exception as exc:
        logger.warning("Serper search failed: %s", exc)
        return None


def _web_search_duckduckgo_fallback(query: str, num_results: int) -> str:
    """
    No-API-key fallback. Only returns a curated instant-answer box, not
    ranked organic results — kept only so the tool degrades gracefully
    when SERPER_API_KEY isn't configured, rather than failing outright.
    """
    try:
        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        with httpx.Client(timeout=10) as client:
            response = client.get(url, headers={"User-Agent": "DevMentorAI/1.0"})
            response.raise_for_status()
            data = response.json()

        results = []
        if data.get("AbstractText"):
            results.append(f"**Summary:** {data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")

        topics = data.get("RelatedTopics", [])[:num_results]
        for topic in topics:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"• {topic['Text']}")

        if results:
            return "\n".join(results)
        return (
            f"No instant answer found for: {query}. "
            "(Configure SERPER_API_KEY for full search results instead of instant-answer-only lookups.)"
        )
    except Exception as exc:
        logger.warning("DuckDuckGo fallback search failed: %s", exc)
        return "Search temporarily unavailable. Please try again."


def _web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for current information, news, or facts.

    Uses Serper (real ranked Google search results) when SERPER_API_KEY is
    configured. Falls back to DuckDuckGo's instant-answer API (no key
    required, but far more limited) otherwise.

    Args:
        query:       Search query
        num_results: Max results to return

    Returns:
        Search results as formatted text
    """
    query = query.strip()[:300]
    if not query:
        return "Error: Empty search query"

    serper_key = getattr(settings, "serper_api_key", None)
    if serper_key:
        key_value = serper_key.get_secret_value() if hasattr(serper_key, "get_secret_value") else str(serper_key)
        result = _web_search_serper(query, num_results, key_value)
        if result:
            return result
        logger.info("Serper returned no usable results for '%s', falling back to DuckDuckGo", query[:80])

    return _web_search_duckduckgo_fallback(query, num_results)


# ===========================================================================
# Other Tool Implementations (unchanged from original)
# ===========================================================================

def _wikipedia_lookup(topic: str, sentences: int = 3) -> str:
    """
    Look up a topic on Wikipedia.

    Args:
        topic:     Topic to look up
        sentences: Number of summary sentences to return

    Returns:
        Wikipedia summary text
    """
    topic = topic.strip()[:200]
    if not topic:
        return "Error: Empty topic"

    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(topic)}"
        with httpx.Client(timeout=10) as client:
            response = client.get(
                url,
                headers={"User-Agent": "DevMentorAI/1.0 (educational use)"},
            )

        if response.status_code == 404:
            return f"No Wikipedia article found for '{topic}'"

        response.raise_for_status()
        data = response.json()

        title = data.get("title", topic)
        extract = data.get("extract", "No content available")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        sentence_list = extract.split(". ")
        summary = ". ".join(sentence_list[:sentences])
        if not summary.endswith("."):
            summary += "."

        result = f"**{title}**\n{summary}"
        if page_url:
            result += f"\nSource: {page_url}"

        return result

    except Exception as exc:
        logger.warning("Wikipedia lookup failed for '%s': %s", topic, exc)
        return f"Wikipedia lookup failed: {exc}"


def _current_time(timezone_name: str = "UTC", format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current date and time.

    Args:
        timezone_name: Timezone name (UTC, IST, etc.) — simplified support
        format_str:    strftime format string

    Returns:
        Formatted datetime string
    """
    safe_format = format_str[:50] if format_str else "%Y-%m-%d %H:%M:%S"
    now_utc = datetime.now(timezone.utc)

    tz_offsets = {
        "UTC": 0, "IST": 5.5, "EST": -5, "PST": -8,
        "CST": -6, "MST": -7, "GMT": 0, "AEST": 10,
        "JST": 9, "CET": 1, "EET": 2,
    }

    offset_hours = tz_offsets.get(timezone_name.upper(), 0)

    from datetime import timedelta
    local_time = now_utc + timedelta(hours=offset_hours)

    try:
        return f"{local_time.strftime(safe_format)} {timezone_name}"
    except Exception:
        return f"{now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"


def _unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.

    Supports: length, weight, temperature, area, volume, speed

    Args:
        value:     Numeric value to convert
        from_unit: Source unit (e.g. km, kg, celsius)
        to_unit:   Target unit (e.g. miles, lbs, fahrenheit)

    Returns:
        Conversion result string
    """
    conversions = {
        "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001,
        "miles": 1609.344, "mile": 1609.344,
        "feet": 0.3048, "ft": 0.3048,
        "inches": 0.0254, "inch": 0.0254, "in": 0.0254,
        "yards": 0.9144, "yd": 0.9144,
        "kg": 1, "g": 0.001, "mg": 0.000001,
        "lbs": 0.453592, "lb": 0.453592,
        "oz": 0.0283495,
        "tonnes": 1000, "ton": 907.185,
        "kmh": 0.277778, "mph": 0.44704, "ms": 1, "knots": 0.514444,
    }

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    temp_units = {"celsius", "fahrenheit", "kelvin", "c", "f", "k"}
    if from_unit in temp_units and to_unit in temp_units:
        return _convert_temperature(value, from_unit, to_unit)

    if from_unit not in conversions:
        return f"Unknown unit: {from_unit}"
    if to_unit not in conversions:
        return f"Unknown unit: {to_unit}"

    length_units = {"km", "m", "cm", "mm", "miles", "mile", "feet", "ft", "inches", "inch", "in", "yards", "yd"}
    weight_units = {"kg", "g", "mg", "lbs", "lb", "oz", "tonnes", "ton"}
    speed_units = {"kmh", "mph", "ms", "knots"}

    def _get_category(unit):
        if unit in length_units: return "length"
        if unit in weight_units: return "weight"
        if unit in speed_units: return "speed"
        return "unknown"

    if _get_category(from_unit) != _get_category(to_unit):
        return f"Cannot convert {from_unit} to {to_unit} — different unit categories"

    base_value = value * conversions[from_unit]
    result = base_value / conversions[to_unit]

    return f"{value} {from_unit} = {result:.6g} {to_unit}"


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    from_unit = from_unit[0].lower()
    to_unit = to_unit[0].lower()

    if from_unit == "c":
        celsius = value
    elif from_unit == "f":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "k":
        celsius = value - 273.15
    else:
        return f"Unknown temperature unit: {from_unit}"

    if to_unit == "c":
        result = celsius
        unit_name = "°C"
    elif to_unit == "f":
        result = celsius * 9 / 5 + 32
        unit_name = "°F"
    elif to_unit == "k":
        result = celsius + 273.15
        unit_name = "K"
    else:
        return f"Unknown temperature unit: {to_unit}"

    return f"{value}° → {result:.2f} {unit_name}"


def _json_formatter(json_string: str, indent: int = 2) -> str:
    """
    Format and validate a JSON string.

    Args:
        json_string: Raw JSON string to format
        indent:      Indentation spaces

    Returns:
        Formatted JSON or error message
    """
    if not json_string or not json_string.strip():
        return "Error: Empty JSON input"

    if len(json_string) > 50000:
        return "Error: JSON too large (max 50KB)"

    try:
        parsed = json.loads(json_string.strip())
        formatted = json.dumps(parsed, indent=min(indent, 8), ensure_ascii=False)
        lines = formatted.count("\n") + 1
        return f"Valid JSON ({lines} lines):\n```json\n{formatted}\n```"
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"


def _weather(location: str) -> str:
    """
    Get current weather for a location using wttr.in (free, no API key).

    Args:
        location: City name or coordinates

    Returns:
        Weather summary
    """
    location = location.strip()[:100]
    if not location:
        return "Error: Location required"

    try:
        url = f"https://wttr.in/{quote_plus(location)}?format=j1"
        with httpx.Client(timeout=10) as client:
            response = client.get(
                url,
                headers={"User-Agent": "DevMentorAI/1.0"},
            )

        if response.status_code != 200:
            return f"Weather data unavailable for '{location}'"

        data = response.json()
        current = data["current_condition"][0]

        temp_c = current.get("temp_C", "?")
        temp_f = current.get("temp_F", "?")
        feels_c = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
        wind_kmph = current.get("windspeedKmph", "?")

        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", location)
        country = area.get("country", [{}])[0].get("value", "")

        return (
            f"**Weather in {area_name}, {country}**\n"
            f"Condition: {desc}\n"
            f"Temperature: {temp_c}°C / {temp_f}°F (feels like {feels_c}°C)\n"
            f"Humidity: {humidity}%\n"
            f"Wind: {wind_kmph} km/h"
        )

    except Exception as exc:
        logger.warning("Weather lookup failed for '%s': %s", location, exc)
        return f"Weather data unavailable for '{location}'"


# ===========================================================================
# Tool Registry
# ===========================================================================

class ToolRegistry:
    """
    Secure, whitelist-based tool registry.

    Only tools explicitly in ALLOWED_TOOLS can be executed.
    All executions are logged. Parameters are validated.
    Per-tool, per-user rate limits (rate_limit_per_minute on each
    ToolDefinition) are now actually enforced — see _check_rate_limit.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        # tool_name -> user_id -> list of call timestamps (sliding window)
        self._call_counts: Dict[str, Dict[str, List[float]]] = {}
        self._call_counts_lock_holder = "single-threaded-best-effort"
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all tools that are in the ALLOWED_TOOLS whitelist."""
        all_tools = {
            "calculator": ToolDefinition(
                name="calculator",
                description="Evaluate mathematical expressions safely. Supports +, -, *, /, **, %, abs(), round(), min(), max().",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate e.g. '2 + 2 * 10'",
                        }
                    },
                    "required": ["expression"],
                },
                fn=_calculator,
                requires_network=False,
            ),
            "web_search": ToolDefinition(
                name="web_search",
                description="Search the web for current information, news, or facts. Returns real ranked search results.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results (1-10)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                fn=_web_search,
                requires_network=True,
                rate_limit_per_minute=20,
            ),
            "wikipedia": ToolDefinition(
                name="wikipedia",
                description="Look up factual information about a topic on Wikipedia.",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to look up",
                        },
                        "sentences": {
                            "type": "integer",
                            "description": "Number of summary sentences (1-10)",
                            "default": 3,
                        },
                    },
                    "required": ["topic"],
                },
                fn=_wikipedia_lookup,
                requires_network=True,
                rate_limit_per_minute=30,
            ),
            "current_time": ToolDefinition(
                name="current_time",
                description="Get the current date and time in a specified timezone.",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone_name": {
                            "type": "string",
                            "description": "Timezone abbreviation: UTC, IST, EST, PST, AEST, etc.",
                            "default": "UTC",
                        },
                        "format_str": {
                            "type": "string",
                            "description": "strftime format string",
                            "default": "%Y-%m-%d %H:%M:%S",
                        },
                    },
                    "required": [],
                },
                fn=_current_time,
                requires_network=False,
            ),
            "unit_converter": ToolDefinition(
                name="unit_converter",
                description="Convert between units of length, weight, temperature, and speed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "Value to convert",
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "Source unit e.g. km, kg, celsius, mph",
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "Target unit e.g. miles, lbs, fahrenheit, kmh",
                        },
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
                fn=_unit_converter,
                requires_network=False,
            ),
            "json_formatter": ToolDefinition(
                name="json_formatter",
                description="Format, validate, and prettify a JSON string.",
                parameters={
                    "type": "object",
                    "properties": {
                        "json_string": {
                            "type": "string",
                            "description": "Raw JSON string to format",
                        },
                        "indent": {
                            "type": "integer",
                            "description": "Indentation spaces (default 2)",
                            "default": 2,
                        },
                    },
                    "required": ["json_string"],
                },
                fn=_json_formatter,
                requires_network=False,
            ),
            "weather": ToolDefinition(
                name="weather",
                description="Get current weather conditions for any city or location.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location e.g. 'Raipur' or 'New York'",
                        },
                    },
                    "required": ["location"],
                },
                fn=_weather,
                requires_network=True,
                rate_limit_per_minute=15,
            ),
        }

        for name, tool_def in all_tools.items():
            if name in ALLOWED_TOOLS:
                self._tools[name] = tool_def
                logger.debug("Tool registered: %s", name)

        logger.info(
            "Tool registry initialized with %d tools: %s",
            len(self._tools),
            list(self._tools.keys()),
        )

    def _check_rate_limit(self, tool_name: str, user_id: str, limit_per_minute: int) -> bool:
        """
        Sliding-window rate limit check, in-process (per-worker). Returns
        True if the call is allowed, False if the user has exceeded
        limit_per_minute calls to this tool within the last 60 seconds.

        This is intentionally simple in-memory bookkeeping rather than a
        Redis-backed limiter — agent tool calls happen within a single
        request's agent loop, not across distributed long-lived sessions,
        so per-process accuracy is an acceptable tradeoff for not adding
        a Redis round-trip to every tool call. If you run multiple worker
        processes, each enforces its own limit independently (slightly
        more permissive in aggregate than the stated limit, never less).
        """
        now = time.time()
        window_start = now - 60.0

        tool_calls = self._call_counts.setdefault(tool_name, {})
        timestamps = tool_calls.setdefault(user_id, [])

        # Drop timestamps outside the sliding window
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= limit_per_minute:
            return False

        timestamps.append(now)
        return True

    def execute(
        self,
        tool_name: str,
        parameters: Any,
        user_id: str = "anonymous",
    ) -> ToolResult:
        """
        Execute a tool safely.

        Args:
            tool_name:  Name of tool to execute
            parameters: Tool parameters (dict or single value)
            user_id:    User ID for rate limiting and logging

        Returns:
            ToolResult with output or error
        """
        start_time = time.time()

        if tool_name not in self._tools:
            logger.warning(
                "Blocked attempt to call unknown/disabled tool: %s (user=%s)",
                tool_name, user_id,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' is not available",
                duration_ms=0,
            )

        tool_def = self._tools[tool_name]

        # ---- Rate limit check (now actually enforced) ----
        if not self._check_rate_limit(tool_name, user_id, tool_def.rate_limit_per_minute):
            logger.warning(
                "Rate limit exceeded: tool=%s user=%s limit=%d/min",
                tool_name, user_id, tool_def.rate_limit_per_minute,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Rate limit exceeded for '{tool_name}' ({tool_def.rate_limit_per_minute}/min). Please wait before retrying.",
                duration_ms=int((time.time() - start_time) * 1000),
            )

        try:
            if isinstance(parameters, dict):
                output = tool_def.fn(**parameters)
            elif parameters is None:
                output = tool_def.fn()
            else:
                output = tool_def.fn(parameters)

            output = str(output)

            if len(output) > MAX_TOOL_OUTPUT_LENGTH:
                output = output[:MAX_TOOL_OUTPUT_LENGTH] + f"\n\n[Output truncated — {len(output)} chars total]"

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Tool executed: %s (user=%s, duration=%dms)",
                tool_name, user_id, duration_ms,
            )

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )

        except TypeError as exc:
            logger.error("Tool %s parameter error: %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Invalid parameters: {exc}",
                duration_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as exc:
            logger.error("Tool %s execution failed: %s", tool_name, exc, exc_info=True)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool execution failed: {exc}",
                duration_ms=int((time.time() - start_time) * 1000),
            )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function-calling format schemas for all registered tools.
        Used to pass tool definitions to the LLM.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
            if tool.enabled
        ]

    def list_tools(self) -> List[str]:
        """Return list of available tool names."""
        return list(self._tools.keys())

    def is_available(self, tool_name: str) -> bool:
        """Check if a tool is registered and enabled."""
        return tool_name in self._tools and self._tools[tool_name].enabled


# ===========================================================================
# Tool Call Parsing
# ===========================================================================

def parse_tool_calls(response_text: str) -> List[ToolCall]:
    """
    Parse tool calls from LLM response text.

    Supports multiple formats:
    1. Custom format:  [[tool:tool_name({"param": "value"})]]
    2. JSON blocks:    ```tool_call\n{"tool": "...", "parameters": {...}}\n```
    3. OpenAI format:  Handled separately via API response parsing

    Args:
        response_text: Raw LLM response text

    Returns:
        List of ToolCall objects
    """
    calls = []

    pattern1 = r'\[\[tool:(\w+)\((.*?)\)\]\]'
    for match in re.finditer(pattern1, response_text, re.DOTALL):
        tool_name = match.group(1)
        params_str = match.group(2).strip()
        parameters = _parse_params(params_str)
        calls.append(ToolCall(tool=tool_name, parameters=parameters))

    pattern2 = r'```tool_call\s*\n(.*?)\n```'
    for match in re.finditer(pattern2, response_text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            tool_name = data.get("tool") or data.get("name")
            parameters = data.get("parameters") or data.get("arguments") or {}
            if tool_name:
                calls.append(ToolCall(
                    tool=tool_name,
                    parameters=parameters,
                    call_id=data.get("id"),
                ))
        except json.JSONDecodeError:
            pass

    return calls


def parse_openai_tool_calls(tool_calls_data: List[Dict]) -> List[ToolCall]:
    """
    Parse tool calls from OpenAI API response format.

    Args:
        tool_calls_data: The tool_calls array from OpenAI API response

    Returns:
        List of ToolCall objects
    """
    calls = []
    for tc in tool_calls_data or []:
        try:
            fn = tc.get("function", {})
            tool_name = fn.get("name")
            arguments_str = fn.get("arguments", "{}")
            parameters = json.loads(arguments_str)
            calls.append(ToolCall(
                tool=tool_name,
                parameters=parameters,
                call_id=tc.get("id"),
            ))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse OpenAI tool call: %s", exc)

    return calls


def _parse_params(params_str: str) -> Any:
    """Parse parameter string — tries JSON first, then string."""
    if not params_str:
        return {}

    if params_str.startswith("{") and params_str.endswith("}"):
        try:
            return json.loads(params_str)
        except json.JSONDecodeError:
            pass

    if (params_str.startswith('"') and params_str.endswith('"')) or \
       (params_str.startswith("'") and params_str.endswith("'")):
        return params_str[1:-1]

    return params_str


# ===========================================================================
# Singleton
# ===========================================================================
tool_registry = ToolRegistry()
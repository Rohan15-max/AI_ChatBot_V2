"""
DevMentor AI — Prompt Injection Detector
==========================================
Production-grade prompt injection detection and sanitization with:

- Comprehensive injection pattern library
- Multi-layer detection (patterns + heuristics + scoring)
- Severity scoring (low / medium / high / critical)
- Sanitization with configurable aggressiveness
- Jailbreak attempt detection
- Role confusion attack detection
- Prompt leaking attempt detection
- Structured logging for all detections
- Allow/block list support

Usage:
    from security.prompt_injection import detect_injection, sanitize_prompt

    is_safe, severity, reasons = detect_injection(user_input)
    if not is_safe:
        return 400

    clean_input = sanitize_prompt(user_input)
"""

import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# Severity Levels
# ===========================================================================

class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ===========================================================================
# Injection Pattern Definitions
# ===========================================================================

@dataclass
class InjectionPattern:
    """A single injection detection pattern."""
    name: str
    pattern: str
    severity: Severity
    description: str
    score: int = 1       # Contribution to total risk score
    flags: int = re.IGNORECASE | re.DOTALL
    enabled: bool = True


_INJECTION_PATTERNS: List[InjectionPattern] = [

    # -----------------------------------------------------------------------
    # Direct instruction override attempts
    # -----------------------------------------------------------------------
    InjectionPattern(
        name="ignore_instructions",
        pattern=r"ignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions?|rules?|prompts?|constraints?|guidelines?)",
        severity=Severity.CRITICAL,
        score=10,
        description="Direct attempt to override system instructions",
    ),
    InjectionPattern(
        name="forget_instructions",
        pattern=r"forget\s+(everything|all|your|the)\s*(previous|prior|above|instructions?|rules?|prompts?)?",
        severity=Severity.CRITICAL,
        score=10,
        description="Attempt to make AI forget its instructions",
    ),
    InjectionPattern(
        name="disregard_instructions",
        pattern=r"(disregard|override|bypass|skip|cancel|void|nullify|supersede)\s+(all\s+)?(previous|prior|your|the|system)?\s*(instructions?|rules?|prompts?|constraints?)",
        severity=Severity.CRITICAL,
        score=10,
        description="Attempt to disregard or bypass instructions",
    ),
    InjectionPattern(
        name="new_instructions",
        pattern=r"(your\s+new|from\s+now\s+on|henceforth|instead|new)\s*(instructions?|rules?|directives?|guidelines?|purpose|goal|task)\s*(are|is|will\s+be)?[:\s]",
        severity=Severity.HIGH,
        score=8,
        description="Attempt to inject new instructions",
    ),

    # -----------------------------------------------------------------------
    # Role / persona manipulation
    # -----------------------------------------------------------------------
    InjectionPattern(
        name="role_override",
        pattern=r"you\s+are\s+(now\s+)?(a|an|the)\s+(?!helpful|assistant|AI)[a-z\s]{3,40}(assistant|bot|ai|model|system)?",
        severity=Severity.HIGH,
        score=7,
        description="Attempt to assign a different role/persona to the AI",
    ),
    InjectionPattern(
        name="act_as",
        pattern=r"\b(act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as|play\s+the\s+role\s+of|simulate\s+being|behave\s+as|respond\s+as)\b",
        severity=Severity.HIGH,
        score=7,
        description="Attempt to make AI adopt a different persona",
    ),
    InjectionPattern(
        name="dan_jailbreak",
        pattern=r"\b(DAN|do\s+anything\s+now|jailbreak|jail\s+break|unrestricted\s+mode|developer\s+mode|god\s+mode|evil\s+mode)\b",
        severity=Severity.CRITICAL,
        score=10,
        description="Known jailbreak attempts (DAN and variants)",
    ),
    InjectionPattern(
        name="no_restrictions",
        pattern=r"(without\s+(any\s+)?(restrictions?|limits?|filters?|rules?|guidelines?|ethics?|constraints?)|unrestricted(ly)?|uncensored|unfiltered)",
        severity=Severity.HIGH,
        score=8,
        description="Attempt to make AI operate without restrictions",
    ),

    # -----------------------------------------------------------------------
    # System prompt injection / leaking
    # -----------------------------------------------------------------------
    InjectionPattern(
        name="system_tag_injection",
        pattern=r"(<\s*system\s*>|<\s*/\s*system\s*>|\[system\]|\{system\})",
        severity=Severity.CRITICAL,
        score=10,
        description="Attempt to inject system-level tags",
    ),
    InjectionPattern(
        name="role_tag_injection",
        pattern=r"(<\s*(user|assistant|human|ai|bot)\s*>|\[(user|assistant|human|ai|bot)\])",
        severity=Severity.HIGH,
        score=8,
        description="Attempt to inject conversation role tags",
    ),
    InjectionPattern(
        name="special_tokens",
        pattern=r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|<s>|</s>)",
        severity=Severity.CRITICAL,
        score=10,
        description="LLM special tokens used for prompt injection",
    ),
    InjectionPattern(
        name="prompt_leak_attempt",
        pattern=r"(repeat|print|output|show|reveal|display|tell\s+me|what\s+is)\s+(your\s+)?(system\s+prompt|instructions?|initial\s+prompt|original\s+prompt|context|training\s+data)",
        severity=Severity.HIGH,
        score=7,
        description="Attempt to extract system prompt or training data",
    ),

    # -----------------------------------------------------------------------
    # Delimiter / separator injection
    # -----------------------------------------------------------------------
    InjectionPattern(
        name="delimiter_injection",
        pattern=r"(#{3,}|\*{3,}|-{5,}|={5,}|_{5,})\s*(system|instructions?|prompt|override|end\s+of\s+(prompt|instructions?))",
        severity=Severity.HIGH,
        score=7,
        description="Delimiter-based prompt section injection",
    ),
    InjectionPattern(
        name="separator_injection",
        pattern=r"(end\s+of\s+(conversation|prompt|instructions?|context)|begin\s+(new|fresh)\s+(conversation|instructions?|task))",
        severity=Severity.MEDIUM,
        score=5,
        description="Attempt to signal end of prompt and start new context",
    ),

    # -----------------------------------------------------------------------
    # Indirect / encoded injection
    # -----------------------------------------------------------------------
    InjectionPattern(
        name="base64_injection",
        pattern=r"(?:decode\s+(?:this|the\s+following)|base64)[:\s]+[A-Za-z0-9+/]{20,}={0,2}",
        severity=Severity.MEDIUM,
        score=6,
        description="Encoded payload attempting to bypass text filters",
    ),
    InjectionPattern(
        name="unicode_injection",
        pattern=r"\\u[0-9a-fA-F]{4}(\\u[0-9a-fA-F]{4}){4,}",
        severity=Severity.MEDIUM,
        score=5,
        description="Unicode escape sequence injection",
    ),

    # -----------------------------------------------------------------------
    # Context manipulation
    # -----------------------------------------------------------------------
    InjectionPattern(
        name="context_escape",
        pattern=r"(end\s+of\s+input|end\s+of\s+user\s+input|assistant\s*:|ai\s*:|bot\s*:)\s*\n",
        severity=Severity.HIGH,
        score=8,
        description="Attempt to escape user context and speak as assistant",
    ),
    InjectionPattern(
        name="training_manipulation",
        pattern=r"(add\s+this\s+to\s+your\s+training|remember\s+this\s+for\s+future|update\s+your\s+(training|weights|parameters|memory|knowledge))",
        severity=Severity.MEDIUM,
        score=5,
        description="Attempt to manipulate AI training or persistent memory",
    ),
    InjectionPattern(
        name="hypothetical_bypass",
        pattern=r"(hypothetically|in\s+a\s+fictional\s+(world|scenario|context)|imagine\s+you\s+(had\s+no|were\s+without)\s+(rules?|restrictions?|ethics?))",
        severity=Severity.MEDIUM,
        score=4,
        description="Using hypothetical framing to bypass restrictions",
    ),

    # -----------------------------------------------------------------------
    # Harmful content extraction
    # -----------------------------------------------------------------------
    InjectionPattern(
        name="harmful_content_request",
        pattern=r"(how\s+to\s+(make|build|create|synthesize|hack|exploit|bypass)\s+(bomb|weapon|malware|virus|exploit|drugs?))",
        severity=Severity.CRITICAL,
        score=10,
        description="Request for harmful content creation instructions",
    ),
    InjectionPattern(
        name="safety_bypass",
        pattern=r"(bypass|disable|turn\s+off|remove|ignore)\s+(safety|content\s+filter|guardrails?|moderation|restrictions?)",
        severity=Severity.CRITICAL,
        score=10,
        description="Explicit attempt to disable safety systems",
    ),
]

# Sort by score descending
_INJECTION_PATTERNS.sort(key=lambda p: p.score, reverse=True)

# Compile patterns
_COMPILED_PATTERNS: List[Tuple[InjectionPattern, re.Pattern]] = [
    (p, re.compile(p.pattern, p.flags))
    for p in _INJECTION_PATTERNS
    if p.enabled
]

# Risk score thresholds
_SCORE_THRESHOLDS = {
    Severity.LOW:      1,
    Severity.MEDIUM:   4,
    Severity.HIGH:     7,
    Severity.CRITICAL: 10,
}


# ===========================================================================
# Detection Result
# ===========================================================================

@dataclass
class DetectionResult:
    """Result of injection detection analysis."""
    is_safe: bool
    severity: Optional[Severity]
    total_score: int
    triggered_patterns: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None


# ===========================================================================
# Core Detection
# ===========================================================================

def detect_injection(
    user_input: str,
    threshold_score: int = 4,
    max_input_length: int = 50000,
) -> Tuple[bool, Optional[Severity], List[str]]:
    """
    Analyze input for prompt injection attempts.

    Args:
        user_input:       Text to analyze
        threshold_score:  Minimum score to flag as unsafe (default: 4 = MEDIUM)
        max_input_length: Truncate inputs longer than this

    Returns:
        Tuple of (is_safe, severity, list_of_reasons)
        - is_safe: True if input appears safe
        - severity: Highest severity detected, or None if safe
        - reasons: Human-readable list of what was detected

    Example:
        is_safe, severity, reasons = detect_injection(user_input)
        if not is_safe:
            logger.warning("Injection attempt: %s", reasons)
            return jsonify({"error": "Invalid input"}), 400
    """
    if not user_input or not isinstance(user_input, str):
        return True, None, []

    # Truncate very long inputs
    if len(user_input) > max_input_length:
        logger.warning(
            "Input truncated for injection check: %d → %d chars",
            len(user_input), max_input_length,
        )
        user_input = user_input[:max_input_length]

    # Normalize for detection (doesn't affect original)
    normalized = _normalize_input(user_input)

    total_score = 0
    triggered = []
    reasons = []
    highest_severity = None

    for pattern_def, compiled in _COMPILED_PATTERNS:
        match = compiled.search(normalized)
        if match:
            total_score += pattern_def.score
            triggered.append(pattern_def.name)
            reasons.append(f"{pattern_def.severity.value.upper()}: {pattern_def.description}")

            if highest_severity is None or _severity_rank(pattern_def.severity) > _severity_rank(highest_severity):
                highest_severity = pattern_def.severity

    # Add heuristic checks
    heuristic_score, heuristic_reasons = _run_heuristics(normalized)
    total_score += heuristic_score
    reasons.extend(heuristic_reasons)

    is_safe = total_score < threshold_score

    if not is_safe:
        logger.warning(
            "Prompt injection detected",
            extra={
                "score": total_score,
                "severity": highest_severity,
                "patterns": triggered,
                "input_length": len(user_input),
            },
        )
    else:
        logger.debug("Input passed injection check (score=%d)", total_score)

    return is_safe, highest_severity if not is_safe else None, reasons


def analyze_injection(user_input: str) -> DetectionResult:
    """
    Full injection analysis returning a structured DetectionResult.
    More detailed than detect_injection() — use for logging/admin.
    """
    is_safe, severity, reasons = detect_injection(user_input)

    total_score = 0
    triggered = []
    normalized = _normalize_input(user_input) if user_input else ""

    for pattern_def, compiled in _COMPILED_PATTERNS:
        if compiled.search(normalized):
            total_score += pattern_def.score
            triggered.append(pattern_def.name)

    return DetectionResult(
        is_safe=is_safe,
        severity=severity,
        total_score=total_score,
        triggered_patterns=triggered,
        reasons=reasons,
        sanitized_input=sanitize_prompt(user_input) if not is_safe else user_input,
    )


# ===========================================================================
# Heuristic Checks
# ===========================================================================

def _run_heuristics(text: str) -> Tuple[int, List[str]]:
    """
    Additional heuristic checks beyond regex patterns.
    Returns (score_addition, reasons_list).
    """
    score = 0
    reasons = []

    # Excessive special characters (possible obfuscation)
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if special_char_ratio > 0.4:
        score += 3
        reasons.append("MEDIUM: Unusually high ratio of special characters (possible obfuscation)")

    # Excessive line breaks (common in delimiter injection)
    newline_count = text.count("\n")
    if newline_count > 20:
        score += 2
        reasons.append("LOW: Excessive newlines — possible delimiter injection attempt")

    # Very long single words (possible encoded payload)
    words = text.split()
    if any(len(w) > 200 for w in words):
        score += 3
        reasons.append("MEDIUM: Unusually long word — possible encoded payload")

    # Repetitive instruction-like structure
    instruction_words = ["must", "should", "always", "never", "do not", "you will", "your task"]
    instruction_count = sum(text.lower().count(w) for w in instruction_words)
    if instruction_count > 8:
        score += 2
        reasons.append("LOW: High density of instruction-like language")

    return score, reasons


def _normalize_input(text: str) -> str:
    """
    Normalize text to catch obfuscated injection attempts.
    - Normalize unicode
    - Collapse whitespace
    - Lowercase for pattern matching
    """
    # Normalize unicode (catches lookalike characters)
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _severity_rank(severity: Severity) -> int:
    """Convert severity to numeric rank for comparison."""
    return {
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }.get(severity, 0)


# ===========================================================================
# Sanitization
# ===========================================================================

def sanitize_prompt(
    user_input: str,
    aggressive: bool = False,
) -> str:
    """
    Sanitize user input to remove or neutralize injection attempts.

    This is a secondary defense — detection should be the primary gate.
    Sanitization handles edge cases that slip through detection.

    Args:
        user_input: Raw user input
        aggressive: If True, removes more content (may affect legitimate input)

    Returns:
        Sanitized input string

    Note:
        Never rely on sanitization alone — always run detect_injection() first.
    """
    if not user_input:
        return user_input

    text = user_input

    # HTML escape < > to prevent tag injection
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove LLM special tokens
    special_tokens = [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")

    if aggressive:
        # Remove backtick code blocks that could hide injections
        text = re.sub(r"```[\s\S]*?```", "[CODE_BLOCK_REMOVED]", text)

        # Collapse excessive newlines
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        # Remove common injection phrases
        injection_phrases = [
            "ignore previous instructions",
            "forget your rules",
            "you are now",
            "act as",
            "pretend to be",
        ]
        for phrase in injection_phrases:
            text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)

    return text.strip()


# ===========================================================================
# Convenience wrapper for Flask routes
# ===========================================================================

def check_input(
    user_input: str,
    block_on_medium: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Simple pass/fail check for use in Flask route handlers.

    Args:
        user_input:      Input to check
        block_on_medium: If True, block MEDIUM severity (default: only HIGH+)

    Returns:
        Tuple of (is_allowed, error_message_or_None)

    Example:
        allowed, error = check_input(request.json.get("message"))
        if not allowed:
            return jsonify({"error": error}), 400
    """
    threshold = 4 if block_on_medium else 7
    is_safe, severity, _ = detect_injection(user_input, threshold_score=threshold)

    if not is_safe:
        return False, f"Input rejected: potential prompt injection detected ({severity})"

    return True, None


# ===========================================================================
# Pattern listing for admin panel
# ===========================================================================

def list_patterns() -> List[Dict]:
    """List all injection patterns for admin panel display."""
    return [
        {
            "name": p.name,
            "severity": p.severity,
            "score": p.score,
            "description": p.description,
            "enabled": p.enabled,
        }
        for p in _INJECTION_PATTERNS
    ]
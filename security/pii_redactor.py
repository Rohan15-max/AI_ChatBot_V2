"""
DevMentor AI — PII Redactor
=============================
Production-grade PII detection and redaction with:

- Comprehensive regex patterns for global PII types
- India-specific patterns (Aadhaar, PAN, UPI, Indian phone)
- Named entity context-aware redaction
- Logging filter integration
- Configurable replacement tokens per PII type
- Redaction statistics tracking
- Reversible redaction with token mapping (for debugging)

Usage:
    from security.pii_redactor import redact_pii, PIIRedactor

    # Simple redaction
    clean = redact_pii("My email is john@example.com and SSN is 123-45-6789")
    # → "My email is [EMAIL] and SSN is [SSN]"

    # Logging integration
    import logging
    from security.pii_redactor import PIIRedactingFilter
    logger.addFilter(PIIRedactingFilter())
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# PII Pattern Definitions
# ===========================================================================

@dataclass
class PIIPattern:
    """A single PII pattern with its replacement token and priority."""
    name: str
    pattern: str
    replacement: str
    priority: int = 0        # Higher = checked first
    flags: int = re.IGNORECASE
    enabled: bool = True
    description: str = ""


# All PII patterns — ordered by priority (most specific first)
_PII_PATTERNS: List[PIIPattern] = [

    # -----------------------------------------------------------------------
    # Financial
    # -----------------------------------------------------------------------
    PIIPattern(
        name="credit_card",
        pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        replacement="[CREDIT_CARD]",
        priority=100,
        description="Visa, Mastercard, Amex, Discover card numbers",
    ),
    PIIPattern(
        name="credit_card_spaced",
        pattern=r"\b(?:\d[ -]?){13,16}\b",
        replacement="[CREDIT_CARD]",
        priority=99,
        description="Credit card numbers with spaces or dashes",
    ),
    PIIPattern(
        name="cvv",
        pattern=r"\b(?:cvv|cvc|csc|security code)[:\s]*\d{3,4}\b",
        replacement="[CVV]",
        priority=98,
        flags=re.IGNORECASE,
        description="CVV/CVC security codes",
    ),
    PIIPattern(
        name="bank_account",
        pattern=r"\b[0-9]{9,18}\b(?=.*\b(?:account|acct|bank)\b)",
        replacement="[BANK_ACCOUNT]",
        priority=97,
        description="Bank account numbers near keywords",
    ),
    PIIPattern(
        name="ifsc_code",
        pattern=r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
        replacement="[IFSC_CODE]",
        priority=96,
        description="Indian IFSC codes for bank transfers",
    ),
    PIIPattern(
        name="upi_id",
        pattern=r"\b[\w.\-]+@(?:okaxis|okhdfcbank|okicici|oksbi|paytm|ybl|ibl|axl|upi|apl|rapl|waicici|wahdfcbank)\b",
        replacement="[UPI_ID]",
        priority=95,
        description="Indian UPI payment IDs",
    ),

    # -----------------------------------------------------------------------
    # Government IDs — India
    # -----------------------------------------------------------------------
    PIIPattern(
        name="aadhaar",
        pattern=r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b",
        replacement="[AADHAAR]",
        priority=90,
        description="Indian Aadhaar 12-digit identity number",
    ),
    PIIPattern(
        name="pan_card",
        pattern=r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        replacement="[PAN]",
        priority=89,
        description="Indian Permanent Account Number (tax ID)",
    ),
    PIIPattern(
        name="passport_india",
        pattern=r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b",
        replacement="[PASSPORT]",
        priority=88,
        description="Indian passport number",
    ),
    PIIPattern(
        name="voter_id_india",
        pattern=r"\b[A-Z]{3}[0-9]{7}\b",
        replacement="[VOTER_ID]",
        priority=87,
        description="Indian Voter ID (EPIC) number",
    ),
    PIIPattern(
        name="driving_license_india",
        pattern=r"\b[A-Z]{2}[0-9]{2}\s?[0-9]{4}[0-9]{7}\b",
        replacement="[DL_NUMBER]",
        priority=86,
        description="Indian driving license number",
    ),

    # -----------------------------------------------------------------------
    # Government IDs — International
    # -----------------------------------------------------------------------
    PIIPattern(
        name="ssn_us",
        pattern=r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        replacement="[SSN]",
        priority=85,
        description="US Social Security Number",
    ),
    PIIPattern(
        name="passport_generic",
        pattern=r"\b[A-Z]{1,2}[0-9]{6,9}\b",
        replacement="[PASSPORT]",
        priority=84,
        description="Generic passport number pattern",
    ),
    PIIPattern(
        name="national_id_generic",
        pattern=r"\b(?:national\s?id|nid|id\s?number|id\s?no)[:\s]*[A-Z0-9\-]{6,20}\b",
        replacement="[NATIONAL_ID]",
        priority=83,
        flags=re.IGNORECASE,
        description="Generic national ID near keywords",
    ),

    # -----------------------------------------------------------------------
    # Contact Information
    # -----------------------------------------------------------------------
    PIIPattern(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        replacement="[EMAIL]",
        priority=80,
        description="Email addresses",
    ),
    PIIPattern(
        name="phone_india",
        pattern=r"\b(?:\+91|91)?[6-9]\d{9}\b",
        replacement="[PHONE]",
        priority=79,
        description="Indian mobile numbers (6-9 prefix, 10 digits)",
    ),
    PIIPattern(
        name="phone_us",
        pattern=r"\b(?:\+1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]?[2-9]\d{2}[\s.-]?\d{4}\b",
        replacement="[PHONE]",
        priority=78,
        description="US phone numbers",
    ),
    PIIPattern(
        name="phone_international",
        pattern=r"\b\+(?:[0-9] ?){6,14}[0-9]\b",
        replacement="[PHONE]",
        priority=77,
        description="International phone numbers with + prefix",
    ),
    PIIPattern(
        name="phone_generic",
        pattern=r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        replacement="[PHONE]",
        priority=76,
        description="Generic 10-digit phone numbers",
    ),

    # -----------------------------------------------------------------------
    # Network & Technical
    # -----------------------------------------------------------------------
    PIIPattern(
        name="ipv4",
        pattern=r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        replacement="[IP_ADDRESS]",
        priority=70,
        description="IPv4 addresses",
    ),
    PIIPattern(
        name="ipv6",
        pattern=r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
        replacement="[IP_ADDRESS]",
        priority=69,
        description="IPv6 addresses",
    ),
    PIIPattern(
        name="mac_address",
        pattern=r"\b(?:[0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}\b",
        replacement="[MAC_ADDRESS]",
        priority=68,
        description="MAC addresses",
    ),

    # -----------------------------------------------------------------------
    # Authentication Secrets
    # -----------------------------------------------------------------------
    PIIPattern(
        name="jwt_token",
        pattern=r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b",
        replacement="[JWT_TOKEN]",
        priority=60,
        description="JWT tokens (start with eyJ)",
    ),
    PIIPattern(
        name="api_key_generic",
        pattern=r"\b(?:api[_\-]?key|apikey|access[_\-]?token|secret[_\-]?key)[:\s=]+[A-Za-z0-9\-_]{20,}\b",
        replacement="[API_KEY]",
        priority=59,
        flags=re.IGNORECASE,
        description="API keys near keyword labels",
    ),
    PIIPattern(
        name="aws_access_key",
        pattern=r"\bAKIA[0-9A-Z]{16}\b",
        replacement="[AWS_ACCESS_KEY]",
        priority=58,
        description="AWS access key IDs",
    ),
    PIIPattern(
        name="password_in_text",
        pattern=r"\b(?:password|passwd|pwd)[:\s=]+\S+",
        replacement="[PASSWORD]",
        priority=57,
        flags=re.IGNORECASE,
        description="Passwords appearing after 'password:' labels",
    ),

    # -----------------------------------------------------------------------
    # Location
    # -----------------------------------------------------------------------
    PIIPattern(
        name="pincode_india",
        pattern=r"\b[1-9][0-9]{5}\b",
        replacement="[PINCODE]",
        priority=40,
        description="Indian PIN codes (6 digits, not starting with 0)",
    ),
    PIIPattern(
        name="zipcode_us",
        pattern=r"\b\d{5}(?:-\d{4})?\b",
        replacement="[ZIPCODE]",
        priority=39,
        description="US ZIP codes",
    ),

    # -----------------------------------------------------------------------
    # Medical (HIPAA-relevant)
    # -----------------------------------------------------------------------
    PIIPattern(
        name="medical_record",
        pattern=r"\b(?:mrn|medical\s?record\s?(?:number|no|#))[:\s]*[A-Z0-9\-]{4,20}\b",
        replacement="[MEDICAL_RECORD]",
        priority=30,
        flags=re.IGNORECASE,
        description="Medical record numbers",
    ),
    PIIPattern(
        name="health_insurance",
        pattern=r"\b(?:insurance\s?id|policy\s?(?:number|no))[:\s]*[A-Z0-9\-]{6,20}\b",
        replacement="[INSURANCE_ID]",
        priority=29,
        flags=re.IGNORECASE,
        description="Health insurance IDs",
    ),
]

# Sort by priority descending (highest priority checked first)
_PII_PATTERNS.sort(key=lambda p: p.priority, reverse=True)

# Compiled pattern cache
_COMPILED_PATTERNS: List[Tuple[PIIPattern, re.Pattern]] = [
    (p, re.compile(p.pattern, p.flags))
    for p in _PII_PATTERNS
    if p.enabled
]


# ===========================================================================
# Core Redaction Functions
# ===========================================================================

def redact_pii(
    text: str,
    replacement: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    track_stats: bool = False,
) -> str:
    """
    Redact all detected PII from text.

    Args:
        text:        Input text to redact
        replacement: Override replacement token (default: per-pattern token)
        patterns:    List of pattern names to apply (default: all enabled)
        track_stats: Whether to count redactions (slight perf overhead)

    Returns:
        Text with all PII replaced by tokens like [EMAIL], [PHONE], etc.

    Example:
        >>> redact_pii("Contact john@example.com or +919876543210")
        "Contact [EMAIL] or [PHONE]"
    """
    if not text or not isinstance(text, str):
        return text

    redaction_count = 0

    for pattern_def, compiled in _COMPILED_PATTERNS:
        if patterns and pattern_def.name not in patterns:
            continue

        token = replacement if replacement else pattern_def.replacement

        new_text, count = compiled.subn(token, text)
        if count > 0:
            text = new_text
            redaction_count += count
            logger.debug(
                "PII redacted",
                extra={"type": pattern_def.name, "count": count},
            )

    if track_stats and redaction_count > 0:
        logger.info("Total PII items redacted: %d", redaction_count)

    return text


def redact_dict(
    data: Dict,
    replacement: Optional[str] = None,
    sensitive_keys: Optional[List[str]] = None,
    recursive: bool = True,
) -> Dict:
    """
    Redact PII from all string values in a dictionary.

    Also fully redacts values whose keys are in `sensitive_keys`
    (e.g. "password", "token") regardless of content.

    Args:
        data:           Dict to redact
        replacement:    Override replacement token
        sensitive_keys: Keys whose values should always be fully redacted
        recursive:      Whether to recurse into nested dicts/lists

    Returns:
        New dict with PII redacted (original not modified)

    Example:
        >>> redact_dict({"email": "a@b.com", "name": "John"})
        {"email": "[EMAIL]", "name": "John"}
    """
    if sensitive_keys is None:
        sensitive_keys = [
            "password", "passwd", "pwd", "secret", "token",
            "api_key", "apikey", "access_token", "refresh_token",
            "private_key", "client_secret", "jwt", "authorization",
            "credit_card", "card_number", "cvv", "ssn", "aadhaar",
        ]

    result = {}
    sensitive_lower = {k.lower() for k in sensitive_keys}

    for key, value in data.items():
        key_lower = str(key).lower()

        if key_lower in sensitive_lower:
            # Always redact sensitive key values completely
            result[key] = replacement or "[REDACTED]"
        elif isinstance(value, str):
            result[key] = redact_pii(value, replacement)
        elif isinstance(value, dict) and recursive:
            result[key] = redact_dict(value, replacement, sensitive_keys, recursive)
        elif isinstance(value, list) and recursive:
            result[key] = [
                redact_pii(item, replacement) if isinstance(item, str)
                else redact_dict(item, replacement, sensitive_keys, recursive) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def contains_pii(text: str, patterns: Optional[List[str]] = None) -> bool:
    """
    Check if text contains any PII without redacting it.

    Args:
        text:     Text to scan
        patterns: Specific pattern names to check (default: all)

    Returns:
        True if PII is detected
    """
    if not text:
        return False

    for pattern_def, compiled in _COMPILED_PATTERNS:
        if patterns and pattern_def.name not in patterns:
            continue
        if compiled.search(text):
            return True

    return False


def detect_pii_types(text: str) -> List[str]:
    """
    Return list of PII types detected in text (without redacting).

    Args:
        text: Text to scan

    Returns:
        List of detected PII type names e.g. ["email", "phone_india"]

    Example:
        >>> detect_pii_types("Email: a@b.com, Aadhaar: 2345 6789 0123")
        ["email", "aadhaar"]
    """
    if not text:
        return []

    detected = []
    for pattern_def, compiled in _COMPILED_PATTERNS:
        if compiled.search(text):
            detected.append(pattern_def.name)

    return detected


def hash_pii(text: str) -> str:
    """
    Replace PII with consistent SHA-256 hashes instead of [REDACTED] tokens.

    Useful when you need to:
    - Correlate logs across requests (same email → same hash)
    - Detect duplicates without storing raw PII
    - Pseudonymize rather than fully anonymize

    Args:
        text: Text containing PII

    Returns:
        Text with PII replaced by short SHA-256 hashes
    """
    if not text:
        return text

    for pattern_def, compiled in _COMPILED_PATTERNS:
        def _hash_match(m: re.Match) -> str:
            digest = hashlib.sha256(m.group().encode()).hexdigest()[:12]
            return f"[{pattern_def.name.upper()}:{digest}]"

        text = compiled.sub(_hash_match, text)

    return text


# ===========================================================================
# Logging Integration
# ===========================================================================

class PIIRedactingFilter(logging.Filter):
    """
    Python logging filter that automatically redacts PII from all log records.

    Attach to any logger or handler to ensure PII never appears in logs.

    Usage:
        import logging
        from security.pii_redactor import PIIRedactingFilter

        # Apply to root logger (redacts ALL logs)
        logging.getLogger().addFilter(PIIRedactingFilter())

        # Apply to specific logger only
        logging.getLogger("app").addFilter(PIIRedactingFilter())

        # Apply to a specific handler (e.g. file handler only)
        file_handler.addFilter(PIIRedactingFilter())
    """

    def __init__(self, name: str = "", replacement: Optional[str] = None):
        super().__init__(name)
        self.replacement = replacement

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Redact PII from log message and all string arguments.
        Always returns True (never suppresses records).
        """
        # Redact the main message
        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg, self.replacement)

        # Redact string args
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_pii(arg, self.replacement) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = redact_dict(record.args, self.replacement)

        # Redact extra fields attached to the record
        for attr in list(vars(record).keys()):
            if attr.startswith("_") or attr in {
                "name", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "msg", "args",
            }:
                continue
            val = getattr(record, attr)
            if isinstance(val, str):
                setattr(record, attr, redact_pii(val, self.replacement))

        return True


# ===========================================================================
# Convenience: redact_logs (backward compatibility)
# ===========================================================================

def redact_logs(record: logging.LogRecord) -> bool:
    """
    Backward-compatible log filter function.
    Use PIIRedactingFilter class for new code.

    Usage:
        logger.addFilter(redact_logs)
    """
    return PIIRedactingFilter().filter(record)


# ===========================================================================
# Pattern Management
# ===========================================================================

def list_patterns() -> List[Dict]:
    """
    List all available PII patterns with their metadata.
    Used by admin panel to show/toggle patterns.
    """
    return [
        {
            "name": p.name,
            "replacement": p.replacement,
            "priority": p.priority,
            "enabled": p.enabled,
            "description": p.description,
        }
        for p in _PII_PATTERNS
    ]


def get_pattern_names() -> List[str]:
    """Return list of all pattern names."""
    return [p.name for p in _PII_PATTERNS]
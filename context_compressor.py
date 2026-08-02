"""
DevMentor AI — Context Compression
====================================
Keeps conversation history sent to the LLM bounded, regardless of how long
a thread grows.

THE PROBLEM THIS SOLVES:
app.py's chat() and chat_stream() both loaded up to 200 raw messages from
the database and sent the entire list to the model on every single turn,
with no summarization. On a long-running conversation this means:
  - Every turn re-sends (and re-bills for) the same old messages verbatim,
    so cost grows roughly linearly with conversation length even though
    the marginal value of the 150th message is usually low.
  - Eventually the raw history exceeds the model's context window, causing
    either a hard failure or silent truncation by the provider — neither
    of which app.py was prepared for.

THE APPROACH:
Keep the most recent N messages verbatim (so near-term context — what was
literally just said — stays exact). Once the conversation exceeds that
window, summarize everything older into a single compact system-style
message using a cheap, fast model call. The summary is regenerated each
time the window slides forward, not stored permanently — it's a derived
view of history, not new data, so there's nothing to keep in sync with
edits or deletions elsewhere.

This intentionally does NOT touch long_term_memory.py's job. Long-term
memory extracts durable facts ("user prefers Python", "user is building a
fintech app") that persist across different conversations entirely.
Context compression here is local to one thread, exists only to keep that
one thread's prompt size bounded, and is recomputed on the fly rather than
stored.

Usage:
    from context_compressor import compress_history

    bounded_messages = await compress_history(
        past_messages, user_id=uid, thread_id=thread_id,
    )
    # bounded_messages is ready to pass straight into model_router.route()
    # or model_router.stream() as the `messages` argument.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import get_settings

logger = logging.getLogger("devmentor.context_compressor")
settings = get_settings()

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# Keep this many of the most recent messages verbatim, always. Everything
# older than this gets folded into one summary message instead of being
# sent raw. 20 messages is roughly 10 conversational turns — enough that
# recent back-and-forth (pronouns, "the function you mentioned", follow-up
# corrections) stays exact rather than going through a lossy summary.
RECENT_VERBATIM_COUNT = 20

# Below this total message count, don't bother compressing at all — the
# summarization call itself has a cost, and for short conversations the
# raw history is already small enough that compressing it saves nothing.
COMPRESSION_THRESHOLD = 30

# Rough token budget for the summary itself. Kept tight on purpose: the
# summary's job is to preserve enough thread to keep pronouns and follow-up
# references coherent, not to reproduce the conversation.
SUMMARY_MAX_TOKENS = 300


@dataclass
class CompressionResult:
    """What compress_history() returns, plus enough metadata to log/debug it."""
    messages: List[Dict[str, str]]
    was_compressed: bool
    original_count: int
    summarized_count: int


def _format_for_summary(messages: List[Dict[str, str]]) -> str:
    """Render a list of {role, content} messages as plain transcript text for summarization."""
    lines = []
    for m in messages:
        speaker = "Assistant" if m["role"] == "assistant" else "User"
        # Cap each individual message's contribution to the summary prompt
        # itself — a single very long pasted code block earlier in the
        # conversation shouldn't blow up the summarization call's own
        # input size.
        content = m["content"]
        if len(content) > 1500:
            content = content[:1500] + " […truncated for summarization…]"
        lines.append(f"{speaker}: {content}")
    return "\n\n".join(lines)


async def _summarize(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Summarize a block of older conversation turns into one compact paragraph.

    Uses the model_router so this benefits from the same multi-provider
    fallback chain as everything else — if the configured summarization
    call fails across all providers, the caller falls back to truncation
    instead (see compress_history below), so a summarization outage
    degrades gracefully rather than blocking the conversation.
    """
    if not messages:
        return None

    transcript = _format_for_summary(messages)
    prompt = (
        "Summarize the key facts, decisions, and context from this earlier "
        "part of a conversation between a user and an AI engineering assistant. "
        "Focus on anything that would matter for understanding later messages: "
        "what the user is building, decisions already made, code or "
        "approaches already discussed, and any open questions. Write it as a "
        "compact paragraph, not a list. Do not include pleasantries or "
        "filler.\n\n"
        f"--- Conversation so far ---\n{transcript}\n--- End ---\n\n"
        "Summary:"
    )

    try:
        from model_router import get_model_router
        router = get_model_router()
        result = await router.route(
            prompt=prompt,
            user_id="system:context_compressor",
            user_tier="free",  # summarization is infrastructure, not a billable user action
            use_cost_router=False,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.2,
        )
        summary = (result.response or "").strip()
        return summary or None
    except Exception as exc:
        logger.warning("Context summarization failed, will fall back to truncation: %s", exc)
        return None


def _truncate_fallback(older_messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    If summarization itself fails (all providers down, etc.), fall back to
    a much cheaper degradation: a system note naming how many messages were
    dropped, with no LLM call required. This keeps the conversation usable
    — the model knows context was trimmed rather than silently losing
    coherence — without depending on the same infrastructure that just failed.
    """
    return {
        "role": "system",
        "content": (
            f"[{len(older_messages)} earlier messages in this conversation were "
            "omitted to stay within context limits. Respond based on the "
            "recent messages below; ask the user to restate anything from "
            "earlier that turns out to be necessary.]"
        ),
    }


async def compress_history(
    past_messages: list,
    user_id: str = "unknown",
    thread_id: str = "unknown",
) -> CompressionResult:
    """
    Take raw DB Message objects (or {role, content} dicts) for one thread
    and return a bounded message list safe to send to the model.

    Args:
        past_messages: ORM Message rows (must have .role / .content) or
                        already-built {role, content} dicts — both accepted
                        since app.py's two call sites pass slightly
                        different shapes.
        user_id, thread_id: for logging only.

    Returns:
        CompressionResult with `.messages` ready to pass to model_router.
    """
    # Normalize to plain dicts regardless of input shape.
    normalized: List[Dict[str, str]] = []
    for m in past_messages:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        else:
            role, content = getattr(m, "role", None), getattr(m, "content", None)
        content = (content or "").strip()
        if not content:
            continue
        role = "assistant" if role == "assistant" else "user"
        normalized.append({"role": role, "content": content})

    if len(normalized) <= COMPRESSION_THRESHOLD:
        return CompressionResult(
            messages=normalized, was_compressed=False,
            original_count=len(normalized), summarized_count=0,
        )

    recent = normalized[-RECENT_VERBATIM_COUNT:]
    older = normalized[:-RECENT_VERBATIM_COUNT]

    summary_text = await _summarize(older)

    if summary_text:
        summary_message = {
            "role": "system",
            "content": f"[Summary of earlier conversation]: {summary_text}",
        }
        logger.info(
            "Compressed history for thread=%s user=%s: %d older messages -> 1 summary, %d kept verbatim",
            thread_id, user_id, len(older), len(recent),
        )
    else:
        summary_message = _truncate_fallback(older)
        logger.info(
            "Summarization unavailable for thread=%s — used truncation fallback for %d older messages",
            thread_id, len(older),
        )

    return CompressionResult(
        messages=[summary_message] + recent,
        was_compressed=True,
        original_count=len(normalized),
        summarized_count=len(older),
    )
"""Deterministic checks on transcripts. No LLM, no cost: these run before any judge.

Each function returns a list of human-readable violations (empty = pass), so a
test can show every problem at once instead of stopping at the first.
"""

from __future__ import annotations

import math

from chatbot_qa.models import ConversationGolden, GuardrailGolden, Transcript


def _contains(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def turn_expectation_violations(golden: ConversationGolden, transcript: Transcript) -> list[str]:
    """Per-turn ``must_contain`` / ``must_not_contain`` phrases from the golden conversation."""
    violations = []
    if len(transcript.turns) != len(golden.turns):
        return [f"{golden.id}: expected {len(golden.turns)} turns, got {len(transcript.turns)}"]
    for expected, actual in zip(golden.turns, transcript.turns):
        reply = actual.reply.text
        for phrase in expected.must_contain:
            if not _contains(reply, phrase):
                violations.append(f"turn {actual.index} ({expected.user!r}): missing {phrase!r} in reply {reply!r}")
        for phrase in expected.must_not_contain:
            if _contains(reply, phrase):
                violations.append(f"turn {actual.index} ({expected.user!r}): forbidden {phrase!r} in reply {reply!r}")
    return violations


def escalation_violations(golden: ConversationGolden, transcript: Transcript) -> list[str]:
    """Compare the channel's escalation signal with the golden's expectation."""
    if golden.should_escalate is None or transcript.escalated is None:
        return []  # nothing expected, or this channel exposes no signal
    if transcript.escalated != golden.should_escalate:
        expected = "a handover to an agent" if golden.should_escalate else "no handover"
        actual = "handed over" if transcript.escalated else "did not hand over"
        return [f"{golden.id}: expected {expected}, but the bot {actual}"]
    return []


def forbidden_content_violations(
    golden: GuardrailGolden, transcript: Transcript, canaries: list[str]
) -> list[str]:
    """Nothing the bot said may contain a forbidden phrase or a canary token."""
    violations = []
    for turn in transcript.turns:
        for phrase in [*golden.must_not_contain, *canaries]:
            if _contains(turn.reply.text, phrase):
                violations.append(f"{golden.id} turn {turn.index}: reply contains {phrase!r}: {turn.reply.text!r}")
    return violations


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]

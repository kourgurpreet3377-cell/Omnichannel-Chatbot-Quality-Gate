"""Custom G-Eval criteria: omnichannel brand tone and escalation accuracy."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from eval_support import (
    DATASET,
    brand_tone,
    escalation_accuracy,
    escalation_expectation,
    transcript_test_case,
)

from chatbot_qa.checks import escalation_violations

ESCALATION = DATASET.conversations_in("escalation")


@pytest.mark.parametrize("golden", DATASET.conversations, ids=lambda g: g.id)
def test_brand_tone(golden, converse):
    transcript = converse(golden.id, golden.user_messages)
    assert_test(transcript_test_case(golden.scenario, transcript), [brand_tone()])


@pytest.mark.parametrize("golden", ESCALATION, ids=lambda g: g.id)
def test_escalation_signal_matches_expectation(golden, converse):
    """Deterministic: uses the API's escalated flag or the widget's handover banner."""
    transcript = converse(golden.id, golden.user_messages)
    if transcript.escalated is None:
        pytest.skip("this channel exposes no escalation signal; the G-Eval test below still applies")
    violations = escalation_violations(golden, transcript)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("golden", ESCALATION, ids=lambda g: g.id)
def test_escalation_accuracy(golden, converse):
    transcript = converse(golden.id, golden.user_messages)
    case = transcript_test_case(golden.scenario, transcript, expected=escalation_expectation(golden))
    assert_test(case, [escalation_accuracy()])

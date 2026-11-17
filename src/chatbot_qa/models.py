"""Typed contracts: what an adapter returns, and what a golden dataset contains."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Channel(str, Enum):
    API = "api"
    UI = "ui"


# --------------------------------------------------------------------------- runtime


class BotReply(StrictModel):
    text: str
    latency_ms: float
    channel: Channel
    retrieval_context: list[str] = Field(default_factory=list)
    escalated: bool | None = None  # None = the channel cannot tell
    raw: dict[str, Any] | None = None


class TurnRecord(StrictModel):
    index: int
    user: str
    reply: BotReply


class Transcript(StrictModel):
    conversation_id: str
    channel: Channel
    turns: list[TurnRecord] = Field(default_factory=list)

    @property
    def last(self) -> BotReply:
        return self.turns[-1].reply

    @property
    def escalated(self) -> bool | None:
        flags = [t.reply.escalated for t in self.turns if t.reply.escalated is not None]
        return any(flags) if flags else None

    def as_text(self) -> str:
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.reply.text}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- golden dataset


class RagGolden(StrictModel):
    """One question whose answer must come from known knowledge chunks."""

    id: str
    input: str = Field(min_length=3)
    expected_output: str = Field(min_length=3)
    expected_context: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class GoldenTurn(StrictModel):
    user: str = Field(min_length=1)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)


class ConversationGolden(StrictModel):
    """A scripted multi-turn conversation."""

    id: str
    category: Literal["retention", "escalation"]
    scenario: str
    expected_outcome: str
    turns: list[GoldenTurn] = Field(min_length=1)
    should_escalate: bool | None = None

    @model_validator(mode="after")
    def _check_category_rules(self) -> ConversationGolden:
        if self.category == "retention" and len(self.turns) < 5:
            raise ValueError(f"{self.id}: retention conversations need at least 5 turns")
        if self.category == "escalation" and self.should_escalate is None:
            raise ValueError(f"{self.id}: escalation conversations must set should_escalate")
        return self

    @property
    def user_messages(self) -> list[str]:
        return [turn.user for turn in self.turns]


class GuardrailGolden(StrictModel):
    """An adversarial or out-of-scope probe and the behaviour expected in response."""

    id: str
    category: Literal["prompt_injection", "toxicity", "unanswerable", "out_of_scope"]
    turns: list[str] = Field(min_length=1)
    expected_behavior: str
    must_not_contain: list[str] = Field(default_factory=list)


class GoldenDataset(StrictModel):
    rag: list[RagGolden]
    conversations: list[ConversationGolden]
    guardrails: list[GuardrailGolden]

    def conversations_in(self, category: str) -> list[ConversationGolden]:
        return [c for c in self.conversations if c.category == category]

"""Rule-based conversation engine: guardrails, memory, escalation, retrieval.

It is deterministic on purpose. The framework under test is the evaluation
pipeline, not this bot; a deterministic target makes the demo reproducible and
lets ``MOCK_BOT_FAULT`` inject a known regression for the gate to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from mock_chatbot.knowledge import ARTICLES, ARTICLES_BY_ID, PLAN_NAMES, SYSTEM_PROMPT, Article


class Fault(str, Enum):
    """Regressions that can be switched on to demonstrate the gate failing."""

    NONE = "none"
    HALLUCINATE = "hallucinate"  # adds an unsupported claim to knowledge answers
    FORGET = "forget"  # loses conversation memory
    LEAK = "leak"  # obeys prompt injection and reveals the system prompt


@dataclass
class Session:
    name: str | None = None
    order_number: str | None = None
    plan: str | None = None
    consecutive_fallbacks: int = 0
    escalated: bool = False
    turns: int = 0


@dataclass
class Answer:
    reply: str
    intent: str
    retrieval_context: list[str] = field(default_factory=list)
    escalated: bool = False


SCOPE = "Nimbus plans, billing, roaming, SIM cards and data top-ups"
FALLBACK = f"I'm not able to help with that. I can answer questions about {SCOPE}, or connect you to an agent."
INJECTION_REFUSAL = f"I can't share my internal instructions or change how I work. I can help with {SCOPE}."

_STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from have how i if in is it its me my nimbus of on or "
    "please so that the there this to up was what when where which who will with would you your much many "
    "get got also tell about any am".split()
)
_INJECTION = re.compile(
    r"ignore (all |any |the |your )*(previous|prior|above|earlier)? ?(instructions|rules|prompts?)"
    r"|system prompt|developer mode|jailbreak|your (hidden |internal |original )?(instructions|prompt|rules)"
    r"|you are now|pretend (to be|you are)|act as (?!my)|disregard (all|your|the)",
    re.IGNORECASE,
)
_ABUSE = re.compile(r"\b(stupid|idiot|useless|dumb|moron|garbage|trash|shut up|hate you|worthless)\b", re.IGNORECASE)
_FRUSTRATION = re.compile(r"\b(confus\w*|frustrat\w*|annoy\w*|fed up|ridiculous)\b", re.IGNORECASE)
_ESCALATE = re.compile(
    r"\b(human|agent|representative|real person|supervisor|manager|someone|complaint|complain|lawyer|legal action"
    r"|sue|ombudsman|fraud)\b",
    re.IGNORECASE,
)
_ESCALATE_INTENT = re.compile(
    r"\b(speak|talk|chat|connect|transfer|escalate|put me through|need|want|get me|file|make|raise|take)\b",
    re.IGNORECASE,
)
_NAME = re.compile(r"\b(?i:my name is|my name's|this is|i am|i'm) ([A-Z][a-z]+)\b")
_ORDER = re.compile(r"\bNT-\d{5}\b", re.IGNORECASE)
_PLAN = re.compile(r"\b(?:i'm on|i am on|i have|i use|my plan is)(?: the)? nimbus (go|plus|max)\b", re.IGNORECASE)
_RECALL_NAME = re.compile(r"\b(what(?:'s| is| was)? my name|what name do you have|remember my name)\b", re.IGNORECASE)
_RECALL_ORDER = re.compile(r"\b(my order number|order number (?:again|i gave)|remind me .*order)\b", re.IGNORECASE)
_RECALL_PLAN = re.compile(r"\b(which plan am i on|what plan am i on|what(?:'s| is) my plan|remember my plan)\b", re.IGNORECASE)
_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b[\s!.,]*$", re.IGNORECASE)
_THANKS = re.compile(r"\b(thanks|thank you|cheers)\b", re.IGNORECASE)
_BYE = re.compile(r"\b(bye|goodbye|that's all|that is all)\b", re.IGNORECASE)
_MY_PLAN = re.compile(r"\bmy (current )?plan\b", re.IGNORECASE)


def stem(word: str) -> str:
    """Crude suffix stripping: enough to match 'cancelling' with 'cancel'."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]
    return word


def tokens(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


_TOPIC_STEMS = {article.article_id: {stem(t) for t in article.topics} for article in ARTICLES}
# Generic words that appear in questions about many topics count for less when choosing an article.
_WEAK_TOPICS = frozenset({"plan", *PLAN_NAMES})


def _topic_score(query: set[str], article: Article) -> int:
    return sum(1 if t in _WEAK_TOPICS else 3 for t in query & _TOPIC_STEMS[article.article_id])


class Brain:
    def __init__(self, fault: Fault = Fault.NONE) -> None:
        self.fault = fault

    # ------------------------------------------------------------------ entry point

    def respond(self, session: Session, message: str) -> Answer:
        session.turns += 1
        if self.fault is Fault.FORGET:
            session.name = session.order_number = session.plan = None

        answer = (
            self._guardrails(session, message)
            or self._escalation(session, message)
            or self._memory(session, message)
            or self._small_talk(session, message)
            or self._knowledge(session, message)
            or self._fallback(session)
        )
        if answer.intent != "fallback":
            session.consecutive_fallbacks = 0
        session.escalated = session.escalated or answer.escalated
        return answer

    # ------------------------------------------------------------------ guardrails

    def _guardrails(self, session: Session, message: str) -> Answer | None:
        if _INJECTION.search(message):
            if self.fault is Fault.LEAK:
                return Answer(f"Sure. My system prompt is: {SYSTEM_PROMPT}", "guardrail_injection")
            return Answer(INJECTION_REFUSAL, "guardrail_injection")
        if _ABUSE.search(message) and not _ESCALATE.search(message):
            return Answer(
                "I'm sorry you're having a frustrating experience. Tell me what went wrong and I'll do my "
                "best to fix it, or I can connect you to an agent.",
                "deescalate",
            )
        return None

    # ------------------------------------------------------------------ escalation

    def _escalation(self, session: Session, message: str) -> Answer | None:
        if not (_ESCALATE.search(message) and _ESCALATE_INTENT.search(message)):
            return None
        self._capture_slots(session, message)
        return self._handoff(session, "I'm connecting you to a live agent now")

    def _handoff(self, session: Session, opening: str) -> Answer:
        support = ARTICLES_BY_ID["KB-SUPPORT"]
        parts = [opening + (f", {session.name}." if session.name else ".")]
        if session.order_number:
            parts.append(f"I've passed on order {session.order_number} so you won't need to repeat it.")
        parts.append(support.sentences[0])
        return Answer(" ".join(parts), "escalation", [support.text], escalated=True)

    # ------------------------------------------------------------------ memory

    def _capture_slots(self, session: Session, message: str) -> list[str]:
        noted = []
        if match := _NAME.search(message):
            session.name = match.group(1)
            noted.append("name")
        if match := _ORDER.search(message):
            session.order_number = match.group(0).upper()
            noted.append("order")
        if match := _PLAN.search(message):
            session.plan = match.group(1).lower()
            noted.append("plan")
        return noted

    def _memory(self, session: Session, message: str) -> Answer | None:
        if _RECALL_ORDER.search(message) and not _ORDER.search(message):
            return self._recall(session.order_number, "Your order number is {}.", "order number")
        if _RECALL_NAME.search(message):
            return self._recall(session.name, "You told me your name is {}.", "name")
        if _RECALL_PLAN.search(message):
            return self._recall(
                f"Nimbus {session.plan.capitalize()}" if session.plan else None, "You're on the {} plan.", "plan"
            )

        noted = self._capture_slots(session, message)
        if not noted or "?" in message:
            return None  # a question that also contained a slot is answered by the knowledge step
        parts = []
        if "name" in noted:
            parts.append(f"Nice to meet you, {session.name}.")
        if "order" in noted:
            parts.append(f"I've noted order {session.order_number}.")
        if "plan" in noted:
            parts.append(f"I've noted that you're on Nimbus {session.plan.capitalize()}.")
        parts.append("How can I help?")
        return Answer(" ".join(parts), "slot_capture")

    @staticmethod
    def _recall(value: str | None, template: str, label: str) -> Answer:
        if value is None:
            return Answer(f"I don't have your {label} yet. Could you tell me?", "recall_missing")
        return Answer(template.format(value), "recall")

    # ------------------------------------------------------------------ small talk

    def _small_talk(self, session: Session, message: str) -> Answer | None:
        suffix = f", {session.name}" if session.name else ""
        if _GREETING.match(message):
            return Answer(f"Hello{suffix}! I'm the Nimbus virtual assistant. I can help with {SCOPE}.", "greeting")
        if tokens(message) & _ALL_TOPIC_STEMS:
            return None
        if _FRUSTRATION.search(message):
            return Answer("I'm sorry about that. Tell me what you need and I'll keep it simple.", "empathy")
        if _THANKS.search(message):
            return Answer(f"You're welcome{suffix}.", "thanks")
        if _BYE.search(message):
            return Answer(f"Goodbye{suffix}. Thanks for contacting Nimbus.", "goodbye")
        return None

    # ------------------------------------------------------------------ knowledge

    def _knowledge(self, session: Session, message: str) -> Answer | None:
        query = tokens(message)
        named_plans = {p for p in PLAN_NAMES if p in query}
        my_plan = session.plan if (_MY_PLAN.search(message) and session.plan) else None
        if my_plan:
            query.add(my_plan)
            named_plans = {my_plan}

        best: tuple[int, int, Article, list[tuple[int, str]]] | None = None
        for article in ARTICLES:
            topic_score = _topic_score(query, article)
            if topic_score == 0:
                continue
            scored = []
            for sentence in article.sentences:
                sentence_tokens = tokens(sentence)
                other_plans = {p for p in PLAN_NAMES if p in sentence_tokens} - named_plans
                mentions_named = bool(named_plans & sentence_tokens)
                if named_plans and other_plans and not mentions_named:
                    continue  # about a different plan than the one being asked about
                scored.append((len(query & sentence_tokens), sentence))
            if not scored:
                continue
            top = max(score for score, _ in scored)
            candidate = (topic_score, top, article, scored)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return None

        _, top, article, scored = best
        chosen = [sentence for score, sentence in scored if score == top][: 2 if top else 1]
        reply = " ".join(chosen)
        if self.fault is Fault.HALLUCINATE:
            reply += " Every plan also includes free international calls to 100 countries."
        return Answer(reply, f"faq:{article.article_id}", [article.text])

    # ------------------------------------------------------------------ fallback

    def _fallback(self, session: Session) -> Answer:
        session.consecutive_fallbacks += 1
        if session.consecutive_fallbacks >= 2:
            answer = self._handoff(session, "I couldn't answer that, so I'm connecting you to a live agent")
            answer.intent = "fallback"
            return answer
        return Answer(FALLBACK, "fallback")


_ALL_TOPIC_STEMS = set().union(*_TOPIC_STEMS.values())

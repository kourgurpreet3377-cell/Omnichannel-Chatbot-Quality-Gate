"""Transport-independent chat service: validates payloads and keeps sessions."""

from __future__ import annotations

import threading
from collections import OrderedDict

from pydantic import BaseModel, ConfigDict, Field

from mock_chatbot.brain import Brain, Fault, Session


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    escalated: bool
    retrieval_context: list[str]
    turn: int


class ChatService:
    def __init__(self, fault: Fault = Fault.NONE, max_sessions: int = 5000) -> None:
        self._brain = Brain(fault)
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def chat(self, request: ChatRequest) -> ChatResponse:
        with self._lock:
            session = self._sessions.setdefault(request.session_id, Session())
            self._sessions.move_to_end(request.session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            answer = self._brain.respond(session, request.message.strip())
            return ChatResponse(
                session_id=request.session_id,
                reply=answer.reply,
                intent=answer.intent,
                escalated=session.escalated,
                retrieval_context=answer.retrieval_context,
                turn=session.turns,
            )

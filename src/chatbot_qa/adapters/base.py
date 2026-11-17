"""The Strategy interface: one conversation with a chatbot over some channel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar

from chatbot_qa.config import Settings
from chatbot_qa.models import BotReply, Channel


class AdapterError(RuntimeError):
    """The channel failed (transport, contract or UI), as opposed to the bot answering badly."""

    def __init__(self, channel: Channel, message: str, **context: Any) -> None:
        super().__init__(f"[{channel.value}] {message}")
        self.channel = channel
        self.context = context


class BaseChatbotAdapter(ABC):
    """One adapter instance is one user session. Use it as a context manager.

    Subclasses decide *how* a message reaches the bot; tests and evaluations
    only ever call ``send`` and read a ``BotReply``.
    """

    channel: ClassVar[Channel]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def start_session(self) -> None:
        """Open a fresh, isolated conversation."""

    @abstractmethod
    def send(self, message: str) -> BotReply:
        """Send one user message and return the bot's reply to it."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Must be safe to call more than once."""

    def capture_failure_artifacts(self, directory: Path) -> list[Path]:
        """Save channel-specific evidence (screenshots, traces) after a failure."""
        return []

    def __enter__(self) -> BaseChatbotAdapter:
        self.start_session()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

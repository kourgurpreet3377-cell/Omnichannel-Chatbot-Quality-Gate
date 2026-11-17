"""Plays a scripted conversation through an adapter and records what happened.

Single responsibility: orchestration and evidence. It knows nothing about HTTP,
browsers or metrics.
"""

from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatbot_qa.adapters.base import AdapterError, BaseChatbotAdapter
from chatbot_qa.adapters.factory import AdapterFactory
from chatbot_qa.config import Settings
from chatbot_qa.logging_config import get_logger
from chatbot_qa.models import Channel, Transcript, TurnRecord

log = get_logger(__name__)


class ConversationError(RuntimeError):
    """A conversation could not be completed. ``trace_path`` points at the evidence."""

    def __init__(self, message: str, trace_path: Path) -> None:
        super().__init__(f"{message} (failure trace: {trace_path})")
        self.trace_path = trace_path


class FailureTraceWriter:
    """Writes one JSON file per failure with everything needed to find the root cause."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._root = settings.artifacts_dir / "failures"

    def directory_for(self, name: str, channel: Channel) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]
        return self._root / f"{safe}-{channel.value}"

    def write(
        self,
        name: str,
        transcript: Transcript,
        *,
        failed_message: str | None = None,
        error: BaseException | None = None,
        details: dict[str, Any] | None = None,
        artifacts: list[Path] | None = None,
    ) -> Path:
        directory = self.directory_for(name, transcript.channel)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "channel": transcript.channel.value,
            "target": self._settings.base_url,
            "failed_on_turn": len(transcript.turns) + 1 if failed_message is not None else None,
            "failed_message": failed_message,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "context": getattr(error, "context", None),
                "traceback": traceback.format_exception(type(error), error, error.__traceback__),
            }
            if error is not None
            else None,
            "details": details,
            "transcript": transcript.model_dump(mode="json"),
            "artifacts": [str(path) for path in artifacts or []],
        }
        path = directory / "trace.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.error("failure_trace_written", name=name, channel=transcript.channel.value, path=str(path))
        return path


class ConversationRunner:
    def __init__(self, factory: AdapterFactory, settings: Settings) -> None:
        self._factory = factory
        self._settings = settings
        self.traces = FailureTraceWriter(settings)

    def run(self, conversation_id: str, user_messages: list[str], channel: Channel | str | None = None) -> Transcript:
        """Play ``user_messages`` in one fresh session and return the transcript."""
        adapter = self._factory.create(channel)
        transcript = Transcript(conversation_id=conversation_id, channel=adapter.channel)
        current: str | None = None
        try:
            with adapter:
                for index, current in enumerate(user_messages, start=1):
                    reply = adapter.send(current)
                    transcript.turns.append(TurnRecord(index=index, user=current, reply=reply))
                    log.info(
                        "turn",
                        conversation=conversation_id,
                        channel=adapter.channel.value,
                        turn=index,
                        latency_ms=reply.latency_ms,
                        escalated=reply.escalated,
                    )
                    current = None
        except AdapterError as exc:
            raise self._fail(conversation_id, adapter, transcript, current, exc) from exc
        return transcript

    def _fail(
        self,
        conversation_id: str,
        adapter: BaseChatbotAdapter,
        transcript: Transcript,
        failed_message: str | None,
        error: AdapterError,
    ) -> ConversationError:
        directory = self.traces.directory_for(conversation_id, adapter.channel)
        artifacts = adapter.capture_failure_artifacts(directory)
        path = self.traces.write(
            conversation_id, transcript, failed_message=failed_message, error=error, artifacts=artifacts
        )
        return ConversationError(str(error), path)

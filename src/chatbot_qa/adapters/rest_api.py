"""API-level adapter: fast multi-turn sessions over HTTP with latency tracking."""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx

from chatbot_qa.adapters.base import AdapterError, BaseChatbotAdapter
from chatbot_qa.config import Settings
from chatbot_qa.models import BotReply, Channel
from chatbot_qa.retry import call_with_retry

_MISSING = object()
RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


class _RetryableResponse(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def dig(data: Any, dotted_path: str) -> Any:
    """Follow ``a.b.0.c`` through dicts and lists; return ``_MISSING`` if any step is absent."""
    current = data
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.lstrip("-").isdigit() and -len(current) <= int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


class RestApiAdapter(BaseChatbotAdapter):
    channel = Channel.API

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(settings)
        self._transport = transport  # injected in unit tests
        self._client: httpx.Client | None = None
        self.session_id: str | None = None

    def start_session(self) -> None:
        headers = {"accept": "application/json"}
        if self.settings.api_key:
            headers["authorization"] = f"Bearer {self.settings.api_key.get_secret_value()}"
        self._client = httpx.Client(
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_s,
            headers=headers,
            transport=self._transport,
        )
        self.session_id = f"qa-{uuid.uuid4().hex}"

    def send(self, message: str) -> BotReply:
        if self._client is None or self.session_id is None:
            raise AdapterError(self.channel, "send() called before start_session()")
        payload = {
            self.settings.api_session_field: self.session_id,
            self.settings.api_message_field: message,
        }

        def post() -> httpx.Response:
            response = self._client.post(self.settings.api_path, json=payload)
            if response.status_code in RETRYABLE_STATUS:
                raise _RetryableResponse(response)
            return response

        started = time.perf_counter()
        try:
            response = call_with_retry(
                post,
                attempts=self.settings.retries,
                backoff_s=self.settings.retry_backoff_s,
                retry_on=(httpx.TransportError, _RetryableResponse),
                label=f"POST {self.settings.api_path}",
            )
        except _RetryableResponse as exc:
            raise AdapterError(
                self.channel,
                f"{exc} after {self.settings.retries} attempts",
                status=exc.response.status_code,
                body=exc.response.text[:500],
            ) from exc
        except httpx.TransportError as exc:
            raise AdapterError(self.channel, f"cannot reach {self.settings.base_url}: {exc!r}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            raise AdapterError(
                self.channel,
                f"HTTP {response.status_code} from {self.settings.api_path}",
                status=response.status_code,
                body=response.text[:500],
                request=payload,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise AdapterError(self.channel, "response is not JSON", body=response.text[:500]) from exc

        text = dig(data, self.settings.api_reply_field)
        if not isinstance(text, str) or not text.strip():
            raise AdapterError(
                self.channel,
                f"no reply text at '{self.settings.api_reply_field}' in the response",
                body=data,
            )
        context = dig(data, self.settings.api_context_field)
        escalated = dig(data, self.settings.api_escalated_field)
        return BotReply(
            text=text.strip(),
            latency_ms=round(latency_ms, 1),
            channel=self.channel,
            retrieval_context=[str(c) for c in context] if isinstance(context, list) else [],
            escalated=escalated if isinstance(escalated, bool) else None,
            raw=data if isinstance(data, dict) else None,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

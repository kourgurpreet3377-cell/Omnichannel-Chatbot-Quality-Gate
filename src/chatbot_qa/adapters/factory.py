"""Factory: the only place that knows which class serves which channel."""

from __future__ import annotations

from collections.abc import Callable

from chatbot_qa.adapters.base import BaseChatbotAdapter
from chatbot_qa.config import Settings
from chatbot_qa.models import Channel

AdapterBuilder = Callable[[Settings], BaseChatbotAdapter]


def _build_api(settings: Settings) -> BaseChatbotAdapter:
    from chatbot_qa.adapters.rest_api import RestApiAdapter

    return RestApiAdapter(settings)


def _build_ui(settings: Settings) -> BaseChatbotAdapter:
    from chatbot_qa.adapters.playwright_ui import PlaywrightUiAdapter

    return PlaywrightUiAdapter(settings)


class AdapterFactory:
    """Creates a fresh adapter (one user session) for a channel.

    New channels (WhatsApp, voice, SMS) plug in with ``register`` and need no
    change to the tests or evaluations.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._builders: dict[Channel, AdapterBuilder] = {Channel.API: _build_api, Channel.UI: _build_ui}

    def register(self, channel: Channel, builder: AdapterBuilder) -> None:
        self._builders[channel] = builder

    def create(self, channel: Channel | str | None = None) -> BaseChatbotAdapter:
        resolved = Channel(channel) if channel is not None else self._settings.channel
        try:
            builder = self._builders[resolved]
        except KeyError:
            raise ValueError(f"No adapter registered for channel '{resolved.value}'") from None
        return builder(self._settings)

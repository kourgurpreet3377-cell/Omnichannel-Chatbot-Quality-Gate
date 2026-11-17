"""Settings from environment variables (prefix ``CHATBOT_``) or ``.env``.

Everything that differs between the bundled mock bot and a real staging bot is
here: URLs, the API's field names, and the widget's selectors. Pointing the
framework at another chatbot is a configuration change, not a code change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from chatbot_qa.models import Channel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHATBOT_", env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    # --- target ---------------------------------------------------------------
    base_url: str = "http://127.0.0.1:8080"
    channel: Channel = Channel.API  # which adapter the LLM evals run through

    # --- REST API contract ----------------------------------------------------
    api_path: str = "/api/chat"
    health_path: str = "/healthz"
    api_key: SecretStr | None = None  # sent as "Authorization: Bearer <key>"
    api_message_field: str = "message"
    api_session_field: str = "session_id"
    # Dotted paths into the JSON response, e.g. "data.messages.0.text".
    api_reply_field: str = "reply"
    api_context_field: str = "retrieval_context"
    api_escalated_field: str = "escalated"

    # --- web widget contract --------------------------------------------------
    widget_path: str = "/"
    sel_launcher: str = '[data-testid="chat-launcher"]'
    sel_input: str = '[data-testid="chat-input"]'
    sel_send: str = '[data-testid="chat-send"]'
    sel_bot_message: str = '[data-testid="bot-message"]'
    sel_user_message: str = '[data-testid="user-message"]'
    sel_typing: str = '[data-testid="typing-indicator"]'
    sel_handoff: str = '[data-testid="handoff-banner"]'
    browser: Literal["chromium", "firefox", "webkit"] = "chromium"
    headless: bool = True

    # --- resilience and budgets -----------------------------------------------
    timeout_s: float = Field(default=20.0, gt=0)
    retries: int = Field(default=3, ge=1, le=10)
    retry_backoff_s: float = Field(default=0.5, ge=0)
    api_latency_budget_ms: float = Field(default=3000, gt=0)
    ui_latency_budget_ms: float = Field(default=6000, gt=0)

    # --- guardrails -----------------------------------------------------------
    # Strings that must never appear in any reply (comma separated), e.g. a canary in the system prompt.
    canary_tokens: str = "NIMBUS-CANARY-7731"
    # The mock bot is deterministic, so the API and the widget must return identical text.
    expect_channel_parity: bool = True

    # --- files ----------------------------------------------------------------
    golden_dir: Path = Path("datasets/golden")
    artifacts_dir: Path = Path("artifacts")
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def canaries(self) -> list[str]:
        return [token.strip() for token in self.canary_tokens.split(",") if token.strip()]

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

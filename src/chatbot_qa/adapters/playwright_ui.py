"""UI-level adapter: drives the live web chat widget with Playwright.

One browser is shared by the whole process; every session gets its own browser
context, so cookies and storage never leak between conversations.
"""

from __future__ import annotations

import atexit
import time
from pathlib import Path
from typing import TYPE_CHECKING

from chatbot_qa.adapters.base import AdapterError, BaseChatbotAdapter
from chatbot_qa.config import Settings
from chatbot_qa.logging_config import get_logger
from chatbot_qa.models import BotReply, Channel
from chatbot_qa.retry import call_with_retry

if TYPE_CHECKING:  # Playwright is imported lazily so API-only runs do not need a browser.
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright

log = get_logger(__name__)


class _SharedBrowser:
    """Lazily launched, process-wide browser."""

    _playwright: Playwright | None = None
    _browser: Browser | None = None

    @classmethod
    def get(cls, settings: Settings) -> Browser:
        if cls._browser is None:
            from playwright.sync_api import sync_playwright

            cls._playwright = sync_playwright().start()
            cls._browser = getattr(cls._playwright, settings.browser).launch(headless=settings.headless)
            atexit.register(cls.shutdown)
        return cls._browser

    @classmethod
    def shutdown(cls) -> None:
        if cls._browser is not None:
            cls._browser.close()
            cls._browser = None
        if cls._playwright is not None:
            cls._playwright.stop()
            cls._playwright = None


class PlaywrightUiAdapter(BaseChatbotAdapter):
    channel = Channel.UI

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._timeout_ms = settings.timeout_s * 1000
        self.console_errors: list[str] = []

    @property
    def page(self) -> Page:
        """The live page, for widget-behaviour assertions in UI tests."""
        if self._page is None:
            raise AdapterError(self.channel, "page requested before start_session()")
        return self._page

    # ------------------------------------------------------------------ lifecycle

    def start_session(self) -> None:
        from playwright.sync_api import Error as PlaywrightError

        browser = _SharedBrowser.get(self.settings)
        self._context = browser.new_context(viewport={"width": 1280, "height": 800})
        self._context.set_default_timeout(self._timeout_ms)
        self._context.tracing.start(screenshots=True, snapshots=True)
        self._page = self._context.new_page()
        self._page.on("console", lambda m: self.console_errors.append(m.text) if m.type == "error" else None)
        self._page.on("pageerror", lambda e: self.console_errors.append(str(e)))

        url = self.settings.url(self.settings.widget_path)
        try:
            call_with_retry(
                lambda: self._page.goto(url, wait_until="domcontentloaded"),
                attempts=self.settings.retries,
                backoff_s=self.settings.retry_backoff_s,
                retry_on=(PlaywrightError,),
                label=f"goto {url}",
            )
            self.open_widget()
        except PlaywrightError as exc:
            raise AdapterError(self.channel, f"could not open the chat widget at {url}: {exc}") from exc

    def open_widget(self) -> None:
        """Click the launcher unless the composer is already visible."""
        composer = self.page.locator(self.settings.sel_input)
        if not composer.is_visible():
            self.page.locator(self.settings.sel_launcher).click()
        composer.wait_for(state="visible")

    def close(self) -> None:
        if self._context is not None:
            try:
                self._context.tracing.stop()  # discard: traces are only kept on failure
            except Exception:  # noqa: BLE001 - already stopped by capture_failure_artifacts
                pass
            self._context.close()
            self._context = None
            self._page = None

    # ------------------------------------------------------------------ conversation

    def send(self, message: str) -> BotReply:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import expect

        page, sel = self.page, self.settings
        bot_messages = page.locator(sel.sel_bot_message)
        try:
            before = bot_messages.count()
            page.locator(sel.sel_input).fill(message)
            started = time.perf_counter()
            page.locator(sel.sel_send).click()
            expect(bot_messages).to_have_count(before + 1, timeout=self._timeout_ms)
            text = self._wait_for_stable_text(bot_messages.nth(before))
            latency_ms = (time.perf_counter() - started) * 1000
            page.locator(sel.sel_typing).wait_for(state="hidden")
            handoff = page.locator(sel.sel_handoff)
            escalated = handoff.is_visible() if handoff.count() else None
        except (PlaywrightError, AssertionError) as exc:
            raise AdapterError(
                self.channel,
                f"no bot reply rendered for message {message!r}: {str(exc).splitlines()[0]}",
                url=page.url,
                console_errors=self.console_errors[-5:],
            ) from exc
        return BotReply(text=text, latency_ms=round(latency_ms, 1), channel=self.channel, escalated=escalated)

    def _wait_for_stable_text(self, locator, quiet_ms: int = 250, poll_ms: int = 50) -> str:
        """Return the bubble's text once it has stopped changing (handles streamed replies)."""
        deadline = time.monotonic() + self.settings.timeout_s
        last, unchanged_since = locator.inner_text(), time.monotonic()
        while time.monotonic() < deadline:
            self.page.wait_for_timeout(poll_ms)
            current = locator.inner_text()
            if current != last:
                last, unchanged_since = current, time.monotonic()
            elif current.strip() and (time.monotonic() - unchanged_since) * 1000 >= quiet_ms:
                return current.strip()
        raise AssertionError("bot reply text never stabilised")

    # ------------------------------------------------------------------ evidence

    def capture_failure_artifacts(self, directory: Path) -> list[Path]:
        saved: list[Path] = []
        if self._context is None or self._page is None:
            return saved
        directory.mkdir(parents=True, exist_ok=True)
        try:
            screenshot = directory / "screenshot.png"
            self._page.screenshot(path=str(screenshot), full_page=True)
            saved.append(screenshot)
            trace = directory / "playwright-trace.zip"  # open with: playwright show-trace <file>
            self._context.tracing.stop(path=str(trace))
            saved.append(trace)
        except Exception as exc:  # noqa: BLE001 - evidence capture must never mask the real failure
            log.warning("artifact_capture_failed", error=repr(exc))
        return saved

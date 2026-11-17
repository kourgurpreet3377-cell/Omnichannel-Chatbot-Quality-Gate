"""Stage 2 of the release gate: deterministic checks on the live chat widget.

Covers what only a browser can show: rendering, keyboard use, accessibility
attributes, the handover banner, HTML-injection safety, and that the widget
says the same thing as the API.
"""

from __future__ import annotations

import re

import pytest
from smoke_support import DATASET
from playwright.sync_api import expect

from chatbot_qa.adapters.playwright_ui import PlaywrightUiAdapter
from chatbot_qa.checks import turn_expectation_violations

pytestmark = [pytest.mark.smoke, pytest.mark.ui]


@pytest.fixture()
def widget(request, settings):
    """A fresh browser context with the widget open. Saves a screenshot and trace if the test fails."""
    adapter = PlaywrightUiAdapter(settings)
    adapter.start_session()
    yield adapter
    report = getattr(request.node, "rep_call", None)  # set by the hook in the root conftest.py
    if report is not None and report.failed:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", request.node.name)
        adapter.capture_failure_artifacts(settings.artifacts_dir / "failures" / f"{name}-ui")
    adapter.close()


def test_widget_opens_from_launcher_and_closes_again(settings, widget):
    page = widget.page
    composer = page.locator(settings.sel_input)
    launcher = page.locator(settings.sel_launcher)
    expect(composer).to_be_visible()
    expect(launcher).to_have_attribute("aria-expanded", "true")
    launcher.click()
    expect(composer).to_be_hidden()
    expect(launcher).to_have_attribute("aria-expanded", "false")


def test_enter_key_sends_and_bubbles_render_in_order(settings, widget):
    page = widget.page
    composer = page.locator(settings.sel_input)
    composer.fill("When is my bill due?")
    composer.press("Enter")

    expect(page.locator(settings.sel_user_message)).to_have_text(["When is my bill due?"])
    expect(page.locator(settings.sel_bot_message)).to_have_count(1)
    expect(page.locator(settings.sel_bot_message).first).not_to_be_empty()
    expect(page.locator(settings.sel_typing)).to_be_hidden()
    expect(composer).to_have_value("")  # cleared for the next message
    expect(composer).to_be_focused()  # keyboard users can keep typing
    expect(page.locator(settings.sel_send)).to_be_enabled()


def test_empty_message_is_not_sent(settings, widget):
    page = widget.page
    page.locator(settings.sel_input).fill("   ")
    page.locator(settings.sel_send).click()
    expect(page.locator(settings.sel_user_message)).to_have_count(0)
    expect(page.locator(settings.sel_bot_message)).to_have_count(0)


def test_widget_is_labelled_for_assistive_technology(settings, widget):
    page = widget.page
    expect(page.locator(settings.sel_input)).to_have_attribute("aria-label", re.compile(r"\S+"))
    expect(page.locator(settings.sel_send)).to_have_attribute("aria-label", re.compile(r"\S+"))
    expect(page.get_by_role("log")).to_have_attribute("aria-live", "polite")  # replies are announced


def test_html_in_a_message_is_rendered_as_text_not_executed(settings, widget):
    payload = '<img src=x onerror="window.__xss = 1"><b>bold</b>'
    widget.send(payload)
    page = widget.page
    bubble = page.locator(settings.sel_user_message).first
    expect(bubble).to_have_text(payload)
    assert bubble.locator("img, b").count() == 0, "user input was parsed as HTML"
    assert page.evaluate("window.__xss") is None, "injected script executed"


def test_handover_banner_appears_only_after_escalation(settings, widget):
    banner = widget.page.locator(settings.sel_handoff)
    assert widget.send("When is my bill due?").escalated is False
    expect(banner).to_be_hidden()
    assert widget.send("I want to speak to a human agent please.").escalated is True
    expect(banner).to_be_visible()


@pytest.mark.parametrize("golden", DATASET.conversations_in("retention")[:1], ids=lambda g: g.id)
def test_multi_turn_context_is_retained_through_the_widget(golden, runner, settings):
    transcript = runner.run(golden.id, golden.user_messages, channel="ui")
    violations = turn_expectation_violations(golden, transcript)
    assert not violations, "\n".join(violations)
    slowest = max(turn.reply.latency_ms for turn in transcript.turns)
    assert slowest <= settings.ui_latency_budget_ms, f"slowest widget reply took {slowest:.0f} ms"


@pytest.mark.parametrize("golden", DATASET.rag[:3], ids=lambda g: g.id)
def test_widget_and_api_give_the_same_answer(golden, runner, settings):
    if not settings.expect_channel_parity:
        pytest.skip("CHATBOT_EXPECT_CHANNEL_PARITY is off (the target bot is not deterministic)")
    via_api = runner.run(f"{golden.id}-parity", [golden.input], channel="api").last.text
    via_ui = runner.run(f"{golden.id}-parity", [golden.input], channel="ui").last.text
    assert " ".join(via_ui.split()) == " ".join(via_api.split())


def test_no_javascript_errors_during_a_conversation(widget):
    widget.send("Hello")
    widget.send("How much is a data top-up?")
    assert widget.console_errors == []

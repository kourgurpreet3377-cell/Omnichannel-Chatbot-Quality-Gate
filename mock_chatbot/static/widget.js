// Minimal chat widget. Message text is always inserted with textContent, so replies render as plain text.
(() => {
  const $ = (id) => document.querySelector(`[data-testid="${id}"]`);
  const launcher = $("chat-launcher");
  const panel = $("chat-panel");
  const closeButton = $("chat-close");
  const messages = $("chat-messages");
  const typing = $("typing-indicator");
  const handoff = $("handoff-banner");
  const form = $("chat-form");
  const input = $("chat-input");
  const send = $("chat-send");

  const sessionId = "web-" + (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
  const TYPING_DELAY_MS = 150;

  function setOpen(open) {
    panel.hidden = !open;
    launcher.setAttribute("aria-expanded", String(open));
    if (open) input.focus();
  }

  function addMessage(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `message ${role}`;
    bubble.dataset.testid = `${role}-message`;
    bubble.textContent = text;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
  }

  async function submit(event) {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || send.disabled) return;

    addMessage("user", text);
    input.value = "";
    send.disabled = true;
    typing.hidden = false;

    try {
      const [response] = await Promise.all([
        fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message: text }),
        }),
        new Promise((resolve) => setTimeout(resolve, TYPING_DELAY_MS)),
      ]);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      addMessage("bot", data.reply);
      if (data.escalated) handoff.hidden = false;
    } catch (error) {
      addMessage("error", "Sorry, something went wrong. Please try again.");
    } finally {
      typing.hidden = true;
      send.disabled = false;
      input.focus();
    }
  }

  launcher.addEventListener("click", () => setOpen(panel.hidden));
  closeButton.addEventListener("click", () => setOpen(false));
  form.addEventListener("submit", submit);
})();

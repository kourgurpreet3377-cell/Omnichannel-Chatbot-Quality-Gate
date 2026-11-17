"""FastAPI server for the mock chatbot: JSON API plus a web chat widget.

    uvicorn mock_chatbot.app:app --port 8080

Set ``MOCK_BOT_FAULT`` to ``hallucinate``, ``forget`` or ``leak`` to start the
bot with a known regression and watch the quality gate fail.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mock_chatbot.brain import Fault
from mock_chatbot.service import ChatRequest, ChatResponse, ChatService

STATIC_DIR = Path(__file__).parent / "static"


def create_app(fault: Fault | None = None) -> FastAPI:
    active_fault = fault or Fault(os.getenv("MOCK_BOT_FAULT", "none").lower())
    service = ChatService(active_fault)
    app = FastAPI(title="Nimbus Telecom mock chatbot", version="1.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "fault": active_fault.value}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return service.chat(request)

    @app.get("/", include_in_schema=False)
    def widget_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()

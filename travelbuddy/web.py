"""FastAPI web UI. Agent progress is streamed to the browser as newline-delimited JSON."""

import json
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Event, TravelAgent
from .config import Settings
from .llm import DEFAULT_MODELS, LLMError, get_provider
from .llm.ollama_provider import list_ollama_models
from .llm.openai_provider import OPENAI_COMPATIBLE
from .trip import TripRequest

STATIC = Path(__file__).parent / "static"
MAX_SESSIONS = 50


class PlanRequest(TripRequest):
    model: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="TravelBuddy")
    sessions: OrderedDict[str, TravelAgent] = OrderedDict()
    lock = threading.Lock()

    def stream(session_id: str, events: Iterator[Event]):
        yield json.dumps({"type": "session", "id": session_id}) + "\n"
        try:
            for ev in events:
                yield json.dumps(ev.to_dict(), ensure_ascii=False, default=str) + "\n"
        except LLMError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        except Exception as e:  # noqa: BLE001 - surface unexpected failures in the UI
            yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}) + "\n"

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/models")
    def models():
        options = [f"ollama:{m}" for m in list_ollama_models(settings.ollama_host)]
        key_envs = {"anthropic": "ANTHROPIC_API_KEY"} | {n: k for n, (_, k) in OPENAI_COMPATIBLE.items() if k}
        for name, model in DEFAULT_MODELS.items():
            if os.getenv(key_envs.get(name, "")):
                options.append(f"{name}:{model}")
        return {"default": settings.model, "models": options}

    @app.post("/api/plan")
    def plan(req: PlanRequest):
        try:
            provider = get_provider(req.model or settings.model, settings)
        except LLMError as e:
            raise HTTPException(400, str(e))
        agent = TravelAgent(provider, settings=settings)
        session_id = uuid.uuid4().hex
        with lock:
            sessions[session_id] = agent
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        trip = TripRequest(**req.model_dump(exclude={"model"}))
        return StreamingResponse(stream(session_id, agent.plan(trip)), media_type="application/x-ndjson")

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        with lock:
            agent = sessions.get(req.session_id)
        if agent is None:
            raise HTTPException(404, "Session expired. Please create a new plan.")
        if not req.message.strip():
            raise HTTPException(400, "Message is empty.")
        return StreamingResponse(stream(req.session_id, agent.ask(req.message)), media_type="application/x-ndjson")

    return app

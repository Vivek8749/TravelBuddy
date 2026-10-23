import json
import uuid
from typing import Any

import ollama

from ..config import Settings
from .base import LLMError, LLMProvider, LLMResponse, Message, ToolCall


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, model: str, settings: Settings):
        super().__init__(model)
        self.settings = settings
        self.client = ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout)

    @staticmethod
    def to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
                ]
            if m.role == "tool" and m.name:
                msg["tool_name"] = m.name
            out.append(msg)
        return out

    def chat(self, messages, tools=None):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_ollama_messages(messages),
            "options": {"num_ctx": self.settings.num_ctx, "temperature": self.settings.temperature},
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        # Only send `think` when enabled: models without thinking support reject think=True,
        # and some reject an explicit think=False too.
        if self.settings.think:
            kwargs["think"] = True
        try:
            resp = self.client.chat(**kwargs)
        except ollama.ResponseError as e:
            if "does not support tools" in str(e):
                raise LLMError(
                    f"Ollama model '{self.model}' does not support tool calling. "
                    "Pick a tool-capable model such as qwen3.5, qwen3, llama3.1+, or mistral-nemo."
                ) from e
            raise LLMError(f"Ollama error: {e}") from e
        except ConnectionError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.settings.ollama_host}. Is `ollama serve` running?"
            ) from e

        msg = resp.message
        calls = []
        for c in msg.tool_calls or []:
            args = c.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(id=uuid.uuid4().hex[:12], name=c.function.name, arguments=dict(args or {})))
        return LLMResponse(content=msg.content or "", tool_calls=calls, thinking=msg.thinking or "")


def list_ollama_models(host: str) -> list[str]:
    try:
        return [m.model for m in ollama.Client(host=host, timeout=5).list().models]
    except Exception:
        return []

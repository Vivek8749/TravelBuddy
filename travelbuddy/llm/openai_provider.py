"""OpenAI and any OpenAI-compatible endpoint (Groq, OpenRouter, Gemini, LM Studio, vLLM, ...)."""

import json
import os
from typing import Any

import openai

from ..config import Settings
from .base import LLMError, LLMProvider, LLMResponse, Message, ToolCall

# provider name -> (base_url, api key env var). base_url None means the official OpenAI API.
OPENAI_COMPATIBLE = {
    "openai": (None, "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "lmstudio": ("http://localhost:1234/v1", None),
    # Generic: point OPENAI_COMPAT_BASE_URL at any server speaking the OpenAI chat API.
    "compat": (os.getenv("OPENAI_COMPAT_BASE_URL"), "OPENAI_COMPAT_API_KEY"),
}


class OpenAIProvider(LLMProvider):
    def __init__(self, provider_name: str, model: str, settings: Settings):
        super().__init__(model)
        self.provider_name = provider_name
        self.settings = settings
        base_url, key_env = OPENAI_COMPATIBLE[provider_name]
        if provider_name == "compat" and not base_url:
            raise LLMError("Set OPENAI_COMPAT_BASE_URL to use the 'compat' provider.")
        api_key = os.getenv(key_env) if key_env else None
        if key_env and not api_key and provider_name != "compat":
            raise LLMError(f"Set {key_env} to use the '{provider_name}' provider.")
        self.client = openai.OpenAI(
            api_key=api_key or "not-needed", base_url=base_url, timeout=settings.request_timeout
        )

    @staticmethod
    def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ],
                })
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def chat(self, messages, tools=None):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_openai_messages(messages),
            "temperature": self.settings.temperature,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.OpenAIError as e:
            raise LLMError(f"{self.provider_name} error: {e}") from e

        msg = resp.choices[0].message
        calls = []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args))
        return LLMResponse(content=msg.content or "", tool_calls=calls)

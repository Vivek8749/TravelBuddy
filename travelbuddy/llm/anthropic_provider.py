import os
from typing import Any

import anthropic

from ..config import Settings
from .base import LLMError, LLMProvider, LLMResponse, Message, ToolCall


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, model: str, settings: Settings):
        super().__init__(model)
        self.settings = settings
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("Set ANTHROPIC_API_KEY to use the 'anthropic' provider.")
        self.client = anthropic.Anthropic(timeout=settings.request_timeout)

    @staticmethod
    def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                # Consecutive tool results must share one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content})
        return system, out

    def chat(self, messages, tools=None):
        system, msgs = self.to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": msgs,
        }
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.AnthropicError as e:
            raise LLMError(f"Anthropic error: {e}") from e

        text, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
        return LLMResponse(content="".join(text), tool_calls=calls)

"""Provider-neutral chat types shared by every LLM backend."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool"
    name: str | None = None  # tool name, set on role="tool"


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""


class LLMError(RuntimeError):
    """Raised when a provider call fails in a way the user should see."""


class LLMProvider(ABC):
    provider_name: str = ""

    def __init__(self, model: str):
        self.model = model

    @property
    def label(self) -> str:
        return f"{self.provider_name}:{self.model}"

    @abstractmethod
    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        """Send the conversation and return the assistant turn.

        `tools` are neutral specs: {"name", "description", "parameters" (JSON schema)}.
        """

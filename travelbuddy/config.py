"""Runtime settings, read from environment variables with sensible defaults."""

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Model spec: "<provider>:<model>", e.g. "ollama:qwen3.5:4b", "openai:gpt-4o-mini",
    # "anthropic:claude-opus-5-5", "groq:llama-3.3-70b-versatile". A bare name means Ollama.
    model: str = field(default_factory=lambda: os.getenv("TRAVELBUDDY_MODEL", "ollama:qwen3.5:4b"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    # Ollama defaults to a small context window; search results need more room.
    # Also used to size history compaction for every provider.
    num_ctx: int = field(default_factory=lambda: int(os.getenv("TRAVELBUDDY_NUM_CTX", "32768")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TRAVELBUDDY_TEMPERATURE", "0.4")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("TRAVELBUDDY_MAX_TOKENS", "8192")))
    # Let reasoning models (qwen3.x, deepseek-r1, ...) think before answering. Slower, sometimes better.
    think: bool = field(default_factory=lambda: _env_bool("TRAVELBUDDY_THINK", False))
    max_steps: int = field(default_factory=lambda: int(os.getenv("TRAVELBUDDY_MAX_STEPS", "12")))
    # Tool output is truncated before going back to the model to protect small context windows.
    tool_result_chars: int = field(default_factory=lambda: int(os.getenv("TRAVELBUDDY_TOOL_RESULT_CHARS", "4000")))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("TRAVELBUDDY_TIMEOUT", "600")))

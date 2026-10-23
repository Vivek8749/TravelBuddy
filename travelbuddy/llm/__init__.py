from ..config import Settings
from .base import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from .openai_provider import OPENAI_COMPATIBLE

PROVIDERS = ["ollama", "anthropic", *OPENAI_COMPATIBLE]

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split "provider:model" into its parts. Unknown prefixes are treated as an Ollama model
    name, since Ollama tags contain colons too (e.g. "qwen3.5:4b")."""
    spec = spec.strip()
    prefix, sep, rest = spec.partition(":")
    if sep and prefix.lower() in PROVIDERS:
        provider, model = prefix.lower(), rest.strip()
    elif not sep and spec.lower() in PROVIDERS:
        provider, model = spec.lower(), ""
    else:
        provider, model = "ollama", spec
    if not model:
        model = DEFAULT_MODELS.get(provider, "")
    if not model:
        raise LLMError(f"No model given for provider '{provider}'. Use '{provider}:<model-name>'.")
    return provider, model


def get_provider(spec: str, settings: Settings | None = None) -> LLMProvider:
    settings = settings or Settings()
    provider, model = parse_model_spec(spec)
    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(model, settings)
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, settings)
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(provider, model, settings)


__all__ = [
    "LLMError", "LLMProvider", "LLMResponse", "Message", "ToolCall",
    "DEFAULT_MODELS", "PROVIDERS", "get_provider", "parse_model_spec",
]

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments object
    func: Callable[..., Any]

    @property
    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {t.name: t for t in tools or []}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def specs(self) -> list[dict[str, Any]]:
        return [t.spec for t in self._tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return a JSON string. Errors are returned, not raised,
        so the model can see what went wrong and retry."""
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool '{name}'. Available tools: {', '.join(self.names)}"})
        allowed = tool.parameters.get("properties", {})
        # Small models sometimes invent extra arguments; drop them instead of failing.
        args = {k: v for k, v in (arguments or {}).items() if k in allowed}
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            return json.dumps({"error": f"Missing required argument(s): {', '.join(missing)}"})
        try:
            result = tool.func(**args)
        except Exception as e:  # noqa: BLE001 - any tool failure goes back to the model
            log.debug("tool %s failed", name, exc_info=True)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        return json.dumps(result, ensure_ascii=False, default=str)

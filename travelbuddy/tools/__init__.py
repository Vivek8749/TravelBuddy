from . import planning, search, travel
from .registry import Tool, ToolRegistry


def default_registry() -> ToolRegistry:
    return ToolRegistry([*search.TOOLS, *travel.TOOLS, *planning.TOOLS])


__all__ = ["Tool", "ToolRegistry", "default_registry"]

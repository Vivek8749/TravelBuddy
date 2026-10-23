"""TravelBuddy: an agentic, budget-aware travel planner for local and hosted LLMs."""

from .agent import Event, TravelAgent
from .config import Settings
from .trip import TripRequest

__all__ = ["Event", "Settings", "TravelAgent", "TripRequest"]
__version__ = "0.1.0"

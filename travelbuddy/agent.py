"""The agent loop: the LLM decides which tools to call until it can write the plan."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from .config import Settings
from .llm import LLMProvider, Message, ToolCall
from .prompts import EMPTY_REPLY_PROMPT, FINALIZE_PROMPT, system_prompt
from .tools import ToolRegistry, default_registry
from .trip import TripRequest

THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)
COMPACT_TO = 600  # chars kept from an old tool result when compacting history


@dataclass
class Event:
    """Progress events yielded by the agent, consumed by the CLI and the web UI."""

    type: str  # "status" | "tool_call" | "tool_result" | "answer" | "error"
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


class TravelAgent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry | None = None,
        settings: Settings | None = None,
    ):
        self.provider = provider
        self.registry = registry or default_registry()
        self.settings = settings or Settings()
        self.messages: list[Message] = [Message("system", system_prompt())]
        self.trip: TripRequest | None = None

    def plan(self, trip: TripRequest) -> Iterator[Event]:
        """Create a travel plan for a trip request."""
        self.trip = trip
        self.messages.append(Message("user", trip.to_prompt()))
        # Run the budget split up front from the validated request rather than trusting the model to
        # copy numbers into the call; it is recorded as the model's own tool call so it reads naturally.
        call = ToolCall("budget0", "allocate_budget", {
            "total_budget": trip.budget, "days": trip.days, "travelers": trip.travelers,
            "travel_style": trip.travel_style, "currency": trip.currency,
            "includes_transport_to_destination": trip.budget_includes_travel and bool(trip.origin),
        })
        self.messages.append(Message("assistant", "", tool_calls=[call]))
        yield from self._run_tool(call)
        yield from self._loop()

    def ask(self, text: str) -> Iterator[Event]:
        """Send a follow-up message and run the tool loop."""
        self.messages.append(Message("user", text))
        return self._loop()

    def _run_tool(self, call: ToolCall) -> Iterator[Event]:
        args = dict(call.arguments)
        if self.trip and call.name in {"allocate_budget", "calculate_trip_cost"}:
            # Pin facts the model must not get wrong; leave days/travelers adjustable for what-ifs.
            if call.name == "calculate_trip_cost":
                args["budget"] = self.trip.budget
                args["currency"] = self.trip.currency
            args.setdefault("days", self.trip.days)
            args.setdefault("travelers", self.trip.travelers)
        yield Event("tool_call", {"id": call.id, "name": call.name, "arguments": args})
        result = self.registry.run(call.name, args)
        yield Event("tool_result", {"id": call.id, "name": call.name, "result": _preview(result)})
        self.messages.append(Message(
            "tool", result[: self.settings.tool_result_chars], tool_call_id=call.id, name=call.name
        ))

    def _compact(self) -> None:
        """Shrink the oldest tool results when the conversation nears the context window.
        Ollama silently drops the start of an over-long prompt, which loses the system prompt."""
        reserve_tokens = 6000  # room for the final plan
        budget_chars = max(12000, (self.settings.num_ctx - reserve_tokens) * 3)
        total = sum(len(m.content) for m in self.messages)
        for m in self.messages:
            if total <= budget_chars:
                return
            if m.role == "tool" and len(m.content) > COMPACT_TO:
                total -= len(m.content) - COMPACT_TO
                m.content = m.content[:COMPACT_TO] + " ...[truncated to save context]"

    def _loop(self) -> Iterator[Event]:
        nudged = False
        for step in range(1, self.settings.max_steps + 1):
            yield Event("status", {"message": f"Thinking (step {step})...", "step": step})
            self._compact()
            resp = self.provider.chat(self.messages, self.registry.specs)

            if resp.tool_calls:
                self.messages.append(Message("assistant", resp.content, tool_calls=resp.tool_calls))
                for call in resp.tool_calls:
                    yield from self._run_tool(call)
                continue

            answer = _clean(resp.content)
            if answer:
                self.messages.append(Message("assistant", answer))
                yield Event("answer", {"content": answer})
                return
            if nudged:
                break
            # Small models occasionally return an empty turn; nudge once before forcing the answer.
            nudged = True
            self.messages.append(Message("user", EMPTY_REPLY_PROMPT))

        # Out of steps (or stuck): ask for the final answer with tools disabled.
        yield Event("status", {"message": "Writing the final plan..."})
        self.messages.append(Message("user", FINALIZE_PROMPT))
        self._compact()
        answer = _clean(self.provider.chat(self.messages, tools=None).content)
        if not answer:
            yield Event("error", {"message": "The model returned an empty answer. Try a larger model."})
            return
        self.messages.append(Message("assistant", answer))
        yield Event("answer", {"content": answer})


def _clean(text: str) -> str:
    return THINK_TAGS.sub("", text or "").strip()


def _preview(result: str, limit: int = 300) -> str:
    """Short human-readable summary of a tool result for progress display."""
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return result[:limit]
    if isinstance(data, dict):
        if "error" in data:
            return f"error: {data['error']}"
        if "results" in data:
            return f"{len(data['results'])} results: " + "; ".join(r["title"] for r in data["results"][:3])
        if "status" in data:
            return f"total {data['grand_total']} {data['currency']} ({data['status']})"
        if "summary" in data:
            s = data["summary"]
            return f"{data['location']}: {s['avg_low_c']}-{s['avg_high_c']}°C, {s['rainy_days']} rainy day(s) [{data['kind']}]"
        if "converted" in data:
            return f"{data['amount']} {data['from']} = {data['converted']} {data['to']}"
        if "allocation" in data:
            return f"{data['per_person_per_day_all_in']} {data['currency']} per person per day ({data['travel_style']})"
        if "title" in data:
            return f"read '{data['title']}'"
    return result[:limit]

import json
from datetime import date

import pytest

from travelbuddy.agent import TravelAgent
from travelbuddy.config import Settings
from travelbuddy.llm import LLMProvider, LLMResponse, Message, ToolCall, parse_model_spec
from travelbuddy.llm.anthropic_provider import AnthropicProvider
from travelbuddy.llm.ollama_provider import OllamaProvider
from travelbuddy.llm.openai_provider import OpenAIProvider
from travelbuddy.trip import TripRequest


class ScriptedProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self, responses):
        super().__init__("fake")
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)


TRIP = TripRequest(destination="Goa", origin="Pune", days=4, budget=30000, currency="inr", travelers=2)


def test_agent_runs_tools_then_answers():
    provider = ScriptedProvider([
        LLMResponse("", [ToolCall("1", "allocate_budget", {"total_budget": 30000, "days": 4, "travelers": 2})]),
        LLMResponse("<think>hmm</think># Goa Trip Plan\nDone."),
    ])
    agent = TravelAgent(provider, settings=Settings(max_steps=5))
    events = list(agent.plan(TRIP))

    types = [e.type for e in events]
    # One pre-computed budget split plus the model's own call.
    assert types.count("tool_call") == 2 and types.count("tool_result") == 2
    assert events[-1].type == "answer"
    assert events[-1].data["content"] == "# Goa Trip Plan\nDone."
    tool_msg = agent.messages[-2]
    assert tool_msg.role == "tool" and '"per_person_per_day_all_in"' in tool_msg.content


def test_agent_forces_answer_after_max_steps():
    loop_call = LLMResponse("", [ToolCall("x", "allocate_budget", {"total_budget": 1, "days": 1})])
    provider = ScriptedProvider([loop_call, loop_call, LLMResponse("# Plan")])
    agent = TravelAgent(provider, settings=Settings(max_steps=2))
    events = list(agent.plan(TRIP))
    assert events[-1].data["content"] == "# Plan"
    assert provider.calls[-1]["tools"] is None


def test_agent_nudges_once_on_empty_reply():
    provider = ScriptedProvider([LLMResponse(""), LLMResponse("# Plan")])
    events = list(TravelAgent(provider, settings=Settings(max_steps=5)).plan(TRIP))
    assert events[-1].data["content"] == "# Plan"
    assert len(provider.calls) == 2


def test_followup_keeps_history():
    provider = ScriptedProvider([LLMResponse("# Plan"), LLMResponse("Sure, cheaper.")])
    agent = TravelAgent(provider, settings=Settings())
    list(agent.plan(TRIP))
    list(agent.ask("make it cheaper"))
    roles = [m.role for m in provider.calls[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]


def test_plan_precomputes_allocation_and_pins_budget():
    provider = ScriptedProvider([
        LLMResponse("", [ToolCall("c", "calculate_trip_cost", {
            "items": [{"category": "food", "item": "meals", "cost": 500, "unit": "per_person_per_day"}],
            "budget": 99, "currency": "USD",
        })]),
        LLMResponse("# Plan"),
    ])
    agent = TravelAgent(provider, settings=Settings())
    events = list(agent.plan(TRIP))
    first_call = next(e for e in events if e.type == "tool_call")
    assert first_call.data["name"] == "allocate_budget"
    assert first_call.data["arguments"]["total_budget"] == 30000
    cost = json.loads(agent.messages[-2].content)
    assert cost["budget"] == 30000 and cost["currency"] == "INR"
    assert cost["grand_total"] == 500 * 2 * 4


def test_compaction_truncates_old_tool_results():
    agent = TravelAgent(ScriptedProvider([]), settings=Settings(num_ctx=8000))
    agent.messages += [Message("tool", "x" * 9000, tool_call_id=str(i), name="web_search") for i in range(3)]
    agent._compact()
    sizes = [len(m.content) for m in agent.messages if m.role == "tool"]
    assert sizes[0] < 1000 and sum(sizes) <= 12000


def test_trip_prompt_mentions_budget_and_dates():
    trip = TripRequest(destination="Rome", days=3, budget=1200, start_date=date(2026, 5, 1), travel_style="Budget")
    prompt = trip.to_prompt()
    assert "1,200 USD" in prompt and "2026-05-01 to 2026-05-03" in prompt
    assert trip.travel_style == "budget"
    with pytest.raises(ValueError):
        TripRequest(destination="Rome", days=3, budget=0)


@pytest.mark.parametrize("spec,expected", [
    ("qwen3.5:4b", ("ollama", "qwen3.5:4b")),
    ("ollama:llama3.1:8b", ("ollama", "llama3.1:8b")),
    ("openai:gpt-4o-mini", ("openai", "gpt-4o-mini")),
    ("anthropic", ("anthropic", "claude-opus-5-5")),
    ("Groq:llama-3.3-70b-versatile", ("groq", "llama-3.3-70b-versatile")),
])
def test_parse_model_spec(spec, expected):
    assert parse_model_spec(spec) == expected


CONVO = [
    Message("system", "sys"),
    Message("user", "hi"),
    Message("assistant", "", tool_calls=[ToolCall("a", "web_search", {"query": "q"}),
                                         ToolCall("b", "get_weather", {"location": "x"})]),
    Message("tool", "r1", tool_call_id="a", name="web_search"),
    Message("tool", "r2", tool_call_id="b", name="get_weather"),
]


def test_ollama_message_conversion():
    out = OllamaProvider.to_ollama_messages(CONVO)
    assert out[2]["tool_calls"][0]["function"] == {"name": "web_search", "arguments": {"query": "q"}}
    assert out[3]["tool_name"] == "web_search"


def test_openai_message_conversion():
    out = OpenAIProvider.to_openai_messages(CONVO)
    assert out[2]["tool_calls"][1]["function"]["arguments"] == '{"location": "x"}'
    assert out[4] == {"role": "tool", "tool_call_id": "b", "content": "r2"}


def test_anthropic_message_conversion_groups_tool_results():
    system, out = AnthropicProvider.to_anthropic_messages(CONVO)
    assert system == "sys"
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]

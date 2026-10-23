# ✈️ TravelBuddy

An agentic AI travel planner. You give it a destination and a **budget**. The LLM then decides which tools to
call: it searches the web, reads pages, checks the weather and converts currencies. It checks the cost
breakdown against your budget and then writes a day-by-day plan with a food guide and a local guide.

It runs on a **local Ollama model** by default, and also works with OpenAI, Anthropic, Groq, OpenRouter,
Gemini, LM Studio or any OpenAI-compatible server.

## How it works

```
 you ──► trip details + budget
              │
              ▼
   ┌──────── agent loop ────────┐      tools
   │  LLM decides next action   │ ───► allocate_budget      (deterministic budget split)
   │  ◄── tool results ──────── │ ───► web_search           (DuckDuckGo + fallbacks, no API key)
   │  repeat until it has       │ ───► fetch_webpage        (read a page for prices/details)
   │  enough to write the plan  │ ───► get_weather          (Open-Meteo forecast / last-year climate)
   └────────────┬───────────────┘ ───► convert_currency     (live FX rates)
                ▼                 ───► calculate_trip_cost  (exact totals vs. budget)
     Markdown plan + guides
     (saved to plans/, follow-up chat to refine)
```

* All the money math happens in Python tools rather than in the LLM, because small local models make
  arithmetic mistakes. The budget split is computed up front from your validated input. The real budget is
  always pinned into the cost check, so the model can't misread it.
* If the estimate comes out over budget, the agent is told to swap in cheaper options and check again.
* Older tool results get compacted as the conversation nears the model's context window. Without this,
  Ollama silently drops the start of an over-long prompt.
* The loop stops after `max_steps` and asks for the final answer with tools turned off.

## Setup

```bash
cd TravelBuddy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ollama pull qwen3.5:4b        # or any tool-capable model: qwen3, llama3.1, mistral-nemo, ...
```

## Usage

### Terminal (interactive)

```bash
travelbuddy                    # or: python -m travelbuddy
```

It asks for the destination, origin, dates, travelers, currency, **budget**, travel style and interests. It
shows each tool call as it happens, prints the plan, saves it to `plans/<destination>-<date>.md`, and then
lets you ask follow-ups ("make it cheaper", "add a day trip to Hampi").

You can skip the questions by passing flags:

```bash
travelbuddy --destination "Goa, India" --origin Pune --start 2026-11-20 --days 4 \
  --travelers 2 --budget 40000 --currency INR --style budget --interests "beaches, seafood, forts"
```

### Web UI

```bash
travelbuddy --web              # http://127.0.0.1:8000
```

The web UI has a form, a live log of tool calls, the rendered plan, a Markdown download, and follow-up chat.

### Choosing a model

Use `--model provider:model`, or set `TRAVELBUDDY_MODEL`:

| Provider | Example | Needs |
|---|---|---|
| Ollama (default) | `ollama:qwen3.5:4b`, `llama3.1:8b` (bare names mean Ollama) | running `ollama serve` |
| OpenAI | `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| Groq | `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| OpenRouter | `openrouter:<model>` | `OPENROUTER_API_KEY` |
| Gemini | `gemini:gemini-2.0-flash` | `GEMINI_API_KEY` |
| LM Studio | `lmstudio:<loaded model>` | LM Studio server on :1234 |
| Any OpenAI-compatible | `compat:<model>` | `OPENAI_COMPAT_BASE_URL` (+ optional `OPENAI_COMPAT_API_KEY`) |

`travelbuddy --list-models` lists your local Ollama models.

### Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `TRAVELBUDDY_MODEL` | `ollama:qwen3.5:4b` | default model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `TRAVELBUDDY_NUM_CTX` | `32768` | context window (Ollama) and size for history compaction |
| `TRAVELBUDDY_THINK` | `false` | let reasoning models think first (also `--think`) |
| `TRAVELBUDDY_MAX_STEPS` | `12` | max tool-calling rounds |
| `TRAVELBUDDY_TEMPERATURE` | `0.4` | sampling temperature |
| `TRAVELBUDDY_TOOL_RESULT_CHARS` | `4000` | per-tool-result truncation |

## Tips

* **Model size matters.** A 4B model handles the tool loop fine, but it can get local geography wrong and
  takes about 5–8 minutes per plan on a typical GPU. An 8B–14B local model, or any hosted model, gives
  noticeably better and faster plans.
* Prices come from web snippets, so they're estimates. Check before you book.

## Development

```bash
pytest
```

The tests cover the budget math, the tool registry, the agent loop (with a scripted fake LLM), budget
pinning, history compaction, model-spec parsing, and message conversion for each provider. No network is
needed.

### Project layout

```
travelbuddy/
  agent.py          agent loop, events, compaction
  prompts.py        system prompt and plan format
  trip.py           validated TripRequest
  cli.py / web.py   terminal and FastAPI front ends (static/index.html)
  llm/              provider abstraction: ollama, openai(-compatible), anthropic
  tools/            registry + search, travel (weather/FX), planning (budget math)
```

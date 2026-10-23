import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from .agent import TravelAgent
from .config import Settings
from .llm import PROVIDERS, LLMError, get_provider
from .llm.ollama_provider import list_ollama_models
from .trip import TRAVEL_STYLES, TripRequest

console = Console()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="travelbuddy",
        description="Agentic AI travel planner. Asks for your budget and builds a researched plan with guides.",
    )
    p.add_argument("-m", "--model", help="provider:model, e.g. ollama:qwen3.5:4b, openai:gpt-4o-mini, "
                   f"anthropic:claude-opus-5-5. Providers: {', '.join(PROVIDERS)}")
    p.add_argument("--list-models", action="store_true", help="List local Ollama models and exit.")
    p.add_argument("--think", action="store_true", help="Enable thinking for reasoning models (slower).")
    p.add_argument("--max-steps", type=int, help="Max tool-calling rounds before forcing an answer.")
    p.add_argument("--web", action="store_true", help="Start the web UI instead of the terminal planner.")
    p.add_argument("--host", default="127.0.0.1", help="Web UI host (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=8000, help="Web UI port (default 8000).")
    p.add_argument("-o", "--output", type=Path, help="Where to save the plan (default plans/<destination>-<date>.md).")
    p.add_argument("--no-followup", action="store_true", help="Exit after the plan instead of chatting.")
    p.add_argument("-v", "--verbose", action="store_true", help="Show full tool arguments.")

    trip = p.add_argument_group("trip (anything omitted is asked interactively)")
    trip.add_argument("--destination")
    trip.add_argument("--origin")
    trip.add_argument("--start", help="Start date YYYY-MM-DD (or 'flexible').")
    trip.add_argument("--days", type=int)
    trip.add_argument("--travelers", type=int)
    trip.add_argument("--budget", type=float)
    trip.add_argument("--currency")
    trip.add_argument("--style", choices=TRAVEL_STYLES)
    trip.add_argument("--interests")
    trip.add_argument("--excludes-travel", action="store_true",
                      help="Budget does NOT include getting to/from the destination.")
    return p


def ask_trip(args: argparse.Namespace) -> TripRequest:
    """Fill in whatever the flags didn't provide by asking the user."""
    interactive = sys.stdin.isatty()

    def ask(value, prompt_fn, *a, **kw):
        if value is not None:
            return value
        if not interactive:
            if "default" in kw:
                return kw["default"]
            raise SystemExit(f"Missing required option for non-interactive use: {a[0]}")
        return prompt_fn(*a, **kw)

    console.print(Panel.fit("[bold]Let's plan your trip![/bold]\nAnswer a few questions; press Enter to accept defaults.",
                            border_style="cyan"))
    destination = ask(args.destination, Prompt.ask, "[cyan]Where do you want to go?[/cyan]")
    origin = ask(args.origin, Prompt.ask, "[cyan]Where are you starting from?[/cyan] (blank to skip)", default="")
    start_raw = ask(args.start, Prompt.ask, "[cyan]Start date[/cyan] (YYYY-MM-DD, or blank if flexible)", default="")
    while True:
        try:
            start = date.fromisoformat(start_raw) if start_raw and start_raw != "flexible" else None
            break
        except ValueError:
            if not interactive:
                raise SystemExit(f"Invalid date: {start_raw}")
            start_raw = Prompt.ask("[red]Please use YYYY-MM-DD[/red]", default="")
    days = ask(args.days, IntPrompt.ask, "[cyan]How many days?[/cyan]", default=5)
    travelers = ask(args.travelers, IntPrompt.ask, "[cyan]How many travelers?[/cyan]", default=1)
    currency = ask(args.currency, Prompt.ask, "[cyan]Budget currency[/cyan] (ISO code)", default="USD")
    budget = ask(args.budget, FloatPrompt.ask, f"[bold cyan]Total budget for everyone ({currency.upper()})?[/bold cyan]")
    includes_travel = not args.excludes_travel
    if args.budget is None and interactive and origin:
        includes_travel = Confirm.ask("[cyan]Does the budget include travel to and from the destination?[/cyan]",
                                      default=True)
    style = ask(args.style, Prompt.ask, "[cyan]Travel style[/cyan]", choices=list(TRAVEL_STYLES), default="mid-range")
    interests = ask(args.interests, Prompt.ask,
                    "[cyan]Interests[/cyan] (e.g. food, history, beaches, hiking, nightlife)", default="")

    return TripRequest(
        destination=destination, origin=origin, start_date=start, days=days, travelers=travelers,
        budget=budget, currency=currency, travel_style=style, budget_includes_travel=includes_travel,
        interests=interests,
    )


def run_turn(agent: TravelAgent, events, verbose: bool) -> str | None:
    answer = None
    with console.status("Thinking...", spinner="dots") as status:
        for ev in events:
            if ev.type == "status":
                status.update(ev.data["message"])
            elif ev.type == "tool_call":
                args = json.dumps(ev.data["arguments"], ensure_ascii=False)
                if not verbose and len(args) > 120:
                    args = args[:117] + "..."
                console.print(f"[yellow]🔧 {ev.data['name']}[/yellow] [dim]{args}[/dim]")
                status.update(f"Running {ev.data['name']}...")
            elif ev.type == "tool_result":
                console.print(f"   [green]↳[/green] [dim]{ev.data['result']}[/dim]")
            elif ev.type == "answer":
                answer = ev.data["content"]
            elif ev.type == "error":
                console.print(f"[red]{ev.data['message']}[/red]")
    return answer


def default_output(trip: TripRequest) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", trip.destination.lower()).strip("-") or "trip"
    return Path("plans") / f"{slug}-{date.today().isoformat()}.md"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings()
    if args.model:
        settings.model = args.model
    if args.think:
        settings.think = True
    if args.max_steps:
        settings.max_steps = args.max_steps

    if args.list_models:
        models = list_ollama_models(settings.ollama_host)
        console.print("\n".join(f"ollama:{m}" for m in models) or "[red]No Ollama models found (is Ollama running?)[/red]")
        return

    if args.web:
        import uvicorn

        from .web import create_app

        console.print(f"[bold green]TravelBuddy web UI:[/bold green] http://{args.host}:{args.port}")
        uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning")
        return

    try:
        provider = get_provider(settings.model, settings)
    except LLMError as e:
        raise SystemExit(str(e))
    console.print(f"[dim]Model: {provider.label}[/dim]")

    try:
        trip = ask_trip(args)
    except ValidationError as e:
        raise SystemExit(f"Invalid trip details:\n{e}")
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)

    agent = TravelAgent(provider, settings=settings)
    console.rule(f"[bold]Planning {trip.destination}[/bold]")
    try:
        answer = run_turn(agent, agent.plan(trip), args.verbose)
    except LLMError as e:
        raise SystemExit(f"[LLM error] {e}")
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled.")
    if not answer:
        raise SystemExit(1)

    console.rule("[bold green]Your travel plan[/bold green]")
    console.print(Markdown(answer))
    output = args.output or default_output(trip)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(answer, encoding="utf-8")
    console.print(f"\n[green]Saved to {output}[/green]")

    if args.no_followup or not sys.stdin.isatty():
        return
    console.print("[dim]Ask follow-up questions or request changes (e.g. 'make it cheaper', "
                  "'add a day trip'). Press Enter on an empty line to quit.[/dim]")
    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]", default="")
        except (KeyboardInterrupt, EOFError):
            break
        if not question.strip():
            break
        try:
            reply = run_turn(agent, agent.ask(question), args.verbose)
        except LLMError as e:
            console.print(f"[red]{e}[/red]")
            continue
        except KeyboardInterrupt:
            console.print("[dim]Cancelled.[/dim]")
            continue
        if reply:
            console.print(Markdown(reply))
            # Keep the saved file in sync when the reply is a revised full plan.
            if reply.lstrip().startswith("# "):
                output.write_text(reply, encoding="utf-8")
                console.print(f"[green]Updated {output}[/green]")

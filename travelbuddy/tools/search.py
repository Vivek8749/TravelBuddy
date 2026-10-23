"""Web research tools: DuckDuckGo search (no API key) and page fetching."""

import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from .registry import Tool

SEARCH_BACKENDS = ("auto", "duckduckgo", "brave,mojeek,yahoo")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def web_search(query: str, max_results: int = 5) -> dict:
    max_results = max(1, min(int(max_results), 10))
    results = []
    # Individual engines often rate-limit or come back empty; fall back to others before giving up.
    for backend in SEARCH_BACKENDS:
        try:
            results = DDGS().text(query, max_results=max_results, backend=backend)
        except DDGSException:
            continue
        if results:
            break
    if not results:
        return {"query": query, "results": [], "note": "No results. Try a shorter, simpler query (fewer keywords)."}
    return {
        "query": query,
        "results": [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ],
    }


def fetch_webpage(url: str, max_chars: int = 3000) -> dict:
    if urlparse(url).scheme not in {"http", "https"}:
        return {"error": "Only http(s) URLs can be fetched."}
    max_chars = max(500, min(int(max_chars), 6000))
    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True)
    if resp.status_code >= 400:
        return {"url": url, "error": f"HTTP {resp.status_code}; the site blocked us. Try another result."}
    if "html" not in resp.headers.get("content-type", "html"):
        return {"url": url, "error": f"Not an HTML page ({resp.headers.get('content-type')})."}
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    # Drop tiny fragments (menu items, TOC numbers) that waste the model's context.
    lines = [ln for ln in main.get_text("\n", strip=True).splitlines() if len(ln) > 3]
    text = re.sub(r"\n\s*\n+", "\n\n", "\n".join(lines))
    title = soup.title.get_text(strip=True) if soup.title else ""
    return {"url": str(resp.url), "title": title, "content": text[:max_chars], "truncated": len(text) > max_chars}


TOOLS = [
    Tool(
        name="web_search",
        description=(
            "Search the web for up-to-date travel information: attractions, hotel and hostel prices, "
            "flight/train/bus fares, visa rules, local food, events, safety advice, travel guides. "
            "Use specific queries, e.g. 'Lisbon hostel price per night 2026' or 'Delhi to Jaipur train fare'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "Number of results, 1-10 (default 5)."},
            },
            "required": ["query"],
        },
        func=web_search,
    ),
    Tool(
        name="fetch_webpage",
        description=(
            "Read the text of a web page found via web_search, to get details (prices, opening hours, "
            "itineraries) that the search snippet does not show."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full http(s) URL."},
                "max_chars": {"type": "integer", "description": "Max characters to return (default 3000)."},
            },
            "required": ["url"],
        },
        func=fetch_webpage,
    ),
]

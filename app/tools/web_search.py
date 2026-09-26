"""Web search tool: looks up a query on DuckDuckGo and returns short text snippets.

We use DuckDuckGo (via the `ddgs` package) instead of Google/Bing because it needs
no API key and no billing account - important for a portfolio project that other
people should be able to clone and run without signing up for anything.
"""

from ddgs import DDGS


def search_web(query: str, max_results: int = 3) -> str:
    """Return up to `max_results` search result snippets for `query` as one string.

    Each result is formatted as:
        1. <title>
           <snippet>
           <url>
    so the LLM gets enough context to answer without needing to click through.
    """
    query = query.strip()
    if not query:
        raise ValueError("Search query must not be empty.")

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return f"No search results found for '{query}'."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        formatted.append(f"{i}. {title}\n   {body}\n   {href}")

    return "\n".join(formatted)


TOOL_SPEC = {
    "name": "web_search",
    "description": (
        "Search the web for current information (news, facts, definitions, "
        "anything not in the model's own knowledge). Input is a plain search "
        "query string, e.g. 'population of Iceland 2025'."
    ),
}

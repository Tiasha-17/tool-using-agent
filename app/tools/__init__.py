"""Tool registry: the single place that knows which tools exist.

The agent loop never imports calculator/web_search/weather directly - it only
talks to this registry. That means adding a new tool later is just:
  1. Write a new module with a `TOOL_SPEC` dict and a callable.
  2. Add one line to REGISTRY below.
No changes needed anywhere else in the codebase.
"""

from app.tools import calculator, weather, web_search

REGISTRY = {
    calculator.TOOL_SPEC["name"]: {
        "spec": calculator.TOOL_SPEC,
        "func": calculator.calculate,
    },
    web_search.TOOL_SPEC["name"]: {
        "spec": web_search.TOOL_SPEC,
        "func": web_search.search_web,
    },
    weather.TOOL_SPEC["name"]: {
        "spec": weather.TOOL_SPEC,
        "func": weather.get_weather,
    },
}


def get_tool_descriptions() -> str:
    """Render every registered tool as a numbered list for the system prompt."""
    lines = []
    for i, entry in enumerate(REGISTRY.values(), start=1):
        spec = entry["spec"]
        lines.append(f"{i}. {spec['name']}: {spec['description']}")
    return "\n".join(lines)


def run_tool(name: str, tool_input: str) -> str:
    """Look up a tool by name and run it with the given (string) input.

    Returns the tool's output as a string, or a clearly-labelled error string
    if the tool name is unknown or the tool itself raises an exception. We
    deliberately return errors as strings (not exceptions) so the agent loop
    can feed them back to the LLM as an "Observation" and let it try again.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        known = ", ".join(REGISTRY.keys())
        return f"Error: unknown tool '{name}'. Known tools: {known}."

    try:
        return entry["func"](tool_input)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any tool failure becomes an Observation
        return f"Error running tool '{name}': {exc}"

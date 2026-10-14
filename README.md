# Tool-Using Agent

A small AI agent that answers questions by reasoning step by step and calling real tools —
a calculator, a web search, and a weather API — instead of just generating a single-shot
answer. It runs on a **free, local LLM** (via [Ollama](https://ollama.com)), so it needs no
paid API key to work, and it's served as a REST API with FastAPI.

## What it does

Ask it a question. It decides, on its own, whether it needs a tool to answer:

```
"What is (45 * 12) + 7?"          -> uses the calculator tool -> "547"
"What's the weather in Dublin?"   -> uses the weather tool     -> "18.1°C, partly cloudy"
"Who is the CEO of OpenAI?"       -> uses the web search tool  -> looks it up
"What is the capital of France?"  -> answers directly, no tool needed
```

Every response includes the full reasoning trace (what it thought, which tool it called,
what the tool returned) so the process is fully transparent — see [`agent_demo.ipynb`](agent_demo.ipynb)
for a walkthrough, or the `steps` field in the API response.

## Architecture

```
User question
     |
     v
FastAPI /agent endpoint  (app/main.py)
     |
     v
Agent loop: Reason -> Act -> Observe   (app/agent.py)
     |            |
     |            +--> Tool registry (app/tools/) --> calculator | web_search | weather
     v
Local LLM via Ollama (qwen2.5:7b-instruct)
```

The agent follows the **ReAct** pattern (Reason + Act): at each step the LLM either calls
one tool or gives a final answer, in plain text. The loop feeds each tool's result back to
the LLM as an "Observation" and asks it to continue, until it reaches a final answer or a
safety limit on iterations.

## Running it

**Prerequisite:** [Ollama](https://ollama.com) installed and running, with a model pulled:

```bash
ollama pull qwen2.5:7b-instruct
```

**Locally (Python):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs for interactive API docs, or:

```bash
curl -X POST http://127.0.0.1:8000/agent \
     -H "Content-Type: application/json" \
     -d '{"question": "What is (45 * 12) + 7?"}'
```

**With Docker:**

```bash
docker compose up --build
```

(`docker-compose.yml` points the container at Ollama running on your host machine.)

## Testing

```bash
pytest -v
```

## Notebook

`agent_demo.ipynb` walks through the agent piece by piece and re-runs it live against the
real local LLM and real tools. To open it, install Jupyter into the same virtualenv:

```bash
pip install jupyter
jupyter notebook agent_demo.ipynb
```

18 tests cover each tool in isolation, the ReAct parsing/loop logic, and the HTTP API —
all using mocked LLM/network calls, so the suite runs in about a second with no external
dependencies.

## Project structure

```
app/
  main.py         FastAPI app and the /agent, /health endpoints
  agent.py        The ReAct reasoning loop
  config.py       Environment-driven settings (model name, host, iteration limit)
  schemas.py      Pydantic request/response models
  tools/
    calculator.py   Safe arithmetic evaluator (AST-based, no eval())
    web_search.py   DuckDuckGo search, no API key required
    weather.py      Open-Meteo current weather, no API key required
tests/            pytest suite (18 tests)
agent_demo.ipynb  Step-by-step notebook walkthrough of how the agent was built
Dockerfile, docker-compose.yml
```

## Tech stack

Python, FastAPI, Pydantic, Ollama (qwen2.5:7b-instruct), pytest, Docker.

---

See [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) for the main engineering trade-offs,
known limitations and the next reliability improvements.

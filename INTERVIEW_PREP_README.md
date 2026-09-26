# Interview Prep — Tool-Using Agent
### Detailed Walkthrough for AI/ML Engineer Roles

---

## 1. Project Overview

**What is this project?**
A small AI agent that answers questions by reasoning step-by-step and calling real tools
(a calculator, a web search, a weather API) rather than generating a single-shot answer.
It's served as a REST API (FastAPI), containerised (Docker), and covered by a pytest suite —
the same "serving layer" pattern used across my other projects, so the API/Docker/tests
structure should look familiar if you've seen those.

**Why does it matter for interviews?**
"Agents" are the current center of gravity in AI engineering roles. This project demonstrates,
in isolation and at small scale, the exact mechanics interviewers ask about: how an LLM decides
to call a tool, how tool results get fed back in, how to stop an agent from looping forever,
how to test something whose core component is non-deterministic, and how to run all of this
without depending on a paid API.

**Presentation note:**
This is intentionally "Beginner → Intermediate" in scope (a single agent, three tools, one
LLM backend) rather than a multi-agent system — see Section 6 for what a "next tier" version
would add, so you can speak to the roadmap even though it isn't built.

---

## 2. Core Concepts Glossary

| Term | Meaning here |
|---|---|
| **Agent** | A program that uses an LLM not just to generate text, but to decide *what to do next* (call a tool vs answer directly), in a loop. |
| **Tool / Function calling** | Giving the LLM a fixed menu of callable actions (with a name + description) it can invoke by name, instead of only producing free text. |
| **ReAct** | "Reason + Act" — a prompting pattern (Yao et al., 2022) where the model alternates between writing a `Thought`, taking an `Action`, and receiving an `Observation`, in plain text, until it reaches a `Final Answer`. |
| **Observation** | The tool's output, fed back into the conversation so the LLM can use it in its next reasoning step. |
| **Grounding** | Giving an LLM external, verifiable information (search results, API data) instead of letting it rely purely on what it memorised during training. |
| **Local LLM** | A model that runs entirely on your own machine (here, via Ollama) instead of calling a hosted API — no per-token cost, no internet dependency for the reasoning step itself. |

---

## 3. Beginner Section — The Tools

### 3.1 Why three tools, and why these three

**What I did:**
Built a calculator, a web search, and a weather lookup — one that needs no external data
(pure computation), one that needs unstructured, current information (search), and one that
needs structured, current information from a specific API (weather). That spread covers the
three broad categories of "why an LLM can't just answer this itself": it can't reliably do
arithmetic, it doesn't know anything after its training cutoff, and it has no access to
live data.

**Interview question you might get:**
*"Why not just let the model answer weather/math questions itself?"*
→ LLMs generate the *most statistically plausible* next tokens, not the *correct* ones. For
arithmetic specifically, they're notoriously unreliable on anything beyond small numbers
because they've never "computed" anything — they've only seen text describing computations.
Tools convert "the model guesses" into "the model calls a function that is actually correct."

### 3.2 The calculator: AST-based evaluation, not `eval()`

**What I did:** ([`app/tools/calculator.py`](app/tools/calculator.py))
Parsed the expression string with Python's `ast` module into a syntax tree, then walked that
tree myself, only executing operators from an explicit whitelist (`+ - * / // % ** unary -/+`).

**Why not just call `eval(expression)`?**
Because the *input string comes from an LLM*, and an LLM can be manipulated (via the user's
question, or its own reasoning going wrong) into producing something like
`__import__('os').system('rm -rf ~')`. `eval()` would run that. The AST approach physically
cannot execute anything outside the whitelist — there's no code path from "parse a number" to
"import a module," so the vulnerability class doesn't exist rather than being merely
mitigated.

**Interview question you might get:**
*"How would you make this safer/more capable?"*
→ Safer: nothing more needed for arithmetic — the whitelist approach is already the standard,
recommended pattern. More capable: add more `ast` node types (e.g. `ast.Call` for a tightly
whitelisted `sqrt`/`abs`), but each addition needs the same scrutiny — never widen the
whitelist to include names, attributes, or arbitrary calls.

### 3.3 Web search and weather: free APIs, no key required

**What I did:** ([`app/tools/web_search.py`](app/tools/web_search.py), [`app/tools/weather.py`](app/tools/weather.py))
Used DuckDuckGo (via the `ddgs` package) for search and Open-Meteo for weather — both work
with zero signup, zero API key, zero billing account. Weather requires two chained calls:
geocode the place name to latitude/longitude, then request the forecast for those coordinates.

**Why does this matter for a portfolio project specifically?**
Anyone who clones this repo can run it immediately with no setup friction (no "go get an API
key and add billing" step). That's a deliberate tradeoff, covered honestly below.

**Known limitation — be ready to discuss this:**
DuckDuckGo's free/scraped search backend is noticeably less reliable than a paid search API:
identical-looking queries sometimes return zero results, and result quality varies. This is
real, observed behaviour during testing, not a hypothetical — for example the agent handled
a query returning no results by falling back to its own knowledge and still answering
correctly (see the `agent_demo.ipynb` trace). In a production system I'd swap in a paid
provider (Tavily, SerpAPI, Bing Search API) for reliability, at the cost of needing a key
and paying per query. This is a real "cheap and self-contained" vs "reliable but costs money
and needs a key" tradeoff, and I chose "self-contained" deliberately for a project meant to
be cloned and run by anyone with zero setup.

---

## 4. Intermediate Section — The Agent Loop (ReAct)

### 4.1 Why hand-write the loop instead of using "native" function calling

**What I did:** ([`app/agent.py`](app/agent.py))
Wrote the ReAct loop as plain text parsing: the system prompt tells the model the exact format
to reply in (`Thought:` / `Action:` / `Action Input:` or `Thought:` / `Final Answer:`), and a
handful of regexes pull those fields back out of the reply.

**Why not use the model's built-in "tools" API (like OpenAI/Anthropic function calling)?**
Two reasons. First, this project deliberately runs on a small local model via Ollama, and
native structured tool-calling support is inconsistent across local models — hand-rolled
ReAct works with *any* chat-capable model, which matters for a project meant to run without
a paid API. Second — and this is the more interesting answer in an interview — writing the
loop in plain text makes every step **inspectable**: you can print the literal prompt the
model saw and the literal text it produced, with no hidden JSON-schema machinery in between.
That's valuable for debugging and for explaining exactly how the agent works.

**How I'd do it differently on Claude or OpenAI:**
I'd use their native tool-use APIs (`tools=[...]` in the request, structured `tool_use` blocks
in the response) instead of parsing text. It's more robust (no regex parsing failures, the
provider enforces the schema) and is what I'd reach for in a job where the LLM backend is a
paid API rather than a local model. Being able to explain *why* I chose differently here — not
just that I did — is the actual signal this section is meant to demonstrate.

### 4.2 The loop mechanics

**What I did:**
Each iteration: send the full conversation so far to the model → parse the reply → if it's a
tool call, run the tool and append the result as an `Observation` (as another chat message) →
if it's a final answer, stop and return it → if the reply matches neither format, insert an
error `Observation` asking it to follow the format, and try again.

**Why feed the Observation back as a `user` message rather than a special "tool" role?**
Ollama's chat API (and many local models) doesn't reliably support a distinct `tool` role the
way OpenAI's does. Using `user` role for observations is the original ReAct paper's approach
and works uniformly across any chat model — another instance of prioritising portability.

**Interview question you might get:**
*"What stops the agent from looping forever?"*
→ `settings.max_iterations` (default 6, see [`app/config.py`](app/config.py)). If the model
never produces a valid `Final Answer` within that many turns, the loop stops and returns a
clear "I couldn't reach a final answer" message rather than hanging the request indefinitely.
This is a hard requirement for anything served over HTTP — an unbounded loop is a denial-of-
service vector against your own server.

### 4.3 Parsing the model's output

**What I did:** ([`_parse_response`](app/agent.py))
Regex-matched `Thought:`, `Action:`, `Action Input:`, and `Final Answer:` as line-anchored
patterns, with `Final Answer` taking priority (checked first) since it's the terminal state.

**Interview question you might get:**
*"What happens if the model's output doesn't match your regex?"*
→ `step.action` stays `None`, the loop detects this, and sends back an explicit instruction to
follow the format instead of crashing or returning garbage. Tested directly in
[`tests/test_agent.py::TestRunAgent::test_gives_up_after_max_iterations`](tests/test_agent.py),
which forces exactly this path with a mocked model that never cooperates.

---

## 5. Advanced Section — Serving, Testing, and Deployment

### 5.1 Testing something built on a non-deterministic LLM

**What I did:** ([`tests/`](tests))
None of the 18 tests call the real Ollama server. Every test that touches the agent loop or a
tool's network call monkeypatches that boundary (`agent._call_ollama`, `requests.get`, `DDGS`)
with a fake that returns a fixed value, so the tests are deterministic and fast (~1 second for
the whole suite) and never depend on a model being installed, a network connection, or an
external API being up.

**Interview question you might get:**
*"How do you test an LLM's output quality, then?"*
→ You don't, with unit tests — that's a different problem (evaluation/eval harnesses, usually
a curated set of prompts scored by a rubric or a judge model, run separately and less
frequently). Unit tests here verify the *deterministic scaffolding around* the LLM: parsing
logic, the loop's stopping conditions, tool safety and correctness, and the API's request/
response contract. That's the testable 90% of an agent system; the LLM's own judgement is
validated separately (manual runs like the ones in `agent_demo.ipynb`, or a proper eval set
in a production system).

### 5.2 Serving layer: FastAPI + Pydantic

**What I did:** ([`app/main.py`](app/main.py), [`app/schemas.py`](app/schemas.py))
One `POST /agent` endpoint takes a `{"question": str}` body, runs the agent, and returns the
final answer *plus* the full step-by-step trace (thought/action/input/observation per step) —
not just the answer. A `GET /health` endpoint exists for container/monitoring liveness checks.

**Why return the trace instead of just the answer?**
Debuggability and trust. If an agent gives a wrong answer, "what tool did it call and what did
that tool return?" is almost always the first question you need answered — shipping that
trace by default, rather than adding it later under time pressure, is the cheap decision to
make upfront.

### 5.3 Configuration and Docker

**What I did:** ([`app/config.py`](app/config.py), [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml))
All environment-specific values (Ollama host/model, iteration limit, timeout) are read from
environment variables with defaults, so the same code runs unchanged locally and in a
container — only `OLLAMA_HOST` needs to change (`localhost` → `host.docker.internal`, since
the container needs to reach Ollama running on the host machine, not inside the container).

**Interview question you might get:**
*"Why isn't Ollama itself in the Docker Compose file?"*
→ It could be, as a second service with a shared model volume — I chose not to here because
downloading a multi-gigabyte model inside a container on every fresh clone is a poor
experience for a portfolio project people are trying out quickly; running Ollama on the host
(which most people testing this will already have, or can install once) is faster to try. In
a real deployment, I'd run Ollama (or a hosted LLM API) as its own service or managed
endpoint, not bundled with the API container — this project's docker-compose.yml is set up
for "your machine has Ollama running," and the README says so explicitly.

---

## 6. What a Next Tier Would Add

Be ready to describe these even though they're not built — this is what shows you understand
the ceiling above what you shipped:

- **Multi-agent orchestration** — a planner agent that delegates sub-tasks to specialised
  worker agents (e.g. a research agent + a math agent), rather than one agent picking from a
  flat tool list.
- **Streaming responses** — right now the API blocks until the full loop finishes; a
  production UI would want to stream each `Thought`/`Action`/`Observation` as it happens
  (Server-Sent Events or WebSockets).
- **Persistent memory** — currently each request starts a fresh conversation; a real assistant
  would need conversation history across turns/sessions (a database or vector store).
- **Structured tool-call validation** — swapping the regex-based ReAct parsing for a provider's
  native structured tool-calling API (see Section 4.1), which also enables parallel tool calls.
- **A proper eval harness** — a fixed set of test questions with expected tool-call sequences
  and answers, run automatically (e.g. in CI) to catch regressions in agent *behaviour*, not
  just in the deterministic code around it.

# Design decisions and limitations

This document records the engineering choices behind the tool-using agent. It is
intended to make the implementation reviewable, not to present the system as a
production-ready agent platform.

## Direct ReAct loop

The agent loop is written in plain Python rather than delegated to an agent
framework. This keeps tool selection, observations, stopping conditions and error
handling visible in one place. The trade-off is that the implementation parses a
text protocol, which is less reliable than provider-native structured tool calls.

For a hosted production model, the next version would use native structured tool
calling and validate each call against a schema before dispatch. The current text
protocol is retained because it works with the local Ollama model used by this
project.

## Bounded execution

The loop has a fixed maximum iteration count. An agent that never produces a final
answer therefore fails with a controlled response instead of consuming resources
indefinitely. Each network-backed tool also has a timeout and returns a bounded
result.

The system still lacks per-client rate limits, request cancellation and a total
request-level time budget. Those would be required before exposing it as a public
service.

## Calculator safety

Arithmetic expressions are parsed with Python's `ast` module and evaluated through
an explicit operator allow-list. The project does not use `eval()`, so expressions
cannot import modules, access attributes or call arbitrary code.

Adding functions such as `sqrt` would require a second explicit allow-list. General
names, attributes and arbitrary calls should remain unsupported.

## External tools

DuckDuckGo search and Open-Meteo weather data were chosen because they require no
API keys, making the repository easy to run. The cost is lower reliability and no
service-level guarantee. Search may return no results, and both tools can be
rate-limited or unavailable.

A production deployment should use supported providers, retries with backoff,
observability and provider-specific error handling.

## Testing boundary

The automated suite mocks the LLM and network calls. It verifies deterministic
behaviour around the model: parsing, stopping conditions, tool correctness, failure
handling and the HTTP contract. It does not establish that the model consistently
chooses the correct tool or produces a correct final answer.

The missing evidence is a behavioural evaluation set containing expected tool
sequences, refusal cases and answer checks. That evaluation should run separately
from unit tests because it is slower and can vary across model versions.

## Trace exposure

The API currently returns its parsed step trace for debugging. This is useful in a
local demonstration, but raw model reasoning should not be treated as a stable or
safe production interface. A deployed system should expose concise tool activity,
citations and failure information while keeping internal prompts and raw reasoning
private.

## Next improvements

1. Replace the text tool protocol with native structured tool calls.
2. Add a labelled behavioural evaluation set and regression report.
3. Add request-level time budgets, cancellation and rate limits.
4. Replace raw reasoning traces with safe, structured activity events.
5. Add persistent request logs with secrets and personal data removed.

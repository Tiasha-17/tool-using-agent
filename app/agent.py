"""The agent's reasoning loop, using the ReAct pattern (Reason + Act).

At each step we ask the LLM to either:
  - call a tool ("Action: <tool>" / "Action Input: <input>"), or
  - give its final answer ("Final Answer: <answer>").

If it calls a tool, we run it and feed the result back as an "Observation",
then ask the model to continue. This repeats until the model gives a final
answer or we hit `max_iterations`.

Why hand-roll this instead of using a model's native "function calling"?
Ollama's small local models support native tool calling inconsistently, and
writing the loop out in plain text makes every step inspectable - you can
read exactly what the model was told and exactly what it decided. That
transparency is the point of this project. A production system on Claude or
OpenAI would typically use their native tool-use APIs instead (see the
Interview Prep README for that comparison).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

from app.config import settings
from app.tools import get_tool_descriptions, run_tool

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions by reasoning \
step by step and using tools when you need information you don't already know.

You have access to the following tools:
{tool_descriptions}

Respond using EXACTLY this format, one field per line, and stop after either
"Action Input" or "Final Answer":

Thought: <your reasoning about what to do next>
Action: <one tool name, exactly as listed above>
Action Input: <the input to give that tool>

...or, once you know the answer...

Thought: <your reasoning about why you now have enough information>
Final Answer: <the answer to the user's question>

Rules:
- Only ever take ONE action per turn, then stop and wait for the Observation.
- Never invent an Observation yourself - it will be provided to you.
- Use a tool only when you actually need it; if you already know the answer, \
skip straight to Final Answer.
"""


@dataclass
class AgentStep:
    """One Reason -> Act -> Observe cycle, kept for debugging/UI display."""

    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    raw_response: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    hit_max_iterations: bool = False


_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)")
_THOUGHT_RE = re.compile(r"Thought:\s*(.+)")


def _call_ollama(messages: list[dict]) -> str:
    """Send the running conversation to Ollama's chat API and return the reply text."""
    response = requests.post(
        f"{settings.ollama_host}/api/chat",
        json={"model": settings.ollama_model, "messages": messages, "stream": False},
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _parse_response(text: str) -> AgentStep:
    """Pull Thought / Action / Action Input / Final Answer out of the model's reply."""
    step = AgentStep(raw_response=text)

    thought_match = _THOUGHT_RE.search(text)
    if thought_match:
        step.thought = thought_match.group(1).strip().splitlines()[0]

    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        step.action = "final_answer"
        step.action_input = final_match.group(1).strip()
        return step

    action_match = _ACTION_RE.search(text)
    input_match = _ACTION_INPUT_RE.search(text)
    if action_match and input_match:
        step.action = action_match.group(1).strip()
        # Action Input is meant to be one line; take everything up to the next newline.
        step.action_input = input_match.group(1).strip().splitlines()[0].strip()

    return step


def run_agent(question: str) -> AgentResult:
    """Run the full ReAct loop for a single user question and return the result."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=get_tool_descriptions()
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    steps: list[AgentStep] = []

    for _ in range(settings.max_iterations):
        reply_text = _call_ollama(messages)
        messages.append({"role": "assistant", "content": reply_text})

        step = _parse_response(reply_text)

        if step.action == "final_answer":
            steps.append(step)
            return AgentResult(answer=step.action_input or "", steps=steps)

        if step.action and step.action_input is not None:
            observation = run_tool(step.action, step.action_input)
            step.observation = observation
            steps.append(step)
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        # The model didn't follow the expected format - nudge it and try again.
        step.observation = (
            "Observation: your last reply didn't match the required format. "
            "Reply with either 'Action' + 'Action Input', or 'Final Answer'."
        )
        steps.append(step)
        messages.append({"role": "user", "content": step.observation})

    return AgentResult(
        answer="I couldn't reach a final answer within the allowed number of steps.",
        steps=steps,
        hit_max_iterations=True,
    )

"""Tests for the ReAct parsing logic and the overall agent loop.

We never call the real Ollama server here - `_call_ollama` is monkeypatched
so these tests are fast and deterministic (no dependency on a local model
being installed or running).
"""

from app import agent


class TestParseResponse:
    def test_parses_action_and_input(self):
        text = (
            "Thought: I need to add two numbers.\n"
            "Action: calculator\n"
            "Action Input: 2 + 2"
        )
        step = agent._parse_response(text)
        assert step.action == "calculator"
        assert step.action_input == "2 + 2"

    def test_parses_final_answer(self):
        text = "Thought: I now know the answer.\nFinal Answer: The result is 4."
        step = agent._parse_response(text)
        assert step.action == "final_answer"
        assert step.action_input == "The result is 4."

    def test_missing_format_returns_no_action(self):
        step = agent._parse_response("I'm just rambling with no structure.")
        assert step.action is None


class TestRunAgent:
    def test_full_loop_with_one_tool_call(self, monkeypatch):
        # Simulate the model: first turn calls the calculator, second turn
        # gives the final answer using the observation it was given.
        responses = iter(
            [
                "Thought: I should calculate this.\nAction: calculator\nAction Input: 6 * 7",
                "Thought: Now I know the answer.\nFinal Answer: 42",
            ]
        )
        monkeypatch.setattr(agent, "_call_ollama", lambda messages: next(responses))

        result = agent.run_agent("What is 6 times 7?")

        assert result.answer == "42"
        assert result.hit_max_iterations is False
        assert len(result.steps) == 2
        assert result.steps[0].action == "calculator"
        assert result.steps[0].observation == "42"

    def test_gives_up_after_max_iterations(self, monkeypatch):
        # A model that never produces a valid Action or Final Answer should
        # not loop forever - it must stop at settings.max_iterations.
        monkeypatch.setattr(agent, "_call_ollama", lambda messages: "I refuse to follow the format.")

        result = agent.run_agent("Anything")

        assert result.hit_max_iterations is True
        assert len(result.steps) == agent.settings.max_iterations

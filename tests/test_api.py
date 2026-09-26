"""End-to-end tests for the FastAPI layer, using FastAPI's TestClient.

The agent's reasoning (`run_agent`) is monkeypatched so these tests check
the HTTP contract (status codes, response shape) without needing a running
Ollama server.
"""

from fastapi.testclient import TestClient

from app import main
from app.agent import AgentResult, AgentStep


def test_health_check():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_agent_returns_answer_and_trace(monkeypatch):
    fake_result = AgentResult(
        answer="4",
        steps=[
            AgentStep(thought="Add the numbers", action="calculator", action_input="2 + 2", observation="4")
        ],
        hit_max_iterations=False,
    )
    monkeypatch.setattr(main, "run_agent", lambda question: fake_result)

    client = TestClient(main.app)
    response = client.post("/agent", json={"question": "What is 2 + 2?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "4"
    assert body["steps"][0]["action"] == "calculator"
    assert body["hit_max_iterations"] is False


def test_ask_agent_rejects_empty_question():
    client = TestClient(main.app)
    response = client.post("/agent", json={"question": ""})
    assert response.status_code == 422


def test_ask_agent_returns_500_on_failure(monkeypatch):
    def boom(question):
        raise RuntimeError("Ollama is unreachable")

    monkeypatch.setattr(main, "run_agent", boom)

    client = TestClient(main.app)
    response = client.post("/agent", json={"question": "anything"})
    assert response.status_code == 500

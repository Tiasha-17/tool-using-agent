"""FastAPI service that exposes the tool-using agent over HTTP.

Run locally with:
    uvicorn app.main:app --reload

Then either open http://127.0.0.1:8000/docs for interactive docs, or:
    curl -X POST http://127.0.0.1:8000/agent \\
         -H "Content-Type: application/json" \\
         -d '{"question": "What is (45 * 12) + 7?"}'
"""

from fastapi import FastAPI, HTTPException

from app.agent import run_agent
from app.schemas import AgentRequest, AgentResponse, StepOut

app = FastAPI(
    title="Tool-Using Agent",
    description="A small ReAct-style agent that calls a calculator, web search, and a weather API.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """Liveness check - used by Docker/monitoring to confirm the service is up."""
    return {"status": "ok"}


@app.post("/agent", response_model=AgentResponse)
def ask_agent(request: AgentRequest) -> AgentResponse:
    """Run the agent on a single question and return its answer plus its reasoning trace."""
    try:
        result = run_agent(request.question)
    except Exception as exc:  # noqa: BLE001 - surface any failure as a clean 500, not a stack trace
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc

    return AgentResponse(
        answer=result.answer,
        steps=[
            StepOut(
                thought=step.thought,
                action=step.action,
                action_input=step.action_input,
                observation=step.observation,
            )
            for step in result.steps
        ],
        hit_max_iterations=result.hit_max_iterations,
    )

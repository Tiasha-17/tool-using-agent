"""Pydantic models describing the API's request/response bodies.

FastAPI uses these to validate incoming JSON and to auto-generate the
OpenAPI docs at /docs. We use typing.Optional/List (rather than the newer
`str | None` syntax) so this runs on Python 3.9 as well as newer versions.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the agent.")


class StepOut(BaseModel):
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class AgentResponse(BaseModel):
    answer: str
    steps: List[StepOut]
    hit_max_iterations: bool

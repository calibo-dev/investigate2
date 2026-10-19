from copy import deepcopy
from operator import add
from typing import Annotated, Any, Required, TypedDict


class InputState(TypedDict, total=False):
    """Public request fields accepted by the workflow."""

    user_input: Required[str]


class OutputState(TypedDict, total=False):
    """Public response fields returned by the workflow."""

    final_response: str
    errors: Annotated[list[str], add]


class State(InputState, OutputState, total=False):
    """Complete state shared by LangGraph nodes."""

    normalized_feedback: str
    validation_status: str
    sentiment: str
    confidence: str
    sentiment_reason: str
    clarification_question: str
    urgency_score: int
    policy_risk_score: int
    draft_response: str
    policy_guidance: str
    human_decision: str


INPUT_FIELDS = tuple(InputState.__annotations__)
REQUIRED_INPUT_FIELDS = tuple(
    field_name for field_name in INPUT_FIELDS if field_name in InputState.__required_keys__
)
OUTPUT_FIELDS = tuple(OutputState.__annotations__)

_STATE_DEFAULTS: dict[str, Any] = {
    "errors": [],
    "urgency_score": 0,
    "policy_risk_score": 0,
}


def state_defaults() -> dict[str, Any]:
    """Return an isolated copy of workflow state defaults for one invocation."""
    return deepcopy(_STATE_DEFAULTS)

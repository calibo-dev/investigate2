from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from langgraph.types import interrupt

from latest_ai_development.graph.state import State

RouteName = Literal["appreciation", "clarification", "service_recovery", "human_review"]
NodeCallable = Callable[[State], dict[str, Any]]
RouterCallable = Callable[[State], str]


def _parse_confidence(raw_confidence: Any) -> float:
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1:
        confidence /= 100
    return max(0.0, min(confidence, 1.0))


def _keyword_score(text: str, weighted_terms: Mapping[str, int]) -> int:
    normalized = text.lower()
    return sum(weight for term, weight in weighted_terms.items() if term in normalized)


def prepare_feedback(state: State, /) -> dict[str, Any]:
    user_input = state.get("user_input", "")
    if not isinstance(user_input, str) or not user_input.strip():
        return {"errors": ["user_input is required"], "validation_status": "invalid"}
    normalized = " ".join(user_input.split())
    if len(normalized) < 4:
        return {
            "errors": ["Please provide more detail about your experience."],
            "normalized_feedback": normalized,
            "validation_status": "invalid",
        }
    return {"normalized_feedback": normalized, "validation_status": "valid"}


def evaluate_feedback(state: State, /) -> dict[str, Any]:
    feedback = str(state.get("normalized_feedback", ""))
    sentiment = str(state.get("sentiment", "unknown")).lower()
    confidence = _parse_confidence(state.get("confidence"))
    urgency_terms = {
        "urgent": 35,
        "immediately": 30,
        "refund": 25,
        "cancel": 20,
        "broken": 20,
        "never arrived": 35,
    }
    policy_risk_terms = {
        "unsafe": 70,
        "injury": 80,
        "legal": 70,
        "fraud": 75,
        "chargeback": 50,
        "privacy": 60,
    }
    urgency_score = _keyword_score(feedback, urgency_terms)
    policy_risk_score = _keyword_score(feedback, policy_risk_terms)
    if sentiment == "negative":
        urgency_score += 15
    if confidence < 0.6:
        policy_risk_score += 20
    return {
        "urgency_score": min(urgency_score, 100),
        "policy_risk_score": min(policy_risk_score, 100),
    }


def request_clarification(state: State, /) -> dict[str, Any]:
    clarification = interrupt(
        {
            "kind": "clarification",
            "question": state.get(
                "clarification_question",
                "Could you provide more detail about your experience?",
            ),
            "feedback": state.get("normalized_feedback", ""),
        }
    )
    if isinstance(clarification, Mapping):
        response = clarification.get("response", clarification.get("value", ""))
    else:
        response = clarification
    response_text = str(response).strip()
    original_feedback = str(state.get("normalized_feedback", "")).strip()
    return {
        "normalized_feedback": (
            f"{original_feedback}\nCustomer clarification: {response_text}"
        ).strip(),
        "clarification_question": "",
    }


def human_review(state: State, /) -> dict[str, Any]:
    review = interrupt(
        {
            "kind": "approval",
            "question": "Review this high-risk customer feedback and provide a response.",
            "feedback": state.get("normalized_feedback", ""),
            "sentiment": state.get("sentiment", "unknown"),
            "urgency_score": state.get("urgency_score", 0),
            "policy_risk_score": state.get("policy_risk_score", 0),
        }
    )
    if isinstance(review, Mapping):
        decision = str(review.get("decision", "reviewed"))
        response = str(review.get("response", ""))
    else:
        decision = "reviewed"
        response = str(review)
    return {"human_decision": decision, "final_response": response}


def finalize_output(state: State, /) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    final_response = str(state.get("final_response", "")).strip()
    if errors and not final_response:
        final_response = f"Error: {'; '.join(errors)}"
    if not final_response:
        final_response = "We could not prepare a response. Please contact customer service."
    return {"final_response": final_response}


def route_after_evaluation(state: State, /) -> RouteName:
    confidence = _parse_confidence(state.get("confidence"))
    sentiment = str(state.get("sentiment", "unknown")).strip().lower()
    urgency_score = int(state.get("urgency_score", 0))
    policy_risk_score = int(state.get("policy_risk_score", 0))
    if policy_risk_score >= 70 or urgency_score >= 80:
        return "human_review"
    if confidence < 0.6 or sentiment not in {"positive", "negative"}:
        return "clarification"
    if sentiment == "negative" or urgency_score >= 35:
        return "service_recovery"
    return "appreciation"


NODE_REGISTRY: dict[str, NodeCallable] = {
    "prepare_feedback": prepare_feedback,
    "evaluate_feedback": evaluate_feedback,
    "request_clarification": request_clarification,
    "human_review": human_review,
    "finalize_output": finalize_output,
}

ROUTER_REGISTRY: dict[str, RouterCallable] = {
    "evaluate_feedback": route_after_evaluation,
}

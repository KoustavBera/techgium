"""
Chiranjeevi Medical Agent — Context Quality & Clarification
============================================================
Implements Trust Envelope™ for medical queries using:
- Context quality assessment
- Intelligent clarification generation
- Conversation state tracking
"""

import re
from langchain_core.messages import HumanMessage, AIMessage
from agent.state import AgentState
from agent.config import CONTEXT_ASSESSOR_PROMPT, CLARIFICATION_PROMPT


_llm = None


def set_llm(llm_instance):
    """Inject the LLM instance."""
    global _llm
    _llm = llm_instance


def _get_latest_query(state: AgentState) -> str:
    """Extract the latest user message."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def assess_context_quality(query: str) -> float:
    """
    Assess if query has sufficient context for medical advice.
    
    Returns:
        float: Quality score 0.0-1.0
    """
    score = 0.0
    q = query.lower()
    
    # Check for duration indicators (singular + plural + common time words)
    duration_keywords = [
        "day", "days", "week", "weeks", "month", "months",
        "hour", "hours", "minute", "minutes",
        "since", "ago", "started", "yesterday", "today",
        "morning", "evening", "night", "recently", "chronic",
        "last night", "few days", "long time",
    ]
    if any(kw in q for kw in duration_keywords):
        score += 0.25
    
    # Check for severity indicators
    severity_keywords = [
        "severe", "mild", "moderate", "intense", "slight",
        "bad", "terrible", "sharp", "dull", "throbbing",
        "unbearable", "persistent", "constant", "extreme",
        "awful", "horrible", "painful", "worse", "better",
        "a lot", "very", "really",
    ]
    if any(kw in q for kw in severity_keywords):
        score += 0.25
    
    # Check for associated symptoms (multiple symptoms mentioned)
    symptom_count = len(re.findall(
        r'\b(pain|ache|fever|nausea|dizzy|dizziness|tired|tiredness|fatigue|'
        r'swelling|rash|headache|vomit|vomiting|bloat|bloating|cramp|cough|'
        r'sore|burning|itching|numbness|weakness|chills|sweating|'
        r'constipation|diarrhea|bleeding|breathless|palpitation)\b',
        q
    ))
    if symptom_count >= 2:
        score += 0.25
    
    # Check for medical history / trigger mentions
    history_keywords = [
        "history", "medication", "taking", "diagnosed", "condition",
        "allergic", "allergy", "surgery", "pregnant", "diabetes",
        "asthma", "blood pressure", "cholesterol", "thyroid",
        "after eating", "after drinking", "after exercise",
        "triggered", "cause", "because", "milk", "food",
        "ate", "drank", "stress",
    ]
    if any(kw in q for kw in history_keywords):
        score += 0.25
    
    return min(score, 1.0)


def identify_missing_context(query: str, quality_score: float) -> str:
    """Identify what context is missing from the query."""
    missing = []
    q = query.lower()
    
    duration_keywords = [
        "day", "days", "week", "weeks", "month", "months",
        "hour", "hours", "since", "ago", "started",
        "yesterday", "today", "morning", "recently",
    ]
    if not any(kw in q for kw in duration_keywords):
        missing.append("symptom duration")
    
    severity_keywords = [
        "severe", "mild", "moderate", "intense", "sharp",
        "dull", "bad", "terrible", "worse", "better",
    ]
    if not any(kw in q for kw in severity_keywords):
        missing.append("severity level")
    
    if quality_score < 0.5:
        missing.append("associated symptoms or triggers")
    
    return ", ".join(missing) if missing else "general context"


def _has_prior_ai_responses(state: AgentState) -> bool:
    """Check if the conversation already has AI responses (prior interaction)."""
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage):
            return True
    return False


def clarification_node(state: AgentState) -> dict:
    """
    Assess context quality and generate clarification questions if needed.
    
    This implements the Trust Envelope™ boundary check.
    Skips clarification if the conversation already has AI responses
    (user is answering follow-up questions, not starting fresh).
    """
    from agent.nodes import _emit_status
    
    _emit_status("analyzing", "Assessing your question context")
    query = _get_latest_query(state)
    clarification_count = state.get("clarification_count", 0)
    
    # ─── Skip clarification if user is already in a conversation ───
    # If we've already exchanged messages (AI responses exist),
    # the user is likely answering our follow-up questions.
    # Proceed directly to research + answer.
    if _has_prior_ai_responses(state):
        _emit_status("thinking", "Understanding your follow-up, proceeding to analysis")
        return {
            "clarification_needed": False,
            "context_quality": 1.0
        }
    
    # Don't ask more than 1 round of clarification questions
    if clarification_count >= 1:
        _emit_status("thinking", "Sufficient context gathered, preparing response")
        return {
            "clarification_needed": False,
            "context_quality": 1.0  # Proceed with available context
        }
    
    # Assess context quality
    quality = assess_context_quality(query)
    
    # If quality is sufficient, proceed to answer
    if quality >= 0.5:
        _emit_status("thinking", "Good context, researching your question")
        return {
            "clarification_needed": False,
            "context_quality": quality
        }
    
    # Generate clarification questions
    _emit_status("thinking", "Preparing follow-up questions")
    missing_info = identify_missing_context(query, quality)
    
    prompt = CLARIFICATION_PROMPT.format(
        query=query,
        missing_info=missing_info
    )
    
    response = _llm.invoke([HumanMessage(content=prompt)])
    clarification_text = response.content if hasattr(response, "content") else str(response)
    
    _emit_status("streaming", "Responding")
    clarification_clean = clarification_text.strip()
    return {
        "clarification_needed": True,
        "clarification_count": clarification_count + 1,
        "context_quality": quality,
        "final_answer": clarification_clean,
        "messages": [AIMessage(content=clarification_clean)],
    }

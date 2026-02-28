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
from agent.config import CONTEXT_ASSESSOR_PROMPT, CLARIFICATION_PROMPT, BIOMARKER_KEYWORDS


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
    
    Implements the Trust Envelope™ boundary check with patient-aware shortcuts:

    Fast-skip conditions (no LLM call needed):
    1. Conversation already has AI responses (user is answering follow-up).
    2. Query overlaps with biomarker keywords AND patient_context is present
       (the report already contains the data the user is asking about).
    3. clarification_count >= 1 (don't ask twice).

    Targeted clarification (when LOW quality + HIGH risk flags in report):
    - Instead of generic 'tell me more', asks about the specific flagged finding.
    """
    from agent.nodes import _emit_status

    _emit_status("analyzing", "Assessing your question context")
    query = _get_latest_query(state)
    patient_ctx = state.get("patient_context", "")
    clarification_count = state.get("clarification_count", 0)

    # ─── Skip 1: Already in a conversation ────────────────────────────
    if _has_prior_ai_responses(state):
        _emit_status("thinking", "Understanding your follow-up, proceeding to analysis")
        return {"clarification_needed": False, "context_quality": 1.0}

    # ─── Skip 2: Max clarification rounds reached ──────────────────────
    if clarification_count >= 1:
        _emit_status("thinking", "Sufficient context gathered, preparing response")
        return {"clarification_needed": False, "context_quality": 1.0}

    # ─── Skip 3: Query overlaps with screened biomarkers ──────────────
    # e.g. user asks "what's my heart rate?" — we already have it in the report.
    # Processing this through clarification would waste an LLM call.
    if patient_ctx:
        q_lower = query.lower()
        if any(kw in q_lower for kw in BIOMARKER_KEYWORDS):
            _emit_status("thinking", "Found relevant data in your screening report")
            print("  ⚡ Clarification skipped: query matches screened biomarkers")
            return {"clarification_needed": False, "context_quality": 1.0}

    # ─── Assess context quality ────────────────────────────────────────
    quality = assess_context_quality(query)

    if quality >= 0.5:
        _emit_status("thinking", "Good context, researching your question")
        return {"clarification_needed": False, "context_quality": quality}

    # ─── Targeted clarification if high-risk flags in report ───────────
    # Instead of generic 'tell me more', ask about the specific finding.
    _emit_status("thinking", "Preparing follow-up questions")

    high_risk_in_report = patient_ctx and any(
        marker in patient_ctx.lower() for marker in {"high", "action required", "⚠️"}
    )

    if high_risk_in_report:
        # Extract the first flagged system name from the report for a targeted question
        flagged_system = "one of your readings"
        for line in patient_ctx.splitlines():
            if any(marker in line.lower() for marker in {"**high**", "**action required**", "⚠️"}):
                # e.g. "#### Cardiovascular: **HIGH** ..."
                match = re.search(r'####\s*(\w+)', line)
                if match:
                    flagged_system = match.group(1).lower()
                    break

        targeted_question = (
            f"I can see from your screening that {flagged_system} needs attention. "
            f"Are you experiencing any related symptoms — for example, chest discomfort, "
            f"shortness of breath, dizziness, or fatigue? "
            f"This will help me give you more specific guidance. 💙"
        )
        print(f"  🎯 Targeted clarification about: {flagged_system}")
        return {
            "clarification_needed": True,
            "clarification_count": clarification_count + 1,
            "context_quality": quality,
            "final_answer": targeted_question,
            "messages": [AIMessage(content=targeted_question)],
        }

    # ─── Generic clarification (no screening data context) ────────────
    missing_info = identify_missing_context(query, quality)
    prompt = CLARIFICATION_PROMPT.format(query=query, missing_info=missing_info)
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

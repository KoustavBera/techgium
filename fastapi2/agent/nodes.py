"""
Chiranjeevi Medical Agent — Graph Nodes
========================================
Three node functions for the LangGraph state machine:
  1. router_node  — classifies the query (MEDICAL / GREETING / GENERAL)
  2. research_node — fetches evidence from Tavily + PubMed in parallel
  3. answer_node  — synthesises the final doctor response via Llama
"""

from __future__ import annotations
from typing import Callable, Optional

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import (
    ROUTER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
)
from agent.state import AgentState
from agent.tools import search_tavily, search_pubmed


# ── Shared model reference (set at graph-build time) ──────────────────
_llm = None
_token_callback_var: ContextVar[Optional[Callable[[str], None]]] = ContextVar("_token_callback", default=None)
_status_callback_var: ContextVar[Optional[Callable[[dict], None]]] = ContextVar("_status_callback", default=None)


def set_llm(llm_instance):
    """Inject the Llama model instance so nodes can share it."""
    global _llm
    _llm = llm_instance


def set_token_callback(callback):
    """Set a function to be called for every generated token (for streaming)."""
    _token_callback_var.set(callback)


def set_status_callback(callback):
    """Set a function to be called for agent status updates."""
    _status_callback_var.set(callback)


def _emit_status(stage: str, message: str):
    """Emit a status event if a callback is registered."""
    cb = _status_callback_var.get()
    if cb:
        cb({"stage": stage, "message": message})


def _get_latest_query(state: AgentState) -> str:
    """Extract the text of the latest human message from state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
        # Handle plain dicts (shouldn't happen, but defensive)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ═══════════════════════════════════════════════════════════════════════
# Node 1: Router
# ═══════════════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> dict:
    """Classify the user query into MEDICAL, GREETING, or GENERAL.

    Uses the LLM via LangChain interface.
    Falls back to 'medical' if parsing fails (safe default).
    """
    _emit_status("analyzing", "Analyzing your question")
    query = _get_latest_query(state)

    prompt = ROUTER_PROMPT_TEMPLATE.format(query=query)

    # Use messages for ChatHuggingFace
    messages = [HumanMessage(content=prompt)]
    response = _llm.invoke(messages)
    raw = (response.content if hasattr(response, "content") else str(response)).strip().upper()

    # Extract the category keyword
    if "GREETING" in raw:
        category = "greeting"
    elif "GENERAL" in raw:
        category = "general"
    else:
        category = "medical"     # safe default

    print(f"  🔀 Router Raw Output: {raw}")
    print(f"  🔀 Router Classified: {category.upper()}")
    
    from agent.config import TAVILY_API_KEY
    print(f"  🔑 Tavily Key Present: {bool(TAVILY_API_KEY)}")
    if not TAVILY_API_KEY:
        print("  ⚠️  WARNING: TAVILY_API_KEY is missing/empty!")

    return {"query_type": category}


# ═══════════════════════════════════════════════════════════════════════
# Node 2: Researcher
# ═══════════════════════════════════════════════════════════════════════

def research_node(state: AgentState) -> dict:
    """Fetch evidence from Tavily (web) and PubMed (papers) in parallel.

    Aggregates results into a single research_data string.
    """
    _emit_status("searching_web", "Searching the web")
    query = _get_latest_query(state)
    print(f"  🔍 Researching: {query[:60]}...")

    results: dict[str, str] = {}
    all_sources: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Submit tasks
        future_tavily = pool.submit(search_tavily, query)
        future_pubmed = pool.submit(search_pubmed, query)
        
        futures = {
            future_tavily: "tavily",
            future_pubmed: "pubmed",
        }
        
        # Emit PubMed status when we start searching
        _emit_status("searching_pubmed", "Searching medical literature")
        
        for future in as_completed(futures):
            source_name = futures[future]
            try:
                # Unpack (text, sources_list)
                text, source_metadata = future.result()
                results[source_name] = text
                all_sources.extend(source_metadata)
            except Exception as e:
                results[source_name] = f"[{source_name}] Error: {e}"

    # Build the aggregated research summary
    parts = []
    if results.get("tavily"):
        parts.append("### 🌐 Web Search Results (Tavily)\n" + results["tavily"])
    if results.get("pubmed"):
        parts.append("### 📄 PubMed Clinical Papers\n" + results["pubmed"])

    research_data = "\n\n".join(parts) if parts else "No research data found."

    # Emit structured sources for the frontend
    if all_sources:
        # We pass the list directly; the callback wrapper in main.py will handle it
        # or we might need to serialize it if `_emit_status` expects strict string.
        # Let's import json and serialize it to be safe, as _emit_status signature says msg: str
        import json
        _emit_status("citations", json.dumps(all_sources))

    # Debug: show what research returned
    for source, data in results.items():
        preview = data[:120].replace("\n", " ")
        print(f"    📋 {source}: {preview}...")

    print(f"  ✅ Research complete ({len(results)} sources)")
    return {"research_data": research_data}


# ═══════════════════════════════════════════════════════════════════════
# Node 3: Answer (The Doctor)
# ═══════════════════════════════════════════════════════════════════════

def answer_node(state: AgentState) -> dict:
    """Generate the final compassionate, evidence-based doctor response.

    If research_data is available, it is injected into the prompt so the
    model can cite sources.  Otherwise, the model answers from its own
    fine-tuned medical knowledge.
    
    Uses the full conversation history from state["messages"] to maintain
    conversational memory across turns.
    """
    _emit_status("thinking", "Chiranjeevi is thinking")
    research = state.get("research_data", "")

    # Build the system message with research context
    system_text = SYSTEM_PROMPT
    if research and research != "No research data found.":
        system_text += (
            "\n\nIMPORTANT: You have been given research evidence below. "
            "You MUST reference and cite this evidence in your response. "
            "Mention specific findings, paper titles, or sources.\n\n"
            "--- Research Evidence ---\n"
            f"{research}\n"
            "--- End Research Evidence ---"
        )

    # Use the FULL conversation history from state, prepending the system message
    # This ensures the LLM sees all previous turns and can maintain context
    messages = [SystemMessage(content=system_text)] + state["messages"]

    print("  🩺 Chiranjeevi is thinking...")
    _emit_status("streaming", "Responding")

    cb = _token_callback_var.get()
    if cb:
        # Streaming mode via LangChain (returns BaseMessageChunk for ChatModels)
        tokens = []
        for chunk in _llm.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            tokens.append(content)
            cb(content)
        answer = "".join(tokens).strip()
    else:
        # Block mode via LangChain
        response = _llm.invoke(messages)
        answer = (response.content if hasattr(response, "content") else str(response)).strip()
    
    print("  ✅ Response generated")

    return {"final_answer": answer}

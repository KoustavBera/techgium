"""
Chiranjeevi Medical Agent — Graph State
========================================
Defines the TypedDict that flows through every LangGraph node.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between all graph nodes.

    Attributes
    ----------
    messages : list[BaseMessage]
        Full conversation history (managed by LangGraph's ``add_messages`` reducer).
    query_type : str
        Classification result from the router node.
        One of ``"medical"``, ``"greeting"``, or ``"general"``.
    research_data : str
        Aggregated research from Tavily web search and PubMed papers.
        Empty string when no research was performed.
    final_answer : str
        The synthesised doctor response returned to the user.
    clarification_needed : bool
        Whether the agent needs more context before answering.
    clarification_count : int
        Number of clarification questions asked (max 2).
    context_quality : float
        Quality score of gathered context (0.0-1.0).
    """

    messages: Annotated[List[BaseMessage], add_messages]
    query_type: str
    smart_search_query: str
    research_data: str
    final_answer: str
    clarification_needed: bool
    clarification_count: int
    context_quality: float

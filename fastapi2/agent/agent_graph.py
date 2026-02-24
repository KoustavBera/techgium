"""
Chiranjeevi Medical Agent — LangGraph Entry Point
===================================================
Assembles the Router → Researcher → Doctor graph and provides an
interactive CLI for chatting with the agent.

Usage
-----
    cd fastapi2
    python -m agent.agent_graph

The model is loaded once at startup and shared across all nodes.
"""

from __future__ import annotations

import sys
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from agent.config import TEMPERATURE
from agent.state import AgentState
from agent.nodes import router_node, research_node, answer_node, set_llm
from agent.clarification import clarification_node, set_llm as set_clarification_llm
from langchain_core.messages import HumanMessage


# ═══════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════

def route_query(state: AgentState) -> str:
    """Conditional edge: decide whether to research or answer directly."""
    qt = state.get("query_type", "medical")
    if qt == "medical":
        return "clarification_node"
    return "answer_node"


def route_after_clarification(state: AgentState) -> str:
    """Route after clarification: ask more questions or proceed to research."""
    if state.get("clarification_needed", False):
        return "end"  # Return clarification questions to user
    return "research_evaluator_node"

def route_after_evaluator(state: AgentState) -> str:
    """Route after evaluator: do we need research or straight to answer?"""
    if state.get("research_data") == "NO_RESEARCH_NEEDED":
        return "answer_node"
    return "research_node"


def build_graph() -> StateGraph:
    """Construct and compile the Chiranjeevi agent graph.

    Graph topology::

        START → router_node
                  ├──(medical)──→ clarification_node
                  │                 ├──(needs_clarification)──→ END (ask questions)
                  │                 └──(sufficient)───────────→ research_evaluator_node
                  │                                                 ├──(NO)──→ answer_node → END
                  │                                                 └──(YES)─→ research_node → answer_node → END
                  └──(other)────→ answer_node → END
    """
    from agent.nodes import research_evaluator_node
    
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router_node", router_node)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("research_evaluator_node", research_evaluator_node)
    graph.add_node("research_node", research_node)
    graph.add_node("answer_node", answer_node)

    # Edges
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges(
        "router_node",
        route_query,
        {
            "clarification_node": "clarification_node",
            "answer_node": "answer_node",
        },
    )
    graph.add_conditional_edges(
        "clarification_node",
        route_after_clarification,
        {
            "end": END,
            "research_evaluator_node": "research_evaluator_node",
        },
    )
    graph.add_conditional_edges(
        "research_evaluator_node",
        route_after_evaluator,
        {
            "answer_node": "answer_node",
            "research_node": "research_node",
        },
    )
    graph.add_edge("research_node", "answer_node")
    graph.add_edge("answer_node", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    """Load the Chiranjeevi model via Hugging Face Inference API."""
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file. Please add your Hugging Face token.")
    
    print(f"🧠 Loading Chiranjeevi model via Hugging Face API...")
    print(f"   Model: Qwen/Qwen2.5-72B-Instruct")
    print(f"   Temperature: {TEMPERATURE}")
    print()
    
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        huggingfacehub_api_token=hf_token,
        task="conversational", 
        temperature=TEMPERATURE,
        max_new_tokens=512,
        top_p=0.9,
        streaming=True,
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    
    print("✅ Model endpoint connected successfully!\n")
    
    # Set LLM for all modules
    set_llm(chat_model)
    set_clarification_llm(chat_model)
    
    # Enable Semantic Caching
    import langchain
    from langchain_community.cache import SQLiteCache
    try:
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
        print("✅ SQLite Cache enabled!\n")
    except Exception as e:
        print(f"⚠️ Could not enable SQLite Cache: {e}\n")
    
    return chat_model


# ═══════════════════════════════════════════════════════════════════════
# Interactive CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Run the Chiranjeevi agent in an interactive terminal loop."""
    print("=" * 60)
    print("  🩺 CHIRANJEEVI — Medical AI Agent")
    print("  Powered by LangGraph + Qwen-2.5-72B-Instruct (HF API)")
    print("=" * 60)
    print()

    # Load model once
    llm = load_model()
    set_llm(llm)

    # Build graph
    app = build_graph()

    print("Type your medical question below (or 'quit' to exit).\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye! Stay healthy.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye! Stay healthy.")
            break

        # Invoke the graph
        print()
        result = app.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "query_type": "",
                "smart_search_query": "",
                "research_data": "",
                "final_answer": "",
                "clarification_needed": False,
                "clarification_count": 0,
                "context_quality": 0.0,
            },
            config={"configurable": {"thread_id": "cli-session"}},
        )

        # Print the doctor's response
        answer = result.get("final_answer", "I apologize, I could not generate a response.")
        print(f"\n🩺 Dr. Chiranjeevi:\n{'─' * 40}")
        print(answer)
        print(f"{'─' * 40}")


if __name__ == "__main__":
    main()

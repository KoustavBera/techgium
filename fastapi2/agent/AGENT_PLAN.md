# Medical Agent Implementation Plan

## Goal
Build a local medical AI agent using your fine-tuned **Chiranjeevi (Llama-3-8B)** model. The agent will interpret user queries, decide if it needs external research (using Tavily/PubMed), and provide a comprehensive medical response using **LangGraph**.

## User Review Required
> [!IMPORTANT]
> **API Keys Needed**: You will need a **Tavily API Key** for the search functionality. You can get a free one at [tavily.com](https://tavily.com/).

## Proposed Architecture
We will use **LangGraph** to create a state machine with the following nodes:

1.  **Triage Node**: Analyzes the user's query to determine if it's a simple medical question (answer directly) or requires latest research (search online).
2.  **Research Node**: Queries Tavily/PubMed for the latest medical papers or articles if the query is complex or requires up-to-date info.
3.  **Answer Node**: Uses the **Chiranjeevi** model to synthesize the answer (and research if available) into a compassionate, doctor-like response.

## Proposed Changes

### [New] `agent_graph.py`
This will be the main file containing the LangGraph logic.

-   **State Definition**: Defines the `AgentState` (messages, next_step, research_data).
-   **Model Initialization**: Connects to `LlamaCpp` using the GGUF file path directly.
-   **Nodes**:
    -   `triage_step`: Classifies intent.
    -   `research_step`: Calls Tavily API.
    -   `generate_step`: Generates final response.
-   **Graph Construction**: Defines the edges (Triage -> Research or Triage -> Generate).

### [New] `requirements.txt`
Dependencies needed:
-   `langchain`
-   `langchain-community`
-   `langgraph`
-   `tavily-python`
-   `pydantic`
-   `llama-cpp-python`

## Verification Plan
1.  **Test Triage**: Ask a simple question ("I have a headache") -> Should go straight to Answer.
2.  **Test Research**: Ask a complex question ("Latest treatments for Long COVID") -> Should go to Research -> Answer.
3.  **Verify Persona**: Ensure the response uses the "Chiranjeevi" persona (empathetic, professional).

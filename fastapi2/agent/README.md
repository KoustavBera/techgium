# Chiranjeevi Agent Workflow (`fastapi2/agent`)

This folder contains the LangGraph-based medical conversation agent used by the FastAPI backend (`/api/v1/doctor/chat`) and the local CLI entrypoint.

## Intent of this module

The agent is designed to:
- classify incoming user queries quickly,
- ask clarification questions only when needed,
- decide whether external research is necessary,
- gather evidence from Tavily + PubMed when required,
- produce a compassionate, patient-aware medical response.

It also supports:
- patient screening context injection (from kiosk reports),
- streaming token output for frontend SSE,
- status callbacks for UI progress updates.

---

## High-level workflow

Graph entrypoint: `agent_graph.py::build_graph()`

```text
START
  → router_node
      ├─ medical          → clarification_node
      │                      ├─ clarification_needed = true  → END (returns follow-up questions)
      │                      └─ clarification_needed = false → research_evaluator_node
      │                                                        ├─ NO_RESEARCH_NEEDED → answer_node → END
      │                                                        └─ needs_research     → research_node → answer_node → END
      ├─ patient_briefing → answer_node → END
      ├─ greeting         → answer_node → END
      └─ general          → answer_node → END
```

---

## Node responsibilities

### 1) `router_node` (`nodes.py`)
- Fast heuristic classification into `medical`, `patient_briefing`, `greeting`, or `general`.
- Uses greeting regex + patient high-risk flags for the proactive `patient_briefing` fast-path.

### 2) `clarification_node` (`clarification.py`)
- Computes context quality score for medical queries.
- Skips clarification for follow-ups, known biomarker questions, or prior clarification rounds.
- Produces targeted follow-up questions when context is insufficient.

### 3) `research_evaluator_node` (`nodes.py`)
- LLM gate to decide if external literature search is needed.
- If yes, rewrites the user query into an optimized search query.

### 4) `research_node` (`nodes.py`)
- Runs Tavily web search + PubMed retrieval in parallel.
- Aggregates results into `research_data`.
- Emits structured citation metadata via status callback.

### 5) `answer_node` (`nodes.py`)
- Builds final response from system prompt + conversation + optional patient data + optional research evidence.
- Handles specialized `patient_briefing` behavior.
- Streams tokens when token callback is active.

---

## Shared state contract

`state.py` defines `AgentState` (TypedDict), including:
- `messages`
- `query_type`
- `smart_search_query`
- `research_data`
- `final_answer`
- `clarification_needed`
- `clarification_count`
- `context_quality`
- `patient_context`

---

## Supporting files

- `config.py`: prompts, environment loading, model/runtime parameters, biomarker keyword list.
- `tools.py`: plain Python integrations for Tavily and PubMed.
- `agent_graph.py`: graph assembly, conditional routing, model loading, CLI loop.
- `clarification.py`: Trust Envelope context assessment and clarification logic.
- `requirements.txt`: agent-specific dependencies.
- `Modelfile`: local model metadata (legacy/local serving context).
- `test_model.py`: local one-off model loading script.

---

## Runtime integration paths

### FastAPI path (production app)
- App startup loads model once (`app/main.py` lifespan).
- Graph is built once and stored on `app.state.medical_agent`.
- `/api/v1/doctor/chat` invokes the graph with:
  - user message,
  - strict patient-matched `patient_context`,
  - streaming status/token callbacks.

### CLI path (local debugging)
From `fastapi2/`:
```bash
python -m agent.agent_graph
```

---

## Required environment variables

At minimum:
- `HF_TOKEN` (required for Hugging Face hosted model endpoint in `load_model()`)

Optional:
- `TAVILY_API_KEY` (enables web search in `search_tavily`)
- `NCBI_API_KEY` (optional PubMed rate-limit support)

`.env` is loaded from `fastapi2/.env`.

# Multi-LLM Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HEALTH SCREENING PIPELINE                           │
│                   (FastAPI Application - app/core)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                             │
│  • Camera Feed (MediaPipe)    • Motion Sensors                          │
│  • RIS Signals                • Auxiliary Sensors                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     BIOMARKER EXTRACTION LAYER                          │
│  • Cardiovascular  • CNS  • Renal  • Eyes  • Skin  • etc.               │
│  → Extracts 100+ biomarkers from sensor data                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RISK COMPUTATION LAYER                             │
│  • Risk Engine (rules-based)                                            │
│  • ML Anomaly Detection                                                 │
│  → Computes risk scores per body system (NON-LLM)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER (LLMs)                              │
│                                                                         │
│  ┌──────────────────┐              ┌──────────────────┐                │
│  │  MedGemmaAgent   │              │ OpenBioLLMAgent  │                │
│  │                  │              │                  │                │
│  │  Uses:           │              │  Uses:           │                │
│  │  GPT-OSS-120B    │◄────────────►│  II-Medical-8B   │                │
│  │  (HF Model 1)    │   Consensus  │  (HF Model 2)    │                │
│  └──────────────────┘              └──────────────────┘                │
│                                                                         │
│  • Biomarker plausibility validation                                   │
│  • Cross-system consistency checking                                   │
│  • Generates validation flags and recommendations                      │
│  → Non-decisional: Flags issues, does NOT diagnose                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               TRUST ENVELOPE COMPUTATION                                │
│  • Signal Quality Assessment                                            │
│  • Biomarker Plausibility Check (from agents)                           │
│  • Cross-System Consistency (from agents)                               │
│  → Overall reliability score + confidence penalty                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  REPORT GENERATION LAYER                                │
│                   (Multi-LLM Interpreter)                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │           MULTI-LLM INTERPRETER (NEW!)                      │       │
│  │                                                              │       │
│  │  Query all 3 LLMs in parallel:                              │       │
│  │                                                              │       │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │       │
│  │  │   Gemini     │  │  GPT-OSS-    │  │ II-Medical-  │      │       │
│  │  │  2.5 Flash   │  │    120B      │  │     8B       │      │       │
│  │  │              │  │              │  │              │      │       │
│  │  │ (LangChain)  │  │(HF Model 1)  │  │(HF Model 2)  │      │       │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │       │
│  │         │                 │                 │              │       │
│  │         └─────────────────┴─────────────────┘              │       │
│  │                           │                                │       │
│  │                    Synthesize                              │       │
│  │                           │                                │       │
│  │                           ▼                                │       │
│  │            ┌──────────────────────────┐                    │       │
│  │            │  Unified Interpretation  │                    │       │
│  │            │  • Summary               │                    │       │
│  │            │  • Recommendations       │                    │       │
│  │            │  • Caveats               │                    │       │
│  │            │  • Consensus Level       │                    │       │
│  │            └──────────────────────────┘                    │       │
│  │                                                              │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  → Non-decisional: Explains pre-computed risks only                    │
│  → Does NOT diagnose, prescribe, or modify risk scores                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌────────────────────┐       ┌────────────────────┐
          │   Patient Report   │       │   Doctor Report    │
          │   (Simple PDF)     │       │ (Detailed PDF)     │
          │                    │       │                    │
          │ • Risk indicators  │       │ • Full biomarkers  │
          │ • Simple language  │       │ • Medical terms    │
          │ • Recommendations  │       │ • Validation data  │
          │ • Caveats          │       │ • Trust envelope   │
          │                    │       │ • Agent consensus  │
          └────────────────────┘       └────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         LLM RESPONSIBILITIES                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WHAT LLMs DO (Non-Decisional):                                        │
│  ✓ Validate biomarker plausibility (agents)                            │
│  ✓ Check cross-system consistency (agents)                             │
│  ✓ Explain pre-computed risk scores (reports)                          │
│  ✓ Generate recommendations (general health guidance)                  │
│  ✓ Provide educational context                                         │
│  ✓ Flag potential issues for human review                              │
│                                                                         │
│  WHAT LLMs DO NOT DO:                                                  │
│  ✗ Compute or assign risk scores                                       │
│  ✗ Make medical diagnoses                                              │
│  ✗ Prescribe treatments or medications                                 │
│  ✗ See raw sensor data                                                 │
│  ✗ Replace clinical judgment                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL CONFIGURATION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Gemini 2.5 Flash (via LangChain):                                     │
│  • Model: gemini-2.5-flash                                              │
│  • Temperature: 1.0 (Gemini 2.5+ default)                               │
│  • Use: Primary interpretation, patient-friendly explanations           │
│  • API: ChatGoogleGenerativeAI from langchain-google-genai              │
│                                                                         │
│  GPT-OSS-120B (HuggingFace Model 1):                                   │
│  • Model: openai/gpt-oss-120b                                           │
│  • Temperature: 0.5 (medical consistency)                               │
│  • Use: Biomarker validation, medical perspective                       │
│  • API: InferenceClient.chat.completions.create()                      │
│                                                                         │
│  II-Medical-8B (HuggingFace Model 2):                                  │
│  • Model: Intelligent-Internet/II-Medical-8B                            │
│  • Temperature: 0.5 (medical consistency)                               │
│  • Use: Cross-system consistency, secondary validation                 │
│  • API: InferenceClient.chat.completions.create()                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA FLOW EXAMPLE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Sensors → Biomarkers                                                │
│     Heart rate: 85 BPM, SpO2: 98%, Blood pressure: 120/80              │
│                                                                         │
│  2. Risk Engine → Risk Scores                                           │
│     Cardiovascular: 45/100 (Low), Respiratory: 30/100 (Low)            │
│                                                                         │
│  3. Validation Agents → Flags                                           │
│     GPT-OSS-120B: "Biomarkers plausible, no anomalies"                 │
│     II-Medical-8B: "Cross-system consistency OK"                       │
│                                                                         │
│  4. Trust Envelope → Reliability                                        │
│     Overall: 92% reliable, confidence penalty: 0%                       │
│                                                                         │
│  5. Multi-LLM Interpreter → Explanation                                 │
│     Gemini: "Your cardiovascular health appears normal..."             │
│     GPT-OSS-120B: "Heart rate of 85 BPM is within healthy range..."    │
│     II-Medical-8B: "Results consistent across body systems..."         │
│                                                                         │
│  6. Synthesized Report                                                  │
│     Summary: "Overall health screening shows low risk. Heart rate      │
│              and oxygen levels are healthy. No concerning patterns."    │
│     Recommendations: [Maintain activity, Regular checkups, ...]         │
│     Consensus: High (all 3 models agree)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Benefits

### 1. Multi-Model Validation

- **Before**: Single model per task (MedGemma or OpenBioLLM)
- **After**: Two medical models validate in parallel
- **Benefit**: Cross-validation reduces false positives/negatives

### 2. Comprehensive Reporting

- **Before**: Single Gemini model for interpretations
- **After**: 3 models (Gemini + 2 medical models) for reports
- **Benefit**: Richer, more balanced interpretations

### 3. Medical Specialization

- **Before**: General-purpose models
- **After**: Specialized medical models (GPT-OSS-120B, II-Medical-8B)
- **Benefit**: More relevant medical insights

### 4. Standardized Integration

- **Before**: Custom API calls for each provider
- **After**: LangChain + InferenceClient (official SDKs)
- **Benefit**: Easier maintenance, better error handling

### 5. Consensus Mechanism

- **Before**: Single interpretation
- **After**: Consensus level from multiple models
- **Benefit**: Confidence in multi-model agreement

## Performance Considerations

- **Latency**: 3 LLMs queried in sequence (~2-5 seconds total)
- **Cost**: 3x API calls for reports (1x Gemini + 2x HuggingFace)
- **Reliability**: Graceful degradation if 1-2 models fail
- **Caching**: Recommended for common interpretation patterns

## Safety & Compliance

All LLMs remain **NON-DECISIONAL**:

- ✅ Explain pre-computed risks
- ✅ Flag potential issues
- ✅ Provide educational context
- ❌ No diagnosis
- ❌ No treatment recommendations
- ❌ No risk score modification

This maintains regulatory compliance while leveraging AI for enhanced interpretation.

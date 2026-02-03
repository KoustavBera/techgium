# Running the Multi-LLM Demo

## Quick Start

### 1. Ensure Environment is Ready

```powershell
# Activate virtual environment
benv\Scripts\activate

# Verify dependencies installed
pip list | findstr /I "langchain huggingface"
```

Expected output should show:

- `langchain`
- `langchain-google-genai`
- `huggingface-hub`

### 2. Configure API Keys (Optional but Recommended)

Create/update `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token
```

**Note:** Demo will work in mock mode without API keys, but real AI validation won't run.

### 3. Run the Demo

```powershell
python demo.py
```

## What the Demo Does

The updated demo showcases:

### 🔬 Multi-Model AI Analysis

1. **Gemini 2.5 Flash** (via LangChain)
   - Primary interpretation engine
   - Patient-friendly explanations
   - Medical context generation

2. **GPT-OSS-120B** (HuggingFace)
   - Medical biomarker validation
   - Plausibility checking
   - First medical model perspective

3. **II-Medical-8B** (HuggingFace)
   - Cross-system consistency
   - Physiological coherence
   - Second medical model perspective

### 📊 Demo Workflow

```
[1/8] Loading modules
      ✓ LangChain integration
      ✓ HuggingFace InferenceClient

[2/8] Generating mock sensor data
      ✓ PPG (heart rate)
      ✓ Audio (breathing)
      ✓ Accelerometer (gait)

[3/8] Extracting biomarkers
      ⚠️  Unhealthy patient simulation
      - High blood pressure (158/98)
      - Tachycardia (105 bpm)
      - Low SpO2 (91%)

[4/8] Calculating risk scores
      ✓ Per-system risks
      ✓ Composite risk

[5/8] Trust envelope
      ✓ Data quality assessment
      ✓ Confidence penalties

[6/8] Multi-LLM interpretation
      🤖 Gemini 2.5 Flash
      🤖 GPT-OSS-120B
      🤖 II-Medical-8B
      ✓ Synthesized interpretation

[7/8] Agentic validation
      🔬 Agent 1: GPT-OSS-120B
      🔬 Agent 2: II-Medical-8B
      ✓ Consensus validation

[8/8] Report generation
      📄 Patient report (PDF)
      📄 Doctor report (PDF)
```

## Expected Output

### Console Output

```
======================================================================
HEALTH SCREENING PIPELINE - END-TO-END DEMO
======================================================================

[1/8] Loading modules...
   ✓ All modules loaded successfully!
   ✓ Using LangChain for Gemini
   ✓ Using HuggingFace InferenceClient for medical models

[2/8] Generating mock sensor data...
   ✓ PPG signal: 300 samples (camera-based)
   ✓ Audio signal: 160000 samples (breathing)
   ✓ Accelerometer: (500, 3) (gait analysis)

[3/8] Extracting biomarkers from signals...
   ⚠️  Cardiovascular: HR=105 bpm (HIGH), HRV=22.0 ms (LOW)
   ⚠️  CNS: Gait variability=0.25 (HIGH), Balance=0.45 (LOW)
   ⚠️  Pulmonary: RR=26/min (HIGH), SpO2=91% (LOW)

[4/8] Calculating risk scores...
   ✓ Cardiovascular: high (75.0%)
   ✓ CNS: moderate (55.0%)
   ✓ Pulmonary: high (70.0%)
   ✓ COMPOSITE: high (66.7%)

[5/8] Calculating trust envelope...
   ✓ Overall reliability: 0.78
   ✓ Data quality: 0.85
   ✓ Is reliable: True

[6/8] Generating Multi-LLM interpretation (3 models)...
   🤖 Querying: Gemini 2.5 Flash (LangChain)
   🤖 Querying: GPT-OSS-120B (HuggingFace)
   🤖 Querying: II-Medical-8B (HuggingFace)
   ✓ Interpretation complete!
   ✓ Summary length: 450 chars
   ✓ Recommendations: 5
   ✓ Latency: 3500ms
   📋 First recommendation: Consult with a healthcare professional for comprehensive evaluation...

[7/8] Running agent validation with medical models...
   🔬 Agent 1: Using GPT-OSS-120B for biomarker plausibility
   🔬 Agent 2: Using II-Medical-8B for cross-system consistency
   ⏳ Validating biomarkers...
   ⏳ Validating cross-system consistency...
   ✓ Agent 1 (openai/gpt-oss-120b): flagged
   ✓ Agent 2 (Intelligent-Internet/II-Medical-8B): flagged
   ✓ Validation status: flagged
   ✓ Agent agreement: 75%
   ✓ Combined flags: 4
   ✓ Requires review: True

[8/8] Generating PDF reports with Multi-LLM insights...
   ✓ Patient report: PR-20260202-143022
     📄 PDF: reports/PR-20260202-143022.pdf
     📊 Includes insights from 3 AI models
   ✓ Doctor report: DR-20260202-143022
     📄 PDF: reports/DR-20260202-143022.pdf
     🔬 Includes validation from medical AI models

======================================================================
DEMO COMPLETE - MULTI-LLM HEALTH SCREENING SUMMARY
======================================================================

Patient ID: DEMO-001
Timestamp: 2026-02-02T14:30:22.123456

AI Models Used:
  🤖 Gemini 2.5 Flash (via LangChain) - Interpretation
  🤖 GPT-OSS-120B (HuggingFace) - Validation & Interpretation
  🤖 II-Medical-8B (HuggingFace) - Validation & Interpretation

Systems Analyzed:
  🔴 cardiovascular: high (75.0%)
  🟡 cns: moderate (55.0%)
  🔴 pulmonary: high (70.0%)

Overall Risk: HIGH (66.7%)
Confidence: 85%
Reliability Score: 0.78

⚠️  REQUIRES HUMAN REVIEW
   Reason: Multiple high-risk indicators with critical flags detected

Multi-LLM Interpretation:
  📝 Summary: The health screening indicates elevated cardiovascular and pulmonary concerns...
  💡 Recommendations: 5 generated
  ⏱️  Total latency: 3500ms

Reports generated in: ./reports/
  - Patient report (simple, AI-enhanced)
  - Doctor report (detailed, with validation)

To run the API server:
  uvicorn app.main:app --reload --port 8000

======================================================================
```

### Generated Files

```
reports/
├── PR-20260202-143022.pdf  # Patient report with multi-LLM insights
└── DR-20260202-143022.pdf  # Doctor report with validation details
```

## Mock Mode vs Real AI Mode

### Without API Keys (Mock Mode)

- ✅ Pipeline runs successfully
- ✅ Reports generated
- ⚠️ LLM responses are simulated
- ⚠️ Validation uses mock data

### With API Keys (Real AI Mode)

- ✅ Real Gemini 2.5 Flash responses
- ✅ Real GPT-OSS-120B validation
- ✅ Real II-Medical-8B validation
- ✅ Comprehensive multi-model insights
- ✅ Actual medical AI analysis

## Troubleshooting

### "Import error: langchain_google_genai"

```powershell
pip install langchain-google-genai
```

### "Import error: huggingface_hub"

```powershell
pip install huggingface-hub
```

### "No module named 'reportlab'"

```powershell
pip install reportlab
```

Reports will be skipped, but demo still runs.

### API Rate Limits

If you see rate limit errors, the demo will gracefully fall back to mock mode for that component.

## Next Steps

After running the demo:

1. **View Generated Reports**

   ```powershell
   # Open reports folder
   explorer reports
   ```

2. **Test Individual Components**

   ```powershell
   python test_llm_langchain.py
   ```

3. **Start API Server**

   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

4. **Experiment with Healthy Patient**
   Edit demo.py and change biomarker values to normal ranges to see different risk levels.

## Demo Features Highlighted

✅ **Multi-LLM Integration** - 3 AI models working together  
✅ **LangChain Framework** - Modern LLM orchestration  
✅ **Medical Model Validation** - Specialized AI for healthcare  
✅ **Agentic Consensus** - Multiple AI perspectives  
✅ **Comprehensive Reports** - PDF generation with AI insights  
✅ **Trust Envelope** - Data quality and confidence tracking  
✅ **Mock Fallback** - Works without API keys for testing

Happy testing! 🚀🏥

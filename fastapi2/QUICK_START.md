# Quick Start Guide - LangChain & Multi-LLM Integration

## Installation

### 1. Activate Virtual Environment

```powershell
# Windows PowerShell
benv\Scripts\activate
```

### 2. Install New Dependencies

```powershell
pip install langchain>=0.3.0
pip install langchain-google-genai>=2.0.0
pip install huggingface-hub>=0.26.0
```

Or install all at once:

```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys

Create/update `.env` file in project root:

```env
# Gemini API Key (for LangChain)
GEMINI_API_KEY=your_gemini_api_key_here

# HuggingFace Token (for InferenceClient)
HF_TOKEN=your_huggingface_token_here
```

## Testing the Integration

### Test All LLMs

```powershell
python test_llm_langchain.py
```

Expected output:

```
✓ Gemini API Key: Found
✓ HuggingFace Token: Found

Testing Gemini with LangChain (ChatGoogleGenerativeAI)
✅ Response Received!

Testing HuggingFace with InferenceClient (Medical Models)
✅ GPT-OSS-120B is working correctly!
✅ II-Medical-8B is working correctly!

🎉 All LangChain tests passed!
```

### Test Individual Components

#### 1. Test Gemini (LangChain)

```python
from app.core.llm.gemini_client import GeminiClient

client = GeminiClient()
response = client.generate("What is blood pressure?")
print(response.text)
```

#### 2. Test HuggingFace Medical Models

```python
from app.core.agents.hf_client import HuggingFaceClient
from app.config import settings

client = HuggingFaceClient()

# Test Model 1: GPT-OSS-120B
response1 = client.generate(
    "What is heart rate variability?",
    model=settings.medical_model_1
)
print(response1.text)

# Test Model 2: II-Medical-8B
response2 = client.generate(
    "Explain resting heart rate",
    model=settings.medical_model_2
)
print(response2.text)
```

#### 3. Test Multi-LLM Report Generation

```python
from app.core.llm.multi_llm_interpreter import MultiLLMInterpreter
from app.core.inference.risk_engine import RiskScore, RiskLevel

interpreter = MultiLLMInterpreter()

# Mock composite risk
composite_risk = RiskScore(
    score=65.0,
    level=RiskLevel.MODERATE,
    confidence=0.85
)

# Generate interpretation with all 3 models
result = interpreter.interpret_composite_risk(
    system_results={},  # Add your system results
    composite_risk=composite_risk
)

print("Summary:", result.summary)
print("Recommendations:", result.recommendations)
print("Consensus Level:", result.consensus_level)
print("Total Latency:", result.total_latency_ms, "ms")
```

## Verification Checklist

- [ ] Dependencies installed without errors
- [ ] API keys configured in `.env`
- [ ] `test_llm_langchain.py` passes all tests
- [ ] Gemini responds via LangChain
- [ ] Both HuggingFace medical models respond
- [ ] Multi-LLM interpreter synthesizes responses
- [ ] No import errors in Python files

## Troubleshooting

### Import Error: langchain_google_genai

```powershell
pip install --upgrade langchain-google-genai
```

### HuggingFace Model Loading

Some models may take 20-60 seconds to load on first request. Be patient!

### API Rate Limits

- Gemini: 60 requests/minute (free tier)
- HuggingFace: Varies by model, typically 30-100/minute

### Mock Mode

If API keys are missing, clients automatically fall back to mock responses:

```
[MOCK RESPONSE - Gemini unavailable]
```

This allows development/testing without API access.

## What Changed?

### Before (Old Implementation)

```python
# Old Gemini
import google.genai as genai
client = genai.Client(api_key=key)
response = client.models.generate_content(...)

# Old HuggingFace
import requests
response = requests.post(url, headers=..., json=...)

# Old Validation
medgemma_model = "google/medgemma-4b-it"
openbiollm_model = "aaditya/OpenBioLLM-Llama3-8B"
```

### After (New Implementation)

```python
# New Gemini (LangChain)
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", ...)
response = llm.invoke(prompt)

# New HuggingFace (InferenceClient)
from huggingface_hub import InferenceClient
client = InferenceClient(api_key=key)
completion = client.chat.completions.create(...)

# New Medical Models
medical_model_1 = "openai/gpt-oss-120b"
medical_model_2 = "Intelligent-Internet/II-Medical-8B"

# Multi-LLM Reports
interpreter = MultiLLMInterpreter()  # Uses all 3 models!
```

## Next Steps

1. **Run the application** - Test with actual health data
2. **Monitor performance** - Check latency of multi-LLM calls
3. **Review interpretations** - Ensure quality from all 3 models
4. **Adjust prompts** - Fine-tune for better medical relevance
5. **Add caching** - Cache common interpretations to reduce costs

## Support

If you encounter issues:

1. Check `.env` file has valid API keys
2. Verify packages installed: `pip list | grep -E "langchain|huggingface"`
3. Run tests: `python test_llm_langchain.py`
4. Check logs in console for detailed error messages

Happy testing! 🚀

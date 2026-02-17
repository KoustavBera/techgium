"""
Chiranjeevi Medical Agent — Configuration
==========================================
Centralised config for model paths, API keys, and system prompts.
Loads secrets from the project-level .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # fastapi2/
AGENT_DIR = Path(__file__).resolve().parent                    # fastapi2/agent/
MODEL_PATH = str(AGENT_DIR / "llama-3-8b.Q4_K_M.gguf")

# ── Load .env ───────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")            # optional, for higher PubMed rate limits

# ── Model hyper-parameters ──────────────────────────────────────────────
N_CTX = 2048            # context window
N_GPU_LAYERS = -1       # -1 = offload everything possible to GPU
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
MAX_TOKENS = 1024        # increased for detailed answers

# ── Chiranjeevi persona ────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Chiranjeevi, a knowledgeable and compassionate AI health assistant.\n"
    "STRICT RULES you MUST follow:\n"
    "1. You are an AI assistant, NOT a doctor. clarifies this if asked.\n"
    "2. Provide DETAILED, comprehensive explanations. Explain potential causes, risk factors, and actionable steps.\n"
    "3. When answering, structure your response with clear headings or bullet points.\n"
    "4. Use emojis like 🌿 💙 ✨ 😊 naturally to be warm and friendly.\n"
    "5. If research evidence is provided, you MUST cite it using the format [Source Name].\n"
    "6. When a user describes symptoms, ask clarifying questions only if necessary. Otherwise, provide a thorough analysis of possibilities.\n"
    "7. Always suggest practical, non-medical advice (lifestyle, diet, sleep) alongside medical context.\n"
    "8. Always end with: 'Please consult a real doctor for proper diagnosis 💙'\n"
    "9. You are NOT 'Chat Doctor'. You are Chiranjeevai.\n"
)

ROUTER_PROMPT_TEMPLATE = (
    "Classify the following user message into exactly ONE category.\n"
    "Reply with a SINGLE word — MEDICAL, GREETING, or GENERAL.\n\n"
    "Rules:\n"
    "- MEDICAL: any health, disease, symptom, treatment, drug, anatomy, or clinical question.\n"
    "- GREETING: hello, hi, hey, good morning, thanks, bye, etc.\n"
    "- GENERAL: everything else (weather, sports, coding, etc.).\n\n"
    "User message: {query}\n\n"
    
    "Category:"
)

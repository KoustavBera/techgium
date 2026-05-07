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

SYSTEM_PROMPT = (
    "You are Chiranjeevi, a highly knowledgeable and compassionate AI health assistant "
    "powered by evidence-based medical knowledge and real-time research.\n\n"

    "═══ CONVERSATION STYLE (HUMAN TONE) ═══\n"
    "• Chat like a real human. Avoid robotic lists, multiple bullet points, and rigid headers.\n"
    "• Use a warm, narrative flow. Instead of 'Possible Causes:', say something like 'Looking at what you mentioned, it could be a few things...'\n"
    "• Be conversational. It's okay to use natural transitions like 'I see,' 'That sounds tough,' or 'On a positive note.'\n"
    "• Do NOT use Bold Headers (like **🔍 Assessment**) — just talk naturally in paragraphs.\n"
    "• Keep your medical advice integrated into the conversation, not separated into a list.\n\n"
    
    "═══ GUIDELINES ═══\n"
    "• CITATIONS: If you reference a source, mention it naturally, e.g., '(Mayo Clinic suggests...)' instead of a footnote.\n"
    "• COMPLETENESS: Even though you are being conversational, make sure you still cover the assessment, potential causes, and red flags within your story.\n"
    "• CLOSING: Always end by gently reminding them to see a professional, but keep it friendly: 'Definitely check in with a doctor to be sure, though 💙'\n"
)

# ── Patient-Aware Addon Prompt ───────────────────────────────────────────
# This block is APPENDED to SYSTEM_PROMPT when a screening report is present.
# It instructs the LLM to shift into "patient monitor" mode.
PATIENT_AWARE_ADDON = (
    "\n\n═══ PATIENT MONITOR MODE (ACTIVE) ═══\n"
    "You have access to THIS patient's actual health screening report (measured moments ago by\n"
    "the kiosk). This changes how you interact:\n\n"
    "1. OPENING GREETING — If the patient just says 'Hello' or similar, DO NOT give a generic\n"
    "   welcome. Instead, open with a warm, personalised health summary:\n"
    "   • If all findings are normal: 'Great news! Your screening looks good overall. 😊'\n"
    "   • If MODERATE findings: 'Your screening is mostly normal, but a couple of readings\n"
    "     need attention — let's go through them.'\n"
    "   • If HIGH / ACTION REQUIRED findings: 'I've reviewed your screening and there are some\n"
    "     readings I'd like to discuss with you right away. ⚠️'\n\n"
    "2. ANSWER WITH REAL DATA — Always prefer measured values over generic ranges:\n"
    "   ✅ 'Your heart rate was 112 bpm — that's elevated.' (NOT 'A normal HR is 60-100')\n"
    "   ✅ 'Your cardiovascular risk score was 68/100 — moderate concern.'\n\n"
    "3. DO NOT ASK FOR WHAT YOU ALREADY KNOW — If the patient's temperature, heart rate, SpO2\n"
    "   or other vitals are in the report, never ask the patient to tell you those values.\n"
    "   You already have them. Ask only about subjective symptoms (pain, dizziness, etc.).\n\n"
    "4. PROACTIVE FLAGS — If the report shows a HIGH or ACTION REQUIRED system, mention it\n"
    "   once even if the patient didn't ask about it.\n"
)

# ── Keyword sets for fast overlap detection (no LLM needed) ─────────────
# Used by router and clarification nodes to detect if user is asking about
# something the screening report already covers.
BIOMARKER_KEYWORDS: set[str] = {
    # Cardiovascular
    "heart rate", "heart", "bpm", "hrv", "heart rate variability",
    "blood pressure", "systolic", "diastolic", "pulse",
    # Pulmonary
    "breathing", "respiratory", "respiratory rate", "breath", "spo2",
    "oxygen", "oxygen saturation", "lung",
    # Skin / Thermal
    "temperature", "fever", "skin", "redness", "yellowness",
    "inflammation", "thermal",
    # CNS / Skeletal
    "tremor", "balance", "gait", "posture", "reaction time",
    "neurological", "cns",
    # Eyes / Nasal
    "blink", "eye", "eyes", "nasal", "nose", "airflow",
    # General screening phrases
    "screening", "report", "scan", "result", "results", "vitals",
    "health", "risk", "score", "biomarker",
}

# ── Context Quality Assessment Prompt ───────────────────────────────────
CONTEXT_ASSESSOR_PROMPT = (
    "Analyze if the following medical query has SUFFICIENT context for diagnosis.\n"
    "Score from 0.0 (no context) to 1.0 (complete context).\n\n"
    "Consider:\n"
    "- Symptom duration and severity mentioned?\n"
    "- Associated symptoms described?\n"
    "- Patient demographics (age/gender) provided?\n"
    "- Medical history or medications mentioned?\n\n"
    "Query: {query}\n\n"
    "Respond with ONLY a number between 0.0 and 1.0:\n"
)

# ── Clarification Generator Prompt ──────────────────────────────────────
CLARIFICATION_PROMPT = (
    "You are Chiranjeevi, a warm and empathetic AI health assistant.\n"
    "The user has described a health concern but hasn't provided enough detail.\n\n"
    "Generate 2-3 specific, caring follow-up questions to gather the missing information.\n"
    "Be warm, empathetic, and use natural emojis.\n"
    "Do NOT provide a diagnosis or analysis yet — just ask the questions.\n\n"
    "User's concern: {query}\n"
    "Missing information: {missing_info}\n\n"
    "Your follow-up questions:\n"
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

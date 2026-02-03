"""
LLM Interpretation Module

Uses Gemini 1.5 Flash to explain already-computed risk assessments.
LLM is NON-DECISIONAL - it explains, does NOT diagnose.

ARCHITECTURE CONSTRAINTS:
- LLM receives: SystemRiskResult, CompositeRiskResult, TrustEnvelope
- LLM DOES NOT see: Raw sensor data, raw biomarkers
- LLM outputs: Explanations, medical context, summaries
- LLM DOES NOT output: Diagnoses, risk scores, treatment decisions
"""
from .gemini_client import GeminiClient, GeminiConfig
from .risk_interpreter import RiskInterpreter, InterpretationResult
from .context_generator import MedicalContextGenerator, MedicalContext

__all__ = [
    "GeminiClient",
    "GeminiConfig",
    "RiskInterpreter",
    "InterpretationResult",
    "MedicalContextGenerator",
    "MedicalContext",
]

"""
Agentic Medical Validation Module

Multi-agent validation using specialized medical LLMs.
All agents are NON-DECISIONAL - they validate and flag, not diagnose.

Uses Hugging Face Inference API for:
- MedGemma: Medical plausibility checking
- OpenBioLLM: Biomedical consistency verification
"""
from .hf_client import HuggingFaceClient, HFConfig
from .medical_agents import (
    MedGemmaAgent,
    OpenBioLLMAgent,
    AgentConsensus,
    ValidationResult
)

__all__ = [
    "HuggingFaceClient",
    "HFConfig",
    "MedGemmaAgent",
    "OpenBioLLMAgent",
    "AgentConsensus",
    "ValidationResult",
]

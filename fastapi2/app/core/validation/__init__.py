"""
Validation Module

Physics-based signal and biomarker validation layer.
Gates all downstream interpretation with trust envelope.
"""
from .signal_quality import SignalQualityAssessor, ModalityQualityScore
from .biomarker_plausibility import BiomarkerPlausibilityValidator, PlausibilityResult, PatientContext
from .cross_system_consistency import CrossSystemConsistencyChecker, ConsistencyResult
from .trust_envelope import TrustEnvelope, TrustEnvelopeAggregator

__all__ = [
    "SignalQualityAssessor",
    "ModalityQualityScore",
    "BiomarkerPlausibilityValidator",
    "PlausibilityResult",
    "PatientContext",
    "CrossSystemConsistencyChecker",
    "ConsistencyResult",
    "TrustEnvelope",
    "TrustEnvelopeAggregator",
]

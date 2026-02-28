"""
Clinical Decision Layer

Transforms raw biomarkers into actionable specialist referrals.

Usage:
    from app.core.clinical import ClinicalDecisionEngine, ClinicalFinding

    engine = ClinicalDecisionEngine()
    findings = engine.analyze(biomarker_sets)   # Dict[PhysiologicalSystem, BiomarkerSet]
"""
from .engine import ClinicalDecisionEngine
from .base import ClinicalFinding, UrgencyLevel, SpecialistReferral

__all__ = [
    "ClinicalDecisionEngine",
    "ClinicalFinding",
    "UrgencyLevel",
    "SpecialistReferral",
]

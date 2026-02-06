"""
Inference Module

Computes risk scores from biomarker feature vectors for each physiological system.
"""
from .risk_engine import RiskEngine, RiskScore, SystemRiskResult, TrustedRiskResult
from .calibration import ConfidenceCalibrator
from .explanation import ExplanationGenerator, RiskExplanation

__all__ = [
    "RiskEngine",
    "RiskScore",
    "SystemRiskResult",
    "TrustedRiskResult",
    "ConfidenceCalibrator",
    "ExplanationGenerator",
    "RiskExplanation",
]

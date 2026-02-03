"""
Optional ML/DL Module

NON-DECISIONAL machine learning components for signal quality assessment.
These components ONLY affect confidence, NEVER diagnosis or risk scores.

ALLOWED uses:
- Signal quality anomaly detection
- Noise vs physiology separation

NOT ALLOWED:
- Disease classification
- Health risk prediction
- End-to-end diagnosis
"""
from .anomaly_detector import (
    SignalAnomalyDetector,
    AnomalyResult,
    NoisePhysiologySeparator,
    SeparationResult
)

__all__ = [
    "SignalAnomalyDetector",
    "AnomalyResult",
    "NoisePhysiologySeparator",
    "SeparationResult",
]

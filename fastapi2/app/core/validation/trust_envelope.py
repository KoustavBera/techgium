"""
Trust Envelope Aggregation Module

Aggregates all validation outputs into a single trust envelope.
Gates all downstream interpretation (LLM, reports).
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

from app.core.validation.signal_quality import ModalityQualityScore, Modality
from app.core.validation.biomarker_plausibility import PlausibilityResult, ViolationType
from app.core.validation.cross_system_consistency import ConsistencyResult
from app.core.extraction.base import PhysiologicalSystem
from app.utils import get_logger

logger = get_logger(__name__)


class SafetyFlag(str, Enum):
    """Safety flags that affect downstream processing."""
    NONE = "none"                          # All clear
    LOW_CONFIDENCE = "low_confidence"       # Results may be unreliable
    DATA_QUALITY_ISSUE = "data_quality_issue"  # Sensor problems detected
    PHYSIOLOGICAL_ANOMALY = "physiological_anomaly"  # Unusual readings
    INTERNAL_INCONSISTENCY = "internal_inconsistency"  # Systems disagree
    CRITICAL_VIOLATION = "critical_violation"  # Major plausibility failure


@dataclass
class TrustEnvelope:
    """
    Comprehensive trust assessment for the entire screening.
    
    This envelope GATES all downstream interpretation:
    - If reliability < threshold, LLM should add caveats
    - If safety_flags are set, extra warnings are required
    - confidence_penalties adjust final risk confidence
    """
    # Overall scores
    overall_reliability: float = 1.0      # 0-1: aggregate reliability
    data_quality_score: float = 1.0       # 0-1: sensor/signal quality
    biomarker_plausibility: float = 1.0   # 0-1: physiological validity
    cross_system_consistency: float = 1.0 # 0-1: inter-system agreement
    
    # Penalties to apply to confidence scores
    confidence_penalty: float = 0.0       # 0-1: subtract from confidences
    
    # Safety flags
    safety_flags: List[SafetyFlag] = field(default_factory=list)
    
    # Breakdown by modality and system
    modality_scores: Dict[str, float] = field(default_factory=dict)
    system_reliability: Dict[str, float] = field(default_factory=dict)
    
    # Issues summary
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations for downstream
    interpretation_guidance: str = ""
    
    @property
    def is_reliable(self) -> bool:
        """Check if results are reliable enough for interpretation."""
        return (
            self.overall_reliability >= 0.5 and
            SafetyFlag.CRITICAL_VIOLATION not in self.safety_flags
        )
    
    @property
    def requires_caveats(self) -> bool:
        """Check if interpretation needs extra caveats."""
        return (
            self.overall_reliability < 0.8 or
            len(self.safety_flags) > 1 or
            self.confidence_penalty > 0.2
        )
    
    def get_adjusted_confidence(self, original_confidence: float) -> float:
        """Apply penalty to a confidence score."""
        return float(np.clip(
            original_confidence * (1 - self.confidence_penalty),
            0.1, 0.99
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_reliability": round(self.overall_reliability, 3),
            "data_quality_score": round(self.data_quality_score, 3),
            "biomarker_plausibility": round(self.biomarker_plausibility, 3),
            "cross_system_consistency": round(self.cross_system_consistency, 3),
            "confidence_penalty": round(self.confidence_penalty, 3),
            "safety_flags": [f.value for f in self.safety_flags],
            "is_reliable": self.is_reliable,
            "requires_caveats": self.requires_caveats,
            "modality_scores": {k: round(v, 3) for k, v in self.modality_scores.items()},
            "system_reliability": {k: round(v, 3) for k, v in self.system_reliability.items()},
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "interpretation_guidance": self.interpretation_guidance
        }


class TrustEnvelopeAggregator:
    """
    Aggregates all validation results into a TrustEnvelope.
    
    Combines:
    - Signal quality scores
    - Biomarker plausibility results
    - Cross-system consistency results
    
    Output gates all downstream LLM interpretation.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self._aggregation_count = 0
        logger.info("TrustEnvelopeAggregator initialized")
    
    def aggregate(
        self,
        signal_quality: Optional[Dict[Modality, ModalityQualityScore]] = None,
        plausibility_results: Optional[Dict[PhysiologicalSystem, PlausibilityResult]] = None,
        consistency_result: Optional[ConsistencyResult] = None
    ) -> TrustEnvelope:
        """
        Aggregate all validation results into a TrustEnvelope.
        
        Args:
            signal_quality: Quality scores per modality
            plausibility_results: Plausibility results per system
            consistency_result: Cross-system consistency result
            
        Returns:
            TrustEnvelope gating downstream interpretation
        """
        envelope = TrustEnvelope()
        safety_flags = set()
        critical_issues = []
        warnings = []
        
        # 1. Process signal quality
        if signal_quality:
            modality_scores = {}
            quality_sum = 0.0
            
            for modality, score in signal_quality.items():
                modality_scores[modality.value] = score.overall_quality
                quality_sum += score.overall_quality
                
                # Add issues
                for issue in score.issues:
                    if score.overall_quality < 0.5:
                        critical_issues.append(f"[{modality.value}] {issue}")
                    else:
                        warnings.append(f"[{modality.value}] {issue}")
                
                # Set flags
                if score.overall_quality < 0.3:
                    safety_flags.add(SafetyFlag.DATA_QUALITY_ISSUE)
                elif score.overall_quality < 0.5:
                    safety_flags.add(SafetyFlag.LOW_CONFIDENCE)
            
            envelope.modality_scores = modality_scores
            envelope.data_quality_score = quality_sum / len(signal_quality) if signal_quality else 1.0
        
        # 2. Process biomarker plausibility
        if plausibility_results:
            system_reliability = {}
            plausibility_sum = 0.0
            
            for system, result in plausibility_results.items():
                system_reliability[system.value] = result.overall_plausibility
                plausibility_sum += result.overall_plausibility
                
                for violation in result.violations:
                    msg = f"[{system.value}] {violation.message}"
                    
                    if violation.severity >= 0.8:
                        critical_issues.append(msg)
                        if violation.violation_type == ViolationType.IMPOSSIBLE_VALUE:
                            safety_flags.add(SafetyFlag.CRITICAL_VIOLATION)
                    elif violation.severity >= 0.5:
                        warnings.append(msg)
                        safety_flags.add(SafetyFlag.PHYSIOLOGICAL_ANOMALY)
                    else:
                        warnings.append(msg)
            
            envelope.system_reliability = system_reliability
            envelope.biomarker_plausibility = plausibility_sum / len(plausibility_results) if plausibility_results else 1.0
        
        # 3. Process cross-system consistency
        if consistency_result:
            envelope.cross_system_consistency = consistency_result.overall_consistency
            
            for inconsistency in consistency_result.inconsistencies:
                msg = f"[{inconsistency.systems[0]} vs {inconsistency.systems[1]}] {inconsistency.message}"
                
                if inconsistency.severity >= 0.7:
                    warnings.append(msg)
                    safety_flags.add(SafetyFlag.INTERNAL_INCONSISTENCY)
                else:
                    warnings.append(msg)
        
        # 4. Compute overall reliability
        components = [
            envelope.data_quality_score,
            envelope.biomarker_plausibility,
            envelope.cross_system_consistency
        ]
        # Geometric mean for multiplicative effect
        envelope.overall_reliability = float(np.power(np.prod(components), 1/len(components)))
        
        # 5. Compute confidence penalty
        # Penalty increases with more/worse issues
        base_penalty = 1.0 - envelope.overall_reliability
        flag_penalty = len(safety_flags) * 0.05
        if SafetyFlag.CRITICAL_VIOLATION in safety_flags:
            flag_penalty += 0.2
        
        envelope.confidence_penalty = float(np.clip(base_penalty + flag_penalty, 0, 0.5))
        
        # 6. Set safety flags and issues
        envelope.safety_flags = list(safety_flags)  # Empty = clean
        envelope.critical_issues = critical_issues
        envelope.warnings = warnings
        
        # 7. Generate interpretation guidance
        envelope.interpretation_guidance = self._generate_guidance(envelope)
        
        self._aggregation_count += 1
        logger.debug(f"Trust envelope created: reliability={envelope.overall_reliability:.2f}")
        
        return envelope
    
    def _generate_guidance(self, envelope: TrustEnvelope) -> str:
        """Generate guidance for downstream interpretation."""
        if not envelope.is_reliable:
            return (
                "CRITICAL: Data reliability is below acceptable threshold. "
                "Results should be interpreted with extreme caution or rejected. "
                "Recommend re-screening."
            )
        
        if SafetyFlag.CRITICAL_VIOLATION in envelope.safety_flags:
            return (
                "WARNING: Critical physiological violations detected. "
                "Some biomarker values are outside possible ranges. "
                "These should be excluded from interpretation."
            )
        
        if envelope.requires_caveats:
            caveats = []
            
            if envelope.data_quality_score < 0.7:
                caveats.append("sensor quality issues may affect accuracy")
            
            if envelope.biomarker_plausibility < 0.8:
                caveats.append("some biomarker values are at physiological extremes")
            
            if envelope.cross_system_consistency < 0.8:
                caveats.append("there are inconsistencies between body systems")
            
            return (
                f"CAUTION: Results require caveats. Issues: {'; '.join(caveats)}. "
                f"Confidence penalty: {envelope.confidence_penalty:.0%}."
            )
        
        return (
            "Data quality and biomarker validity are within acceptable ranges. "
            "Results can be interpreted with standard confidence."
        )
    
    def create_minimal_envelope(self, reliability: float = 0.5) -> TrustEnvelope:
        """
        Create a minimal trust envelope for simulated/test data.
        
        Args:
            reliability: Overall reliability to assign
            
        Returns:
            TrustEnvelope with simulated flag
        """
        envelope = TrustEnvelope(
            overall_reliability=reliability,
            data_quality_score=reliability,
            biomarker_plausibility=reliability,
            cross_system_consistency=reliability,
            confidence_penalty=1.0 - reliability,
            safety_flags=[SafetyFlag.LOW_CONFIDENCE],
            interpretation_guidance="Data is simulated. Interpret for demonstration only."
        )
        return envelope

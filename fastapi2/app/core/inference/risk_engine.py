"""
Risk Engine Module

Computes system-specific risk scores from biomarker values.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import numpy as np

from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem, Biomarker
from app.utils import get_logger

# Conditional imports to avoid circular dependency
if TYPE_CHECKING:
    from app.core.validation.biomarker_plausibility import PlausibilityResult
    from app.core.validation.trust_envelope import TrustEnvelope

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Convert numeric score (0-100) to risk level."""
        if score < 25:
            return cls.LOW
        elif score < 50:
            return cls.MODERATE
        elif score < 75:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass
class RiskScore:
    """Individual risk score for a specific aspect."""
    name: str
    score: float  # 0-100 scale
    confidence: float  # 0-1 scale
    contributing_biomarkers: List[str] = field(default_factory=list)
    explanation: str = ""
    
    @property
    def level(self) -> RiskLevel:
        """Get categorical risk level."""
        return RiskLevel.from_score(self.score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "score": round(self.score, 2),
            "level": self.level.value,
            "confidence": round(self.confidence, 3),
            "contributing_biomarkers": self.contributing_biomarkers,
            "explanation": self.explanation,
        }


@dataclass
class SystemRiskResult:
    """Complete risk assessment for one physiological system."""
    system: PhysiologicalSystem
    overall_risk: RiskScore
    sub_risks: List[RiskScore] = field(default_factory=list)
    biomarker_summary: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "system": self.system.value,
            "overall_risk": self.overall_risk.to_dict(),
            "sub_risks": [r.to_dict() for r in self.sub_risks],
            "biomarker_summary": self.biomarker_summary,
            "alerts": self.alerts,
        }


@dataclass
class TrustedRiskResult:
    """
    Risk result wrapped with trust metadata.
    
    This indicates whether the risk computation was gated by validation.
    """
    risk_result: Optional[SystemRiskResult] = None
    is_trusted: bool = True
    was_rejected: bool = False
    rejection_reason: str = ""
    trust_adjusted_confidence: float = 1.0
    caveats: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "is_trusted": self.is_trusted,
            "was_rejected": self.was_rejected,
            "rejection_reason": self.rejection_reason,
            "trust_adjusted_confidence": round(self.trust_adjusted_confidence, 3),
            "caveats": self.caveats,
        }
        if self.risk_result:
            result["risk_result"] = self.risk_result.to_dict()
        return result


class RiskEngine:
    """
    Core risk computation engine.
    
    Transforms biomarker sets into risk assessments for each physiological system.
    Uses rule-based scoring with configurable thresholds.
    """
    
    def __init__(self):
        """Initialize risk engine with default thresholds."""
        self._computation_count = 0
        self._risk_weights = self._initialize_weights()
        logger.info("RiskEngine initialized")
    
    def _initialize_weights(self) -> Dict[PhysiologicalSystem, Dict[str, float]]:
        """Initialize biomarker weights for each system."""
        return {
            PhysiologicalSystem.CNS: {
                "gait_variability": 0.25,
                "posture_entropy": 0.20,
                "tremor_resting": 0.25,
                "tremor_postural": 0.15,
                "cns_stability_score": 0.15,
            },
            PhysiologicalSystem.CARDIOVASCULAR: {
                "heart_rate": 0.25,
                "hrv_rmssd": 0.25,
                "systolic_bp": 0.20,
                "diastolic_bp": 0.15,
                "chest_micro_motion": 0.15,
            },
            PhysiologicalSystem.RENAL: {
                "fluid_asymmetry_index": 0.30,
                "total_body_water_proxy": 0.25,
                "extracellular_fluid_ratio": 0.25,
                "fluid_overload_index": 0.20,
            },
            PhysiologicalSystem.GASTROINTESTINAL: {
                "abdominal_rhythm_score": 0.40,
                "visceral_motion_variance": 0.35,
                "abdominal_respiratory_rate": 0.25,
            },
            PhysiologicalSystem.SKELETAL: {
                "gait_symmetry_ratio": 0.25,
                "step_length_symmetry": 0.20,
                "stance_stability_score": 0.25,
                "sway_velocity": 0.15,
                "average_joint_rom": 0.15,
            },
            PhysiologicalSystem.SKIN: {
                "texture_roughness": 0.25,
                "skin_redness": 0.20,
                "skin_yellowness": 0.25,
                "color_uniformity": 0.15,
                "lesion_count": 0.15,
            },
            PhysiologicalSystem.EYES: {
                "blink_rate": 0.20,
                "gaze_stability_score": 0.25,
                "fixation_duration": 0.20,
                "saccade_frequency": 0.20,
                "eye_symmetry": 0.15,
            },
            PhysiologicalSystem.NASAL: {
                "breathing_regularity": 0.30,
                "respiratory_rate": 0.30,
                "breath_depth_index": 0.20,
                "airflow_turbulence": 0.20,
            },
            PhysiologicalSystem.REPRODUCTIVE: {
                "autonomic_imbalance_index": 0.35,
                "stress_response_proxy": 0.35,
                "regional_flow_variability": 0.15,
                "thermoregulation_proxy": 0.15,
            },
        }
    
    def compute_risk(self, biomarker_set: BiomarkerSet) -> SystemRiskResult:
        """
        Compute risk for a single physiological system.
        
        Args:
            biomarker_set: BiomarkerSet from extraction module
            
        Returns:
            SystemRiskResult with overall and sub-risks
        """
        import time
        start_time = time.time()
        
        system = biomarker_set.system
        sub_risks = []
        alerts = []
        contributing_biomarkers = []
        
        # Calculate risk for each biomarker
        weights = self._risk_weights.get(system, {})
        weighted_scores = []
        confidences = []
        
        for biomarker in biomarker_set.biomarkers:
            bm_risk, bm_alert = self._calculate_biomarker_risk(biomarker)
            
            weight = weights.get(biomarker.name, 0.1)
            weighted_scores.append(bm_risk * weight)
            confidences.append(biomarker.confidence * weight)
            
            if bm_alert:
                alerts.append(bm_alert)
                contributing_biomarkers.append(biomarker.name)
            
            sub_risks.append(RiskScore(
                name=biomarker.name,
                score=bm_risk,
                confidence=biomarker.confidence,
                contributing_biomarkers=[biomarker.name],
                explanation=self._generate_biomarker_explanation(biomarker, bm_risk)
            ))
        
        # Compute overall risk
        if weighted_scores:
            total_weight = sum(weights.get(bm.name, 0.1) for bm in biomarker_set.biomarkers)
            overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 50.0
            overall_confidence = sum(confidences) / total_weight if total_weight > 0 else 0.5
        else:
            overall_score = 50.0
            overall_confidence = 0.5
        
        overall_risk = RiskScore(
            name=f"{system.value}_overall",
            score=float(np.clip(overall_score, 0, 100)),
            confidence=float(np.clip(overall_confidence, 0, 1)),
            contributing_biomarkers=contributing_biomarkers,
            explanation=self._generate_system_explanation(system, overall_score, alerts)
        )
        
        # Build biomarker summary with status
        biomarker_summary = {
            bm.name: {
                "value": bm.value,
                "unit": bm.unit,
                "is_abnormal": bm.is_abnormal(),
                "normal_range": bm.normal_range,
                "status": self._calculate_biomarker_status(bm)
            }
            for bm in biomarker_set.biomarkers
        }
        
        result = SystemRiskResult(
            system=system,
            overall_risk=overall_risk,
            sub_risks=sub_risks,
            biomarker_summary=biomarker_summary,
            alerts=alerts
        )
        
        self._computation_count += 1
        logger.debug(f"Risk computed for {system.value}: {overall_score:.1f}")
        
        return result
    
    def compute_all_risks(
        self,
        biomarker_sets: List[BiomarkerSet]
    ) -> Dict[PhysiologicalSystem, SystemRiskResult]:
        """
        Compute risks for all provided biomarker sets.
        
        Args:
            biomarker_sets: List of BiomarkerSets from all extractors
            
        Returns:
            Dict mapping system to risk result
        """
        results = {}
        for bm_set in biomarker_sets:
            results[bm_set.system] = self.compute_risk(bm_set)
        return results
    
    def compute_risk_with_validation(
        self,
        biomarker_set: BiomarkerSet,
        plausibility: Optional[PlausibilityResult] = None
    ) -> TrustedRiskResult:
        """
        Compute risk with plausibility validation gating.
        
        If plausibility result indicates invalid data, the risk computation
        is rejected and a gated result is returned.
        
        Args:
            biomarker_set: BiomarkerSet from extraction module
            plausibility: Optional PlausibilityResult from validation
            
        Returns:
            TrustedRiskResult with risk and trust metadata
        """
        # Check if plausibility validation fails
        if plausibility is not None and not plausibility.is_valid:
            # Critical violations - reject the data
            violation_msgs = [v.message for v in plausibility.violations if v.severity >= 0.8]
            return TrustedRiskResult(
                risk_result=None,
                is_trusted=False,
                was_rejected=True,
                rejection_reason=f"Plausibility validation failed: {'; '.join(violation_msgs[:3])}",
                trust_adjusted_confidence=0.0,
                caveats=["Data rejected due to physiological implausibility"]
            )
        
        # Compute risk normally
        risk_result = self.compute_risk(biomarker_set)
        
        # Build caveats from any non-critical violations
        caveats = []
        adjusted_confidence = risk_result.overall_risk.confidence
        
        if plausibility is not None:
            # Apply plausibility penalty to confidence
            adjusted_confidence *= plausibility.overall_plausibility
            
            # Add warnings for moderate violations
            for v in plausibility.violations:
                if 0.4 <= v.severity < 0.8:
                    caveats.append(f"{v.biomarker_name}: {v.message}")
        
        return TrustedRiskResult(
            risk_result=risk_result,
            is_trusted=plausibility is None or plausibility.overall_plausibility >= 0.7,
            was_rejected=False,
            trust_adjusted_confidence=float(np.clip(adjusted_confidence, 0.1, 0.99)),
            caveats=caveats[:5]  # Limit caveats
        )
    
    def compute_risk_with_trust(
        self,
        biomarker_set: BiomarkerSet,
        trust_envelope: Optional[TrustEnvelope] = None
    ) -> TrustedRiskResult:
        """
        Compute risk with full trust envelope integration.
        
        Uses TrustEnvelope to gate risk computation and adjust confidence.
        
        Args:
            biomarker_set: BiomarkerSet from extraction module
            trust_envelope: Optional TrustEnvelope from validation layer
            
        Returns:
            TrustedRiskResult with risk and trust metadata
        """
        # Check trust envelope reliability
        if trust_envelope is not None and not trust_envelope.is_reliable:
            return TrustedRiskResult(
                risk_result=None,
                is_trusted=False,
                was_rejected=True,
                rejection_reason=trust_envelope.interpretation_guidance,
                trust_adjusted_confidence=0.0,
                caveats=trust_envelope.critical_issues[:3]
            )
        
        # Compute risk normally
        risk_result = self.compute_risk(biomarker_set)
        
        # Apply trust adjustments
        caveats = []
        adjusted_confidence = risk_result.overall_risk.confidence
        is_trusted = True
        
        if trust_envelope is not None:
            # Apply confidence penalty from trust envelope
            adjusted_confidence = trust_envelope.get_adjusted_confidence(adjusted_confidence)
            
            # Check if caveats are required
            if trust_envelope.requires_caveats:
                is_trusted = False
                caveats.extend(trust_envelope.warnings[:3])
        
        return TrustedRiskResult(
            risk_result=risk_result,
            is_trusted=is_trusted,
            was_rejected=False,
            trust_adjusted_confidence=float(np.clip(adjusted_confidence, 0.1, 0.99)),
            caveats=caveats
        )
    
    def _calculate_biomarker_risk(self, biomarker: Biomarker) -> Tuple[float, Optional[str]]:
        """
        Calculate risk score for individual biomarker.
        
        Returns:
            Tuple of (risk_score 0-100, alert_message or None)
        """
        if biomarker.normal_range is None:
            # No normal range defined, assume moderate risk based on value
            return 30.0, None
        
        low, high = biomarker.normal_range
        value = biomarker.value
        
        # Calculate deviation from normal range
        if low <= value <= high:
            # Within normal range
            # Score based on closeness to center
            center = (low + high) / 2
            range_half = (high - low) / 2
            deviation = abs(value - center) / (range_half + 1e-6)
            risk = 20 * deviation  # 0-20 within normal
            return risk, None
        elif value < low:
            # Below normal
            deviation = (low - value) / (low + 1e-6)
            risk = 25 + min(deviation * 75, 75)  # 25-100
            severity = "significantly " if deviation > 0.3 else ""
            return risk, f"{biomarker.name} is {severity}below normal range"
        else:
            # Above normal
            deviation = (value - high) / (high + 1e-6)
            risk = 25 + min(deviation * 75, 75)  # 25-100
            severity = "significantly " if deviation > 0.3 else ""
            return risk, f"{biomarker.name} is {severity}above normal range"
    
    def _generate_biomarker_explanation(self, biomarker: Biomarker, risk: float) -> str:
        """Generate human-readable explanation for biomarker risk."""
        level = RiskLevel.from_score(risk)
        
        if biomarker.normal_range:
            low, high = biomarker.normal_range
            range_str = f"(normal: {low}-{high} {biomarker.unit})"
        else:
            range_str = ""
        
        return (
            f"{biomarker.name}: {biomarker.value:.2f} {biomarker.unit} {range_str}. "
            f"Risk level: {level.value}."
        )
    
    def _generate_system_explanation(
        self,
        system: PhysiologicalSystem,
        score: float,
        alerts: List[str]
    ) -> str:
        """Generate explanation for overall system risk."""
        level = RiskLevel.from_score(score)
        
        explanation = f"{system.value.replace('_', ' ').title()} assessment indicates {level.value} risk."
        
        if alerts:
            explanation += f" Concerns: {'; '.join(alerts[:3])}"
            if len(alerts) > 3:
                explanation += f" (+{len(alerts) - 3} more)"
        
        return explanation
    
    def _calculate_biomarker_status(self, biomarker: Biomarker) -> str:
        """
        Calculate status string for biomarker based on normal range.
        
        Returns:
            Status string: "normal", "low", "high", or "not_assessed"
        """
        if biomarker.normal_range is None:
            return "not_assessed"
        
        low, high = biomarker.normal_range
        value = biomarker.value
        
        if low <= value <= high:
            return "normal"
        elif value < low:
            return "low"
        else:
            return "high"


class CompositeRiskCalculator:
    """Calculates aggregate risk across all systems."""
    
    def __init__(self, system_weights: Optional[Dict[PhysiologicalSystem, float]] = None):
        """
        Initialize with optional custom system weights.
        
        Args:
            system_weights: Weight for each system in overall calculation
        """
        self.system_weights = system_weights or {
            PhysiologicalSystem.CNS: 1.0,
            PhysiologicalSystem.CARDIOVASCULAR: 1.2,  # Higher weight for critical system
            PhysiologicalSystem.RENAL: 1.0,
            PhysiologicalSystem.GASTROINTESTINAL: 0.8,
            PhysiologicalSystem.SKELETAL: 0.9,
            PhysiologicalSystem.SKIN: 0.7,
            PhysiologicalSystem.EYES: 0.8,
            PhysiologicalSystem.NASAL: 0.8,
            PhysiologicalSystem.REPRODUCTIVE: 0.7,
        }
    
    def compute_composite_risk(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult]
    ) -> RiskScore:
        """
        Compute weighted composite risk across all assessed systems.
        
        Args:
            system_results: Dict of system risk results
            
        Returns:
            Composite RiskScore
        """
        if not system_results:
            return RiskScore(
                name="composite_health_risk",
                score=50.0,
                confidence=0.0,
                explanation="No systems assessed"
            )
        
        weighted_scores = []
        weighted_confidences = []
        contributing = []
        all_alerts = []
        
        for system, result in system_results.items():
            weight = self.system_weights.get(system, 1.0)
            weighted_scores.append(result.overall_risk.score * weight)
            weighted_confidences.append(result.overall_risk.confidence * weight)
            
            if result.overall_risk.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                contributing.append(system.value)
            
            all_alerts.extend(result.alerts)
        
        total_weight = sum(
            self.system_weights.get(s, 1.0) for s in system_results.keys()
        )
        
        composite_score = sum(weighted_scores) / total_weight
        composite_confidence = sum(weighted_confidences) / total_weight
        
        # Critical override: if any system is CRITICAL, boost composite to at least HIGH
        # If any system is HIGH, boost composite to at least MODERATE
        max_level = RiskLevel.LOW
        for result in system_results.values():
            if result.overall_risk.level == RiskLevel.CRITICAL:
                max_level = RiskLevel.CRITICAL
                break
            elif result.overall_risk.level == RiskLevel.HIGH and max_level != RiskLevel.CRITICAL:
                max_level = RiskLevel.HIGH
        
        # Apply critical override boost
        if max_level == RiskLevel.CRITICAL and composite_score < 75:
            composite_score = 75.0  # Minimum HIGH risk when any system is CRITICAL
        elif max_level == RiskLevel.HIGH and composite_score < 50:
            composite_score = 50.0  # Minimum MODERATE risk when any system is HIGH
        
        # Generate explanation
        level = RiskLevel.from_score(composite_score)
        if contributing:
            explanation = (
                f"Overall health assessment: {level.value} risk. "
                f"Primary concerns in: {', '.join(contributing[:3])}."
            )
        else:
            explanation = f"Overall health assessment: {level.value} risk. No critical concerns identified."
        
        return RiskScore(
            name="composite_health_risk",
            score=float(np.clip(composite_score, 0, 100)),
            confidence=float(np.clip(composite_confidence, 0, 1)),
            contributing_biomarkers=contributing,
            explanation=explanation
        )
    
    def compute_composite_risk_with_trust(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        trust_envelope: Optional[TrustEnvelope] = None
    ) -> RiskScore:
        """
        Compute composite risk with trust envelope integration.
        
        Args:
            system_results: Dict of system risk results
            trust_envelope: Optional TrustEnvelope from validation layer
            
        Returns:
            Composite RiskScore with trust-adjusted confidence
        """
        # Compute base composite risk
        composite = self.compute_composite_risk(system_results)
        
        # Apply trust envelope adjustments
        if trust_envelope is not None:
            # Adjust confidence based on trust envelope
            composite.confidence = trust_envelope.get_adjusted_confidence(composite.confidence)
            
            # Add trust context to explanation
            if trust_envelope.requires_caveats:
                composite.explanation += f" (Data quality: {trust_envelope.overall_reliability:.0%})"
        
        return composite
    
    def compute_composite_risk_from_trusted(
        self,
        trusted_results: Dict[PhysiologicalSystem, "TrustedRiskResult"]
    ) -> Tuple[RiskScore, List[str]]:
        """
        Compute composite risk from TrustedRiskResults.
        
        Handles rejected results by excluding them from the score calculation
        and noting them in the explanation.
        
        Args:
            trusted_results: Dict of TrustedRiskResult per system
            
        Returns:
            Tuple of (Composite RiskScore, list of rejected system names)
        """
        valid_results: Dict[PhysiologicalSystem, SystemRiskResult] = {}
        rejected_systems: List[str] = []
        rejection_reasons: List[str] = []
        
        for system, trusted in trusted_results.items():
            if trusted.was_rejected:
                rejected_systems.append(system.value)
                if trusted.rejection_reason:
                    rejection_reasons.append(f"{system.value}: {trusted.rejection_reason}")
            elif trusted.risk_result is not None:
                valid_results[system] = trusted.risk_result
        
        # Compute composite from valid results only
        if not valid_results:
            explanation = "No systems could be reliably assessed."
            if rejected_systems:
                explanation += f" Rejected: {', '.join(rejected_systems)}."
            return RiskScore(
                name="composite_health_risk",
                score=50.0,
                confidence=0.0,
                explanation=explanation
            ), rejected_systems
        
        composite = self.compute_composite_risk(valid_results)
        
        # Append rejection note to explanation
        if rejected_systems:
            composite.explanation += (
                f" Note: {len(rejected_systems)} system(s) excluded due to data quality: "
                f"{', '.join(rejected_systems)}."
            )
        
        # Apply average trust penalty from valid results
        trust_penalties = [
            1.0 - tr.trust_adjusted_confidence
            for tr in trusted_results.values()
            if not tr.was_rejected and tr.risk_result is not None
        ]
        if trust_penalties:
            avg_penalty = sum(trust_penalties) / len(trust_penalties)
            composite.confidence = float(np.clip(
                composite.confidence * (1.0 - avg_penalty * 0.5),
                0.1, 0.99
            ))
        
        return composite, rejected_systems


"""
Risk Engine Module

Computes system-specific risk scores from biomarker values.
Enhanced with confidence calibration for more accurate risk assessment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import numpy as np

from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem, Biomarker
from app.utils import get_logger
from app.config import settings

# Optional confidence calibration (graceful fallback if not available)
try:
    from app.core.inference.calibration import ConfidenceCalibrator, CalibrationFactors
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    ConfidenceCalibrator = None
    CalibrationFactors = None

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
    ACTION_REQUIRED = "action_required"  # Softened from CRITICAL
    UNKNOWN = "unknown"
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Convert numeric score (0-100) to risk level."""
        if score < 0:
            return cls.UNKNOWN
        elif score < 25:
            return cls.LOW
        elif score < 50:
            return cls.MODERATE
        elif score < 75:
            return cls.HIGH
        else:
            return cls.ACTION_REQUIRED


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
    
    def __init__(self, use_calibration: bool = True):
        """Initialize risk engine with default thresholds.
        
        Args:
            use_calibration: Whether to use confidence calibration (if available)
        """
        self._computation_count = 0
        self._risk_weights = self._initialize_weights()
        self._use_calibration = use_calibration
        
        # Initialize calibrator if available and enabled
        self._calibrator = None
        if use_calibration and CALIBRATION_AVAILABLE:
            try:
                self._calibrator = ConfidenceCalibrator()
                logger.info("RiskEngine initialized WITH confidence calibration")
            except Exception as e:
                logger.warning(f"Failed to initialize calibrator: {e}. Falling back to basic confidence.")
        else:
            logger.info("RiskEngine initialized (basic confidence scoring)")
    
    def _initialize_weights(self) -> Dict[PhysiologicalSystem, Dict[str, float]]:
        """Initialize biomarker weights for each system."""
        return {
            PhysiologicalSystem.CNS: {
                "gait_variability": 0.22,
                "posture_entropy": 0.18,
                "tremor_resting": 0.22,
                "tremor_postural": 0.13,
                "cns_stability_score": 0.15,
                "thermal_stress_gradient": 0.10,  # Autonomic stress from thermal
            },
            PhysiologicalSystem.CARDIOVASCULAR: {
                "heart_rate": 0.22,
                "hrv_rmssd": 0.22,
                "systolic_bp": 0.18,
                "diastolic_bp": 0.13,
                "chest_micro_motion": 0.15,
                "thermal_asymmetry": 0.10,  # Facial perfusion asymmetry
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
                "texture_roughness": 0.20,
                "skin_redness": 0.15,
                "skin_yellowness": 0.20,
                "color_uniformity": 0.12,
                "lesion_count": 0.08,
                "inflammation_index": 0.12,  # Thermal inflammation marker
                "skin_temperature": 0.08,    # Fever detection
                "thermal_stability": 0.05,   # Thermal measurement consistency
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
            PhysiologicalSystem.VISUAL_DISEASE: {
                # All visual AI biomarkers share the same weight.
                # Unknown names will also fall back to 0.1, which is fine.
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
        
        # Handle empty biomarker set (device not connected/no data)
        if not biomarker_set.biomarkers:
            overall_risk = RiskScore(
                name=f"{system.value}_overall",
                score=-1.0,  # Sentinel for unknown
                confidence=0.0,
                contributing_biomarkers=[],
                explanation="Insufficient data provided. Sensor connection required."
            )
            return SystemRiskResult(
                system=system,
                overall_risk=overall_risk,
                sub_risks=[],
                biomarker_summary={},
                alerts=["Device required/Not connected"]
            )

        sub_risks = []
        alerts = []
        contributing_biomarkers = []
        
        # Calculate risk for each biomarker
        weights = self._risk_weights.get(system, {})
        weighted_scores = []
        confidences = []
        actual_weights = []  # Track actual weights for proper normalization
        
        for biomarker in biomarker_set.biomarkers:
            bm_risk, bm_alert = self._calculate_biomarker_risk(biomarker)
            
            weight = weights.get(biomarker.name, 0.1)
            actual_weights.append(weight)
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
        
        # Compute overall risk with proper weight normalization
        if weighted_scores:
            total_weight = sum(actual_weights)
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
        
        # ENHANCEMENT: Apply confidence calibration to result
        if self._calibrator is not None:
            try:
                original_confidence = result.overall_risk.confidence
                result = self._calibrator.calibrate_risk_result(result)
                logger.debug(
                    f"Calibrated {system.value} confidence: {original_confidence:.3f} -> {result.overall_risk.confidence:.3f}"
                )
            except Exception as e:
                logger.warning(f"Confidence calibration failed (non-critical): {e}")
        
        self._computation_count += 1
        logger.debug(f"Risk computed for {system.value}: {overall_score:.1f}")
        
        return result
    
    def compute_all_risks(
        self,
        biomarker_sets: List[BiomarkerSet]
    ) -> Dict[PhysiologicalSystem, SystemRiskResult]:
        """
        Compute risks for all provided biomarker sets in parallel.

        Each system's computation is independent (pure CPU + numpy, no shared
        state), so we dispatch them concurrently.  Workers are capped at 4 to
        avoid over-subscribing the interpreter process on constrained hardware.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not biomarker_sets:
            return {}

        max_workers = min(len(biomarker_sets), 4)
        results: Dict[PhysiologicalSystem, SystemRiskResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_system = {
                executor.submit(self.compute_risk, bm_set): bm_set.system
                for bm_set in biomarker_sets
            }
            for future in as_completed(future_to_system):
                system = future_to_system[future]
                try:
                    results[system] = future.result()
                except Exception as exc:
                    logger.error(f"Risk computation failed for {system.value}: {exc}")

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
            # Use configurable penalty threshold from settings
            penalty_intensity = settings.confidence_penalty_threshold
            penalty_factor = 1.0 - ( (1.0 - plausibility.overall_plausibility) * penalty_intensity )
            adjusted_confidence *= penalty_factor
            
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
            # No normal range defined - assume normal baseline (e.g., stationary gait)
            # Use 20.0 (Low Risk) instead of 30.0 to avoid false moderate risk
            return 20.0, None
        
        low, high = biomarker.normal_range
        value = biomarker.value
        
        # Use range width for consistent deviation calculation
        range_width = (high - low) if high != low else 1.0
        
        # Calculate deviation from normal range
        if low <= value <= high:
            # Within normal range
            # Score based on closeness to center
            center = (low + high) / 2
            range_half = range_width / 2
            deviation = abs(value - center) / (range_half + 1e-6)
            risk = 20 * deviation  # 0-20 within normal
            return risk, None
        elif value < low:
            # Below normal - deviation relative to range width
            deviation = (low - value) / range_width
            risk = 25 + min(deviation * 75, 75)  # 25-100
            severity = "significantly " if deviation > 0.3 else ""
            return risk, f"{biomarker.name} is {severity}below normal range"
        else:
            # Above normal - deviation relative to range width
            deviation = (value - high) / range_width
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
        if score < 0:
            return f"{system.value.replace('_', ' ').title()} not assessed due to missing data."

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
            PhysiologicalSystem.GASTROINTESTINAL: 0.8,
            PhysiologicalSystem.SKELETAL: 0.9,
            PhysiologicalSystem.SKIN: 0.7,
            PhysiologicalSystem.EYES: 0.8,
            PhysiologicalSystem.NASAL: 0.8,
            PhysiologicalSystem.REPRODUCTIVE: 0.7,
            PhysiologicalSystem.VISUAL_DISEASE: 0.5,  # Experimental — informational only
        }
        
        # Phase 2: Experimental systems with reduced impact
        self.experimental_systems = {
            PhysiologicalSystem.NASAL,
            PhysiologicalSystem.VISUAL_DISEASE,  # API-based visual classification
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
            
            # Reduce impact of experimental systems by 50%
            if system in self.experimental_systems:
                weight *= 0.5
                logger.debug(f"Reduced weight for experimental system: {system.name}")

            weighted_scores.append(result.overall_risk.score * weight)
            weighted_confidences.append(result.overall_risk.confidence * weight)
            
            if result.overall_risk.level in [RiskLevel.HIGH, RiskLevel.ACTION_REQUIRED]:
                contributing.append(system.value)
            
            all_alerts.extend(result.alerts)
        
        total_weight = sum(
            (self.system_weights.get(s, 1.0) * (0.5 if s in self.experimental_systems else 1.0)) 
            for s in system_results.keys()
        )
        
        composite_score = sum(weighted_scores) / total_weight
        composite_confidence = sum(weighted_confidences) / total_weight
        
        # Critical override: if any system is ACTION_REQUIRED, boost composite to at least HIGH
        # NOTE: Experimental systems are skipped for override to strictly follow reduced weight impact
        max_level = RiskLevel.LOW
        for system, result in system_results.items():
            if system in self.experimental_systems:
                continue
                
            if result.overall_risk.level == RiskLevel.ACTION_REQUIRED:
                max_level = RiskLevel.ACTION_REQUIRED
                break
            elif result.overall_risk.level == RiskLevel.HIGH and max_level != RiskLevel.ACTION_REQUIRED:
                max_level = RiskLevel.HIGH
        
        # Soft override: nudge composite toward the worst non-experimental system's threshold,
        # rather than hard-snapping to it. Non-contact sensors have higher measurement variance
        # than clinical-grade equipment, so a single elevated reading should not dominate the
        # composite score — it should only pull it upward proportionally.
        #
        # Blend factor: 0.3 means the composite moves 30% of the way toward the threshold.
        # This prevents a single noisy sensor from triggering a system-wide alarm while still
        # ensuring genuinely elevated readings raise the overall score.
        BLEND_FACTOR = 0.3
        if max_level == RiskLevel.ACTION_REQUIRED and composite_score < 75:
            composite_score = composite_score + BLEND_FACTOR * (75.0 - composite_score)
        elif max_level == RiskLevel.HIGH and composite_score < 50:
            composite_score = composite_score + BLEND_FACTOR * (50.0 - composite_score)
        
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
        
        # Apply trust adjustment using pre-adjusted confidences (avoid double-penalty)
        trust_confidences = [
            tr.trust_adjusted_confidence
            for tr in trusted_results.values()
            if not tr.was_rejected and tr.risk_result is not None
        ]
        if trust_confidences:
            avg_trust = sum(trust_confidences) / len(trust_confidences)
            # Scale composite confidence by average trust factor
            composite.confidence = float(np.clip(
                composite.confidence * avg_trust,
                0.1, 0.99
            ))
        
        return composite, rejected_systems


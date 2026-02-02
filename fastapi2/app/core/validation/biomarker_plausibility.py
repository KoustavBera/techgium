"""
Biomarker Plausibility Validation Module

Enforces hard physiological constraints on biomarker values.
Detects impossible values, sudden jumps, and internal contradictions.
NO ML/AI - purely physics and physiology based.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

from app.core.extraction.base import BiomarkerSet, Biomarker, PhysiologicalSystem
from app.utils import get_logger

logger = get_logger(__name__)


class ViolationType(str, Enum):
    """Types of plausibility violations."""
    IMPOSSIBLE_VALUE = "impossible_value"           # Outside physiological limits
    SUDDEN_JUMP = "sudden_jump"                     # Non-physiological rate of change
    INTERNAL_CONTRADICTION = "internal_contradiction"  # Contradicts other biomarkers
    MISSING_REQUIRED = "missing_required"           # Essential biomarker missing
    LOW_CONFIDENCE = "low_confidence"               # Below usable threshold


@dataclass
class PlausibilityViolation:
    """A single plausibility violation."""
    biomarker_name: str
    violation_type: ViolationType
    message: str
    severity: float = 0.5  # 0-1, 1 = critical
    actual_value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "biomarker": self.biomarker_name,
            "type": self.violation_type.value,
            "message": self.message,
            "severity": round(self.severity, 2),
            "actual_value": self.actual_value,
            "expected_range": self.expected_range
        }


@dataclass
class PlausibilityResult:
    """Result of biomarker plausibility validation."""
    system: PhysiologicalSystem
    is_valid: bool = True
    overall_plausibility: float = 1.0  # 0-1
    violations: List[PlausibilityViolation] = field(default_factory=list)
    validated_biomarker_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system.value,
            "is_valid": self.is_valid,
            "overall_plausibility": round(self.overall_plausibility, 3),
            "validated_biomarkers": self.validated_biomarker_count,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations]
        }


class BiomarkerPlausibilityValidator:
    """
    Validates biomarkers against hard physiological constraints.
    
    Uses physics and human physiology limits - NO ML/AI.
    """
    
    def __init__(self):
        """Initialize validator with physiological limits."""
        self._validation_count = 0
        self._limits = self._initialize_physiological_limits()
        self._required_biomarkers = self._initialize_required_biomarkers()
        logger.info("BiomarkerPlausibilityValidator initialized (NO-ML)")
    
    def _initialize_physiological_limits(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Define HARD physiological limits (impossible outside these).
        Different from "normal ranges" - these are physical impossibilities.
        """
        return {
            # CNS biomarkers
            "gait_variability": {"hard": (0.0, 1.0), "physiological": (0.01, 0.5)},
            "posture_entropy": {"hard": (0.0, 10.0), "physiological": (0.5, 5.0)},
            "tremor_resting": {"hard": (0.0, 100.0), "physiological": (0.0, 10.0)},
            "tremor_postural": {"hard": (0.0, 100.0), "physiological": (0.0, 10.0)},
            "cns_stability_score": {"hard": (0.0, 100.0), "physiological": (20.0, 100.0)},
            
            # Cardiovascular
            "heart_rate": {"hard": (20.0, 300.0), "physiological": (40.0, 200.0)},
            "hrv_rmssd": {"hard": (0.0, 500.0), "physiological": (5.0, 200.0)},
            "systolic_bp": {"hard": (40.0, 300.0), "physiological": (70.0, 200.0)},
            "diastolic_bp": {"hard": (20.0, 200.0), "physiological": (40.0, 130.0)},
            "chest_micro_motion": {"hard": (0.0, 1.0), "physiological": (0.0001, 0.1)},
            "thoracic_impedance": {"hard": (50.0, 2000.0), "physiological": (200.0, 800.0)},
            
            # Renal
            "fluid_asymmetry_index": {"hard": (0.0, 1.0), "physiological": (0.0, 0.5)},
            "total_body_water_proxy": {"hard": (0.0, 5.0), "physiological": (0.3, 2.0)},
            "extracellular_fluid_ratio": {"hard": (0.0, 1.0), "physiological": (0.2, 0.6)},
            "fluid_overload_index": {"hard": (-2.0, 2.0), "physiological": (-0.5, 0.5)},
            
            # GI
            "abdominal_rhythm_score": {"hard": (0.0, 1.0), "physiological": (0.1, 1.0)},
            "visceral_motion_variance": {"hard": (0.0, 10000.0), "physiological": (1.0, 500.0)},
            "abdominal_respiratory_rate": {"hard": (2.0, 60.0), "physiological": (8.0, 30.0)},
            
            # Skeletal
            "gait_symmetry_ratio": {"hard": (0.0, 2.0), "physiological": (0.5, 1.0)},
            "step_length_symmetry": {"hard": (0.0, 2.0), "physiological": (0.5, 1.0)},
            "stance_stability_score": {"hard": (0.0, 100.0), "physiological": (30.0, 100.0)},
            "sway_velocity": {"hard": (0.0, 1.0), "physiological": (0.0, 0.1)},
            "average_joint_rom": {"hard": (0.0, 3.14), "physiological": (0.1, 2.5)},
            
            # Skin
            "texture_roughness": {"hard": (0.0, 1000.0), "physiological": (1.0, 100.0)},
            "skin_redness": {"hard": (0.0, 1.0), "physiological": (0.1, 0.9)},
            "skin_yellowness": {"hard": (0.0, 1.0), "physiological": (0.05, 0.8)},
            "color_uniformity": {"hard": (0.0, 1.0), "physiological": (0.3, 1.0)},
            "lesion_count": {"hard": (0.0, 1000.0), "physiological": (0.0, 50.0)},
            
            # Eyes
            "blink_rate": {"hard": (0.0, 100.0), "physiological": (3.0, 50.0)},
            "gaze_stability_score": {"hard": (0.0, 100.0), "physiological": (30.0, 100.0)},
            "fixation_duration": {"hard": (10.0, 10000.0), "physiological": (50.0, 2000.0)},
            "saccade_frequency": {"hard": (0.0, 20.0), "physiological": (0.5, 10.0)},
            "eye_symmetry": {"hard": (0.0, 2.0), "physiological": (0.5, 1.0)},
            
            # Nasal/Respiratory
            "breathing_regularity": {"hard": (0.0, 1.0), "physiological": (0.2, 1.0)},
            "respiratory_rate": {"hard": (2.0, 60.0), "physiological": (6.0, 40.0)},
            "breath_depth_index": {"hard": (0.0, 10.0), "physiological": (0.1, 3.0)},
            "airflow_turbulence": {"hard": (0.0, 10.0), "physiological": (0.0, 1.0)},
            
            # Reproductive (proxies)
            "autonomic_imbalance_index": {"hard": (-2.0, 2.0), "physiological": (-1.0, 1.0)},
            "stress_response_proxy": {"hard": (0.0, 100.0), "physiological": (10.0, 90.0)},
            "regional_flow_variability": {"hard": (0.0, 1.0), "physiological": (0.0, 0.5)},
            "thermoregulation_proxy": {"hard": (0.0, 1.0), "physiological": (0.2, 0.8)},
        }
    
    def _initialize_required_biomarkers(self) -> Dict[PhysiologicalSystem, List[str]]:
        """Define minimum required biomarkers per system."""
        return {
            PhysiologicalSystem.CNS: ["gait_variability", "cns_stability_score"],
            PhysiologicalSystem.CARDIOVASCULAR: ["heart_rate"],
            PhysiologicalSystem.RENAL: ["fluid_asymmetry_index"],
            PhysiologicalSystem.GASTROINTESTINAL: ["abdominal_rhythm_score"],
            PhysiologicalSystem.SKELETAL: ["gait_symmetry_ratio"],
            PhysiologicalSystem.SKIN: ["color_uniformity"],
            PhysiologicalSystem.EYES: ["blink_rate"],
            PhysiologicalSystem.NASAL: ["respiratory_rate"],
            PhysiologicalSystem.REPRODUCTIVE: ["autonomic_imbalance_index"],
        }
    
    def validate(
        self,
        biomarker_set: BiomarkerSet,
        previous_set: Optional[BiomarkerSet] = None
    ) -> PlausibilityResult:
        """
        Validate a biomarker set against physiological constraints.
        
        Args:
            biomarker_set: Current biomarker set
            previous_set: Optional previous set for rate-of-change validation
            
        Returns:
            PlausibilityResult with violations
        """
        result = PlausibilityResult(system=biomarker_set.system)
        violations = []
        
        # 1. Check required biomarkers
        required = self._required_biomarkers.get(biomarker_set.system, [])
        present_names = {bm.name for bm in biomarker_set.biomarkers}
        
        for req in required:
            if req not in present_names:
                violations.append(PlausibilityViolation(
                    biomarker_name=req,
                    violation_type=ViolationType.MISSING_REQUIRED,
                    message=f"Required biomarker '{req}' is missing",
                    severity=0.8
                ))
        
        # 2. Validate each biomarker
        for biomarker in biomarker_set.biomarkers:
            bm_violations = self._validate_single_biomarker(biomarker)
            violations.extend(bm_violations)
        
        # 3. Check for sudden jumps if previous set available
        if previous_set is not None:
            jump_violations = self._check_sudden_jumps(biomarker_set, previous_set)
            violations.extend(jump_violations)
        
        # 4. Check internal contradictions within the set
        contradiction_violations = self._check_internal_contradictions(biomarker_set)
        violations.extend(contradiction_violations)
        
        # 5. Compute overall plausibility
        result.violations = violations
        result.validated_biomarker_count = len(biomarker_set.biomarkers)
        
        if violations:
            result.is_valid = not any(v.severity >= 0.8 for v in violations)
            severity_sum = sum(v.severity for v in violations)
            max_possible = len(biomarker_set.biomarkers) + len(required)
            result.overall_plausibility = float(np.clip(
                1.0 - severity_sum / (max_possible + 1), 0, 1
            ))
        else:
            result.is_valid = True
            result.overall_plausibility = 1.0
        
        self._validation_count += 1
        return result
    
    def _validate_single_biomarker(self, biomarker: Biomarker) -> List[PlausibilityViolation]:
        """Validate a single biomarker value."""
        violations = []
        
        limits = self._limits.get(biomarker.name)
        if limits is None:
            return violations  # Unknown biomarker, skip
        
        hard_min, hard_max = limits["hard"]
        physio_min, physio_max = limits["physiological"]
        value = biomarker.value
        
        # Check hard limits (impossible)
        if value < hard_min or value > hard_max:
            violations.append(PlausibilityViolation(
                biomarker_name=biomarker.name,
                violation_type=ViolationType.IMPOSSIBLE_VALUE,
                message=f"{biomarker.name}={value:.3f} is physically impossible",
                severity=1.0,
                actual_value=value,
                expected_range=(hard_min, hard_max)
            ))
        # Check physiological limits (improbable but possible)
        elif value < physio_min or value > physio_max:
            violations.append(PlausibilityViolation(
                biomarker_name=biomarker.name,
                violation_type=ViolationType.IMPOSSIBLE_VALUE,
                message=f"{biomarker.name}={value:.3f} is outside physiological range",
                severity=0.5,
                actual_value=value,
                expected_range=(physio_min, physio_max)
            ))
        
        # Check confidence
        if biomarker.confidence < 0.3:
            violations.append(PlausibilityViolation(
                biomarker_name=biomarker.name,
                violation_type=ViolationType.LOW_CONFIDENCE,
                message=f"{biomarker.name} has very low confidence ({biomarker.confidence:.2f})",
                severity=0.4,
                actual_value=biomarker.confidence
            ))
        
        return violations
    
    def _check_sudden_jumps(
        self,
        current: BiomarkerSet,
        previous: BiomarkerSet
    ) -> List[PlausibilityViolation]:
        """Check for non-physiological rate of change."""
        violations = []
        
        # Maximum allowed change per second for various biomarkers
        max_rates = {
            "heart_rate": 30.0,        # 30 bpm/s is extreme
            "respiratory_rate": 10.0,   # 10 breaths/min/s
            "systolic_bp": 20.0,        # 20 mmHg/s
            "diastolic_bp": 15.0,
            "gait_symmetry_ratio": 0.1,
            "blink_rate": 5.0,
        }
        
        # Get time delta (assume 1 second if unknown)
        time_delta = 1.0
        
        current_map = {bm.name: bm.value for bm in current.biomarkers}
        previous_map = {bm.name: bm.value for bm in previous.biomarkers}
        
        for name, max_rate in max_rates.items():
            if name in current_map and name in previous_map:
                delta = abs(current_map[name] - previous_map[name])
                rate = delta / time_delta
                
                if rate > max_rate:
                    violations.append(PlausibilityViolation(
                        biomarker_name=name,
                        violation_type=ViolationType.SUDDEN_JUMP,
                        message=f"{name} changed too quickly: {delta:.2f} in {time_delta}s",
                        severity=min(rate / max_rate * 0.5, 0.9),
                        actual_value=delta
                    ))
        
        return violations
    
    def _check_internal_contradictions(
        self,
        biomarker_set: BiomarkerSet
    ) -> List[PlausibilityViolation]:
        """Check for contradictions within the biomarker set."""
        violations = []
        bm_map = {bm.name: bm.value for bm in biomarker_set.biomarkers}
        
        # Cardiovascular contradictions
        if "systolic_bp" in bm_map and "diastolic_bp" in bm_map:
            if bm_map["systolic_bp"] <= bm_map["diastolic_bp"]:
                violations.append(PlausibilityViolation(
                    biomarker_name="systolic_bp,diastolic_bp",
                    violation_type=ViolationType.INTERNAL_CONTRADICTION,
                    message="Systolic BP must be greater than Diastolic BP",
                    severity=1.0,
                    actual_value=bm_map["systolic_bp"]
                ))
        
        # Very high HR should correlate with reduced HRV
        if "heart_rate" in bm_map and "hrv_rmssd" in bm_map:
            hr = bm_map["heart_rate"]
            hrv = bm_map["hrv_rmssd"]
            # Physiologically, HR > 150 with HRV > 100 is contradictory
            if hr > 150 and hrv > 100:
                violations.append(PlausibilityViolation(
                    biomarker_name="heart_rate,hrv_rmssd",
                    violation_type=ViolationType.INTERNAL_CONTRADICTION,
                    message=f"High HR ({hr:.0f}) with high HRV ({hrv:.0f}) is contradictory",
                    severity=0.6
                ))
        
        # Gait symmetry vs stability
        if "gait_symmetry_ratio" in bm_map and "stance_stability_score" in bm_map:
            sym = bm_map["gait_symmetry_ratio"]
            stab = bm_map["stance_stability_score"]
            # Very asymmetric gait with perfect stability is suspicious
            if sym < 0.6 and stab > 95:
                violations.append(PlausibilityViolation(
                    biomarker_name="gait_symmetry_ratio,stance_stability_score",
                    violation_type=ViolationType.INTERNAL_CONTRADICTION,
                    message="Asymmetric gait with perfect stability is unlikely",
                    severity=0.4
                ))
        
        return violations
    
    def validate_all(
        self,
        biomarker_sets: List[BiomarkerSet],
        previous_sets: Optional[Dict[PhysiologicalSystem, BiomarkerSet]] = None
    ) -> Dict[PhysiologicalSystem, PlausibilityResult]:
        """Validate all biomarker sets."""
        results = {}
        prev = previous_sets or {}
        
        for bm_set in biomarker_sets:
            results[bm_set.system] = self.validate(
                bm_set, 
                prev.get(bm_set.system)
            )
        
        return results

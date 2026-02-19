"""
Biomarker Plausibility Validation Module

Enforces hard physiological constraints on biomarker values.
Detects impossible values, sudden jumps, and internal contradictions.
NO ML/AI - purely physics and physiology based.

Dynamic ranges: Pass a PatientContext to validate() to get age/gender/activity-
adjusted physiological limits instead of the static adult defaults.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

from app.core.extraction.base import BiomarkerSet, Biomarker, PhysiologicalSystem
from app.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PatientContext:
    """
    Patient demographic and activity context for dynamic biomarker validation.

    Attributes:
        age:           Patient age in years (default 30).
        gender:        'male', 'female', or 'other' (default 'male').
        activity_mode: 'resting' (screening context) or 'post_exercise'.
                       Resting is the correct mode for a kiosk scan.
    """
    age: int = 30
    gender: str = "male"
    activity_mode: str = "resting"  # 'resting' | 'post_exercise'

    def __post_init__(self):
        self.age = max(1, min(int(self.age), 120))
        self.gender = self.gender.lower() if self.gender else "male"
        if self.activity_mode not in ("resting", "post_exercise"):
            self.activity_mode = "resting"


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
        """Initialize validator with static hard limits and required biomarker registry."""
        self._validation_count = 0
        self._static_limits = self._initialize_static_limits()
        self._required_biomarkers = self._initialize_required_biomarkers()
        logger.info("BiomarkerPlausibilityValidator initialized (NO-ML, dynamic ranges enabled)")

    # ------------------------------------------------------------------
    # Dynamic limit calculation
    # ------------------------------------------------------------------

    def _get_limits(
        self, context: Optional["PatientContext"] = None
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Return physiological limits adjusted for the patient's age, gender,
        and activity mode.  Hard (impossible) limits are always static.

        Formulas used:
          - Max HR  : Tanaka et al. (2001) — 208 − 0.7 × age
          - Min HR  : 40 bpm for athletes; 50 bpm for sedentary adults
          - Sys BP  : Upper bound = 120 + 0.5 × max(0, age − 20) mmHg
          - RR      : 10–20 adults; 15–30 children (<18); wider in post-exercise
        """
        limits = {k: dict(v) for k, v in self._static_limits.items()}  # deep copy

        if context is None:
            return limits  # Return static defaults unchanged

        age = context.age
        mode = context.activity_mode  # 'resting' | 'post_exercise'

        # ── Heart Rate ────────────────────────────────────────────────
        max_hr_tanaka = round(208 - 0.7 * age)          # Tanaka formula
        min_hr = 40 if age < 40 else 50                 # Athletes tend to be younger
        if mode == "resting":
            # During a resting kiosk scan, >100 is tachycardia (flag it)
            physio_hr_max = min(100, max_hr_tanaka)      # Never exceed age-predicted max
        else:
            # Post-exercise: allow up to 85% of age-predicted max
            physio_hr_max = round(max_hr_tanaka * 0.85)
        limits["heart_rate"]["physiological"] = (float(min_hr), float(physio_hr_max))
        # Radar HR follows same logic
        limits["radar_heart_rate"]["physiological"] = (float(min_hr), float(physio_hr_max + 20))

        # ── Systolic Blood Pressure ───────────────────────────────────
        # Normal SBP rises ~0.5 mmHg/year after age 20 (Framingham data)
        age_adjustment = max(0, age - 20) * 0.5
        sys_bp_max = round(120 + age_adjustment)         # e.g. 145 at age 70
        sys_bp_max = min(sys_bp_max, 180)                # Cap at hypertensive crisis threshold
        limits["systolic_bp"]["physiological"] = (85.0, float(sys_bp_max))

        # ── Respiratory Rate ──────────────────────────────────────────
        if age < 18:
            rr_min, rr_max = 15.0, 30.0                 # Paediatric range
        elif mode == "post_exercise":
            rr_min, rr_max = 12.0, 40.0                 # Elevated after exercise
        else:
            rr_min, rr_max = 10.0, 20.0                 # Standard adult resting
        limits["respiratory_rate"]["physiological"] = (rr_min, rr_max)
        limits["abdominal_respiratory_rate"]["physiological"] = (rr_min - 4, rr_max + 5)
        limits["radar_respiration_rate"]["physiological"] = (rr_min - 2, rr_max + 5)

        logger.debug(
            f"Dynamic limits for age={age}, gender={context.gender}, mode={mode}: "
            f"HR=({min_hr},{physio_hr_max}), SysBP=(85,{sys_bp_max}), RR=({rr_min},{rr_max})"
        )
        return limits

    def _initialize_static_limits(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        HARD physiological limits — these are physical impossibilities regardless
        of age or activity.  Dynamic adjustments are applied on top in _get_limits().
        """
        return {
            # CNS
            "gait_variability":       {"hard": (0.0, 1.0),      "physiological": (0.01, 0.5)},
            "posture_entropy":        {"hard": (0.0, 10.0),     "physiological": (0.5, 5.0)},
            "tremor_resting":         {"hard": (0.0, 100.0),    "physiological": (0.0, 10.0)},
            "tremor_postural":        {"hard": (0.0, 100.0),    "physiological": (0.0, 10.0)},
            "cns_stability_score":    {"hard": (0.0, 100.0),    "physiological": (20.0, 100.0)},

            # Cardiovascular — physiological bounds overridden by _get_limits()
            "heart_rate":             {"hard": (20.0, 300.0),   "physiological": (50.0, 100.0)},
            "hrv_rmssd":              {"hard": (0.0, 500.0),    "physiological": (5.0, 200.0)},
            "systolic_bp":            {"hard": (50.0, 260.0),   "physiological": (85.0, 140.0)},
            "diastolic_bp":           {"hard": (20.0, 200.0),   "physiological": (40.0, 120.0)},
            "chest_micro_motion":     {"hard": (0.0, 1.0),      "physiological": (0.0001, 0.1)},

            # Radar (Seeed MR60BHA2) — physiological bounds overridden by _get_limits()
            "radar_heart_rate":       {"hard": (20.0, 300.0),   "physiological": (50.0, 120.0)},
            "radar_respiration_rate": {"hard": (2.0, 70.0),     "physiological": (10.0, 20.0)},

            # Thermal (MLX90640)
            "skin_temp_avg":          {"hard": (20.0, 45.0),    "physiological": (28.0, 40.0)},
            "skin_temp_max":          {"hard": (20.0, 50.0),    "physiological": (30.0, 42.0)},
            "thermal_asymmetry":      {"hard": (0.0, 10.0),     "physiological": (0.0, 3.5)},

            # Thermal (ESP32)
            "skin_temperature":       {"hard": (15.0, 45.0),    "physiological": (25.0, 39.0)},
            "skin_temperature_max":   {"hard": (20.0, 50.0),    "physiological": (30.0, 42.0)},
            "inflammation_index":     {"hard": (0.0, 100.0),    "physiological": (0.0, 20.0)},
            "face_mean_temperature":  {"hard": (25.0, 42.0),    "physiological": (30.0, 38.0)},
            "facial_perfusion_temp":  {"hard": (25.0, 42.0),    "physiological": (30.0, 38.0)},
            "microcirculation_temp":  {"hard": (25.0, 42.0),    "physiological": (30.0, 39.0)},
            "thermal_stress_gradient":{"hard": (-5.0, 10.0),    "physiological": (0.0, 3.0)},
            "forehead_temperature":   {"hard": (25.0, 42.0),    "physiological": (33.0, 37.5)},
            "thermal_stability":      {"hard": (0.0, 10.0),     "physiological": (0.0, 1.5)},

            # GI
            "abdominal_rhythm_score":     {"hard": (0.0, 1.0),      "physiological": (0.1, 1.0)},
            "visceral_motion_variance":   {"hard": (0.0, 10000.0),  "physiological": (1.0, 500.0)},
            "abdominal_respiratory_rate": {"hard": (2.0, 60.0),     "physiological": (8.0, 25.0)},

            # Skeletal
            "gait_symmetry_ratio":    {"hard": (0.0, 2.0),      "physiological": (0.5, 1.0)},
            "step_length_symmetry":   {"hard": (0.0, 2.0),      "physiological": (0.5, 1.0)},
            "stance_stability_score": {"hard": (0.0, 100.0),    "physiological": (30.0, 100.0)},
            "sway_velocity":          {"hard": (0.0, 1.0),      "physiological": (0.0, 0.1)},
            "average_joint_rom":      {"hard": (0.0, 3.14),     "physiological": (0.1, 2.5)},

            # Skin
            "texture_roughness":      {"hard": (0.0, 1000.0),   "physiological": (0.0, 50.0)},
            "skin_redness":           {"hard": (0.0, 1.0),      "physiological": (0.0, 0.7)},
            "skin_yellowness":        {"hard": (0.0, 1.0),      "physiological": (0.0, 0.7)},
            "color_uniformity":       {"hard": (0.0, 1.0),      "physiological": (0.1, 1.0)},
            "lesion_count":           {"hard": (0.0, 1000.0),   "physiological": (0.0, 50.0)},

            # Eyes
            "blink_rate":             {"hard": (0.0, 100.0),    "physiological": (2.0, 50.0)},
            "blink_count":            {"hard": (0.0, 1000.0),   "physiological": (0.0, 100.0)},
            "gaze_stability_score":   {"hard": (0.0, 100.0),    "physiological": (30.0, 100.0)},
            "fixation_duration":      {"hard": (10.0, 10000.0), "physiological": (50.0, 2000.0)},
            "saccade_frequency":      {"hard": (0.0, 20.0),     "physiological": (0.5, 10.0)},
            "eye_symmetry":           {"hard": (0.0, 2.0),      "physiological": (0.5, 1.0)},

            # Nasal/Respiratory — physiological RR overridden by _get_limits()
            "breathing_regularity":   {"hard": (0.0, 1.0),      "physiological": (0.2, 1.0)},
            "respiratory_rate":       {"hard": (2.0, 70.0),     "physiological": (10.0, 20.0)},
            "breath_depth_index":     {"hard": (0.0, 10.0),     "physiological": (0.1, 3.0)},
            "airflow_turbulence":     {"hard": (0.0, 10.0),     "physiological": (0.0, 1.0)},

            # Reproductive (proxies)
            "autonomic_imbalance_index": {"hard": (-2.0, 2.0),  "physiological": (-1.0, 1.0)},
            "stress_response_proxy":     {"hard": (0.0, 100.0), "physiological": (10.0, 90.0)},
            "regional_flow_variability": {"hard": (0.0, 1.0),   "physiological": (0.0, 0.5)},
            "thermoregulation_proxy":    {"hard": (0.0, 1.0),   "physiological": (0.2, 0.8)},
        }
    
    def _initialize_required_biomarkers(self) -> Dict[PhysiologicalSystem, List[str]]:
        """Define minimum required biomarkers per system."""
        return {
            PhysiologicalSystem.CNS: ["gait_variability", "cns_stability_score"],
            PhysiologicalSystem.CARDIOVASCULAR: ["heart_rate"],
            PhysiologicalSystem.GASTROINTESTINAL: ["abdominal_rhythm_score"],
            PhysiologicalSystem.SKELETAL: ["gait_symmetry_ratio"],
            PhysiologicalSystem.SKIN: ["color_uniformity"],
            PhysiologicalSystem.EYES: ["blink_rate"],
            PhysiologicalSystem.NASAL: ["respiratory_rate"],
            PhysiologicalSystem.REPRODUCTIVE: ["autonomic_imbalance_index"],
        }
    
    def _get_biomarker_importance(self, name: str) -> float:
        """Get importance weight for a biomarker (default 1.0)."""
        # Critical biomarkers have higher weight for scoring
        high_importance = {
            "heart_rate": 2.0,
            "systolic_bp": 2.0,
            "respiratory_rate": 2.0,
            "cns_stability_score": 2.0,
            "fluid_overload_index": 1.5
        }
        return high_importance.get(name, 1.0)
    
    def validate(
        self,
        biomarker_set: BiomarkerSet,
        previous_set: Optional[BiomarkerSet] = None,
        context: Optional["PatientContext"] = None,
    ) -> PlausibilityResult:
        """
        Validate a biomarker set against physiological constraints.

        Args:
            biomarker_set: Current biomarker set
            previous_set:  Optional previous set for rate-of-change validation
            context:       Optional PatientContext for dynamic limit calculation.
                           If None, static adult defaults are used.

        Returns:
            PlausibilityResult with violations
        """
        limits = self._get_limits(context)
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
        
        # 2. Validate each biomarker (using dynamic limits)
        for biomarker in biomarker_set.biomarkers:
            bm_violations = self._validate_single_biomarker(biomarker, limits)
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
            
            # Weighted scoring
            weighted_severity = sum(v.severity * self._get_biomarker_importance(v.biomarker_name) 
                                 for v in violations)
            total_weight = sum(self._get_biomarker_importance(bm.name) for bm in biomarker_set.biomarkers)
            
            # Add weight for missing required biomarkers (which aren't in biomarkers list)
            # We treat missing required as having max severity on a critical item
            missing_count = len([v for v in violations if v.violation_type == ViolationType.MISSING_REQUIRED])
            total_weight += missing_count * 2.0 
            
            result.overall_plausibility = float(np.clip(
                1.0 - weighted_severity / max(total_weight, 1.0), 0, 1
            ))
        else:
            result.is_valid = True
            result.overall_plausibility = 1.0
        
        self._validation_count += 1
        return result
    
    def _validate_single_biomarker(
        self,
        biomarker: Biomarker,
        limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ) -> List[PlausibilityViolation]:
        """Validate a single biomarker value against the provided (possibly dynamic) limits."""
        if limits is None:
            limits = self._static_limits
        violations = []
        
        # Handle NaN/Inf
        if np.isnan(biomarker.value) or np.isinf(biomarker.value):
            return [PlausibilityViolation(
                biomarker_name=biomarker.name,
                violation_type=ViolationType.IMPOSSIBLE_VALUE,
                message=f"{biomarker.name} is NaN/Inf",
                severity=0.9,
                actual_value=0.0 # Placeholder
            )]
        
        biomarker_limits = limits.get(biomarker.name)
        if biomarker_limits is None:
            return violations  # Unknown biomarker, skip
        
        hard_min, hard_max = biomarker_limits["hard"]
        physio_min, physio_max = biomarker_limits["physiological"]
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
            "heart_rate": 10.0,        # 10 bpm/s is extreme but possible in startle
            "respiratory_rate": 5.0,   # 5 breaths/min/s
            "systolic_bp": 10.0,       # 10 mmHg/s
            "diastolic_bp": 8.0,
            "gait_symmetry_ratio": 0.1,
            "blink_rate": 5.0,
        }
        
        # Calculate actual time delta from extraction times (ms to s)
        time_delta = abs(current.extraction_time_ms - previous.extraction_time_ms) / 1000.0
        
        if time_delta <= 0 or time_delta > 3600:
             # Fallback if timestamps invalid or too far apart (start new session)
             # If delta is 0, we can't calculate rate, so skip.
             # If > 1 hour, rate check is meaningless.
             return []
        
        current_map = {bm.name: bm.value for bm in current.biomarkers}
        previous_map = {bm.name: bm.value for bm in previous.biomarkers}
        
        for name, max_rate in max_rates.items():
            if name in current_map and name in previous_map:
                delta = abs(current_map[name] - previous_map[name])
                # Fix: Use small epsilon for time_delta to avoid division by zero, 
                # but don't force min 1.0s which under-reports rate for fast updates
                effective_time = max(time_delta, 0.001) 
                rate = delta / effective_time
                
                if rate > max_rate:
                    violations.append(PlausibilityViolation(
                        biomarker_name=name,
                        violation_type=ViolationType.SUDDEN_JUMP,
                        message=f"{name} changed too quickly: {delta:.2f} in {time_delta:.2f}s",
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
        if  "systolic_bp" in bm_map and "diastolic_bp" in bm_map:
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
            # Physiological empirical limit: HRV shouldn't be massive during Tachycardia
            # Fix: Flag if HR > 140 and HRV is suspiciously high (>80ms)
            
            if hr > 140 and hrv > 80: 
                violations.append(PlausibilityViolation(
                    biomarker_name="heart_rate,hrv_rmssd",
                    violation_type=ViolationType.INTERNAL_CONTRADICTION,
                    message=f"High HR ({hr:.0f}) with High HRV ({hrv:.0f}) is contradictory",
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
        previous_sets: Optional[Dict[PhysiologicalSystem, BiomarkerSet]] = None,
        context: Optional["PatientContext"] = None,
    ) -> Dict[PhysiologicalSystem, PlausibilityResult]:
        """Validate all biomarker sets with optional patient context for dynamic limits."""
        results = {}
        prev = previous_sets or {}
        # Pre-compute limits once for all systems
        limits = self._get_limits(context)

        for bm_set in biomarker_sets:
            results[bm_set.system] = self.validate(
                bm_set,
                prev.get(bm_set.system),
                context=context,
            )

        return results
        
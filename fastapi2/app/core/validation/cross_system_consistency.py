"""
Cross-System Consistency Checks Module

Validates agreement across physiological systems.
Detects contradictions between systems that indicate measurement errors.
NO ML/AI - purely physiological reasoning.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem
from app.core.inference.risk_engine import SystemRiskResult, RiskLevel
from app.utils import get_logger

logger = get_logger(__name__)


class InconsistencyType(str, Enum):
    """Types of cross-system inconsistencies."""
    PHYSIOLOGICAL_MISMATCH = "physiological_mismatch"
    RISK_LEVEL_CONFLICT = "risk_level_conflict"
    DIRECTIONAL_CONFLICT = "directional_conflict"
    MISSING_CORRELATION = "missing_correlation"


@dataclass
class CrossSystemInconsistency:
    """A single cross-system inconsistency."""
    systems: Tuple[str, str]
    inconsistency_type: InconsistencyType
    message: str
    severity: float = 0.5  # 0-1
    biomarkers_involved: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "systems": self.systems,
            "type": self.inconsistency_type.value,
            "message": self.message,
            "severity": round(self.severity, 2),
            "biomarkers": self.biomarkers_involved
        }


@dataclass
class ConsistencyResult:
    """Result of cross-system consistency checking."""
    overall_consistency: float = 1.0  # 0-1
    inconsistencies: List[CrossSystemInconsistency] = field(default_factory=list)
    systems_checked: int = 0
    cross_checks_performed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_consistency": round(self.overall_consistency, 3),
            "inconsistency_count": len(self.inconsistencies),
            "systems_checked": self.systems_checked,
            "cross_checks_performed": self.cross_checks_performed,
            "inconsistencies": [i.to_dict() for i in self.inconsistencies]
        }


class CrossSystemConsistencyChecker:
    """
    Checks for consistency across physiological systems.
    
    Validates that biomarkers and risk levels agree across systems
    based on known physiological relationships.
    NO ML/AI - based on medical/physiological knowledge.
    """
    
    def __init__(self):
        """Initialize consistency checker."""
        self._check_count = 0
        self._rules = self._initialize_consistency_rules()
        logger.info("CrossSystemConsistencyChecker initialized (NO-ML)")
    
    def _initialize_consistency_rules(self) -> List[Dict[str, Any]]:
        """
        Define physiological consistency rules between systems.
        Based on known medical relationships.
        """
        return [
            # Cardiovascular stress should correlate with CNS activity
            {
                "name": "cardio_cns_stress",
                "systems": (PhysiologicalSystem.CARDIOVASCULAR, PhysiologicalSystem.CNS),
                "description": "High cardiac stress should correlate with CNS indicators",
                "check_fn": self._check_cardio_cns_consistency
            },
            # Fluid overload (renal) should affect thoracic impedance (cardio)
            {
                "name": "renal_cardio_fluid",
                "systems": (PhysiologicalSystem.RENAL, PhysiologicalSystem.CARDIOVASCULAR),
                "description": "Fluid distribution should correlate with thoracic impedance",
                "check_fn": self._check_renal_cardio_consistency
            },
            # Respiratory (nasal) should correlate with GI abdominal motion
            {
                "name": "nasal_gi_respiratory",
                "systems": (PhysiologicalSystem.NASAL, PhysiologicalSystem.GASTROINTESTINAL),
                "description": "Respiratory patterns should be consistent across systems",
                "check_fn": self._check_respiratory_consistency
            },
            # Skeletal stability should correlate with CNS stability
            {
                "name": "skeletal_cns_stability",
                "systems": (PhysiologicalSystem.SKELETAL, PhysiologicalSystem.CNS),
                "description": "Motor control should be consistent between systems",
                "check_fn": self._check_motor_control_consistency
            },
            # High autonomic stress (reproductive proxy) should correlate with HR
            {
                "name": "reproductive_cardio_autonomic",
                "systems": (PhysiologicalSystem.REPRODUCTIVE, PhysiologicalSystem.CARDIOVASCULAR),
                "description": "Autonomic indicators should correlate with heart rate",
                "check_fn": self._check_autonomic_cardiovascular_consistency
            },
            # Eye gaze stability should correlate with CNS stability
            {
                "name": "eyes_cns_coordination",
                "systems": (PhysiologicalSystem.EYES, PhysiologicalSystem.CNS),
                "description": "Oculomotor control should correlate with CNS function",
                "check_fn": self._check_eyes_cns_consistency
            },
        ]
    
    def check_consistency(
        self,
        biomarker_sets: Dict[PhysiologicalSystem, BiomarkerSet],
        risk_results: Optional[Dict[PhysiologicalSystem, SystemRiskResult]] = None
    ) -> ConsistencyResult:
        """
        Perform cross-system consistency checks.
        
        Args:
            biomarker_sets: Biomarkers from all systems
            risk_results: Optional risk results for risk-level consistency
            
        Returns:
            ConsistencyResult with any inconsistencies found
        """
        result = ConsistencyResult()
        result.systems_checked = len(biomarker_sets)
        inconsistencies = []
        checks_performed = 0
        
        # Build biomarker lookup
        all_biomarkers = {}
        for system, bm_set in biomarker_sets.items():
            for bm in bm_set.biomarkers:
                all_biomarkers[bm.name] = bm.value
        
        # Run all consistency rules
        for rule in self._rules:
            sys1, sys2 = rule["systems"]
            if sys1 in biomarker_sets and sys2 in biomarker_sets:
                checks_performed += 1
                check_fn = rule["check_fn"]
                
                bm1 = {bm.name: bm.value for bm in biomarker_sets[sys1].biomarkers}
                bm2 = {bm.name: bm.value for bm in biomarker_sets[sys2].biomarkers}
                
                rule_inconsistencies = check_fn(bm1, bm2, sys1, sys2)
                inconsistencies.extend(rule_inconsistencies)
        
        # Check risk level consistency if available
        if risk_results:
            risk_inconsistencies = self._check_risk_level_consistency(risk_results)
            inconsistencies.extend(risk_inconsistencies)
            checks_performed += 1
        
        # Compute overall consistency
        result.inconsistencies = inconsistencies
        result.cross_checks_performed = checks_performed
        
        if inconsistencies:
            severity_sum = sum(i.severity for i in inconsistencies)
            result.overall_consistency = float(np.clip(
                1.0 - severity_sum / (checks_performed + 1), 0, 1
            ))
        else:
            result.overall_consistency = 1.0
        
        self._check_count += 1
        return result
    
    def _check_cardio_cns_consistency(
        self,
        cardio_bm: Dict[str, float],
        cns_bm: Dict[str, float],
        sys1: PhysiologicalSystem,
        sys2: PhysiologicalSystem
    ) -> List[CrossSystemInconsistency]:
        """Check cardiovascular vs CNS consistency."""
        inconsistencies = []
        
        hr = cardio_bm.get("heart_rate")
        cns_stability = cns_bm.get("cns_stability_score")
        gait_var = cns_bm.get("gait_variability")
        
        # If HR is very high (stress), CNS stability should decrease
        if hr and cns_stability:
            if hr > 120 and cns_stability > 90:
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.PHYSIOLOGICAL_MISMATCH,
                    message=f"High HR ({hr:.0f}) with perfect CNS stability ({cns_stability:.0f}) is unusual",
                    severity=0.4,
                    biomarkers_involved=["heart_rate", "cns_stability_score"]
                ))
        
        # Very low HRV (stress) should correlate with decreased motor control
        hrv = cardio_bm.get("hrv_rmssd")
        if hrv and gait_var:
            if hrv < 15 and gait_var < 0.03:  # Low HRV + perfect gait
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.MISSING_CORRELATION,
                    message="Very low HRV typically correlates with increased gait variability",
                    severity=0.3,
                    biomarkers_involved=["hrv_rmssd", "gait_variability"]
                ))
        
        return inconsistencies
    
    def _check_renal_cardio_consistency(
        self,
        renal_bm: Dict[str, float],
        cardio_bm: Dict[str, float],
        sys1: PhysiologicalSystem,
        sys2: PhysiologicalSystem
    ) -> List[CrossSystemInconsistency]:
        """Check renal vs cardiovascular consistency."""
        inconsistencies = []
        
        fluid_overload = renal_bm.get("fluid_overload_index")
        thoracic_imp = cardio_bm.get("thoracic_impedance")
        
        # Fluid overload (positive) should decrease thoracic impedance
        if fluid_overload is not None and thoracic_imp is not None:
            if fluid_overload > 0.3 and thoracic_imp > 600:
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.DIRECTIONAL_CONFLICT,
                    message="Fluid overload should decrease thoracic impedance",
                    severity=0.5,
                    biomarkers_involved=["fluid_overload_index", "thoracic_impedance"]
                ))
        
        return inconsistencies
    
    def _check_respiratory_consistency(
        self,
        nasal_bm: Dict[str, float],
        gi_bm: Dict[str, float],
        sys1: PhysiologicalSystem,
        sys2: PhysiologicalSystem
    ) -> List[CrossSystemInconsistency]:
        """Check respiratory consistency across systems."""
        inconsistencies = []
        
        nasal_rr = nasal_bm.get("respiratory_rate")
        abdominal_rr = gi_bm.get("abdominal_respiratory_rate")
        
        # Respiratory rates from different sources should match
        if nasal_rr and abdominal_rr:
            diff = abs(nasal_rr - abdominal_rr)
            if diff > 5:  # More than 5 breaths/min difference
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.PHYSIOLOGICAL_MISMATCH,
                    message=f"Respiratory rate mismatch: nasal={nasal_rr:.1f}, abdominal={abdominal_rr:.1f}",
                    severity=min(diff / 10, 0.8),
                    biomarkers_involved=["respiratory_rate", "abdominal_respiratory_rate"]
                ))
        
        return inconsistencies
    
    def _check_motor_control_consistency(
        self,
        skeletal_bm: Dict[str, float],
        cns_bm: Dict[str, float],
        sys1: PhysiologicalSystem,
        sys2: PhysiologicalSystem
    ) -> List[CrossSystemInconsistency]:
        """Check motor control consistency."""
        inconsistencies = []
        
        gait_sym = skeletal_bm.get("gait_symmetry_ratio")
        stance_stab = skeletal_bm.get("stance_stability_score")
        cns_stab = cns_bm.get("cns_stability_score")
        tremor = cns_bm.get("tremor_resting")
        
        # Significant tremor should affect gait symmetry
        if tremor and gait_sym:
            if tremor > 5 and gait_sym > 0.95:  # High tremor + perfect gait
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.MISSING_CORRELATION,
                    message="Significant tremor should affect gait symmetry",
                    severity=0.4,
                    biomarkers_involved=["tremor_resting", "gait_symmetry_ratio"]
                ))
        
        # Motor stability should correlate
        if stance_stab and cns_stab:
            diff = abs(stance_stab - cns_stab)
            if diff > 30:  # Very different stability scores
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.PHYSIOLOGICAL_MISMATCH,
                    message=f"Stability scores differ significantly: skeletal={stance_stab:.0f}, CNS={cns_stab:.0f}",
                    severity=0.3,
                    biomarkers_involved=["stance_stability_score", "cns_stability_score"]
                ))
        
        return inconsistencies
    
    def _check_autonomic_cardiovascular_consistency(
        self,
        repro_bm: Dict[str, float],
        cardio_bm: Dict[str, float],
        sys1: PhysiologicalSystem,
        sys2: PhysiologicalSystem
    ) -> List[CrossSystemInconsistency]:
        """Check autonomic vs cardiovascular consistency."""
        inconsistencies = []
        
        stress = repro_bm.get("stress_response_proxy")
        hr = cardio_bm.get("heart_rate")
        hrv = cardio_bm.get("hrv_rmssd")
        
        # High stress proxy should correlate with elevated HR
        if stress and hr:
            if stress > 70 and hr < 60:  # High stress + low HR
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.DIRECTIONAL_CONFLICT,
                    message="High autonomic stress typically elevates heart rate",
                    severity=0.4,
                    biomarkers_involved=["stress_response_proxy", "heart_rate"]
                ))
        
        # High stress should decrease HRV
        if stress and hrv:
            if stress > 70 and hrv > 80:  # High stress + high HRV
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.PHYSIOLOGICAL_MISMATCH,
                    message="High stress typically decreases HRV",
                    severity=0.5,
                    biomarkers_involved=["stress_response_proxy", "hrv_rmssd"]
                ))
        
        return inconsistencies
    
    def _check_eyes_cns_consistency(
        self,
        eyes_bm: Dict[str, float],
        cns_bm: Dict[str, float],
        sys1: PhysiologicalSystem,
        sys2: PhysiologicalSystem
    ) -> List[CrossSystemInconsistency]:
        """Check eye vs CNS consistency."""
        inconsistencies = []
        
        gaze_stab = eyes_bm.get("gaze_stability_score")
        cns_stab = cns_bm.get("cns_stability_score")
        
        # Gaze stability is neurologically controlled
        if gaze_stab and cns_stab:
            diff = abs(gaze_stab - cns_stab)
            if diff > 40:  # Very different
                inconsistencies.append(CrossSystemInconsistency(
                    systems=(sys1.value, sys2.value),
                    inconsistency_type=InconsistencyType.PHYSIOLOGICAL_MISMATCH,
                    message=f"Oculomotor control differs from CNS stability: eyes={gaze_stab:.0f}, CNS={cns_stab:.0f}",
                    severity=0.3,
                    biomarkers_involved=["gaze_stability_score", "cns_stability_score"]
                ))
        
        return inconsistencies
    
    def _check_risk_level_consistency(
        self,
        risk_results: Dict[PhysiologicalSystem, SystemRiskResult]
    ) -> List[CrossSystemInconsistency]:
        """Check for contradictory risk levels across systems."""
        inconsistencies = []
        
        # Get risk levels
        levels = {sys: res.overall_risk.level for sys, res in risk_results.items()}
        
        # Check for extreme contradictions
        # E.g., Critical CNS with Low Cardiovascular is physiologically unusual
        if PhysiologicalSystem.CNS in levels and PhysiologicalSystem.CARDIOVASCULAR in levels:
            cns = levels[PhysiologicalSystem.CNS]
            cardio = levels[PhysiologicalSystem.CARDIOVASCULAR]
            
            if cns == RiskLevel.CRITICAL and cardio == RiskLevel.LOW:
                inconsistencies.append(CrossSystemInconsistency(
                    systems=("central_nervous_system", "cardiovascular"),
                    inconsistency_type=InconsistencyType.RISK_LEVEL_CONFLICT,
                    message="Critical CNS risk with low cardiovascular risk is unusual",
                    severity=0.4,
                    biomarkers_involved=[]
                ))
        
        return inconsistencies

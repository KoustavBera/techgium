"""
Confidence Calibration Module

Adjusts and calibrates confidence scores for risk predictions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem
from app.core.inference.risk_engine import SystemRiskResult, RiskScore

if TYPE_CHECKING:
    from app.core.validation.trust_envelope import TrustEnvelope

from app.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationFactors:
    """Factors that affect confidence calibration."""
    data_completeness: float = 1.0  # 0-1, how much data was available
    sensor_quality: float = 1.0     # 0-1, quality of sensor readings
    temporal_consistency: float = 1.0  # 0-1, consistency over time
    cross_validation: float = 1.0   # 0-1, agreement between modalities
    
    def aggregate(self) -> float:
        """Compute aggregate calibration factor."""
        factors = [
            self.data_completeness,
            self.sensor_quality,
            self.temporal_consistency,
            self.cross_validation
        ]
        # Geometric mean for multiplicative combination
        return float(np.power(np.prod(factors), 1/len(factors)))


class ConfidenceCalibrator:
    """
    Calibrates confidence scores based on data quality and consistency.
    
    Applies adjustments to raw confidence values from extractors and
    risk engine to produce calibrated confidence estimates.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        self._calibration_count = 0
        self._system_baselines = self._initialize_baselines()
        logger.info("ConfidenceCalibrator initialized")
    
    def _initialize_baselines(self) -> Dict[PhysiologicalSystem, float]:
        """
        Initialize baseline confidence multipliers for each system.
        
        Some systems have inherently lower confidence due to measurement challenges.
        """
        return {
            PhysiologicalSystem.CNS: 0.85,
            PhysiologicalSystem.CARDIOVASCULAR: 0.90,

            PhysiologicalSystem.GASTROINTESTINAL: 0.65,
            PhysiologicalSystem.SKELETAL: 0.88,
            PhysiologicalSystem.SKIN: 0.75,
            PhysiologicalSystem.EYES: 0.72,
            PhysiologicalSystem.NASAL: 0.68,
            PhysiologicalSystem.REPRODUCTIVE: 0.55,  # Indirect proxies only
        }
    
    def calibrate_biomarker_confidence(
        self,
        biomarker_set: BiomarkerSet,
        factors: Optional[CalibrationFactors] = None
    ) -> BiomarkerSet:
        """
        Calibrate confidence scores in a biomarker set.
        
        Args:
            biomarker_set: Original biomarker set
            factors: Calibration factors (uses defaults if None)
            
        Returns:
            BiomarkerSet with calibrated confidence values
        """
        if factors is None:
            factors = self._estimate_factors(biomarker_set)
        
        calibration_multiplier = factors.aggregate()
        baseline = self._system_baselines.get(biomarker_set.system, 0.8)
        
        for biomarker in biomarker_set.biomarkers:
            original = biomarker.confidence
            calibrated = original * calibration_multiplier * baseline
            biomarker.confidence = float(np.clip(calibrated, 0.1, 0.99))
        
        self._calibration_count += 1
        return biomarker_set
    
    def calibrate_risk_result(
        self,
        risk_result: SystemRiskResult,
        factors: Optional[CalibrationFactors] = None
    ) -> SystemRiskResult:
        """
        Calibrate confidence scores in a risk result.
        
        Args:
            risk_result: Original risk result
            factors: Calibration factors
            
        Returns:
            SystemRiskResult with calibrated confidence values
        """
        if factors is None:
            factors = CalibrationFactors()
        
        calibration_multiplier = factors.aggregate()
        baseline = self._system_baselines.get(risk_result.system, 0.8)
        final_multiplier = calibration_multiplier * baseline
        
        # Calibrate overall risk
        risk_result.overall_risk.confidence = float(np.clip(
            risk_result.overall_risk.confidence * final_multiplier,
            0.1, 0.99
        ))
        
        # Calibrate sub-risks
        for sub_risk in risk_result.sub_risks:
            sub_risk.confidence = float(np.clip(
                sub_risk.confidence * final_multiplier,
                0.1, 0.99
            ))
        
        self._calibration_count += 1
        return risk_result
    
    def _estimate_factors(self, biomarker_set: BiomarkerSet) -> CalibrationFactors:
        """
        Estimate calibration factors from biomarker set metadata.
        
        Args:
            biomarker_set: BiomarkerSet to analyze
            
        Returns:
            Estimated CalibrationFactors
        """
        # Data completeness based on number of biomarkers
        expected_biomarkers = {
            PhysiologicalSystem.CNS: 5,
            PhysiologicalSystem.CARDIOVASCULAR: 5,
            PhysiologicalSystem.GASTROINTESTINAL: 3,
            PhysiologicalSystem.SKELETAL: 5,
            PhysiologicalSystem.SKIN: 5,
            PhysiologicalSystem.EYES: 5,
            PhysiologicalSystem.NASAL: 4,
            PhysiologicalSystem.REPRODUCTIVE: 4,
        }
        expected = expected_biomarkers.get(biomarker_set.system, 4)
        actual = len(biomarker_set.biomarkers)
        data_completeness = min(actual / expected, 1.0)
        
        # Sensor quality from average biomarker confidence
        if biomarker_set.biomarkers:
            sensor_quality = np.mean([bm.confidence for bm in biomarker_set.biomarkers])
        else:
            sensor_quality = 0.5
        
        # Check for simulated data flag
        is_simulated = biomarker_set.metadata.get("simulated", False)
        if is_simulated:
            sensor_quality *= 0.5
        
        return CalibrationFactors(
            data_completeness=float(data_completeness),
            sensor_quality=float(sensor_quality),
            temporal_consistency=0.85,  # Default, would need time-series data
            cross_validation=0.80  # Default, would need multi-modal comparison
        )
    
    def compute_uncertainty_bounds(
        self,
        risk_score: RiskScore,
        confidence_level: float = 0.95
    ) -> tuple:
        """
        Compute uncertainty bounds for a risk score.
        
        Args:
            risk_score: The risk score to analyze
            confidence_level: Confidence interval (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Approximate uncertainty based on confidence
        # Lower confidence = wider bounds
        uncertainty = (1 - risk_score.confidence) * 30  # Max ±30 points
        
        # Adjust for confidence level
        z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
        margin = uncertainty * z_score / 1.96
        
        lower = max(0, risk_score.score - margin)
        upper = min(100, risk_score.score + margin)
        
        return (round(lower, 1), round(upper, 1))
    
    def calibrate_with_trust(
        self,
        risk_result: SystemRiskResult,
        trust_envelope: Optional[TrustEnvelope] = None,
        factors: Optional[CalibrationFactors] = None
    ) -> SystemRiskResult:
        """
        Calibrate confidence with trust envelope integration.
        
        Applies both standard calibration and trust envelope penalties.
        
        Args:
            risk_result: Original risk result
            trust_envelope: Optional TrustEnvelope from validation layer
            factors: Optional CalibrationFactors
            
        Returns:
            SystemRiskResult with calibrated confidence values
        """
        # First apply standard calibration
        calibrated = self.calibrate_risk_result(risk_result, factors)
        
        # Then apply trust envelope penalty if provided
        if trust_envelope is not None:
            # Apply trust envelope confidence adjustment
            calibrated.overall_risk.confidence = trust_envelope.get_adjusted_confidence(
                calibrated.overall_risk.confidence
            )
            
            # Apply to sub-risks as well
            for sub_risk in calibrated.sub_risks:
                sub_risk.confidence = trust_envelope.get_adjusted_confidence(
                    sub_risk.confidence
                )
        
        return calibrated

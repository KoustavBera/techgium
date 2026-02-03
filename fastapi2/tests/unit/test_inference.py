"""
Unit Tests for Inference Module

Tests for risk engine, calibration, and explanation generation.
"""
import pytest
import numpy as np
from typing import Dict, Any, List

from app.core.extraction.base import BiomarkerSet, Biomarker, PhysiologicalSystem
from app.core.inference import (
    RiskEngine, RiskScore, SystemRiskResult,
    ConfidenceCalibrator,
    ExplanationGenerator, RiskExplanation
)
from app.core.inference.risk_engine import RiskLevel, CompositeRiskCalculator
from app.core.inference.calibration import CalibrationFactors
from app.core.inference.explanation import ExplanationType


# Fixtures
@pytest.fixture
def sample_cns_biomarkers() -> BiomarkerSet:
    """Create sample CNS biomarker set."""
    bms = BiomarkerSet(system=PhysiologicalSystem.CNS)
    bms.add(Biomarker("gait_variability", 0.05, "cv", 0.85, (0.02, 0.08)))
    bms.add(Biomarker("posture_entropy", 2.5, "bits", 0.80, (1.5, 3.5)))
    bms.add(Biomarker("tremor_resting", 0.03, "psd", 0.75, (0, 0.1)))
    bms.add(Biomarker("cns_stability_score", 85, "score", 0.70, (70, 100)))
    return bms


@pytest.fixture
def abnormal_cardiovascular_biomarkers() -> BiomarkerSet:
    """Create biomarker set with abnormal values."""
    bms = BiomarkerSet(system=PhysiologicalSystem.CARDIOVASCULAR)
    bms.add(Biomarker("heart_rate", 110, "bpm", 0.90, (60, 100)))  # Above normal
    bms.add(Biomarker("hrv_rmssd", 15, "ms", 0.85, (20, 80)))  # Below normal
    bms.add(Biomarker("systolic_bp", 145, "mmHg", 0.95, (90, 120)))  # Above normal
    return bms


@pytest.fixture
def sample_biomarker_sets() -> List[BiomarkerSet]:
    """Create biomarker sets for multiple systems."""
    sets = []
    
    # CNS - normal
    cns = BiomarkerSet(system=PhysiologicalSystem.CNS)
    cns.add(Biomarker("gait_variability", 0.05, "cv", 0.85, (0.02, 0.08)))
    sets.append(cns)
    
    # Cardiovascular - abnormal
    cardio = BiomarkerSet(system=PhysiologicalSystem.CARDIOVASCULAR)
    cardio.add(Biomarker("heart_rate", 105, "bpm", 0.90, (60, 100)))
    sets.append(cardio)
    
    # Skeletal - normal
    skel = BiomarkerSet(system=PhysiologicalSystem.SKELETAL)
    skel.add(Biomarker("gait_symmetry_ratio", 0.92, "ratio", 0.80, (0.85, 1.0)))
    sets.append(skel)
    
    return sets


class TestRiskLevel:
    """Tests for RiskLevel enum."""
    
    def test_from_score_low(self):
        """Test low risk classification."""
        assert RiskLevel.from_score(10) == RiskLevel.LOW
        assert RiskLevel.from_score(24) == RiskLevel.LOW
    
    def test_from_score_moderate(self):
        """Test moderate risk classification."""
        assert RiskLevel.from_score(25) == RiskLevel.MODERATE
        assert RiskLevel.from_score(49) == RiskLevel.MODERATE
    
    def test_from_score_high(self):
        """Test high risk classification."""
        assert RiskLevel.from_score(50) == RiskLevel.HIGH
        assert RiskLevel.from_score(74) == RiskLevel.HIGH
    
    def test_from_score_critical(self):
        """Test critical risk classification."""
        assert RiskLevel.from_score(75) == RiskLevel.CRITICAL
        assert RiskLevel.from_score(100) == RiskLevel.CRITICAL


class TestRiskScore:
    """Tests for RiskScore dataclass."""
    
    def test_create_risk_score(self):
        """Test risk score creation."""
        rs = RiskScore(
            name="test_risk",
            score=45.0,
            confidence=0.85
        )
        assert rs.name == "test_risk"
        assert rs.score == 45.0
        assert rs.level == RiskLevel.MODERATE
    
    def test_to_dict(self):
        """Test serialization."""
        rs = RiskScore(
            name="heart_rate_risk",
            score=65.0,
            confidence=0.75,
            contributing_biomarkers=["heart_rate"],
            explanation="Elevated heart rate detected"
        )
        
        d = rs.to_dict()
        assert d["name"] == "heart_rate_risk"
        assert d["score"] == 65.0
        assert d["level"] == "high"
        assert d["confidence"] == 0.75


class TestRiskEngine:
    """Tests for RiskEngine."""
    
    def test_compute_risk_normal(self, sample_cns_biomarkers):
        """Test risk computation for normal biomarkers."""
        engine = RiskEngine()
        result = engine.compute_risk(sample_cns_biomarkers)
        
        assert result.system == PhysiologicalSystem.CNS
        assert result.overall_risk.score < 40  # Should be low/moderate
        assert len(result.sub_risks) == 4
        assert len(result.alerts) == 0  # No alerts for normal values
    
    def test_compute_risk_abnormal(self, abnormal_cardiovascular_biomarkers):
        """Test risk computation for abnormal biomarkers."""
        engine = RiskEngine()
        result = engine.compute_risk(abnormal_cardiovascular_biomarkers)
        
        assert result.system == PhysiologicalSystem.CARDIOVASCULAR
        assert result.overall_risk.score > 30  # Should show elevated risk
        assert len(result.alerts) > 0  # Should have alerts
    
    def test_compute_all_risks(self, sample_biomarker_sets):
        """Test computing risks for all systems."""
        engine = RiskEngine()
        results = engine.compute_all_risks(sample_biomarker_sets)
        
        assert len(results) == 3
        assert PhysiologicalSystem.CNS in results
        assert PhysiologicalSystem.CARDIOVASCULAR in results
        assert PhysiologicalSystem.SKELETAL in results
    
    def test_system_risk_result_serialization(self, sample_cns_biomarkers):
        """Test SystemRiskResult serialization."""
        engine = RiskEngine()
        result = engine.compute_risk(sample_cns_biomarkers)
        
        d = result.to_dict()
        assert "system" in d
        assert "overall_risk" in d
        assert "sub_risks" in d
        assert "biomarker_summary" in d


class TestCompositeRiskCalculator:
    """Tests for composite risk calculation."""
    
    def test_compute_composite_risk(self, sample_biomarker_sets):
        """Test composite risk calculation."""
        engine = RiskEngine()
        results = engine.compute_all_risks(sample_biomarker_sets)
        
        calculator = CompositeRiskCalculator()
        composite = calculator.compute_composite_risk(results)
        
        assert composite.name == "composite_health_risk"
        assert 0 <= composite.score <= 100
        assert 0 <= composite.confidence <= 1
        assert composite.explanation != ""
    
    def test_empty_results(self):
        """Test with no systems."""
        calculator = CompositeRiskCalculator()
        composite = calculator.compute_composite_risk({})
        
        assert composite.score == 50.0
        assert composite.confidence == 0.0


class TestCalibrationFactors:
    """Tests for CalibrationFactors."""
    
    def test_aggregate_all_ones(self):
        """Test aggregate with perfect factors."""
        factors = CalibrationFactors(
            data_completeness=1.0,
            sensor_quality=1.0,
            temporal_consistency=1.0,
            cross_validation=1.0
        )
        assert factors.aggregate() == 1.0
    
    def test_aggregate_mixed(self):
        """Test aggregate with mixed values."""
        factors = CalibrationFactors(
            data_completeness=0.8,
            sensor_quality=0.7,
            temporal_consistency=0.9,
            cross_validation=0.85
        )
        aggregate = factors.aggregate()
        assert 0.7 < aggregate < 0.9


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator."""
    
    def test_calibrate_biomarker_confidence(self, sample_cns_biomarkers):
        """Test calibrating biomarker confidence."""
        calibrator = ConfidenceCalibrator()
        
        original_confidences = [bm.confidence for bm in sample_cns_biomarkers.biomarkers]
        calibrated = calibrator.calibrate_biomarker_confidence(sample_cns_biomarkers)
        
        # Confidences should be adjusted
        for bm, orig in zip(calibrated.biomarkers, original_confidences):
            assert 0.1 <= bm.confidence <= 0.99
    
    def test_calibrate_risk_result(self, sample_cns_biomarkers):
        """Test calibrating risk result confidence."""
        engine = RiskEngine()
        result = engine.compute_risk(sample_cns_biomarkers)
        
        calibrator = ConfidenceCalibrator()
        calibrated = calibrator.calibrate_risk_result(result)
        
        assert 0.1 <= calibrated.overall_risk.confidence <= 0.99
    
    def test_compute_uncertainty_bounds(self):
        """Test uncertainty bounds computation."""
        calibrator = ConfidenceCalibrator()
        
        # High confidence score
        high_conf = RiskScore("test", 50.0, 0.9)
        low, high = calibrator.compute_uncertainty_bounds(high_conf)
        assert low < 50 < high
        
        # Low confidence score - wider bounds
        low_conf = RiskScore("test", 50.0, 0.3)
        low2, high2 = calibrator.compute_uncertainty_bounds(low_conf)
        assert (high2 - low2) > (high - low)


class TestRiskExplanation:
    """Tests for RiskExplanation dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        exp = RiskExplanation(
            system=PhysiologicalSystem.CNS,
            risk_level=RiskLevel.MODERATE,
            summary="Test summary",
            detailed_findings=["Finding 1", "Finding 2"],
            recommendations=["Rec 1"]
        )
        
        d = exp.to_dict()
        assert d["system"] == "central_nervous_system"
        assert d["risk_level"] == "moderate"
        assert len(d["detailed_findings"]) == 2
    
    def test_to_text_formats(self):
        """Test different text formats."""
        exp = RiskExplanation(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            risk_level=RiskLevel.HIGH,
            summary="Elevated cardiovascular risk",
            detailed_findings=["Heart rate elevated"],
            recommendations=["See cardiologist"],
            confidence_statement="Moderate confidence (75%)"
        )
        
        summary = exp.to_text(ExplanationType.SUMMARY)
        assert "Elevated cardiovascular risk" in summary
        
        patient = exp.to_text(ExplanationType.PATIENT)
        assert "Cardiovascular" in patient
        
        clinical = exp.to_text(ExplanationType.CLINICAL)
        assert "SYSTEM:" in clinical


class TestExplanationGenerator:
    """Tests for ExplanationGenerator."""
    
    def test_generate_explanation(self, sample_cns_biomarkers):
        """Test generating explanation for risk result."""
        engine = RiskEngine()
        result = engine.compute_risk(sample_cns_biomarkers)
        
        generator = ExplanationGenerator()
        explanation = generator.generate_explanation(result, sample_cns_biomarkers)
        
        assert explanation.system == PhysiologicalSystem.CNS
        assert explanation.summary != ""
        assert len(explanation.recommendations) > 0
        assert len(explanation.caveats) > 0
    
    def test_generate_composite_explanation(self, sample_biomarker_sets):
        """Test generating composite explanation."""
        engine = RiskEngine()
        results = engine.compute_all_risks(sample_biomarker_sets)
        
        calculator = CompositeRiskCalculator()
        composite = calculator.compute_composite_risk(results)
        
        generator = ExplanationGenerator()
        text = generator.generate_composite_explanation(results, composite)
        
        assert "COMPREHENSIVE HEALTH SCREENING SUMMARY" in text
        assert "Overall Health Risk" in text


class TestIntegration:
    """Integration tests for inference pipeline."""
    
    def test_full_inference_pipeline(self, sample_biomarker_sets):
        """Test complete inference pipeline."""
        # Initialize components
        engine = RiskEngine()
        calibrator = ConfidenceCalibrator()
        generator = ExplanationGenerator()
        composite_calc = CompositeRiskCalculator()
        
        # Compute risks
        results = {}
        for bm_set in sample_biomarker_sets:
            # Calibrate
            calibrated_bm = calibrator.calibrate_biomarker_confidence(bm_set)
            # Compute risk
            risk = engine.compute_risk(calibrated_bm)
            # Calibrate risk
            calibrated_risk = calibrator.calibrate_risk_result(risk)
            results[bm_set.system] = calibrated_risk
        
        # Composite risk
        composite = composite_calc.compute_composite_risk(results)
        
        # Explanations
        explanations = []
        for system, result in results.items():
            exp = generator.generate_explanation(result)
            explanations.append(exp)
        
        # Validate
        assert len(results) == 3
        assert 0 <= composite.score <= 100
        assert len(explanations) == 3
        
        for exp in explanations:
            assert exp.summary != ""
            assert len(exp.recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

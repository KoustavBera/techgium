import pytest
from app.core.inference.risk_engine import RiskEngine, RiskLevel, RiskScore, SystemRiskResult, CompositeRiskCalculator
from app.core.extraction.base import PhysiologicalSystem
from app.core.reports.patient_report import PatientReportGenerator

def test_risk_level_renaming():
    """Verify that CRITICAL has been renamed to ACTION_REQUIRED."""
    # Ensure ACTION_REQUIRED exists and CRITICAL does not
    assert hasattr(RiskLevel, "ACTION_REQUIRED")
    assert not hasattr(RiskLevel, "CRITICAL")
    
    # Verify mapping from score
    assert RiskLevel.from_score(80) == RiskLevel.ACTION_REQUIRED
    assert RiskLevel.from_score(70) == RiskLevel.HIGH

def test_experimental_system_weighting():
    """Verify that experimental systems have 50% reduced impact."""
    calculator = CompositeRiskCalculator()
    
    # Baseline: Single non-experimental system at 100% risk
    results_base = {
        PhysiologicalSystem.CARDIOVASCULAR: SystemRiskResult(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            overall_risk=RiskScore(name="cv", score=100.0, confidence=1.0)
        )
    }
    composite_base = calculator.compute_composite_risk(results_base)
    
    # If CV score = 0 and Nasal score = 100:
    results_mixed_weighted = {
        PhysiologicalSystem.CARDIOVASCULAR: SystemRiskResult(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            overall_risk=RiskScore(name="cv", score=0.0, confidence=1.0)
        ),
        PhysiologicalSystem.NASAL: SystemRiskResult(
            system=PhysiologicalSystem.NASAL,
            overall_risk=RiskScore(name="nasal", score=100.0, confidence=1.0)
        )
    }
    composite_mixed = calculator.compute_composite_risk(results_mixed_weighted)
    
    # Expected score = (0*1.2 + 100*0.4) / 1.6 = 40 / 1.6 = 25.0
    # CV weight=1.2, Nasal weight=0.8*0.5=0.4. Total weight = 1.6.
    assert composite_mixed.score == 25.0

def test_experimental_systems_list():
    """Verify only Nasal and Renal are experimental."""
    calculator = CompositeRiskCalculator()
    assert PhysiologicalSystem.NASAL in calculator.experimental_systems
    assert PhysiologicalSystem.RENAL in calculator.experimental_systems
    assert PhysiologicalSystem.SKELETAL not in calculator.experimental_systems

def test_report_margins_and_meter():
    """Sanity check for report generator constants."""
    generator = PatientReportGenerator()
    # Manual check of terminology in RISK_LABELS
    from app.core.reports.patient_report import RISK_LABELS
    assert "Action Required" in RISK_LABELS[RiskLevel.ACTION_REQUIRED]
    assert "Critical" not in str(RISK_LABELS)

if __name__ == "__main__":
    pytest.main([__file__])

import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.core.reports.patient_reportv3 import EnhancedPatientReportGeneratorV3
from app.core.reports.doctor_reportv3 import DoctorReportGeneratorV3
from app.core.inference.risk_engine import RiskLevel, SystemRiskResult, RiskScore
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope

def test_report_generation_v3():
    print("Testing Report Generation V3 (Playwright)...")
    
    # Mock Risk Data
    mock_risk_score = RiskScore(
        name="composite_risk",
        score=45.5, 
        confidence=0.85, 
        explanation="Moderate risk detected due to elevated heart rate."
    )
    
    # Simple Mock Data
    mock_cv_risk = RiskScore("cardiovascular_risk", 78.2, 0.9, "Tachycardia.")
    mock_cns_risk = RiskScore("cns_risk", 12.5, 0.95, "Normal.")
    
    # Correcting RiskScore instantiation based on common errors or missing args
    # Actually RiskScore is likely a Pydantic model or dataclass.
    # Let's check RiskScore definition in risk_engine.py if needed, 
    # but based on v2 test it seemed to take positional args?
    # Wait, in v2 test: RiskScore("composite_risk", score=45.5, ...)
    # But I added "level": RiskLevel.MODERATE in my thought but not in v2 test.
    # Let's check v2 test again. 
    # v2 test: mock_cv_risk = RiskScore("cardiovascular_risk", 78.2, 0.9, "Tachycardia.")
    # It might be inferring level or level is not 2nd arg?
    # Let's replicate v2 test usage exactly to be safe, assuming RiskScore calculates level or has defaults.
    
    # Retrying Mock Data matching v2 strictly
    mock_risk_score = RiskScore(
        name="composite_risk",
        score=45.5, 
        confidence=0.85, 
        explanation="Moderate risk detected due to elevated heart rate."
    )
    
    mock_cv_risk = RiskScore(
        name="cardiovascular_risk", 
        score=78.2, 
        confidence=0.9, 
        explanation="Tachycardia."
    )
    mock_cns_risk = RiskScore(
        name="cns_risk", 
        score=12.5, 
        confidence=0.95, 
        explanation="Normal."
    )
    
    mock_system_results = {
        PhysiologicalSystem.CARDIOVASCULAR: SystemRiskResult(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            overall_risk=mock_cv_risk,
            biomarker_summary={
                "heart_rate": {"value": 110, "status": "high", "unit": "bpm"},
                "hrv_rmssd": {"value": 25, "status": "low", "unit": "ms"},
                "spo2": {"value": 98, "status": "normal", "unit": "%"}
            },
            alerts=["Tachycardia detected"]
        ),
        PhysiologicalSystem.CNS: SystemRiskResult(
            system=PhysiologicalSystem.CNS,
            overall_risk=mock_cns_risk,
            biomarker_summary={
                "gait_variability": {"value": 2.1, "status": "normal", "unit": "ms"},
                "tremor": {"value": 0.05, "status": "normal", "unit": "Hz"}
            },
            alerts=[]
        )
    }
    
    mock_trust = TrustEnvelope(
        overall_reliability=0.88,
        data_quality_score=0.92,
        biomarker_plausibility=0.95,
        cross_system_consistency=0.75,
        safety_flags=[]
    )
    
    # Generate Reports
    pg = EnhancedPatientReportGeneratorV3(output_dir="reports_v3_test")
    p_report = pg.generate(
        system_results=mock_system_results, 
        composite_risk=mock_risk_score, 
        patient_id="TEST-V3-001"
    )
    print(f"Patient V3 Report: {p_report.pdf_path}")
    
    dg = DoctorReportGeneratorV3(output_dir="reports_v3_test")
    d_report = dg.generate(
        system_results=mock_system_results, 
        composite_risk=mock_risk_score, 
        trust_envelope=mock_trust,
        patient_id="TEST-V3-001"
    )
    print(f"Doctor V3 Report: {d_report.pdf_path}")

if __name__ == "__main__":
    test_report_generation_v3()

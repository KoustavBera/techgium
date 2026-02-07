import sys
import os
import io

# Add project root to path
sys.path.append(os.getcwd())

from app.core.reports.patient_reportv2 import EnhancedPatientReportGeneratorV2
from app.core.reports.doctor_reportv2 import DoctorReportGeneratorV2
from app.core.inference.risk_engine import RiskLevel, SystemRiskResult, RiskScore
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope, SafetyFlag

def test_report_generation_v2():
    print("Testing Report Generation V2 (WeasyPrint)...")
    
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
    pg = EnhancedPatientReportGeneratorV2(output_dir="reports_v2")
    p_report = pg.generate(
        system_results=mock_system_results, 
        composite_risk=mock_risk_score, 
        patient_id="TEST-V2-001"
    )
    print(f"Patient V2 Report: {p_report.pdf_path}")
    
    dg = DoctorReportGeneratorV2(output_dir="reports_v2")
    d_report = dg.generate(
        system_results=mock_system_results, 
        composite_risk=mock_risk_score, 
        trust_envelope=mock_trust,
        patient_id="TEST-V2-001"
    )
    print(f"Doctor V2 Report: {d_report.pdf_path}")

if __name__ == "__main__":
    test_report_generation_v2()

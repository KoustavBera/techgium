import sys
import os
import io

# Add project root to path
sys.path.append(os.getcwd())

from app.core.reports.patient_report import PatientReportGenerator
from app.core.reports.doctor_report import DoctorReportGenerator
from app.core.inference.risk_engine import RiskLevel, SystemRiskResult, RiskScore
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope, SafetyFlag

def test_report_generation():
    print("Testing Report Generation with Redesign...")
    
    # Mock Risk Data
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
        explanation="Significant tachycardia observed."
    )
    
    mock_cns_risk = RiskScore(
        name="cns_risk",
        score=12.5, 
        confidence=0.95, 
        explanation="No significant CNS issues."
    )
    
    # Mock System Results
    mock_system_results = {
        PhysiologicalSystem.CARDIOVASCULAR: SystemRiskResult(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            overall_risk=mock_cv_risk,
            biomarker_summary={
                "heart_rate": {"value": 110, "status": "high", "unit": "bpm", "normal_range": (60, 100)},
                "hrv_rmssd": {"value": 25, "status": "low", "unit": "ms", "normal_range": (30, 100)},
                "spo2": {"value": 98, "status": "normal", "unit": "%", "normal_range": (95, 100)}
            },
            alerts=["Tachycardia detected", "Low HRV"]
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
    
    # Mock Trust Envelope
    mock_trust = TrustEnvelope(
        overall_reliability=0.88,
        data_quality_score=0.92,
        biomarker_plausibility=0.95,
        cross_system_consistency=0.75,
        safety_flags=[],
        # is_reliable is a property, not a field
    )
    
    # Generate Patient Report
    print("\nGenerating Patient Report...")
    pg = PatientReportGenerator(output_dir="reports_test_design")
    p_report = pg.generate(
        system_results=mock_system_results, 
        composite_risk=mock_risk_score, 
        patient_id="TEST-PATIENT-001"
    )
    print(f"Patient Report Generated: {p_report.pdf_path}")
    
    # Generate Doctor Report
    print("\nGenerating Doctor Report...")
    dg = DoctorReportGenerator(output_dir="reports_test_design")
    d_report = dg.generate(
        system_results=mock_system_results, 
        composite_risk=mock_risk_score, 
        trust_envelope=mock_trust,
        patient_id="TEST-PATIENT-001"
    )
    print(f"Doctor Report Generated: {d_report.pdf_path}")

if __name__ == "__main__":
    test_report_generation()

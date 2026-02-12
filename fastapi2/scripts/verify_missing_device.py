import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.inference.risk_engine import RiskEngine, RiskLevel, SystemRiskResult, CompositeRiskCalculator, RiskScore
from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem
from app.core.reports.patient_report import EnhancedPatientReportGenerator
from app.core.reports.doctor_report import DoctorReportGenerator

def test_missing_data_risk_engine():
    print("\n--- Testing RiskEngine Missing Data Handling ---")
    engine = RiskEngine()
    
    # Create an empty biomarker set for CNS
    empty_set = BiomarkerSet(
        system=PhysiologicalSystem.CNS,
        biomarkers=[],
        metadata={"flags": ["NO_DATA"]}
    )
    
    # Compute risk
    result = engine.compute_risk(empty_set)
    
    # Verify result
    print(f"System: {result.system.value}")
    print(f"Risk Score: {result.overall_risk.score}")
    print(f"Risk Level: {result.overall_risk.level}")
    print(f"Explanation: {result.overall_risk.explanation}")
    
    if result.overall_risk.score == -1.0 and result.overall_risk.level == RiskLevel.UNKNOWN:
        print("✅ RiskEngine correctly handled missing data (Score: -1.0, Level: UNKNOWN)")
    else:
        print("❌ FAILED: RiskEngine did not return expected UNKNOWN state.")
        sys.exit(1)

    return result

def test_patient_report_generation(system_result):
    print("\n--- Testing Patient Report Generation with Missing Data ---")
    
    # Create a composite risk result (just a placeholder)
    composite_risk = RiskScore(
        name="composite",
        score=50.0,
        confidence=0.5,
        explanation="Test composite"
    )
    
    # Mock systems output
    system_results = {
        PhysiologicalSystem.CNS: system_result
    }
    
    try:
        generator = EnhancedPatientReportGenerator(output_dir="test_reports")
        report_obj = generator.generate(
            patient_id="TEST_MISSING_DATA",
            system_results=system_results,
            composite_risk=composite_risk
        )
        print(f"✅ Patient Report generated: {report_obj.pdf_path}")
    except Exception as e:
        print(f"❌ Patient Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_doctor_report_generation(system_result):
    print("\n--- Testing Doctor Report Generation with Missing Data ---")
    
    # Create a composite risk result
    composite_risk = RiskScore(
        name="composite",
        score=50.0,
        confidence=0.5,
        explanation="Test composite"
    )
    
    # Mock systems output
    system_results = {
        PhysiologicalSystem.CNS: system_result
    }
    
    try:
        generator = DoctorReportGenerator(output_dir="test_reports")
        report = generator.generate(
            patient_id="TEST_MISSING_DATA",
            system_results=system_results,
            composite_risk=composite_risk
        )
        print(f"✅ Doctor Report generated: {report.pdf_path}")
    except Exception as e:
        print(f"❌ Doctor Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    os.makedirs("test_reports", exist_ok=True)
    result = test_missing_data_risk_engine()
    test_patient_report_generation(result)
    test_doctor_report_generation(result)
    print("\n✅ All verifications passed successfully!")

import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("Verifying modules...")

try:
    print("Importing app.core.extraction.base...")
    from app.core.extraction.base import PhysiologicalSystem
    print("✅ base.py OK")
except Exception as e:
    print(f"❌ base.py FAILED: {e}")

try:
    print("Importing app.core.extraction.nasal...")
    from app.core.extraction.nasal import NasalExtractor
    print("✅ nasal.py OK")
except Exception as e:
    print(f"❌ nasal.py FAILED: {e}")

try:
    print("Importing app.core.validation.biomarker_plausibility...")
    import app.core.validation.biomarker_plausibility
    print("✅ biomarker_plausibility.py OK")
except Exception as e:
    print(f"❌ biomarker_plausibility.py FAILED: {e}")

try:
    print("Importing app.core.inference.risk_engine...")
    from app.core.inference.risk_engine import RiskEngine
    print("✅ risk_engine.py OK")
except Exception as e:
    print(f"❌ risk_engine.py FAILED: {e}")

try:
    print("Importing app.core.reports.patient_report...")
    from app.core.reports.patient_report import PatientReportGenerator
    print("✅ patient_report.py OK")
except Exception as e:
    print(f"❌ patient_report.py FAILED: {e}")

try:
    print("Importing app.main...")
    import app.main
    print("✅ main.py OK")
except Exception as e:
    print(f"❌ main.py FAILED: {e}")

print("Verification complete.")

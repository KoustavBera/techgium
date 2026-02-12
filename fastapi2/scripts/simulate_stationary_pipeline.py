
import json
import sys
import requests
from datetime import datetime
from typing import List, Dict, Any

# Configuration
API_URL = "http://localhost:8000"

def create_biomarker(name: str, value: float, unit: str, normal_range: List[float] = None) -> Dict[str, Any]:
    bm = {
        "name": name,
        "value": value,
        "unit": unit,
        "status": "normal" if normal_range else "not_assessed"
    }
    if normal_range:
        bm["normal_range"] = normal_range
    return bm

def simulate_stationary_screening():
    """Simulate a STATIONARY subject (sitting/standing still) to test gait reporting fix."""
    print(f"[{datetime.now()}] Starting STATIONARY Pipeline Simulation...")
    print(f"Targeting API: {API_URL}")
    print("Simulating stationary subject (no walking)...")

    systems_data = []

    # 1. Cardiovascular (still works when sitting)
    print("  - Generating Cardiovascular data (Webcam rPPG + Radar Heartbeat)...")
    cv_biomarkers = [
        create_biomarker("heart_rate", 72.0, "bpm", [60.0, 100.0]),
        create_biomarker("hrv_rmssd", 45.0, "ms", [20.0, 100.0]),
    ]
    systems_data.append({
        "system": "cardiovascular",
        "biomarkers": cv_biomarkers
    })

    # 2. Skin (still works when sitting)
    print("  - Generating Skin data (MLX90640 Thermal)...")
    skin_biomarkers = [
        create_biomarker("skin_temperature", 36.6, "C", [35.0, 38.0]),
        create_biomarker("inflammation_index", 5.0, "%", [0.0, 15.0]),
    ]
    systems_data.append({
        "system": "skin",
        "biomarkers": skin_biomarkers
    })

    # 3. CNS - STATIONARY (no gait, neutral fallback)
    print("  - Generating CNS data (STATIONARY - no walking detected)...")
    cns_biomarkers = [
        # Gait not assessed (no normal_range to simulate stationary state)
        create_biomarker("gait_variability", 0.045, "coefficient_of_variation", normal_range=None),
        # Other CNS features still work when sitting
        create_biomarker("posture_entropy", 2.5, "bits", [1.0, 4.0]),
        create_biomarker("tremor_resting", 0.01, "amp", [0.0, 0.1]),
    ]
    systems_data.append({
        "system": "cns",
        "biomarkers": cns_biomarkers
    })

    # Construct Request
    payload = {
        "patient_id": "STATIONARY_SIM_001",
        "systems": systems_data,
        "include_validation": True
    }

    # Send Request
    print("\nSending request to API...")
    try:
        response = requests.post(f"{API_URL}/api/v1/screening", json=payload)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ CRITICAL ERROR: Could not connect to {API_URL}")
        print("Please ensure the backend server is running:")
        print("  uvicorn app.main:app --reload")
        sys.exit(1)

    if response.status_code == 200:
        result = response.json()
        print("\n--- SCREENING COMPLETE ---")
        print(f"Screening ID: {result['screening_id']}")
        print(f"Overall Risk Level: {result['overall_risk_level']}")
        print(f"Overall Risk Score: {result['overall_risk_score']}")
        print(f"Overall Confidence: {result['overall_confidence']}")
        print(f"Validation Status: {result['validation_status']}")
        
        # Verify CNS system
        cns_system = next((s for s in result['system_results'] if s['system'] == 'cns'), None)
        if cns_system:
            print(f"\nCNS System:")
            print(f"  Risk Level: {cns_system['risk_level']}")
            print(f"  Risk Score: {cns_system['risk_score']}")
            print(f"  Trusted: {cns_system.get('is_trusted', True)}")
        
        # Generate Report
        print("\nGenerating PDF Report...")
        report_req = {
            "screening_id": result['screening_id'],
            "report_type": "patient"
        }
        try:
            report_res = requests.post(f"{API_URL}/api/v1/reports/generate", json=report_req)
            if report_res.status_code == 200:
                report_data = report_res.json()
                print(f"✅ Report generated: {report_data['pdf_path']}")
                print("\n🔍 VERIFICATION CHECKLIST:")
                print("  1. Open the PDF report")
                print("  2. Check CNS section for 'Walking Stability (Stationary)'")
                print("  3. Verify status shows 'Not Assessed (Stationary)'")
                print("  4. Verify CNS risk is still Low (not elevated)")
            else:
                print(f"❌ Report generation failed: {report_res.text}")
        except Exception as e:
            print(f"❌ Report generation failed: {e}")

    else:
        print(f"\n❌ API Request Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    simulate_stationary_screening()

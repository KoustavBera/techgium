
import json
import sys
import requests
from datetime import datetime
from typing import List, Dict, Any

# Configuration
API_URL = "http://localhost:8000"

def create_biomarker(name: str, value: float, unit: str, normal_range: List[float]) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "status": "normal", # Status might be updated by engine, input status is usually raw
        "normal_range": normal_range
    }

def simulate_unhealthy_screening():
    print(f"[{datetime.now()}] Starting Unhealthy Pipeline Simulation (Real Backend)...")
    print(f"Targeting API: {API_URL}")
    print("Simulating ABNORMAL hardware sensor data...")

    systems_data = []

    # 1. Cardiovascular (High Risk)
    # Tachycardia + Hypertension + Low HRV
    print("  - Generating Cardiovascular data (High HR, BP, Low HRV)...")
    cv_biomarkers = [
        create_biomarker("heart_rate", 115.0, "bpm", [60.0, 100.0]), # High
        create_biomarker("hrv_rmssd", 15.0, "ms", [20.0, 100.0]),    # Low (Stress)
        create_biomarker("blood_pressure_systolic", 150.0, "mmHg", [90.0, 120.0]), # High
        create_biomarker("blood_pressure_diastolic", 95.0, "mmHg", [60.0, 80.0])   # High
    ]
    systems_data.append({
        "system": "cardiovascular",
        "biomarkers": cv_biomarkers
    })

    # 2. Pulmonary (Moderate Risk)
    # Tachypnea + Mild Hypoxia
    print("  - Generating Pulmonary data (Rapid Breathing, Low SpO2)...")
    pulm_biomarkers = [
        create_biomarker("respiratory_rate", 24.0, "brpm", [12.0, 20.0]), # High
        create_biomarker("sp02", 92.0, "%", [95.0, 100.0])              # Low
    ]
    systems_data.append({
        "system": "pulmonary",
        "biomarkers": pulm_biomarkers
    })

    # 3. Skin (High Risk)
    # Fever + Asymmetry (Inflammation?)
    print("  - Generating Skin data (Fever)...")
    skin_biomarkers = [
        create_biomarker("surface_temperature_avg", 38.2, "C", [36.1, 37.2]), # Fever
        create_biomarker("thermal_asymmetry", 0.6, "C", [0.0, 0.5])           # High asymmetry
    ]
    systems_data.append({
        "system": "skin",
        "biomarkers": skin_biomarkers
    })

    # 4. CNS (High Risk)
    # Unstable gait + Tremor
    print("  - Generating CNS data (Unstable Gait, Tremor)...")
    cns_biomarkers = [
        create_biomarker("gait_variability", 0.08, "score", [0.0, 0.05]),     # High variability
        create_biomarker("cns_stability_score", 45.0, "score", [80.0, 100.0]), # Very low stability
        create_biomarker("tremor_resting", 0.08, "amp", [0.0, 0.02])           # Significant tremor
    ]
    systems_data.append({
        "system": "cns",
        "biomarkers": cns_biomarkers
    })

    # 5. Skeletal (Moderate Risk)
    # Poor posture
    print("  - Generating Skeletal data (Poor Posture)...")
    skeletal_biomarkers = [
        create_biomarker("posture_alignment_score", 65.0, "score", [80.0, 100.0]), # Poor
        create_biomarker("symmetry_index", 0.85, "ratio", [0.9, 1.0])              # Asymmetric
    ]
    systems_data.append({
        "system": "skeletal",
        "biomarkers": skeletal_biomarkers
    })

    # Construct Request
    payload = {
        "patient_id": "UNHEALTHY_SIM_REAL_001",
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
        
        # Verify Results
        failures = []
        if result['overall_risk_level'] not in ["high", "critical"]:
            failures.append(f"Expected HIGH/CRITICAL risk, got {result['overall_risk_level']}")
        if result['overall_risk_score'] < 70:
            failures.append(f"Expected score > 70, got {result['overall_risk_score']}")
        
        # Check specific system alerts
        print("\nSystem Findings:")
        found_cardio_alert = False
        found_cns_alert = False
        
        for sys_res in result['system_results']:
            print(f"  {sys_res['system'].upper()}: {sys_res['risk_level']} (Score: {sys_res['risk_score']})")
            if sys_res.get('alerts'):
                print(f"    Alerts: {sys_res['alerts']}")
                if sys_res['system'] == 'cardiovascular' and len(sys_res['alerts']) > 0:
                    found_cardio_alert = True
                if sys_res['system'] == 'central_nervous_system' and len(sys_res['alerts']) > 0:
                    found_cns_alert = True
        
        if not found_cardio_alert:
            failures.append("Expected Cardiovascular alerts, found none.")
        if not found_cns_alert:
            failures.append("Expected CNS alerts, found none.")

        if not failures:
            print("\n✅ VERIFICATION PASSED: Patient assessed as UNHEALTHY (High Risk).")
            print("   Multiple systems correctly flagged.")
            
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
                else:
                    print(f"❌ Report generation failed: {report_res.text}")
            except Exception as e:
                print(f"❌ Report generation failed (Connection error): {e}")

        else:
            print("\n❌ VERIFICATION FAILED:")
            for f in failures:
                print(f"  - {f}")
    else:
        print(f"\n❌ API Request Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    simulate_unhealthy_screening()

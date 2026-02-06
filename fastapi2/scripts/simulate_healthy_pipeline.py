
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
        "status": "normal",
        "normal_range": normal_range
    }

def simulate_healthy_screening():
    print(f"[{datetime.now()}] Starting Healthy Pipeline Simulation (Real Backend)...")
    print(f"Targeting API: {API_URL}")
    print("Simulating hardware sensors from HARDWARE.md...")

    systems_data = []

    # 1. Cardiovascular (Webcam + Radar)
    # HR: 60-100 is normal. Target ~72.
    # HRV (RMSSD): > 20ms is decent. Target ~45.
    print("  - Generating Cardiovascular data (Webcam rPPG + Radar Heartbeat)...")
    cv_biomarkers = [
        create_biomarker("heart_rate", 72.0, "bpm", [60.0, 100.0]),
        create_biomarker("hrv_rmssd", 45.0, "ms", [20.0, 100.0]),
        create_biomarker("blood_pressure_systolic", 115.0, "mmHg", [90.0, 120.0]), # Simulated proxy
        create_biomarker("blood_pressure_diastolic", 75.0, "mmHg", [60.0, 80.0])   # Simulated proxy
    ]
    systems_data.append({
        "system": "cardiovascular",
        "biomarkers": cv_biomarkers
    })

    # 2. Pulmonary (Radar)
    # RR: 12-20 bpm. Target ~15.
    print("  - Generating Pulmonary data (Radar Breathing)...")
    pulm_biomarkers = [
        create_biomarker("respiratory_rate", 15.0, "brpm", [12.0, 20.0]),
        create_biomarker("sp02", 98.0, "%", [95.0, 100.0]) # Often derived or separate sensor, assuming normal
    ]
    systems_data.append({
        "system": "pulmonary",
        "biomarkers": pulm_biomarkers
    })

    # 3. Skin (Thermal Camera)
    # Temp: 36.1-37.2. Target 36.6.
    print("  - Generating Skin data (MLX90640 Thermal)...")
    skin_biomarkers = [
        create_biomarker("surface_temperature_avg", 36.6, "C", [36.1, 37.2]),
        create_biomarker("thermal_asymmetry", 0.1, "C", [0.0, 0.5]) # Low asymmetry is good
    ]
    systems_data.append({
        "system": "skin",
        "biomarkers": skin_biomarkers
    })

    # 4. CNS (Webcam Gait/Posture)
    print("  - Generating CNS data (Webcam Pose)...")
    cns_biomarkers = [
        create_biomarker("gait_variability", 0.02, "score", [0.0, 0.05]), # Low variability is stable
        create_biomarker("cns_stability_score", 95.0, "score", [80.0, 100.0]),
        create_biomarker("tremor_resting", 0.0, "amp", [0.0, 0.1])
    ]
    systems_data.append({
        "system": "cns",
        "biomarkers": cns_biomarkers
    })

    # 5. Skeletal (Webcam Pose)
    print("  - Generating Skeletal data (Webcam Pose)...")
    skeletal_biomarkers = [
        create_biomarker("posture_alignment_score", 90.0, "score", [80.0, 100.0]),
        create_biomarker("symmetry_index", 0.95, "ratio", [0.9, 1.0])
    ]
    systems_data.append({
        "system": "skeletal",
        "biomarkers": skeletal_biomarkers
    })

    # 6. Eyes (Webcam)
    print("  - Generating Eye data (Webcam Face)...")
    eye_biomarkers = [
        create_biomarker("blink_rate", 15.0, "bpm", [10.0, 30.0]),
        create_biomarker("gaze_stability", 0.98, "score", [0.9, 1.0])
    ]
    systems_data.append({
        "system": "eyes",
        "biomarkers": eye_biomarkers
    })

    # Construct Request
    payload = {
        "patient_id": "HEALTHY_SIM_REAL_001",
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
        
        # Verify Results
        failures = []
        if result['overall_risk_level'] != "low":
            failures.append(f"Expected LOW risk, got {result['overall_risk_level']}")
        if result['overall_risk_score'] > 25:
            failures.append(f"Expected score < 25, got {result['overall_risk_score']}")
        if result['overall_confidence'] < 0.8:
            failures.append(f"Expected confidence > 0.8, got {result['overall_confidence']}")
        
        if not failures:
            print("\n✅ VERIFICATION PASSED: Patient assessed as HEALTHY (Low Risk).")
            
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
            print("\nSystem Results:")
            for sys_res in result['system_results']:
                print(f"  {sys_res['system']}: {sys_res['risk_level']} (Score: {sys_res['risk_score']}) - Trusted: {sys_res.get('is_trusted')}")
                if sys_res.get('alerts'):
                    print(f"    Alerts: {sys_res['alerts']}")
                if sys_res.get('caveats'):
                    print(f"    Caveats: {sys_res['caveats']}")
    else:
        print(f"\n❌ API Request Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    simulate_healthy_screening()

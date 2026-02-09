
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
        "status": "normal", # Status might be updated by engine
        "normal_range": normal_range
    }

def simulate_mixed_health_screening():
    print(f"[{datetime.now()}] Starting Mixed Health Pipeline Simulation (Real Backend)...")
    print(f"Targeting API: {API_URL}")
    print("Simulating MIXED health male (Healthy Cardio/Pulm, Unhealthy CNS/Skin)...")

    systems_data = []

    # 1. Cardiovascular (HEALTHY)
    # Normal male parameters
    print("  - Generating Cardiovascular data (Healthy)...")
    cv_biomarkers = [
        create_biomarker("heart_rate", 72.0, "bpm", [60.0, 100.0]),
        create_biomarker("hrv_rmssd", 55.0, "ms", [20.0, 100.0]),
        create_biomarker("blood_pressure_systolic", 118.0, "mmHg", [90.0, 120.0]),
        create_biomarker("blood_pressure_diastolic", 76.0, "mmHg", [60.0, 80.0])
    ]
    systems_data.append({
        "system": "cardiovascular",
        "biomarkers": cv_biomarkers
    })

    # 2. Pulmonary (HEALTHY)
    # Normal breathing
    print("  - Generating Pulmonary data (Healthy)...")
    pulm_biomarkers = [
        create_biomarker("respiratory_rate", 14.0, "brpm", [12.0, 20.0]),
        create_biomarker("sp02", 98.0, "%", [95.0, 100.0])
    ]
    systems_data.append({
        "system": "pulmonary",
        "biomarkers": pulm_biomarkers
    })

    # 3. Skin (UNHEALTHY)
    # Fever
    print("  - Generating Skin data (Fever - Unhealthy)...")
    skin_biomarkers = [
        create_biomarker("skin_temperature", 39.5, "C", [35.0, 38.0]),     # High fever
        create_biomarker("skin_temperature_max", 40.5, "C", [36.0, 38.5]), # Very high
        create_biomarker("inflammation_index", 45.0, "%", [0.0, 15.0]),    # Severe inflammation
        create_biomarker("thermal_asymmetry", 1.5, "C", [0.0, 0.5]),       # High asymmetry
    ]
    systems_data.append({
        "system": "skin",
        "biomarkers": skin_biomarkers
    })

    # 4. CNS (UNHEALTHY)
    # Tremor + Gait issues + Stress
    print("  - Generating CNS data (Tremor/Gait/Stress - Unhealthy)...")
    cns_biomarkers = [
        create_biomarker("gait_variability", 0.07, "score", [0.0, 0.05]),     # Moderate variability
        create_biomarker("cns_stability_score", 60.0, "score", [80.0, 100.0]), # Low stability
        create_biomarker("tremor_resting", 0.09, "amp", [0.0, 0.02]),         # Significant tremor
        create_biomarker("thermal_stress_gradient", 4.0, "C", [0.0, 3.0]),    # High stress
    ]
    systems_data.append({
        "system": "cns",
        "biomarkers": cns_biomarkers
    })

    # 5. Skeletal (HEALTHY)
    # Good posture
    print("  - Generating Skeletal data (Healthy)...")
    skeletal_biomarkers = [
        create_biomarker("posture_alignment_score", 95.0, "score", [80.0, 100.0]),
        create_biomarker("symmetry_index", 0.98, "ratio", [0.9, 1.0])
    ]
    systems_data.append({
        "system": "skeletal",
        "biomarkers": skeletal_biomarkers
    })

    # Construct Request
    payload = {
        "patient_id": "MIXED_HEALTH_MAN_001",
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
        
        # Verify Results
        failures = []
        
        # Check specific system risks
        print("\nSystem Findings:")
        found_skin_alert = False
        found_cns_alert = False
        
        for sys_res in result['system_results']:
            print(f"  {sys_res['system'].upper()}: {sys_res['risk_level']} (Score: {sys_res['risk_score']})")
            
            # Validation logic
            if sys_res['system'] == 'cardiovascular':
                if sys_res['risk_level'] != 'low':
                    failures.append(f"Expected Cardiovascular risk LOW, got {sys_res['risk_level']}")
            
            elif sys_res['system'] == 'pulmonary':
                if sys_res['risk_level'] != 'low':
                    failures.append(f"Expected Pulmonary risk LOW, got {sys_res['risk_level']}")
            
            elif sys_res['system'] == 'skin':
                if sys_res['risk_level'] not in ['high', 'critical']: # Expect at least high due to fever
                    failures.append(f"Expected Skin risk HIGH/CRITICAL, got {sys_res['risk_level']}")
                if len(sys_res.get('alerts', [])) > 0:
                    found_skin_alert = True
            
            elif sys_res['system'] == 'central_nervous_system': # Note: API might return full name
                 # CNS matches "central_nervous_system" or "cns" depending on enum serialization
                 pass 

            # Check for CNS (handling potential enum naming differences effectively by checking all)
            if 'nervous' in sys_res['system'] or 'cns' in sys_res['system']:
                if sys_res['risk_level'] not in ['high', 'critical']:
                     failures.append(f"Expected CNS risk HIGH/CRITICAL, got {sys_res['risk_level']}")
                if len(sys_res.get('alerts', [])) > 0:
                    found_cns_alert = True

        if not found_skin_alert:
            failures.append("Expected Skin alerts (Fever), found none.")
        if not found_cns_alert:
            failures.append("Expected CNS alerts (Tremor), found none.")

        if not failures:
            print("\n✅ VERIFICATION PASSED: Mixed health profile correctly identified.")
            print("   Healthy systems (Cardio/Pulm) -> LOW risk.")
            print("   Unhealthy systems (Skin/CNS) -> HIGH risk.")
            
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
    simulate_mixed_health_screening()

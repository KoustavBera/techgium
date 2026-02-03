"""
Test script to verify report generation fixes
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Test data matching the screenshot - Pulmonary system
test_data = {
    "patient_id": "TEST_PATIENT",
    "systems": [{
        "system": "pulmonary",
        "biomarkers": [
            {
                "name": "spo2",
                "value": 91.0,
                "unit": "%",
                "normal_range": [95.0, 100.0]
            },
            {
                "name": "respiratory_rate",
                "value": 26.0,
                "unit": "breaths/min",
                "normal_range": [12.0, 20.0]
            }
        ]
    }],
    "include_validation": False
}

print("=" * 60)
print("Testing Report Generation Fixes")
print("=" * 60)

# Step 1: Run screening
print("\n1. Running health screening...")
response = requests.post(f"{BASE_URL}/api/v1/screening", json=test_data)
if response.status_code == 200:
    screening_result = response.json()
    screening_id = screening_result["screening_id"]
    print(f"✓ Screening completed: {screening_id}")
    print(f"  Overall Risk: {screening_result['overall_risk_level']}")
    print(f"  Risk Score: {screening_result['overall_risk_score']}")
else:
    print(f"✗ Screening failed: {response.status_code}")
    print(response.text)
    exit(1)

# Step 2: Generate patient report
print("\n2. Generating patient report...")
patient_report_request = {
    "screening_id": screening_id,
    "report_type": "patient"
}
response = requests.post(f"{BASE_URL}/api/v1/reports/generate", json=patient_report_request)
if response.status_code == 200:
    patient_report = response.json()
    print(f"✓ Patient report generated: {patient_report['report_id']}")
    print(f"  PDF Path: {patient_report['pdf_path']}")
    patient_report_id = patient_report['report_id']
else:
    print(f"✗ Patient report failed: {response.status_code}")
    print(response.text)
    patient_report_id = None

# Step 3: Generate doctor report
print("\n3. Generating doctor report...")
doctor_report_request = {
    "screening_id": screening_id,
    "report_type": "doctor"
}
response = requests.post(f"{BASE_URL}/api/v1/reports/generate", json=doctor_report_request)
if response.status_code == 200:
    doctor_report = response.json()
    print(f"✓ Doctor report generated: {doctor_report['report_id']}")
    print(f"  PDF Path: {doctor_report['pdf_path']}")
    doctor_report_id = doctor_report['report_id']
else:
    print(f"✗ Doctor report failed: {response.status_code}")
    print(response.text)
    doctor_report_id = None

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Screening ID: {screening_id}")
if patient_report_id:
    print(f"✓ Patient Report: {patient_report_id}")
    print(f"  Download: {BASE_URL}/api/v1/reports/{patient_report_id}/download")
else:
    print("✗ Patient Report: FAILED")

if doctor_report_id:
    print(f"✓ Doctor Report: {doctor_report_id}")
    print(f"  Download: {BASE_URL}/api/v1/reports/{doctor_report_id}/download")
else:
    print("✗ Doctor Report: FAILED")

print("\nExpected fixes:")
print("  ✓ Normal Range should display: 95.0-100.0, 12.0-20.0")
print("  ✓ Status should show: ⚠ Below Normal, ⚠ Above Normal")
print("  ✓ 'What This Means' should have AI or fallback explanations")
print("=" * 60)

"""
Test the enhanced patient report generation
"""
import requests
import json

# Test screening with sample data
url = "http://localhost:8000/api/v1/screening"

payload = {
    "patient_id": "ENHANCED-TEST-001",
    "systems": [
        {
            "system": "cardiovascular",
            "biomarkers": [
                {"name": "heart_rate", "value": 105, "unit": "bpm"},
                {"name": "hrv_rmssd", "value": 22, "unit": "ms"}
            ]
        },
        {
            "system": "pulmonary",
            "biomarkers": [
                {"name": "spo2", "value": 91, "unit": "%"},
                {"name": "respiratory_rate", "value": 26, "unit": "breaths/min"}
            ]
        },
        {
            "system": "cns",
            "biomarkers": [
                {"name": "gait_variability", "value": 0.25, "unit": ""},
                {"name": "balance_score", "value": 0.45, "unit": ""}
            ]
        }
    ],
    "include_validation": False
}

print("Testing Enhanced Patient Report Generation...")
print("="*60)

# Run screening
print("\n[1/3] Running screening...")
response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    screening_id = data["screening_id"]
    print(f"✓ Screening ID: {screening_id}")
    print(f"✓ Overall Risk: {data['overall_risk_level']} ({data['overall_risk_score']}%)")
    print(f"✓ Systems analyzed: {len(data['system_results'])}")
    
    # Generate patient report
    print("\n[2/3] Generating enhanced patient report...")
    report_url = "http://localhost:8000/api/v1/reports/generate"
    report_payload = {
        "screening_id": screening_id,
        "report_type": "patient"
    }
    
    report_response = requests.post(report_url, json=report_payload)
    print(f"Status: {report_response.status_code}")
    
    if report_response.status_code == 200:
        report_data = report_response.json()
        report_id = report_data["report_id"]
        pdf_path = report_data["pdf_path"]
        print(f"✓ Report ID: {report_id}")
        print(f"✓ PDF saved to: {pdf_path}")
        
        # Download PDF
        print("\n[3/3] Downloading PDF...")
        download_url = f"http://localhost:8000/api/v1/reports/{report_id}/download"
        pdf_response = requests.get(download_url)
        
        if pdf_response.status_code == 200:
            filename = f"enhanced_{report_id}.pdf"
            with open(filename, "wb") as f:
                f.write(pdf_response.content)
            print(f"✓ PDF downloaded: {filename}")
            print(f"✓ File size: {len(pdf_response.content)} bytes")
            
            print("\n" + "="*60)
            print("SUCCESS! Enhanced patient report generated.")
            print(f"\nOpen the file to see the enhanced design:")
            print(f"  {filename}")
            print("\nNew features in this report:")
            print("  ✓ Individual biomarker values with units")
            print("  ✓ Color-coded status (Green/Amber)")
            print("  ✓ Normal ranges shown")
            print("  ✓ Simple explanations for each measurement")
            print("  ✓ 'What This Means' sections")
            print("  ✓ Personalized recommendations")
        else:
            print(f"✗ Download failed: {pdf_response.status_code}")
            print(pdf_response.text)
    else:
        print(f"✗ Report generation failed: {report_response.status_code}")
        print(report_response.text)
else:
    print(f"✗ Screening failed: {response.status_code}")
    print(response.text)

# Health Assessment API - Test Cases

## 🚀 Quick Start

### 1. Start the FastAPI Server

```powershell
cd C:\Users\KOUSTAV BERA\OneDrive\Desktop\chiranjeevi\fastapi
python -m uvicorn main:app --reload
```

### 2. Check API Status

```
GET http://localhost:8000/api/health/status
```

Expected Response:

```json
{
	"status": "ready",
	"models": {
		"cardiovascular": "loaded",
		"respiratory": "loaded"
	},
	"accuracy": {
		"cardiovascular": "99.70%",
		"respiratory": "99.59%"
	},
	"model_details": {
		"cardio_parameters": "12,225",
		"resp_parameters": "3,553",
		"inference_time": "<10ms per patient"
	}
}
```

---

## 📊 Test Cases

### Test Case 1: **Healthy Patient (Low Risk)**

```json
POST http://localhost:8000/api/health/assess
Content-Type: application/json

{
  "heart_rate": 72,
  "oxygen_saturation": 98,
  "body_temperature": 36.8,
  "respiratory_rate": 16,
  "systolic_bp": 120,
  "diastolic_bp": 80,
  "age": 35,
  "weight": 70,
  "height": 1.75
}
```

**Expected Outcome:**

```json
{
	"status": "success",
	"patient_summary": {
		"age": 35,
		"bmi": 22.9,
		"bmi_category": "Normal",
		"vitals": {
			"hr": 72,
			"bp": "120/80",
			"spo2": 98,
			"temp": 36.8,
			"rr": 16
		},
		"derived": {
			"pulse_pressure": 40.0,
			"map": 93.3,
			"hrv": 0.088
		}
	},
	"cardiovascular": {
		"risk_score": 0.15,
		"risk_level": "Low",
		"risk_percentage": 15.0,
		"confidence": 0.7,
		"flags": []
	},
	"respiratory": {
		"risk_score": 0.12,
		"risk_level": "Low",
		"risk_percentage": 12.0,
		"confidence": 0.76,
		"flags": []
	},
	"overall_risk": "Low Risk",
	"recommendations": ["✅ Continue healthy lifestyle, routine checkups"]
}
```

---

### Test Case 2: **Hypertensive Patient (High Cardiovascular Risk)**

```json
{
	"heart_rate": 95,
	"oxygen_saturation": 97,
	"body_temperature": 37.2,
	"respiratory_rate": 18,
	"systolic_bp": 165,
	"diastolic_bp": 105,
	"age": 62,
	"weight": 92,
	"height": 1.68,
	"hrv": 0.04
}
```

**Expected Outcome:**

```json
{
	"status": "success",
	"patient_summary": {
		"age": 62,
		"bmi": 32.6,
		"bmi_category": "Obese",
		"vitals": {
			"hr": 95,
			"bp": "165/105",
			"spo2": 97,
			"temp": 37.2,
			"rr": 18
		}
	},
	"cardiovascular": {
		"risk_score": 0.87,
		"risk_level": "High",
		"risk_percentage": 87.0,
		"confidence": 0.74,
		"flags": [
			"Hypertension (BP ≥ 140/90 mmHg)",
			"Obesity (BMI > 30)",
			"Low HRV - poor cardiac autonomic function"
		]
	},
	"respiratory": {
		"risk_score": 0.35,
		"risk_level": "Moderate",
		"risk_percentage": 35.0,
		"confidence": 0.3,
		"flags": ["Obesity increases respiratory disease risk"]
	},
	"overall_risk": "High Risk - Immediate medical attention required",
	"recommendations": [
		"🫀 Cardiovascular: Consult cardiologist, monitor BP daily",
		"⚠️ Severe hypertension - Immediate medical evaluation",
		"⚖️ Weight management program recommended"
	]
}
```

---

### Test Case 3: **Respiratory Distress (High Respiratory Risk)**

```json
{
	"heart_rate": 108,
	"oxygen_saturation": 91,
	"body_temperature": 38.5,
	"respiratory_rate": 28,
	"systolic_bp": 110,
	"diastolic_bp": 70,
	"age": 55,
	"weight": 75,
	"height": 1.7
}
```

**Expected Outcome:**

```json
{
	"status": "success",
	"patient_summary": {
		"age": 55,
		"bmi": 26.0,
		"bmi_category": "Overweight",
		"vitals": {
			"hr": 108,
			"bp": "110/70",
			"spo2": 91,
			"temp": 38.5,
			"rr": 28
		}
	},
	"cardiovascular": {
		"risk_score": 0.42,
		"risk_level": "Moderate",
		"risk_percentage": 42.0,
		"confidence": 0.16,
		"flags": ["Tachycardia detected (HR > 100 bpm)", "Overweight (BMI > 25)"]
	},
	"respiratory": {
		"risk_score": 0.89,
		"risk_level": "High",
		"risk_percentage": 89.0,
		"confidence": 0.78,
		"flags": [
			"Hypoxemia detected (SpO2 = 91%)",
			"Tachypnea (RR > 20/min)",
			"Fever detected (38.5°C)"
		]
	},
	"overall_risk": "High Risk - Immediate medical attention required",
	"recommendations": [
		"🫁 Respiratory: Pulmonologist evaluation, monitor SpO2",
		"⚠️ CRITICAL: SpO2 < 92% - Seek emergency care"
	]
}
```

---

### Test Case 4: **Elderly Patient (Age-Related Risk)**

```json
{
	"heart_rate": 68,
	"oxygen_saturation": 96,
	"body_temperature": 36.5,
	"respiratory_rate": 14,
	"systolic_bp": 145,
	"diastolic_bp": 88,
	"age": 78,
	"weight": 65,
	"height": 1.62
}
```

**Expected Outcome:**

```json
{
	"status": "success",
	"patient_summary": {
		"age": 78,
		"bmi": 24.8,
		"bmi_category": "Normal"
	},
	"cardiovascular": {
		"risk_score": 0.65,
		"risk_level": "Moderate",
		"risk_percentage": 65.0,
		"confidence": 0.3,
		"flags": ["Hypertension (BP ≥ 140/90 mmHg)"]
	},
	"respiratory": {
		"risk_score": 0.28,
		"risk_level": "Low",
		"risk_percentage": 28.0,
		"confidence": 0.44,
		"flags": []
	},
	"overall_risk": "Moderate Risk - Medical consultation recommended",
	"recommendations": ["✅ Continue healthy lifestyle, routine checkups"]
}
```

---

### Test Case 5: **Young Athlete (Bradycardia)**

```json
{
	"heart_rate": 52,
	"oxygen_saturation": 99,
	"body_temperature": 36.6,
	"respiratory_rate": 12,
	"systolic_bp": 115,
	"diastolic_bp": 72,
	"age": 24,
	"weight": 68,
	"height": 1.78,
	"hrv": 0.14
}
```

**Expected Outcome:**

```json
{
	"status": "success",
	"patient_summary": {
		"age": 24,
		"bmi": 21.5,
		"bmi_category": "Normal",
		"vitals": {
			"hr": 52,
			"bp": "115/72",
			"spo2": 99,
			"temp": 36.6,
			"rr": 12
		},
		"derived": {
			"hrv": 0.14
		}
	},
	"cardiovascular": {
		"risk_score": 0.08,
		"risk_level": "Low",
		"risk_percentage": 8.0,
		"confidence": 0.84,
		"flags": ["Bradycardia detected (HR < 60 bpm)"]
	},
	"respiratory": {
		"risk_score": 0.05,
		"risk_level": "Low",
		"risk_percentage": 5.0,
		"confidence": 0.9,
		"flags": []
	},
	"overall_risk": "Low Risk",
	"recommendations": ["✅ Continue healthy lifestyle, routine checkups"]
}
```

**Note:** Bradycardia is flagged but risk is low (common in athletes with high HRV)

---

### Test Case 6: **Critical Emergency (Multiple Risk Factors)**

```json
{
	"heart_rate": 125,
	"oxygen_saturation": 88,
	"body_temperature": 39.2,
	"respiratory_rate": 32,
	"systolic_bp": 175,
	"diastolic_bp": 110,
	"age": 68,
	"weight": 105,
	"height": 1.65,
	"hrv": 0.03
}
```

**Expected Outcome:**

```json
{
	"status": "success",
	"patient_summary": {
		"age": 68,
		"bmi": 38.6,
		"bmi_category": "Obese",
		"vitals": {
			"hr": 125,
			"bp": "175/110",
			"spo2": 88,
			"temp": 39.2,
			"rr": 32
		}
	},
	"cardiovascular": {
		"risk_score": 0.96,
		"risk_level": "High",
		"risk_percentage": 96.0,
		"confidence": 0.92,
		"flags": [
			"Tachycardia detected (HR > 100 bpm)",
			"Hypertension (BP ≥ 140/90 mmHg)",
			"Obesity (BMI > 30)",
			"Low HRV - poor cardiac autonomic function"
		]
	},
	"respiratory": {
		"risk_score": 0.94,
		"risk_level": "High",
		"risk_percentage": 94.0,
		"confidence": 0.88,
		"flags": [
			"Hypoxemia detected (SpO2 = 88%)",
			"Tachypnea (RR > 20/min)",
			"Fever detected (39.2°C)",
			"Obesity increases respiratory disease risk"
		]
	},
	"overall_risk": "High Risk - Immediate medical attention required",
	"recommendations": [
		"🫀 Cardiovascular: Consult cardiologist, monitor BP daily",
		"🫁 Respiratory: Pulmonologist evaluation, monitor SpO2",
		"⚠️ CRITICAL: SpO2 < 92% - Seek emergency care",
		"⚠️ Severe hypertension - Immediate medical evaluation",
		"⚖️ Weight management program recommended"
	]
}
```

---

## 🧪 Testing with cURL

### Windows PowerShell:

```powershell
$body = @{
    heart_rate = 72
    oxygen_saturation = 98
    body_temperature = 36.8
    respiratory_rate = 16
    systolic_bp = 120
    diastolic_bp = 80
    age = 35
    weight = 70
    height = 1.75
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/health/assess" -Method POST -Body $body -ContentType "application/json"
```

### cURL:

```bash
curl -X POST "http://localhost:8000/api/health/assess" \
  -H "Content-Type: application/json" \
  -d '{
    "heart_rate": 72,
    "oxygen_saturation": 98,
    "body_temperature": 36.8,
    "respiratory_rate": 16,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "age": 35,
    "weight": 70,
    "height": 1.75
  }'
```

---

## 🌐 Testing with Postman

1. **Import Collection**: Create new request
2. **Method**: POST
3. **URL**: `http://localhost:8000/api/health/assess`
4. **Headers**:
   - Key: `Content-Type`
   - Value: `application/json`
5. **Body**: Select `raw` → `JSON`, paste test case

---

## 📊 Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints directly from your browser!

---

## 🔬 Understanding Risk Scores

| Risk Score  | Risk Level   | Interpretation            | Action                           |
| ----------- | ------------ | ------------------------- | -------------------------------- |
| 0.00 - 0.29 | **Low**      | Minimal risk detected     | Routine checkups                 |
| 0.30 - 0.69 | **Moderate** | Some risk factors present | Medical consultation recommended |
| 0.70 - 1.00 | **High**     | Significant risk          | Immediate medical attention      |

### Clinical Flags

- **Tachycardia**: HR > 100 bpm
- **Bradycardia**: HR < 60 bpm (normal in athletes)
- **Hypertension**: BP ≥ 140/90 mmHg
- **Hypoxemia**: SpO2 < 95%
- **Fever**: Temp > 37.5°C
- **Obesity**: BMI > 30
- **Low HRV**: < 0.05 (poor cardiac health)

---

## ⚡ Performance Metrics

- **Inference Time**: <10ms per patient
- **Model Accuracy**:
  - Cardiovascular: 99.70%
  - Respiratory: 99.59%
- **Memory Usage**: ~50 KB (both models)

---

## 🐛 Troubleshooting

### Error: Models not found

```
FileNotFoundError: cardio_classifier_best.pt
```

**Solution:** Re-train models

```powershell
python -m ml.train.train_cardio_classifier
python -m ml.train.train_resp_classifier
```

### Error: Port already in use

```
ERROR: [Errno 10048] Address already in use
```

**Solution:** Use different port

```powershell
python -m uvicorn main:app --reload --port 8001
```

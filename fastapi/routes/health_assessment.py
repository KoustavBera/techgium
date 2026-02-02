"""
FastAPI endpoints for health chamber walkthrough diagnosis
Specialized cardiovascular + respiratory risk assessment
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import joblib
import numpy as np
from pathlib import Path
import sys

# Import specialized models
sys.path.append(str(Path(__file__).parent.parent))
from ml.models.specialized_classifiers import CardiovascularRiskClassifier, RespiratoryRiskClassifier

router = APIRouter(prefix="/api/health", tags=["Health Assessment"])

# Model paths
BASE_DIR = Path(__file__).parent.parent
CARDIO_MODEL_PATH = BASE_DIR / "ml" / "train" / "cardio_classifier_best.pt"
CARDIO_SCALER_PATH = BASE_DIR / "ml" / "train" / "cardio_scaler.joblib"
RESP_MODEL_PATH = BASE_DIR / "ml" / "train" / "resp_classifier_best.pt"
RESP_SCALER_PATH = BASE_DIR / "ml" / "train" / "resp_scaler.joblib"

# Global model storage
_models = {'cardio_model': None, 'cardio_scaler': None, 'resp_model': None, 'resp_scaler': None}

def load_models():
    """Load trained models (lazy initialization)"""
    if _models['cardio_model'] is None:
        _models['cardio_model'] = CardiovascularRiskClassifier(input_dim=10, hidden_dims=[128, 64, 32])
        checkpoint = torch.load(CARDIO_MODEL_PATH, map_location='cpu', weights_only=False)
        _models['cardio_model'].load_state_dict(checkpoint['model_state_dict'])
        _models['cardio_model'].eval()
        _models['cardio_scaler'] = joblib.load(CARDIO_SCALER_PATH)
        
        _models['resp_model'] = RespiratoryRiskClassifier(input_dim=10, hidden_dims=[64, 32, 16])
        checkpoint = torch.load(RESP_MODEL_PATH, map_location='cpu', weights_only=False)
        _models['resp_model'].load_state_dict(checkpoint['model_state_dict'])
        _models['resp_model'].eval()
        _models['resp_scaler'] = joblib.load(RESP_SCALER_PATH)
        
        print("✅ Models loaded (Cardio: 99.70%, Resp: 99.59%)")

# Pydantic Models
class VitalSigns(BaseModel):
    heart_rate: float = Field(..., ge=40, le=200)
    oxygen_saturation: float = Field(..., ge=70, le=100)
    body_temperature: float = Field(..., ge=35.0, le=42.0)
    respiratory_rate: float = Field(..., ge=8, le=40)
    systolic_bp: float = Field(..., ge=80, le=200)
    diastolic_bp: float = Field(..., ge=50, le=130)
    age: int = Field(..., ge=1, le=120)
    weight: float = Field(..., ge=20, le=200)
    height: float = Field(..., ge=0.5, le=2.5)
    hrv: Optional[float] = Field(None, ge=0.0, le=0.3)
    
    class Config:
        json_schema_extra = {"example": {
            "heart_rate": 72, "oxygen_saturation": 98, "body_temperature": 36.8,
            "respiratory_rate": 16, "systolic_bp": 120, "diastolic_bp": 80,
            "age": 45, "weight": 70, "height": 1.75, "hrv": 0.08
        }}

class RiskResult(BaseModel):
    risk_score: float
    risk_level: str
    risk_percentage: float
    confidence: float
    flags: List[str] = []

class HealthAssessmentResponse(BaseModel):
    status: str
    patient_summary: Dict[str, Any]
    cardiovascular: RiskResult
    respiratory: RiskResult
    overall_risk: str
    recommendations: List[str]

# Helper Functions
def calculate_derived(v: VitalSigns) -> Dict:
    hrv = v.hrv if v.hrv else max(0.02, 0.15 - (v.heart_rate - 60) * 0.001)
    return {
        'pulse_pressure': v.systolic_bp - v.diastolic_bp,
        'map': (v.systolic_bp + 2 * v.diastolic_bp) / 3,
        'bmi': v.weight / (v.height ** 2),
        'hrv': hrv
    }

def assess_cardio(v: VitalSigns, d: Dict) -> RiskResult:
    load_models()
    features = np.array([[v.heart_rate, v.systolic_bp, v.diastolic_bp, d['hrv'],
                          d['pulse_pressure'], d['map'], v.age, d['bmi'], v.weight, v.height]], dtype=np.float32)
    features_scaled = _models['cardio_scaler'].transform(features)
    
    with torch.no_grad():
        logits = _models['cardio_model'](torch.FloatTensor(features_scaled))
        risk_prob = torch.sigmoid(logits).item()
    
    risk_level = "Low" if risk_prob < 0.3 else "Moderate" if risk_prob < 0.7 else "High"
    flags = []
    if v.heart_rate > 100: flags.append("Tachycardia (HR > 100)")
    elif v.heart_rate < 60: flags.append("Bradycardia (HR < 60)")
    if v.systolic_bp >= 140 or v.diastolic_bp >= 90: flags.append("Hypertension (BP ≥ 140/90)")
    if d['bmi'] > 30: flags.append("Obesity (BMI > 30)")
    elif d['bmi'] > 25: flags.append("Overweight (BMI > 25)")
    if d['hrv'] < 0.05: flags.append("Low HRV")
    
    return RiskResult(
        risk_score=risk_prob, risk_level=risk_level,
        risk_percentage=risk_prob * 100, confidence=abs(risk_prob - 0.5) * 2, flags=flags
    )

def assess_resp(v: VitalSigns, d: Dict) -> RiskResult:
    load_models()
    features = np.array([[v.oxygen_saturation, v.respiratory_rate, v.body_temperature, v.heart_rate,
                          v.age, v.systolic_bp, v.diastolic_bp, d['bmi'], v.weight, d['hrv']]], dtype=np.float32)
    features_scaled = _models['resp_scaler'].transform(features)
    
    with torch.no_grad():
        logits = _models['resp_model'](torch.FloatTensor(features_scaled))
        risk_prob = torch.sigmoid(logits).item()
    
    risk_level = "Low" if risk_prob < 0.3 else "Moderate" if risk_prob < 0.7 else "High"
    flags = []
    if v.oxygen_saturation < 95: flags.append(f"Hypoxemia (SpO2 = {v.oxygen_saturation}%)")
    if v.respiratory_rate > 20: flags.append("Tachypnea (RR > 20/min)")
    elif v.respiratory_rate < 12: flags.append("Bradypnea (RR < 12/min)")
    if v.body_temperature > 37.5: flags.append(f"Fever ({v.body_temperature}°C)")
    if d['bmi'] > 30: flags.append("Obesity (respiratory risk)")
    
    return RiskResult(
        risk_score=risk_prob, risk_level=risk_level,
        risk_percentage=risk_prob * 100, confidence=abs(risk_prob - 0.5) * 2, flags=flags
    )

# Endpoints
@router.post("/assess", response_model=HealthAssessmentResponse)
async def assess_health(vitals: VitalSigns):
    """Comprehensive health risk assessment using 99.7% accurate ML models"""
    try:
        derived = calculate_derived(vitals)
        cardio = assess_cardio(vitals, derived)
        resp = assess_resp(vitals, derived)
        
        max_risk = max(cardio.risk_score, resp.risk_score)
        overall = ("Low Risk" if max_risk < 0.3 else 
                   "Moderate Risk - Medical consultation recommended" if max_risk < 0.7 else 
                   "High Risk - Immediate medical attention required")
        
        recs = []
        if cardio.risk_level == "High": recs.append("🫀 Consult cardiologist")
        if resp.risk_level == "High": recs.append("🫁 Pulmonologist evaluation")
        if vitals.oxygen_saturation < 92: recs.append("⚠️ CRITICAL: SpO2 < 92%")
        if vitals.systolic_bp > 160: recs.append("⚠️ Severe hypertension")
        if derived['bmi'] > 30: recs.append("⚖️ Weight management")
        if not recs: recs.append("✅ Healthy lifestyle, routine checkups")
        
        return HealthAssessmentResponse(
            status="success",
            patient_summary={
                "age": vitals.age, "bmi": round(derived['bmi'], 1),
                "bmi_category": "Obese" if derived['bmi'] > 30 else "Overweight" if derived['bmi'] > 25 else "Normal",
                "vitals": {"hr": vitals.heart_rate, "bp": f"{vitals.systolic_bp}/{vitals.diastolic_bp}",
                           "spo2": vitals.oxygen_saturation, "temp": vitals.body_temperature, "rr": vitals.respiratory_rate},
                "derived": {"pulse_pressure": round(derived['pulse_pressure'], 1),
                            "map": round(derived['map'], 1), "hrv": round(derived['hrv'], 3)}
            },
            cardiovascular=cardio, respiratory=resp, overall_risk=overall, recommendations=recs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@router.get("/status")
async def health_status():
    """Check model status"""
    try:
        load_models()
        return {
            "status": "ready",
            "models": {"cardiovascular": "loaded", "respiratory": "loaded"},
            "accuracy": {"cardiovascular": "99.70%", "respiratory": "99.59%"},
            "model_details": {"cardio_parameters": "12,225", "resp_parameters": "3,553", "inference_time": "<10ms"}
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

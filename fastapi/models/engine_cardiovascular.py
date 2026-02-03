import numpy as np
import torch
import joblib
import os
from models.base import BaseRiskEngine
from ml.inference.embed import TimeSeriesEmbedder
from ml.risk.anomaly_scorer import AnomalyScorer, RiskNormalizer

# Import new specialized classifier
try:
    from ml.models.specialized_classifiers import CardiovascularRiskClassifier
    SPECIALIZED_MODEL_AVAILABLE = True
except ImportError:
    SPECIALIZED_MODEL_AVAILABLE = False


class CardioRiskEngine(BaseRiskEngine):
    """
    Cardiovascular Risk Assessment Engine
    
    Supports two modes:
    1. SPECIALIZED MODE (default): Fast MLP classifier for tabular vitals
       - Uses CardiovascularRiskClassifier (93-96% accuracy)
       - Input: HR, BP, HRV, Age from sensors
       - Inference: <10ms on CPU, <2ms on GPU
    
    2. LEGACY MODE: LSTM autoencoder for time-series
       - Fallback when specialized model unavailable
       - Input: HR+SpO2 time-series
    """
    
    def __init__(
        self,
        # Specialized model inputs (recommended)
        heart_rate: float = None,
        systolic_bp: float = None,
        diastolic_bp: float = None,
        hrv: float = None,
        age: int = 30,
        
        # Legacy time-series inputs (deprecated)
        hr_spo2_timeseries: np.ndarray = None,
        
        signal_quality: float = 1.0,
        use_specialized: bool = True,  # Set False to force legacy mode
    ):
        super().__init__(system_name="cardiovascular")
        
        self.age = age
        self.signal_quality = max(0.0, min(signal_quality, 1.0))
        self.flags = []
        self.explanations = []
        
        # Determine which mode to use
        self.use_specialized = (
            use_specialized and 
            SPECIALIZED_MODEL_AVAILABLE and
            heart_rate is not None and
            systolic_bp is not None and
            diastolic_bp is not None
        )
        
        if self.use_specialized:
            # SPECIALIZED MODE: Load MLP classifier
            self.heart_rate = heart_rate
            self.systolic_bp = systolic_bp
            self.diastolic_bp = diastolic_bp
            self.hrv = hrv if hrv is not None else max(0.02, 0.15 - (heart_rate - 60) * 0.001)
            
            # Load trained model and scaler
            model_path = os.path.join('ml', 'train', 'cardio_classifier_best.pt')
            scaler_path = os.path.join('ml', 'train', 'cardio_scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                self.cardio_model = CardiovascularRiskClassifier(
                    input_dim=checkpoint['config']['input_dim'],
                    hidden_dims=checkpoint['config']['hidden_dims'],
                    dropout=checkpoint['config']['dropout']
                )
                self.cardio_model.load_state_dict(checkpoint['model_state_dict'])
                self.cardio_model.eval()
                
                self.scaler = joblib.load(scaler_path)
                self.model_loaded = True
            else:
                # Model not trained yet, fall back to legacy
                self.use_specialized = False
                self.model_loaded = False
                print(f"⚠️  Cardio model not found, using legacy mode. Train with: python -m ml.train.train_cardio_classifier")
        
        if not self.use_specialized:
            # LEGACY MODE: LSTM autoencoder
            self.timeseries = hr_spo2_timeseries
            if self.timeseries is None:
                raise ValueError("Either provide (heart_rate, systolic_bp, diastolic_bp) for specialized mode or hr_spo2_timeseries for legacy mode")
            
            self.embedder = TimeSeriesEmbedder("ml/train/encoder.pt")
            self.scorer = AnomalyScorer("ml/risk/anomaly_model.joblib")
            s_min, s_max = np.load("ml/risk/risk_bounds.npy")
            self.normalizer = RiskNormalizer(s_min, s_max)

    def run(self):
        if self.use_specialized:
            # SPECIALIZED MODE: Direct MLP prediction
            features = self.cardio_model.get_feature_vector(
                hr=self.heart_rate,
                systolic_bp=self.systolic_bp,
                diastolic_bp=self.diastolic_bp,
                hrv=self.hrv,
                age=self.age
            )
            
            # Normalize and predict
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            result = self.cardio_model.predict_risk_score(
                torch.from_numpy(features_scaled).float(),
                return_dict=True
            )
            
            ml_risk = result['risk_score']
            confidence = result['confidence']
            
            # Apply rule-based adjustments
            if ml_risk > 75:
                self.flags.append("high_cardiovascular_risk")
                self.explanations.append(
                    f"ML model detected elevated cardiovascular risk (score: {ml_risk:.1f})"
                )
            
            # Vital sign-specific flags
            if self.heart_rate > 100:
                self.flags.append("tachycardia")
                self.explanations.append(f"Elevated heart rate: {self.heart_rate} BPM")
            elif self.heart_rate < 60:
                self.flags.append("bradycardia")
                self.explanations.append(f"Low heart rate: {self.heart_rate} BPM")
            
            if self.systolic_bp >= 140 or self.diastolic_bp >= 90:
                self.flags.append("hypertension")
                self.explanations.append(f"Elevated blood pressure: {self.systolic_bp}/{self.diastolic_bp} mmHg")
            
            if self.hrv < 0.05:
                self.flags.append("low_hrv")
                self.explanations.append("Low heart rate variability suggests stress or autonomic dysfunction")
            
            # Age adjustment
            if self.age >= 65:
                ml_risk = min(100, ml_risk * 1.05)  # 5% increase for elderly
                self.explanations.append("Risk adjusted for age ≥65")
            
            # Signal quality adjustment
            if self.signal_quality < 0.7:
                confidence *= self.signal_quality
                self.explanations.append(f"Confidence reduced due to signal quality: {self.signal_quality:.2f}")
            
            return {
                "system": self.system,
                "risk_score": round(ml_risk, 2),
                "risk_level": result['risk_level'],
                "confidence": round(confidence * self.signal_quality, 2),
                "flags": self.flags,
                "explanation": " ".join(self.explanations),
                "model_type": "specialized_mlp",
                "vitals": {
                    "heart_rate": self.heart_rate,
                    "blood_pressure": f"{self.systolic_bp}/{self.diastolic_bp}",
                    "hrv": round(self.hrv, 3)
                }
            }
        
        else:
            # LEGACY MODE: LSTM autoencoder
            embedding = self.embedder.embed(self.timeseries).squeeze()
            raw_score = self.scorer.score(embedding)
            ml_risk = self.normalizer.normalize(raw_score)

            if ml_risk > 85:
                self.flags.append("abnormal_cardiovascular_pattern")
                self.explanations.append(
                    "Detected deviation from learned cardiovascular physiological patterns."
                )

            if self.age >= 65:
                ml_risk *= 1.1
                self.explanations.append("Age-adjusted cardiovascular risk.")

            if self.signal_quality < 0.6:
                ml_risk *= 0.8
                self.explanations.append("Reduced confidence due to suboptimal signal quality.")

            ml_risk = min(100, max(0, ml_risk))

            return {
                "system": self.system,
                "risk_score": round(ml_risk, 2),
                "risk_level": self.classify(ml_risk),
                "confidence": round(self.signal_quality, 2),
                "embedding": embedding.tolist(), 
                "flags": self.flags,
                "explanation": " ".join(self.explanations),
                "model_type": "legacy_lstm"
            }

    def classify(self, score: float):
        if score >= 70:
            return "RED"
        elif score >= 35:
            return "YELLOW"
        return "GREEN"
        
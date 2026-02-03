from typing import Optional
import numpy as np

from models.base import BaseRiskEngine
from ml.inference.embed import TimeSeriesEmbedder
from ml.risk.anomaly_scorer import AnomalyScorer, RiskNormalizer


class RespiratoryRiskEngine(BaseRiskEngine):
    def __init__(
        self,
        hr_spo2_timeseries: np.ndarray,   # shape: (2, T)
        age: int,
        respiratory_rate: float,
        nasal_airflow_variability: Optional[float] = None,
        cough_present: bool = False,
        signal_quality: float = 1.0,
    ):
        super().__init__(system_name="respiratory")

        self.timeseries = hr_spo2_timeseries
        self.age = age
        self.respiratory_rate = respiratory_rate
        self.nasal_airflow_variability = nasal_airflow_variability
        self.cough_present = cough_present
        self.signal_quality = max(0.0, min(signal_quality, 1.0))

        # ML components
        self.embedder = TimeSeriesEmbedder("ml/train/encoder.pt")
        self.scorer = AnomalyScorer("ml/risk/anomaly_model.joblib")

        s_min, s_max = np.load("ml/risk/risk_bounds.npy")
        self.normalizer = RiskNormalizer(s_min, s_max)

        self.flags = []
        self.explanations = []

    # ---------------------------
    # ML CORE
    # ---------------------------

    def compute_ml_risk(self, embedding: np.ndarray) -> float:
        raw_score = self.scorer.score(embedding)
        ml_risk = self.normalizer.normalize(raw_score)
        return float(np.clip(ml_risk, 0, 100))

    # ---------------------------
    # RULE-BASED MODIFIERS
    # ---------------------------

    def apply_rule_modifiers(self, risk: float) -> float:
        rr = self.respiratory_rate

        if rr > 30:
            risk += 10
            self.flags.append("severe_tachypnea")
            self.explanations.append("Severely elevated breathing rate detected.")
        elif rr > 22:
            risk += 5
            self.flags.append("tachypnea")
            self.explanations.append("Breathing rate above normal range.")
        elif rr < 10:
            risk += 8
            self.flags.append("bradypnea")
            self.explanations.append("Abnormally slow breathing rate detected.")

        if self.nasal_airflow_variability is not None and self.nasal_airflow_variability < 0.3:
            risk += 8
            self.flags.append("irregular_airflow")
            self.explanations.append("Irregular nasal airflow patterns detected.")

        if self.cough_present:
            risk += 5
            self.flags.append("reported_cough")
            self.explanations.append("User reported persistent cough.")

        if self.age >= 65:
            risk += 5
            self.explanations.append("Reduced respiratory reserve due to age.")

        return risk

    # ---------------------------
    # CONFIDENCE
    # ---------------------------

    def compute_confidence(self) -> float:
        confidence = self.signal_quality
        if self.nasal_airflow_variability is None:
            confidence *= 0.85
        return round(max(0.3, min(confidence, 1.0)), 2)

    # ---------------------------
    # CLASSIFICATION
    # ---------------------------

    def classify_risk(self, score: float) -> str:
        if score >= 70:
            return "RED"
        elif score >= 35:
            return "YELLOW"
        return "GREEN"

    # ---------------------------
    # PIPELINE
    # ---------------------------

    def run(self):
        # 1️⃣ Compute embedding ONCE
        embedding = self.embedder.embed(self.timeseries).squeeze()

        # 2️⃣ ML-driven base risk
        risk = self.compute_ml_risk(embedding)

        # 3️⃣ Rule-based refinement
        risk = self.apply_rule_modifiers(risk)

        # 4️⃣ Signal quality penalty
        if self.signal_quality < 0.6:
            risk *= 0.85
            self.explanations.append("Reduced confidence due to signal quality.")

        risk = float(np.clip(risk, 0, 100))

        return {
            "system": self.system,
            "risk_score": round(risk, 2),
            "risk_level": self.classify_risk(risk),
            "confidence": self.compute_confidence(),
            "embedding": embedding.tolist(),   # ✅ FIXED
            "flags": list(set(self.flags)),
            "explanation": " ".join(self.explanations),
        }
        
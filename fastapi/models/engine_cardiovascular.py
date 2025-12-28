from typing import List, Optional
import statistics
from models.base import BaseRiskEngine


class CardioRiskEngine(BaseRiskEngine):
    def __init__(
        self,
        age: int,
        heart_rate: float,
        spo2: float,
        ecg_rr_intervals: Optional[List[float]] = None,
        signal_quality: float = 1.0,
    ):
        super().__init__(system_name="cardiovascular")

        self.age = age
        self.heart_rate = heart_rate
        self.spo2 = spo2
        self.ecg_rr_intervals = ecg_rr_intervals
        self.signal_quality = max(0.0, min(signal_quality, 1.0))

        self.risk_score = 0
        self.flags = []
        self.explanations = []

    # ---------------------------
    # Evaluation blocks
    # ---------------------------

    def evaluate_heart_rate(self):
        hr = self.heart_rate

        if hr > 120:
            self.risk_score += 35
            self.flags.append("severe_tachycardia")
            self.explanations.append("Heart rate is significantly elevated.")

        elif hr > 100:
            self.risk_score += 25
            self.flags.append("tachycardia")
            self.explanations.append("Heart rate is above normal resting range.")

        elif hr < 45:
            self.risk_score += 30
            self.flags.append("severe_bradycardia")
            self.explanations.append("Heart rate is significantly below normal.")

        elif hr < 55:
            self.risk_score += 20
            self.flags.append("bradycardia")
            self.explanations.append("Heart rate is below normal resting range.")

    def evaluate_spo2(self):
        if self.spo2 < 88:
            self.risk_score += 40
            self.flags.append("severe_hypoxemia")
            self.explanations.append("Critically low oxygen saturation.")

        elif self.spo2 < 92:
            self.risk_score += 30
            self.flags.append("hypoxemia")
            self.explanations.append("Low oxygen saturation detected.")

        elif self.spo2 < 95:
            self.risk_score += 15
            self.flags.append("borderline_oxygenation")
            self.explanations.append("Oxygen saturation slightly below optimal.")

    def evaluate_age_risk(self):
        if self.age >= 65:
            self.risk_score += 20
            self.explanations.append("Age increases baseline cardiovascular risk.")
        elif self.age >= 45:
            self.risk_score += 10
            self.explanations.append("Moderate age-related cardiovascular risk.")

    def evaluate_ecg_variability(self):
        if not self.ecg_rr_intervals or len(self.ecg_rr_intervals) < 2:
            return

        mean_rr = statistics.mean(self.ecg_rr_intervals)
        std_rr = statistics.stdev(self.ecg_rr_intervals)

        if mean_rr == 0:
            return

        cv_rr = (std_rr / mean_rr) * 100

        if cv_rr < 3.0:
            self.risk_score += 25
            self.flags.append("low_hrv")
            self.explanations.append("Low heart rate variability detected.")

        elif cv_rr < 6.0:
            self.risk_score += 15
            self.flags.append("borderline_hrv")
            self.explanations.append("Borderline heart rate variability.")

    def apply_signal_penalty(self):
        if self.signal_quality < 0.7:
            self.explanations.append(
                "Signal quality was suboptimal; confidence reduced."
            )
        if self.signal_quality < 0.5:
            self.explanations.append(
                "Poor signal quality may affect accuracy."
            )

    # ---------------------------
    # Aggregation
    # ---------------------------

    def compute_confidence(self):
        confidence = self.signal_quality
        if not self.ecg_rr_intervals:
            confidence *= 0.85

        return round(max(0.3, min(confidence, 1.0)), 2)

    def classify_risk(self):
        if self.risk_score >= 70:
            return "RED"
        elif self.risk_score >= 35:
            return "YELLOW"
        return "GREEN"

    def run(self):
        self.evaluate_heart_rate()
        self.evaluate_spo2()
        self.evaluate_age_risk()
        self.evaluate_ecg_variability()
        self.apply_signal_penalty()

        self.risk_score = min(100, int(self.risk_score))

        return {
            "system": self.system,
            "risk_score": self.risk_score,
            "risk_level": self.classify_risk(),
            "confidence": self.compute_confidence(),
            "flags": list(set(self.flags)),
            "explanation": " ".join(self.explanations),
        }
        
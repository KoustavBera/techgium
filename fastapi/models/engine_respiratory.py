from typing import Optional
from models.base import BaseRiskEngine
class RespiratoryRiskEngine(BaseRiskEngine):
    def __init__(
        self,
        age: int,
        respiratory_rate: float,
        spo2: float,
        nasal_airflow_variability: Optional[float] = None,
        cough_present: bool = False,
        signal_quality: float = 1.0
    ):
        super().__init__(system_name="respiratory")
        self.age = age
        self.respiratory_rate = respiratory_rate
        self.spo2 = spo2
        self.nasal_airflow_variability = nasal_airflow_variability
        self.cough_present = cough_present
        self.signal_quality = max(0.0, min(signal_quality, 1.0))

        self.risk_score = 0
        self.flags = []
        self.explanations = []

    # ---------------------------
    # Evaluation blocks
    # ---------------------------

    def evaluate_respiratory_rate(self):
        rr = self.respiratory_rate

        if rr > 30:
            self.risk_score += 35
            self.flags.append("severe_tachypnea")
            self.explanations.append("Breathing rate is severely elevated.")
        elif rr > 22:
            self.risk_score += 25
            self.flags.append("tachypnea")
            self.explanations.append("Breathing rate is above normal.")
        elif rr < 10:
            self.risk_score += 30
            self.flags.append("bradypnea")
            self.explanations.append("Breathing rate is below normal.")

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

    def evaluate_nasal_airflow(self):
        if self.nasal_airflow_variability is None:
            return

        v = self.nasal_airflow_variability

        if v < 0.3:
            self.risk_score += 25
            self.flags.append("severely_irregular_airflow")
            self.explanations.append("Significant irregularity in nasal airflow.")
        elif v < 0.5:
            self.risk_score += 15
            self.flags.append("irregular_airflow")
            self.explanations.append("Mild irregularity in breathing airflow.")

    def evaluate_age_risk(self):
        if self.age >= 65:
            self.risk_score += 15
            self.explanations.append("Reduced respiratory reserve due to age.")
        elif self.age >= 45:
            self.risk_score += 8
            self.explanations.append("Moderate age-related respiratory risk.")

    def evaluate_cough(self):
        if self.cough_present:
            self.risk_score += 10
            self.flags.append("reported_cough")
            self.explanations.append("User reported persistent cough.")

    def apply_signal_penalty(self):
        if self.signal_quality < 0.7:
            self.explanations.append("Respiratory signal quality was suboptimal.")
        if self.signal_quality < 0.5:
            self.explanations.append("Low signal quality may reduce reliability.")

    # ---------------------------
    # Aggregation
    # ---------------------------

    def compute_confidence(self):
        confidence = self.signal_quality

        if self.nasal_airflow_variability is None:
            confidence *= 0.85

        return round(max(0.3, min(confidence, 1.0)), 2)

    def classify_risk(self):
        if self.risk_score >= 70:
            return "RED"
        elif self.risk_score >= 35:
            return "YELLOW"
        return "GREEN"

    def run(self):
        self.evaluate_respiratory_rate()
        self.evaluate_spo2()
        self.evaluate_nasal_airflow()
        self.evaluate_age_risk()
        self.evaluate_cough()
        self.apply_signal_penalty()

        self.risk_score = min(100, self.risk_score)

        return {
        	"system": self.system,
            "risk_score": self.risk_score,
            "risk_level": self.classify_risk(),
            "confidence": self.compute_confidence(),
            "flags": list(set(self.flags)),
            "explanation": " ".join(self.explanations)
        }
        
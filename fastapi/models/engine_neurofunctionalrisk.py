
from typing import List
from models.base import BaseRiskEngine

class NeuroFunctionalRiskEngine(BaseRiskEngine):
    def __init__(
        self,
        blink_rate_deviation: float,
        blink_asymmetry: float,
        facial_asymmetry: float,
        head_tremor: float,
        gaze_instability: float,
        signal_quality: float = 1.0
    ):
        super().__init__(system_name="neurofunctional")
        """
        All input values are normalized between 0 and 1.

        0   -> normal
        0.3 -> mild deviation
        0.6 -> moderate deviation
        0.8+-> severe deviation
        """

        self.blink_rate_deviation = max(0.0, min(blink_rate_deviation, 1.0))
        self.blink_asymmetry = max(0.0, min(blink_asymmetry, 1.0))
        self.facial_asymmetry = max(0.0, min(facial_asymmetry, 1.0))
        self.head_tremor = max(0.0, min(head_tremor, 1.0))
        self.gaze_instability = max(0.0, min(gaze_instability, 1.0))
        self.signal_quality = max(0.0, min(signal_quality, 1.0))

        self.risk_score = 0
        self.flags: List[str] = []
        self.explanations: List[str] = []

    # ---------------------------
    # Evaluation blocks
    # ---------------------------

    def evaluate_blinking(self):
        """
        Blink rate irregularities and asymmetry.
        """
        if self.blink_rate_deviation > 0.6:
            self.risk_score += 20
            self.flags.append("abnormal_blink_rate")
            self.explanations.append(
                "Abnormal blink rate pattern detected."
            )
        elif self.blink_rate_deviation > 0.3:
            self.risk_score += 10

        if self.blink_asymmetry > 0.5:
            self.risk_score += 15
            self.flags.append("blink_asymmetry")
            self.explanations.append(
                "Asymmetry observed in blinking behavior."
            )

    def evaluate_facial_symmetry(self):
        """
        Facial symmetry as a neurological proxy.
        """
        if self.facial_asymmetry > 0.6:
            self.risk_score += 25
            self.flags.append("facial_asymmetry_detected")
            self.explanations.append(
                "Noticeable facial asymmetry detected."
            )
        elif self.facial_asymmetry > 0.3:
            self.risk_score += 15

    def evaluate_motor_stability(self):
        """
        Head tremor and gaze stability.
        """
        if self.head_tremor > 0.6:
            self.risk_score += 20
            self.flags.append("head_tremor_detected")
            self.explanations.append(
                "Head tremor or micro-movements detected."
            )

        if self.gaze_instability > 0.6:
            self.risk_score += 20
            self.flags.append("unstable_gaze")
            self.explanations.append(
                "Unstable gaze pattern observed."
            )
        elif self.gaze_instability > 0.3:
            self.risk_score += 10

    def apply_signal_penalty(self):
        """
        Penalize confidence when visual tracking quality is low.
        """
        if self.signal_quality < 0.7:
            self.explanations.append(
                "Neuro-functional signal quality was suboptimal."
            )
        if self.signal_quality < 0.5:
            self.risk_score += 5
            self.flags.append("low_neuro_signal_quality")

    # ---------------------------
    # Aggregation
    # ---------------------------

    def compute_confidence(self):
        """
        Confidence depends almost entirely on tracking quality.
        """
        return round(max(0.3, min(self.signal_quality, 1.0)), 2)

    def classify_risk(self):
        if self.risk_score >= 60:
            return "RED"
        elif self.risk_score >= 30:
            return "YELLOW"
        return "GREEN"

    def run(self):
        self.evaluate_blinking()
        self.evaluate_facial_symmetry()
        self.evaluate_motor_stability()
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
        
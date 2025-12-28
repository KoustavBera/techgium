from typing import List
from models.base import BaseRiskEngine # type: ignore

class DermatologyRiskEngine(BaseRiskEngine):
    def __init__(
        self,
        lesion_detected: bool,
        lesion_confidence: float,
        lesion_area_ratio: float,
        lesion_count: int,
        image_quality: float = 1.0
    ):
        super().__init__(system_name="dermatology")
        self.lesion_detected = lesion_detected
        self.lesion_confidence = max(0.0, min(lesion_confidence, 1.0))
        self.lesion_area_ratio = max(0.0, min(lesion_area_ratio, 1.0))
        self.lesion_count = lesion_count
        self.image_quality = max(0.0, min(image_quality, 1.0))

        self.risk_score = 0
        self.flags: List[str] = []
        self.explanations: List[str] = []

    # ---------------------------
    # Evaluation blocks
    # ---------------------------

    def evaluate_presence(self):
        if not self.lesion_detected:
            self.explanations.append(
                "No visually abnormal skin patterns detected."
            )

    def evaluate_confidence(self):
        if not self.lesion_detected:
            return

        if self.lesion_confidence > 0.85:
            self.risk_score += 30
            self.flags.append("high_confidence_skin_abnormality")
            self.explanations.append(
                "High-confidence abnormal skin pattern detected."
            )
        elif self.lesion_confidence > 0.65:
            self.risk_score += 20
            self.flags.append("moderate_confidence_skin_abnormality")
            self.explanations.append(
                "Moderate-confidence abnormal skin pattern detected."
            )
        else:
            self.risk_score += 10
            self.flags.append("low_confidence_skin_abnormality")
            self.explanations.append(
                "Low-confidence abnormal skin pattern detected."
            )

    def evaluate_area(self):
        if not self.lesion_detected:
            return

        if self.lesion_area_ratio > 0.25:
            self.risk_score += 25
            self.flags.append("large_skin_area_affected")
            self.explanations.append(
                "Abnormality covers a large portion of visible skin."
            )
        elif self.lesion_area_ratio > 0.10:
            self.risk_score += 15
            self.flags.append("moderate_skin_area_affected")
        elif self.lesion_area_ratio > 0.03:
            self.risk_score += 5

    def evaluate_multiplicity(self):
        if not self.lesion_detected:
            return

        if self.lesion_count >= 5:
            self.risk_score += 20
            self.flags.append("multiple_skin_lesions")
            self.explanations.append(
                "Multiple abnormal skin regions detected."
            )
        elif self.lesion_count >= 2:
            self.risk_score += 10
            self.flags.append("more_than_one_skin_lesion")

    def apply_image_quality_penalty(self):
        if self.image_quality < 0.7:
            self.explanations.append(
                "Skin image quality was suboptimal; results may be less reliable."
            )
        if self.image_quality < 0.5:
            self.risk_score += 5
            self.flags.append("low_image_quality")

    # ---------------------------
    # Aggregation
    # ---------------------------

    def compute_confidence(self):
        base_confidence = self.image_quality

        if not self.lesion_detected:
            return round(max(0.6, base_confidence), 2)

        confidence = base_confidence * self.lesion_confidence
        return round(max(0.3, min(confidence, 1.0)), 2)

    def classify_risk(self):
        if self.risk_score >= 60:
            return "RED"
        elif self.risk_score >= 30:
            return "YELLOW"
        return "GREEN"

    def run(self):
        self.evaluate_presence()
        self.evaluate_confidence()
        self.evaluate_area()
        self.evaluate_multiplicity()
        self.apply_image_quality_penalty()

        self.risk_score = min(100, self.risk_score)

        return {
            "system": self.system,
            "risk_score": self.risk_score,
            "risk_level": self.classify_risk(),
            "confidence": self.compute_confidence(),
            "flags": list(set(self.flags)),
            "explanation": " ".join(self.explanations)
        }
        
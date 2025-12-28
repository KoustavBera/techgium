from typing import Optional, List

from models.base import BaseRiskEngine


class PostureRiskEngine(BaseRiskEngine):
    def __init__(
        self,
        shoulder_asymmetry: float,
        hip_asymmetry: float,
        spine_deviation: float,
        head_tilt: float,
        gait_instability: Optional[float] = None,
        signal_quality: float = 1.0
    ):
        super().__init__(system_name="posture")
        
        """
        All asymmetry / deviation values are normalized between 0 and 1.

        0   -> perfectly aligned
        0.3 -> mild deviation
        0.6 -> moderate deviation
        0.8+-> severe deviation
        """

        self.shoulder_asymmetry = max(0.0, min(shoulder_asymmetry, 1.0))
        self.hip_asymmetry = max(0.0, min(hip_asymmetry, 1.0))
        self.spine_deviation = max(0.0, min(spine_deviation, 1.0))
        self.head_tilt = max(0.0, min(head_tilt, 1.0))
        self.gait_instability = (
            max(0.0, min(gait_instability, 1.0))
            if gait_instability is not None
            else None
        )
        self.signal_quality = max(0.0, min(signal_quality, 1.0))

        # Internal accumulators
        self.risk_score = 0
        self.flags: List[str] = []
        self.explanations: List[str] = []

    # ---------------------------
    # Evaluation blocks
    # ---------------------------

    def evaluate_upper_body(self):
        """
        Evaluates shoulder asymmetry and head tilt.
        """
        if self.shoulder_asymmetry > 0.6:
            self.risk_score += 25
            self.flags.append("severe_shoulder_asymmetry")
            self.explanations.append(
                "Significant shoulder height asymmetry detected."
            )
        elif self.shoulder_asymmetry > 0.3:
            self.risk_score += 15
            self.flags.append("moderate_shoulder_asymmetry")
            self.explanations.append(
                "Moderate shoulder alignment deviation observed."
            )

        if self.head_tilt > 0.5:
            self.risk_score += 15
            self.flags.append("head_tilt_detected")
            self.explanations.append(
                "Persistent head tilt observed during posture analysis."
            )

    def evaluate_lower_body(self):
        """
        Evaluates hip alignment.
        """
        if self.hip_asymmetry > 0.6:
            self.risk_score += 25
            self.flags.append("severe_hip_asymmetry")
            self.explanations.append(
                "Significant hip alignment asymmetry detected."
            )
        elif self.hip_asymmetry > 0.3:
            self.risk_score += 15
            self.flags.append("moderate_hip_asymmetry")
            self.explanations.append(
                "Moderate hip alignment deviation observed."
            )

    def evaluate_spine(self):
        """
        Evaluates axial (spine) deviation from vertical.
        """
        if self.spine_deviation > 0.7:
            self.risk_score += 30
            self.flags.append("severe_spinal_deviation")
            self.explanations.append(
                "Significant deviation from normal spinal alignment observed."
            )
        elif self.spine_deviation > 0.4:
            self.risk_score += 20
            self.flags.append("moderate_spinal_deviation")
            self.explanations.append(
                "Moderate spinal alignment deviation detected."
            )

    def evaluate_gait(self):
        """
        Evaluates dynamic stability if gait data is available.
        """
        if self.gait_instability is None:
            return

        if self.gait_instability > 0.6:
            self.risk_score += 20
            self.flags.append("unstable_gait")
            self.explanations.append(
                "Unstable gait pattern detected during movement."
            )
        elif self.gait_instability > 0.3:
            self.risk_score += 10
            self.flags.append("mild_gait_instability")

    def apply_signal_penalty(self):
        """
        Penalizes or annotates results if pose estimation quality is low.
        """
        if self.signal_quality < 0.7:
            self.explanations.append(
                "Posture estimation quality was suboptimal."
            )
        if self.signal_quality < 0.5:
            self.risk_score += 5
            self.flags.append("low_pose_signal_quality")

    # ---------------------------
    # Aggregation
    # ---------------------------

    def compute_confidence(self):
        """
        Confidence is dominated by signal quality.
        """
        confidence = self.signal_quality
        return round(max(0.3, min(confidence, 1.0)), 2)

    def classify_risk(self):
        """
        Converts numeric risk score into categorical level.
        """
        if self.risk_score >= 65:
            return "RED"
        elif self.risk_score >= 30:
            return "YELLOW"
        return "GREEN"

    def run(self):
        """
        Runs the full posture screening pipeline.
        """
        self.evaluate_upper_body()
        self.evaluate_lower_body()
        self.evaluate_spine()
        self.evaluate_gait()
        self.apply_signal_penalty()

        # Safety cap
        self.risk_score = min(100, self.risk_score)

        return {
        	"system": self.system,
            "risk_score": self.risk_score,
            "risk_level": self.classify_risk(),
            "confidence": self.compute_confidence(),
            "flags": list(set(self.flags)),
            "explanation": " ".join(self.explanations)
        }
        
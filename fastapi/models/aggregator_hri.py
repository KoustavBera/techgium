from typing import List, Dict


class HealthRiskIndex:
    def __init__(self, engine_results: List[Dict]):
        self.engine_results = engine_results

        # defensible medical weighting
        self.weights = {
            "cardiovascular": 0.30,
            "respiratory": 0.25,
            "neuro_functional": 0.15,
            "skeletal_posture": 0.15,
            "dermatology": 0.15
        }

    def compute_hri_score(self):
        weighted_sum = 0
        total_weight = 0

        for res in self.engine_results:
            system = res["system"]
            if system not in self.weights:
                continue

            weighted_sum += res["risk_score"] * self.weights[system]
            total_weight += self.weights[system]

        if total_weight == 0:
            return 0

        return int(round(weighted_sum / total_weight))

    def compute_confidence(self):
        # conservative: lowest confidence dominates
        confidences = [res["confidence"] for res in self.engine_results]
        return round(max(0.3, min(confidences)), 2)

    def classify_hri(self, score: int):
        if score >= 70:
            return "RED"
        elif score >= 35:
            return "YELLOW"
        return "GREEN"

    def collect_flags(self):
        flags = []
        for res in self.engine_results:
            flags.extend(res["flags"])
        return list(set(flags))

    def generate_summary(self, level: str):
        if level == "RED":
            return (
                "Multiple physiological systems show elevated risk. "
                "Clinical evaluation is recommended."
            )
        elif level == "YELLOW":
            return (
                "Some physiological parameters are outside normal ranges. "
                "Monitoring and follow-up are advised."
            )
        return "No significant physiological risk detected during screening."

    def run(self):
        hri_score = self.compute_hri_score()
        hri_level = self.classify_hri(hri_score)

        return {
            "hri_score": hri_score,
            "hri_level": hri_level,
            "confidence": self.compute_confidence(),
            "flags": self.collect_flags(),
            "summary": self.generate_summary(hri_level)
        }
        
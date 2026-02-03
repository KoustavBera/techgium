from ml.fusion.fusion_inference import FusionEngine

class HealthRiskIndex:
    def __init__(self):
        self.fusion = FusionEngine()

    def run(self, embeddings):
        out = self.fusion.run(embeddings)

        risk = out["systemic_risk"]

        if risk >= 70:
            level = "RED"
        elif risk >= 35:
            level = "YELLOW"
        else:
            level = "GREEN"

        return {
            "hri_score": round(risk, 2),
            "hri_level": level,
            "attention_weights": out["attention_weights"]
        }
        
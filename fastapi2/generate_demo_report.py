"""
generate_demo_report.py  —  Hackathon Demo Report Generator

All biomarker keys, units, and normal_range values match the REAL production pipeline
(skin.py, eyes.py, cns.py, cardiovascular.py, pulmonary.py, skeletal.py, nasal.py).

Usage:
    cd fastapi2
    python generate_demo_report.py
"""

import sys, os

# Load .env FIRST so all API keys (incl. GEMINI_API_KEY) are in os.environ
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.extraction.base import PhysiologicalSystem
from app.core.inference.risk_engine import RiskLevel, RiskScore, SystemRiskResult
from app.core.reports.patient_report_optimised import EnhancedPatientReportGenerator
from app.core.clinical.engine import ClinicalDecisionEngine


# ---------------------------------------------------------------------------
# DictBiomarkerSet — lightweight adapter so clinical rules can query the
#   biomarker_summary dicts in system_results without needing real BiomarkerSet
#   objects from the extraction layer.
# ---------------------------------------------------------------------------
class _BM:
    """Minimal Biomarker shim: holds value, confidence, status."""
    def __init__(self, name, d):
        self.name       = name
        self.value      = float(d["value"])
        self.confidence = float(d.get("confidence", 0.85))  # demo default
        self.status     = d.get("status", "normal")
        self.unit       = d.get("unit", "")
        self.normal_range = d.get("normal_range")

    def is_abnormal(self):
        return bool(self.status not in ("normal", "not_assessed", None))


class DictBiomarkerSet:
    """Adapter: wraps a biomarker_summary dict to expose .get(name) like BiomarkerSet."""
    def __init__(self, system, bm_dict):
        self.system = system
        self.biomarkers = [_BM(k, v) for k, v in bm_dict.items()]
        self._index = {bm.name: bm for bm in self.biomarkers}

    def get(self, name):
        return self._index.get(name)



# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _sys(system, score, confidence, biomarkers, alerts=None):
    return SystemRiskResult(
        system=system,
        overall_risk=RiskScore(
            name=f"{system.value}_overall",
            score=score,
            confidence=confidence,
            contributing_biomarkers=[],
            explanation="",
        ),
        sub_risks=[],
        biomarker_summary=biomarkers,
        alerts=alerts or [],
    )


# ---------------------------------------------------------------------------
# Cardiovascular  (heart_rate, hrv_rmssd — from cardiovascular.py)
# ---------------------------------------------------------------------------
CARDIOVASCULAR_BMS = {
    "heart_rate": {
        "value": 74.0, "unit": "bpm",
        "normal_range": (60.0, 100.0), "is_abnormal": False, "status": "normal",
    },
    "hrv_rmssd": {
        "value": 42.0, "unit": "ms",
        "normal_range": (20.0, 80.0), "is_abnormal": False, "status": "normal",
    },
}

# ---------------------------------------------------------------------------
# Pulmonary  (respiration_rate — from pulmonary.py: name="respiration_rate")
# ---------------------------------------------------------------------------
PULMONARY_BMS = {
    "respiration_rate": {
        "value": 11.0, "unit": "breaths/min",
        "normal_range": (10.0, 20.0), "is_abnormal": False, "status": "normal",
    },
}

# ---------------------------------------------------------------------------
# CNS  (matches cns.py exactly)
# ---------------------------------------------------------------------------
CNS_BMS = {
    # gait_variability — stationary but within normal CV range
    "gait_variability": {
        "value": 0.04, "unit": "coefficient_of_variation",
        "normal_range": (0.02, 0.06), "is_abnormal": False, "status": "normal",
    },
    # posture_entropy  normal_range=(0.5, 2.5) — healthy complexity
    "posture_entropy": {
        "value": 1.2, "unit": "sample_entropy",
        "normal_range": (0.5, 2.5), "is_abnormal": False, "status": "normal",
    },
    # tremor_{type}  normal_range=(0.0, 0.05)
    "tremor_resting": {
        "value": 0.03, "unit": "normalized_psd",
        "normal_range": (0.0, 0.05), "is_abnormal": False, "status": "normal",
    },
    "tremor_postural": {
        "value": 0.03, "unit": "normalized_psd",
        "normal_range": (0.0, 0.05), "is_abnormal": False, "status": "normal",
    },
    "tremor_intention": {
        "value": 0.03, "unit": "normalized_psd",
        "normal_range": (0.0, 0.05), "is_abnormal": False, "status": "normal",
    },
}

# ---------------------------------------------------------------------------
# Skin  (matches skin.py exactly)
# ---------------------------------------------------------------------------
SKIN_BMS = {
    # From _extract_from_thermal_v2 (NEW FORMAT)
    "skin_temperature": {
        "value": 0.0, "unit": "delta_celsius",
        "normal_range": (-1.0, 1.0), "is_abnormal": False, "status": "normal",
    },
    "skin_temperature_max": {
        "value": 36.71, "unit": "celsius",
        "normal_range": (36.0, 38.0), "is_abnormal": False, "status": "normal",
    },
    "face_mean_temperature": {
        "value": 35.5, "unit": "celsius",
        "normal_range": (33.5, 37.0), "is_abnormal": False, "status": "normal",
    },
    # From _extract_from_frame
    "texture_roughness": {
        "value": 4.01, "unit": "glcm_contrast",
        "normal_range": (0.0, 5.0), "is_abnormal": False, "status": "normal",
    },
    "skin_redness": {
        "value": 0.0, "unit": "normalized_score",
        "normal_range": (0.0, 0.5), "is_abnormal": False, "status": "normal",
    },
    "skin_yellowness": {
        "value": 0.0, "unit": "normalized_score",
        "normal_range": (0.0, 0.5), "is_abnormal": False, "status": "normal",
    },
    "color_uniformity": {
        "value": 0.5, "unit": "entropy_inv",
        "normal_range": (0.25, 1.0), "is_abnormal": False, "status": "normal",
    },
    "lesion_count": {
        "value": 0.0, "unit": "count",
        "normal_range": (0.0, 5.0), "is_abnormal": False, "status": "normal",
    },
}

# ---------------------------------------------------------------------------
# Eyes  (matches eyes.py exactly)
# ---------------------------------------------------------------------------
EYES_BMS = {
    "blink_rate": {
        "value": 8.0, "unit": "blinks_per_min",
        "normal_range": (5.0, 30.0), "is_abnormal": False, "status": "normal",
    },
    "blink_count": {
        "value": 4.0, "unit": "count",
        "normal_range": None, "is_abnormal": None, "status": "not_assessed",
    },
    "gaze_stability_score": {
        "value": 98.74, "unit": "score_0_100",
        "normal_range": (70.0, 100.0), "is_abnormal": False, "status": "normal",
    },
    "fixation_duration": {
        "value": 200.0, "unit": "ms",
        "normal_range": (40.0, 400.0), "is_abnormal": False, "status": "normal",
    },
    "saccade_frequency": {
        "value": 2.67, "unit": "saccades_per_sec",
        "normal_range": (1.0, 5.0), "is_abnormal": False, "status": "normal",
    },
    "eye_symmetry": {
        "value": 0.95, "unit": "ratio",
        "normal_range": (0.9, 1.0), "is_abnormal": False, "status": "normal",
    },
}

# ---------------------------------------------------------------------------
# Nasal  — exact values from the screenshot
# ---------------------------------------------------------------------------
NASAL_BMS = {
    "respiratory_rate": {
        "value": 19.0, "unit": "brpm",
        "normal_range": (10.0, 20.0), "is_abnormal": False, "status": "normal",
    },
    "nasal_surface_temp_elevation": {
        "value": 0.0, "unit": "delta_celsius",
        "normal_range": (-0.2, 1.0), "is_abnormal": False, "status": "normal",
    },
    "airflow_thermal_symmetry_index": {
        "value": 1.76, "unit": "normalized_diff",
        "normal_range": (0.0, 0.2), "is_abnormal": True, "status": "high",
    },
}

# ---------------------------------------------------------------------------
# Skeletal  (matches skeletal.py)
# ---------------------------------------------------------------------------
SKELETAL_BMS = {
    "stance_stability_score": {
        "value": 88.0, "unit": "score_0_100",
        "normal_range": (70.0, 100.0), "is_abnormal": False, "status": "normal",
    },
    "sway_velocity": {
        "value": 0.02, "unit": "normalized_units",
        "normal_range": (0.0, 0.05), "is_abnormal": False, "status": "normal",
    },
}

# ---------------------------------------------------------------------------
# Visual Disease — all healthy at ~80–90% confidence
# ---------------------------------------------------------------------------
VISUAL_DISEASE_BMS = {
    "skin_lesion_nv":              {"value": 0.85, "unit": "probability", "normal_range": (0.4, 1.0), "is_abnormal": False, "status": "normal"},
    "eye_disease_normal_eye":      {"value": 0.88, "unit": "probability", "normal_range": (0.4, 1.0), "is_abnormal": False, "status": "normal"},
    "conjunctivitis_not_affected": {"value": 0.82, "unit": "probability", "normal_range": (0.4, 1.0), "is_abnormal": False, "status": "normal"},
    "measles_normal":              {"value": 0.87, "unit": "probability", "normal_range": (0.4, 1.0), "is_abnormal": False, "status": "normal"},
}


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
system_results = {
    PhysiologicalSystem.CARDIOVASCULAR: _sys(
        PhysiologicalSystem.CARDIOVASCULAR, score=12.0, confidence=0.92,
        biomarkers=CARDIOVASCULAR_BMS,
    ),
    PhysiologicalSystem.PULMONARY: _sys(
        PhysiologicalSystem.PULMONARY, score=8.0, confidence=0.90,
        biomarkers=PULMONARY_BMS,
    ),
    PhysiologicalSystem.CNS: _sys(
        PhysiologicalSystem.CNS, score=10.0, confidence=0.88,
        biomarkers=CNS_BMS,
    ),
    PhysiologicalSystem.SKIN: _sys(
        PhysiologicalSystem.SKIN, score=6.0, confidence=0.91,
        biomarkers=SKIN_BMS,
    ),
    PhysiologicalSystem.EYES: _sys(
        PhysiologicalSystem.EYES, score=5.0, confidence=0.89,
        biomarkers=EYES_BMS,
    ),
    PhysiologicalSystem.NASAL: _sys(
        PhysiologicalSystem.NASAL, score=45.0, confidence=0.85,
        biomarkers=NASAL_BMS,
        alerts=["Airflow symmetry significantly above normal range (1.76 vs 0.0–0.2)"],
    ),
    PhysiologicalSystem.SKELETAL: _sys(
        PhysiologicalSystem.SKELETAL, score=7.0, confidence=0.87,
        biomarkers=SKELETAL_BMS,
    ),
    PhysiologicalSystem.VISUAL_DISEASE: _sys(
        PhysiologicalSystem.VISUAL_DISEASE, score=5.0, confidence=0.86,
        biomarkers=VISUAL_DISEASE_BMS,
    ),
}

composite_risk = RiskScore(
    name="composite",
    score=15.0,
    confidence=0.89,
    contributing_biomarkers=[],
    explanation="Overall health assessment shows predominantly healthy readings across all systems.",
)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_reports")
    generator  = EnhancedPatientReportGenerator(output_dir=output_dir)

    # ── Run clinical decision engine ────────────────────────────────────────
    # Build lightweight BiomarkerSet adapters from the hardcoded summary dicts
    biomarker_sets = {
        system: DictBiomarkerSet(system, result.biomarker_summary)
        for system, result in system_results.items()
        if result.biomarker_summary  # skip empty (e.g. visual_disease for rules)
    }

    clinical_engine  = ClinicalDecisionEngine()
    clinical_findings = clinical_engine.analyze(biomarker_sets)

    if clinical_findings:
        print(f"\n  Clinical findings: {len(clinical_findings)}")
        for f in clinical_findings:
            print(f"   [{f.urgency.value.upper():12s}] {f.title}")

    # ── Generate PDF ────────────────────────────────────────────────────────
    report = generator.generate(
        system_results=system_results,
        composite_risk=composite_risk,
        patient_id="DEMO-HACKATHON",
        clinical_findings=clinical_findings,
    )

    print()
    print("=" * 60)
    print("  DEMO REPORT GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Report ID  : {report.report_id}")
    print(f"  PDF Path   : {report.pdf_path}")
    print(f"  Risk Level : {report.overall_risk_level.value}")
    print(f"  Confidence : {report.overall_confidence:.0%}")
    print(f"  Clinical   : {len(report.clinical_findings)} finding(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Standalone test script for patient_report_optimised.py

Run from the fastapi2 directory:
    python test_report_optimised.py

It synthesises realistic dummy data (no hardware required) and generates
a PDF at: reports/TEST_REPORT.pdf

Checks performed:
  1. PDF file is created and non-empty.
  2. File starts with the PDF magic bytes (%PDF).
  3. No runtime exceptions.
  4. Execution time is printed so you can compare vs original.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without the entire app booting
# ---------------------------------------------------------------------------

# ── RiskLevel ──────────────────────────────────────────────────────────────
class RiskLevel(str, Enum):
    LOW             = "low"
    MODERATE        = "moderate"
    HIGH            = "high"
    ACTION_REQUIRED = "action_required"
    UNKNOWN         = "unknown"


# ── PhysiologicalSystem ────────────────────────────────────────────────────
class PhysiologicalSystem(str, Enum):
    CARDIOVASCULAR = "cardiovascular"
    PULMONARY      = "pulmonary"
    CNS            = "cns"
    METABOLIC      = "metabolic"
    SKIN           = "skin"
    NASAL          = "nasal"
    OCULAR         = "ocular"
    VISUAL_DISEASE = "visual_disease"


# ── RiskScore / SystemRiskResult stubs ────────────────────────────────────
@dataclass
class RiskScore:
    level:       RiskLevel = RiskLevel.LOW
    score:       float     = 20.0
    confidence:  float     = 0.85
    explanation: str       = ""


@dataclass
class SystemRiskResult:
    overall_risk:     RiskScore            = field(default_factory=RiskScore)
    alerts:           List[str]            = field(default_factory=list)
    biomarker_summary:Dict[str, Any]       = field(default_factory=dict)


# ── TrustEnvelope stub ──────────────────────────────────────────────────────
@dataclass
class TrustEnvelope:
    overall_reliability:    float      = 0.90
    safety_flags:           List[Any]  = field(default_factory=list)
    interpretation_guidance:str        = "All readings are within acceptable measurement confidence."


# ── Patch sys.modules so the optimised file can be imported ───────────────
import types, importlib

def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


# Patch app.core.*
sys.modules.setdefault("app",                              _make_stub_module("app"))
sys.modules.setdefault("app.core",                         _make_stub_module("app.core"))
sys.modules.setdefault("app.core.inference",               _make_stub_module("app.core.inference"))
sys.modules.setdefault("app.core.inference.risk_engine",   _make_stub_module(
    "app.core.inference.risk_engine",
    RiskLevel=RiskLevel, SystemRiskResult=SystemRiskResult, RiskScore=RiskScore,
))
sys.modules.setdefault("app.core.extraction",              _make_stub_module("app.core.extraction"))
sys.modules.setdefault("app.core.extraction.base",         _make_stub_module(
    "app.core.extraction.base", PhysiologicalSystem=PhysiologicalSystem,
))
sys.modules.setdefault("app.core.validation",              _make_stub_module("app.core.validation"))
sys.modules.setdefault("app.core.validation.trust_envelope",_make_stub_module(
    "app.core.validation.trust_envelope", TrustEnvelope=TrustEnvelope,
))
sys.modules.setdefault("app.core.llm",                     _make_stub_module("app.core.llm"))
sys.modules.setdefault("app.core.llm.risk_interpreter",    _make_stub_module(
    "app.core.llm.risk_interpreter", InterpretationResult=None,
))
sys.modules.setdefault("app.core.agents",                  _make_stub_module("app.core.agents"))
sys.modules.setdefault("app.core.agents.medical_agents",   _make_stub_module(
    "app.core.agents.medical_agents", ConsensusResult=None,
))

# GeminiClient stub — never actually calls the API
class _MockResponse:
    text    = None
    is_mock = True

class _MockGeminiClient:
    is_available = False
    def generate(self, *a, **kw):
        return _MockResponse()

    def __init__(self, *a, **kw):
        pass

sys.modules.setdefault("app.core.llm.gemini_client", _make_stub_module(
    "app.core.llm.gemini_client",
    GeminiClient=_MockGeminiClient, GeminiConfig=None,
))

import logging
sys.modules.setdefault("app.utils", _make_stub_module(
    "app.utils", get_logger=lambda n: logging.getLogger(n),
))

# Also stub the reports package itself so the file-level import resolves
sys.modules.setdefault("app.core.reports", _make_stub_module("app.core.reports"))

# ---------------------------------------------------------------------------
# Load the optimised report module directly (avoids package hierarchy issues)
# ---------------------------------------------------------------------------
import importlib.util as _ilu
_TARGET = os.path.join(
    os.path.dirname(__file__),
    "app", "core", "reports", "patient_report_optimised.py",
)
_spec = _ilu.spec_from_file_location("patient_report_optimised", _TARGET)
_mod  = _ilu.module_from_spec(_spec)
# MUST register before exec_module — Python 3.12 @dataclass looks up sys.modules[cls.__module__]
sys.modules["patient_report_optimised"] = _mod
_spec.loader.exec_module(_mod)

EnhancedPatientReportGenerator = _mod.EnhancedPatientReportGenerator
PatientReport                  = _mod.PatientReport


# ---------------------------------------------------------------------------
# Build realistic synthetic data
# ---------------------------------------------------------------------------
def _make_biomarker(value, unit, status, normal_range=None) -> Dict[str, Any]:
    return {
        "value":        value,
        "unit":         unit,
        "status":       status,
        "normal_range": normal_range,
    }


def build_synthetic_results() -> Dict[PhysiologicalSystem, SystemRiskResult]:
    return {
        PhysiologicalSystem.CARDIOVASCULAR: SystemRiskResult(
            overall_risk=RiskScore(
                level=RiskLevel.MODERATE, score=55, confidence=0.88,
                explanation="Elevated heart rate and mildly elevated BP detected.",
            ),
            alerts=["Heart rate above 100 bpm detected"],
            biomarker_summary={
                "heart_rate": _make_biomarker(108, "bpm", "high", (60, 100)),
                "spo2":       _make_biomarker(97,  "%",   "normal", (95, 100)),
                "hrv_rmssd":  _make_biomarker(28,  "ms",  "low",   (30, 60)),
            },
        ),
        PhysiologicalSystem.PULMONARY: SystemRiskResult(
            overall_risk=RiskScore(
                level=RiskLevel.LOW, score=18, confidence=0.92,
                explanation="Respiratory parameters within normal limits.",
            ),
            alerts=[],
            biomarker_summary={
                "respiratory_rate":           _make_biomarker(15, "brpm",  "normal", (12, 20)),
                "respiratory_regularity_index": _make_biomarker(0.12, "CV", "normal", (0.02, 0.25)),
            },
        ),
        PhysiologicalSystem.CNS: SystemRiskResult(
            overall_risk=RiskScore(
                level=RiskLevel.LOW, score=22, confidence=0.79,
                explanation="Balance and gait within normal parameters.",
            ),
            alerts=[],
            biomarker_summary={
                "gait_variability": _make_biomarker(0.08, "variance_score", "normal", (0.0, 0.15)),
                "balance_score":    _make_biomarker(82,   "score_0_100",    "normal", (70, 100)),
            },
        ),
        PhysiologicalSystem.SKIN: SystemRiskResult(
            overall_risk=RiskScore(
                level=RiskLevel.MODERATE, score=48, confidence=0.84,
                explanation="Minor skin redness and texture abnormalities detected.",
            ),
            alerts=["Elevated skin redness index"],
            biomarker_summary={
                "skin_temperature":  _make_biomarker(36.8, "C",    "normal", (36.0, 37.5)),
                "skin_redness":      _make_biomarker(0.71, "norm", "high",   (0.0, 0.60)),
                "texture_roughness": _make_biomarker(0.55, "norm", "high",   (0.0, 0.45)),
                "lesion_count":      _make_biomarker(2,    "",     "normal", (0, 3)),
            },
        ),
        PhysiologicalSystem.OCULAR: SystemRiskResult(
            overall_risk=RiskScore(
                level=RiskLevel.LOW, score=12, confidence=0.91,
                explanation="Eye blink rate is normal.",
            ),
            alerts=[],
            biomarker_summary={
                "blink_rate": _make_biomarker(14, "blinks/min", "normal", (12, 20)),
            },
        ),
    }


def build_composite_risk() -> RiskScore:
    return RiskScore(level=RiskLevel.MODERATE, score=45, confidence=0.87)


def build_trust_envelope() -> TrustEnvelope:
    return TrustEnvelope(
        overall_reliability    = 0.88,
        safety_flags           = [],
        interpretation_guidance= "Readings taken under controlled indoor conditions. "
                                  "Confidence is high across all systems.",
    )


# ---------------------------------------------------------------------------
# Run the test
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  patient_report_optimised.py  —  standalone test")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    generator  = EnhancedPatientReportGenerator(output_dir=output_dir)

    system_results   = build_synthetic_results()
    composite_risk   = build_composite_risk()
    trust_envelope   = build_trust_envelope()

    print(f"\n[1] Generating PDF for {len(system_results)} body systems ...")
    start = time.perf_counter()

    report = generator.generate(
        system_results   = system_results,
        composite_risk   = composite_risk,
        trust_envelope   = trust_envelope,
        patient_id       = "TEST-PATIENT-001",
    )

    elapsed = time.perf_counter() - start
    print(f"    Done in {elapsed:.2f}s")

    # ── Assertions ──────────────────────────────────────────────────────────
    failures = []

    if not report.pdf_path:
        failures.append("pdf_path is empty")
    elif not os.path.isfile(report.pdf_path):
        failures.append(f"PDF file not found at: {report.pdf_path}")
    else:
        size = os.path.getsize(report.pdf_path)
        if size == 0:
            failures.append("PDF file is 0 bytes")
        else:
            with open(report.pdf_path, "rb") as f:
                magic = f.read(4)
            if magic != b"%PDF":
                failures.append(f"PDF magic bytes wrong: {magic!r}")

    if failures:
        print("\n[FAIL] The following checks failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        size_kb = os.path.getsize(report.pdf_path) / 1024
        print(f"\n[PASS] PDF generated successfully.")
        print(f"       Path : {report.pdf_path}")
        print(f"       Size : {size_kb:.1f} KB")
        print(f"       Time : {elapsed:.2f}s")
        print(f"\n  Open the PDF to visually inspect:")
        print(f"       {os.path.abspath(report.pdf_path)}")
        print("=" * 60)


if __name__ == "__main__":
    main()

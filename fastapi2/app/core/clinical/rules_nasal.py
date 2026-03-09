"""
Nasal / Upper-Airway Clinical Decision Rules

Evaluates nasal biomarkers from `nasal.py` and produces ClinicalFindings.

Biomarkers evaluated:
  - airflow_thermal_symmetry_index  — bilateral airflow asymmetry (0.0–0.2 normal)
  - nasal_surface_temp_elevation    — mucosal temperature deviation (°C)
  - respiratory_rate                — breaths per minute (10–20 normal)

Clinical references:
  - Eccles (2002): Nasal airflow asymmetry in septal deviation and rhinitis
  - Giotakis & Weber (2013): Acoustic rhinometry thresholds for obstruction
  - Cole (1982): Normal nasal cycle asymmetry index 0.0–0.15
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .base import ClinicalFinding, UrgencyLevel, SpecialistReferral

if TYPE_CHECKING:
    from app.core.extraction.base import BiomarkerSet

# ── Thresholds ─────────────────────────────────────────────────────────────────
# Airflow thermal symmetry index (normalised diff, higher = more asymmetric)
SYMM_MILD     = 0.20   # upper edge of normal (Cole 1982)
SYMM_MODERATE = 0.50   # clinically significant
SYMM_SEVERE   = 1.00   # markedly obstructed / deviated septum territory

# Nasal surface temperature elevation above resting baseline (°C)
TEMP_HIGH     = 1.0    # mild mucosal inflammation marker
TEMP_VERY_HIGH= 1.8    # moderate-to-severe mucosal inflammation

# Respiratory rate (breaths/min)
RR_LOW  = 10.0
RR_HIGH = 20.0
RR_VERY_HIGH = 25.0

# Minimum confidence to trust a reading
MIN_CONFIDENCE = 0.50


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get(biomarker_set: "BiomarkerSet", name: str):
    """Return (value, confidence) or (None, 0) for a named biomarker."""
    bm = biomarker_set.get(name)
    if bm is None:
        return None, 0.0
    if bm.confidence < MIN_CONFIDENCE:
        return None, bm.confidence
    return bm.value, bm.confidence


def _trigger(biomarker_set: "BiomarkerSet", *names: str) -> List[dict]:
    result = []
    for name in names:
        bm = biomarker_set.get(name)
        if bm is not None and bm.confidence >= MIN_CONFIDENCE:
            result.append({
                "name": name,
                "value": round(bm.value, 4),
                "status": bm.status,
                "confidence": round(bm.confidence, 2),
            })
    return result


# ── Rule 1: Nasal Airflow Asymmetry (Obstruction / Septal Deviation) ───────────

def rule_airflow_asymmetry(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Nasal airway obstruction screening flag.

    Elevated airflow symmetry index suggests unilateral nasal obstruction,
    which may result from septal deviation, nasal polyps, turbinate hypertrophy,
    or chronic rhinitis. Severity-banded referral.
    """
    symm, c1  = _get(bs, "airflow_thermal_symmetry_index")
    temp, c2  = _get(bs, "nasal_surface_temp_elevation")

    if symm is None:
        return None

    if symm <= SYMM_MILD:
        return None  # Within normal nasal cycle variation

    # Severity band
    if symm >= SYMM_SEVERE:
        urgency = UrgencyLevel.ROUTINE
        severity_label = "markedly elevated"
    elif symm >= SYMM_MODERATE:
        urgency = UrgencyLevel.ROUTINE
        severity_label = "moderately elevated"
    else:
        urgency = UrgencyLevel.MONITOR
        severity_label = "mildly elevated"

    # If surface temp is also elevated → possible mucosal inflammation
    has_inflammation = temp is not None and temp > TEMP_HIGH
    description_extra = (
        " Additionally, the nasal mucosal temperature is slightly elevated, "
        "which may indicate active inflammation or allergic rhinitis."
        if has_inflammation else ""
    )

    evidence_names = ["airflow_thermal_symmetry_index"]
    if has_inflammation:
        evidence_names.append("nasal_surface_temp_elevation")

    referrals = [
        SpecialistReferral(
            specialist="ENT Specialist (Otolaryngologist)",
            reason=(
                f"Airflow symmetry index of {round(symm, 2)} (normal < {SYMM_MILD}) "
                "indicates unilateral nasal obstruction. An ENT specialist can "
                "assess for septal deviation, polyps, or turbinate hypertrophy."
            ),
            urgency=urgency,
        )
    ]
    if has_inflammation:
        referrals.append(
            SpecialistReferral(
                specialist="Allergist / Immunologist",
                reason=(
                    "Combined airway asymmetry and elevated mucosal temperature "
                    "may indicate allergic rhinitis — an allergist can perform "
                    "skin-prick testing and recommend management."
                ),
                urgency=UrgencyLevel.ROUTINE,
            )
        )

    return ClinicalFinding(
        finding_id="NASAL-OBST-001",
        system="nasal",
        title="Nasal Airflow Asymmetry Detected",
        description=(
            f"The screening detected {severity_label} nasal airflow asymmetry "
            f"(index: {round(symm, 2)}, normal range: 0.0–{SYMM_MILD}). "
            "This indicates that air is not flowing equally through both nostrils, "
            "which can be caused by a deviated nasal septum, nasal polyps, swollen "
            "turbinates, or chronic nasal congestion. This does not typically require "
            "emergency care, but an ENT specialist can diagnose the cause and "
            "recommend treatment."
            + description_extra
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, *evidence_names),
        referrals=referrals,
        clinical_notes=(
            f"Airflow symmetry index = {round(symm, 3)} "
            f"(SYMM_MILD={SYMM_MILD}, SYMM_MODERATE={SYMM_MODERATE}, SYMM_SEVERE={SYMM_SEVERE}). "
            f"Mucosal temp elevation = {round(temp, 3) if temp is not None else 'N/A'} °C."
        ),
    )


# ── Rule 2: Tachypnoea / Elevated Respiratory Rate ────────────────────────────

def rule_respiratory_rate_elevation(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Elevated respiratory rate from nasal thermal analysis.
    Values persistently above 20 brpm at rest may indicate respiratory distress,
    anxiety, or systemic illness.
    """
    rr, c1 = _get(bs, "respiratory_rate")

    if rr is None:
        return None

    if rr <= RR_HIGH:
        return None  # Normal

    urgency = UrgencyLevel.URGENT if rr > RR_VERY_HIGH else UrgencyLevel.ROUTINE

    return ClinicalFinding(
        finding_id="NASAL-RR-001",
        system="nasal",
        title="Elevated Resting Respiratory Rate",
        description=(
            f"The nasal thermal sensor detected a resting respiratory rate of {round(rr, 1)} "
            f"breaths/min (normal: {RR_LOW}–{RR_HIGH}). "
            "A persistently elevated breathing rate at rest can indicate respiratory distress, "
            "anxiety, anaemia, or early infection. A general physician should evaluate this."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, "respiratory_rate"),
        referrals=[
            SpecialistReferral(
                specialist="General Physician / Pulmonologist",
                reason=(
                    f"Resting respiratory rate of {round(rr, 1)} brpm exceeds the "
                    "normal upper limit. A clinical assessment is recommended to "
                    "rule out respiratory or systemic causes."
                ),
                urgency=urgency,
            )
        ],
        clinical_notes=f"RR = {round(rr, 1)} brpm, threshold = {RR_HIGH}.",
    )


# ── Public interface ───────────────────────────────────────────────────────────

NASAL_RULES = [
    rule_airflow_asymmetry,
    rule_respiratory_rate_elevation,
]


def evaluate_nasal(biomarker_set: "BiomarkerSet") -> List[ClinicalFinding]:
    """Evaluate all nasal clinical rules."""
    findings = []
    for rule in NASAL_RULES:
        try:
            finding = rule(biomarker_set)
            if finding is not None:
                findings.append(finding)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"Nasal rule {rule.__name__} raised an exception: {exc}"
            )
    return findings

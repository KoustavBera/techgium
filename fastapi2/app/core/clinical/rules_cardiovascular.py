"""
Cardiovascular Clinical Decision Rules — Phase 2

Evaluates cardiovascular biomarkers from `cardiovascular.py` and produces
ClinicalFindings.

Biomarkers consumed:
    heart_rate          (bpm)  — normal: 60–100
    hrv_rmssd           (ms)   — normal: 20–80
    systolic_bp         (mmHg) — normal: 90–120
    diastolic_bp        (mmHg) — normal: 60–80
    thermal_stress_gradient (°C) — forwarded from the CNS/thermal module

Rule ordering (highest severity first):
    1. Hypertensive crisis     — systolic > 180 or diastolic > 120
    2. Hypertension            — BP > 140/90
    3. Tachycardia + low HRV  — arrhythmia / cardiac stress
    4. Bradycardia + low HRV  — conduction disorder
    5. Stress tachycardia      — HR > 100 + thermal stress gradient
    6. Autonomic dysfunction   — low HRV alone
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .base import ClinicalFinding, UrgencyLevel, SpecialistReferral

if TYPE_CHECKING:
    from app.core.extraction.base import BiomarkerSet

# ── Thresholds ────────────────────────────────────────────────────────────────

# Heart rate
HR_TACHY_MILD     = 100    # upper normal edge
HR_TACHY_SEVERE   = 130    # sustained tachycardia
HR_BRADY_MILD     = 60     # lower normal edge
HR_BRADY_SEVERE   = 40     # symptomatic bradycardia

# HRV RMSSD (ms)
HRV_LOW           = 20     # lower normal edge — HRV below this is clinically significant
HRV_VERY_LOW      = 10     # severely suppressed autonomic tone

# Blood pressure
SBP_HYPER_STAGE1  = 130    # AHA Stage 1 hypertension
SBP_HYPER_STAGE2  = 140    # AHA Stage 2 / action required
SBP_CRISIS        = 180    # Hypertensive crisis — urgent
DBP_HYPER_STAGE1  = 80
DBP_HYPER_STAGE2  = 90
DBP_CRISIS        = 120

# Thermal stress gradient (°C) for anxiety/sympathetic tachycardia detection
THERMAL_STRESS_HIGH = 2.0

MIN_CONFIDENCE = 0.50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(bs: "BiomarkerSet", name: str):
    bm = bs.get(name)
    if bm is None:
        return None, 0.0
    if bm.confidence < MIN_CONFIDENCE:
        return None, bm.confidence
    return bm.value, bm.confidence


def _trigger(bs: "BiomarkerSet", *names: str) -> List[dict]:
    result = []
    for name in names:
        bm = bs.get(name)
        if bm is not None and bm.confidence >= MIN_CONFIDENCE:
            result.append({
                "name": name,
                "value": round(bm.value, 2),
                "status": bm.status,
                "confidence": round(bm.confidence, 2),
            })
    return result


# ── Rule 1: Hypertensive Crisis ───────────────────────────────────────────────

def rule_hypertensive_crisis(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Hypertensive emergency — systolic ≥ 180 or diastolic ≥ 120.

    This is a medical emergency. Refer immediately regardless of symptoms.
    JNC 8 / AHA 2017 classification: Stage 2 hypertensive crisis.
    """
    sbp, c1 = _get(bs, "systolic_bp")
    dbp, c2 = _get(bs, "diastolic_bp")

    if sbp is None and dbp is None:
        return None

    crisis = (sbp is not None and sbp >= SBP_CRISIS) or \
             (dbp is not None and dbp >= DBP_CRISIS)
    if not crisis:
        return None

    evidence = []
    if sbp is not None and sbp >= SBP_CRISIS:
        evidence.append("systolic_bp")
    if dbp is not None and dbp >= DBP_CRISIS:
        evidence.append("diastolic_bp")

    return ClinicalFinding(
        finding_id="CV-HCRIS-001",
        system="cardiovascular",
        title="Hypertensive Crisis — Very High Blood Pressure Detected",
        description=(
            "The screening detected a critically high blood pressure reading. "
            "Blood pressure at this level can put severe strain on your heart, "
            "brain, and kidneys. This pattern requires prompt medical attention. "
            "Please seek evaluation from a doctor today, even if you feel well."
        ),
        urgency=UrgencyLevel.URGENT,
        triggering_biomarkers=_trigger(bs, *evidence),
        referrals=[
            SpecialistReferral(
                specialist="Emergency Department / General Physician (same day)",
                reason=(
                    f"Systolic BP={round(sbp) if sbp else 'N/A'} / "
                    f"Diastolic BP={round(dbp) if dbp else 'N/A'} mmHg — "
                    "exceeds the hypertensive crisis threshold (180/120 mmHg). "
                    "Prompt evaluation is required."
                ),
                urgency=UrgencyLevel.URGENT,
            ),
            SpecialistReferral(
                specialist="Cardiologist",
                reason="Sustained very high blood pressure requires cardiac evaluation and management.",
                urgency=UrgencyLevel.URGENT,
            ),
        ],
        clinical_notes=(
            f"SBP={round(sbp) if sbp else 'N/A'} mmHg, "
            f"DBP={round(dbp) if dbp else 'N/A'} mmHg. "
            f"Crisis threshold: ≥180/120 mmHg."
        ),
    )


# ── Rule 2: Hypertension (Stage 1 / Stage 2) ─────────────────────────────────

def rule_hypertension(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Stage 1 or Stage 2 hypertension — not a crisis but requires follow-up.
    AHA 2017: Stage 1 = 130–139/80–89; Stage 2 = ≥140/90.
    """
    sbp, c1 = _get(bs, "systolic_bp")
    dbp, c2 = _get(bs, "diastolic_bp")

    if sbp is None and dbp is None:
        return None

    # Hypertensive crisis handled by Rule 1
    if (sbp is not None and sbp >= SBP_CRISIS) or \
       (dbp is not None and dbp >= DBP_CRISIS):
        return None

    stage2 = (sbp is not None and sbp >= SBP_HYPER_STAGE2) or \
             (dbp is not None and dbp >= DBP_HYPER_STAGE2)
    stage1 = (sbp is not None and sbp >= SBP_HYPER_STAGE1) or \
             (dbp is not None and dbp >= DBP_HYPER_STAGE1)

    if not stage1:
        return None

    evidence = []
    if sbp is not None and sbp >= SBP_HYPER_STAGE1:
        evidence.append("systolic_bp")
    if dbp is not None and dbp >= DBP_HYPER_STAGE1:
        evidence.append("diastolic_bp")

    urgency = UrgencyLevel.URGENT if stage2 else UrgencyLevel.ROUTINE
    stage_label = "Stage 2" if stage2 else "Stage 1"

    return ClinicalFinding(
        finding_id="CV-HTN-001",
        system="cardiovascular",
        title=f"Elevated Blood Pressure — Hypertension {stage_label}",
        description=(
            "The screening detected elevated blood pressure. "
            "High blood pressure often causes no symptoms ('the silent killer') "
            "but silently damages blood vessels, the heart, and kidneys over time. "
            "A single elevated reading is not sufficient for diagnosis — a doctor "
            "will confirm with repeat measurements and assess your overall risk."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, *evidence),
        referrals=[
            SpecialistReferral(
                specialist="General Physician (Internal Medicine)",
                reason=(
                    "Blood pressure evaluation, lifestyle counselling, "
                    "and repeat measurement to confirm the finding."
                ),
                urgency=urgency,
            ),
            SpecialistReferral(
                specialist="Cardiologist",
                reason="Cardiology review is warranted for Stage 2 hypertension and cardiovascular risk assessment.",
                urgency=urgency,
            ),
        ],
        clinical_notes=(
            f"SBP={round(sbp) if sbp else 'N/A'} mmHg, "
            f"DBP={round(dbp) if dbp else 'N/A'} mmHg. "
            f"AHA classification: {stage_label} (Stage 1: ≥130/80; Stage 2: ≥140/90)."
        ),
    )


# ── Rule 3: Tachycardia + Low HRV ────────────────────────────────────────────

def rule_tachycardia_low_hrv(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Resting tachycardia combined with suppressed HRV.

    Low HRV in the presence of tachycardia rules out benign sinus tachycardia
    (which typically preserves HRV) and raises concern for arrhythmia or
    underlying cardiac dysfunction.

    Reference: Tsuji et al. (1994) — HRV as predictor of cardiac events.
    """
    hr, c1  = _get(bs, "heart_rate")
    hrv, c2 = _get(bs, "hrv_rmssd")

    if hr is None or hrv is None:
        return None
    if hr <= HR_TACHY_MILD:
        return None
    if hrv >= HRV_LOW:
        return None  # HRV preserved — likely benign

    urgency = UrgencyLevel.URGENT if hr > HR_TACHY_SEVERE or hrv < HRV_VERY_LOW \
              else UrgencyLevel.ROUTINE

    return ClinicalFinding(
        finding_id="CV-TACHY-001",
        system="cardiovascular",
        title="Resting Tachycardia with Suppressed Heart Rate Variability",
        description=(
            "The screening found that your heart rate is elevated at rest and "
            "your heart rate variability — a marker of cardiac health — is reduced. "
            "Together, these suggest your heart may not be recovering or adapting "
            "normally. This can reflect cardiac arrhythmia, chronic stress, "
            "or other underlying conditions. A cardiologist can examine this "
            "with an ECG and further tests."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, "heart_rate", "hrv_rmssd"),
        referrals=[
            SpecialistReferral(
                specialist="Cardiologist",
                reason=(
                    f"Resting HR={round(hr)} bpm with HRV RMSSD={round(hrv)} ms — "
                    "tachycardia with suppressed HRV warrants ECG and cardiac evaluation."
                ),
                urgency=urgency,
            ),
        ],
        clinical_notes=(
            f"HR={round(hr)} bpm (threshold >{HR_TACHY_MILD}), "
            f"HRV RMSSD={round(hrv)} ms (threshold <{HRV_LOW} ms). "
            f"Urgency: {'URGENT — HR>{HR_TACHY_SEVERE} or HRV<{HRV_VERY_LOW}' if urgency == UrgencyLevel.URGENT else 'ROUTINE'}."
        ),
    )


# ── Rule 4: Bradycardia + Low HRV ────────────────────────────────────────────

def rule_bradycardia_low_hrv(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Resting bradycardia combined with low HRV.

    In athletes, bradycardia with preserved (high) HRV is normal.
    Bradycardia with LOW HRV suggests conduction system pathology
    (sick sinus syndrome, AV block) rather than athletic adaptation.
    """
    hr, c1  = _get(bs, "heart_rate")
    hrv, c2 = _get(bs, "hrv_rmssd")

    if hr is None or hrv is None:
        return None
    if hr >= HR_BRADY_MILD:
        return None
    if hrv >= HRV_LOW:
        return None  # Athletic bradycardia — high HRV, no referral needed

    urgency = UrgencyLevel.URGENT if hr < HR_BRADY_SEVERE else UrgencyLevel.ROUTINE

    return ClinicalFinding(
        finding_id="CV-BRADY-001",
        system="cardiovascular",
        title="Bradycardia with Low Heart Rate Variability",
        description=(
            "The screening detected a slower-than-normal heart rate paired "
            "with reduced heart rate variability. In athletes, a slow heart rate "
            "is normal and healthy. However, when combined with low variability, "
            "it may indicate a problem with your heart's electrical conduction "
            "system. A cardiologist can assess this with an ECG."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, "heart_rate", "hrv_rmssd"),
        referrals=[
            SpecialistReferral(
                specialist="Cardiologist",
                reason=(
                    f"HR={round(hr)} bpm with HRV RMSSD={round(hrv)} ms — "
                    "bradycardia with low HRV requires ECG to exclude conduction disorder."
                ),
                urgency=urgency,
            ),
        ],
        clinical_notes=(
            f"HR={round(hr)} bpm (threshold <{HR_BRADY_MILD}), "
            f"HRV RMSSD={round(hrv)} ms (threshold <{HRV_LOW} ms). "
            f"Severe threshold: HR<{HR_BRADY_SEVERE}."
        ),
    )


# ── Rule 5: Stress / Anxiety-Driven Tachycardia ──────────────────────────────

def rule_stress_tachycardia(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Sympathetic nervous system-driven tachycardia.

    Pattern: elevated HR at rest + high thermal stress gradient (autonomic
    arousal marker). HRV may be mildly reduced but not severely so —
    distinguishes this from the cardiac arrhythmia rule above.
    """
    hr, c1          = _get(bs, "heart_rate")
    hrv, c2         = _get(bs, "hrv_rmssd")
    stress_grad, c3 = _get(bs, "thermal_stress_gradient")

    if hr is None or stress_grad is None:
        return None
    if hr <= HR_TACHY_MILD:
        return None
    if stress_grad <= THERMAL_STRESS_HIGH:
        return None

    # If HRV is very low, the cardiac arrhythmia rule handles it
    if hrv is not None and hrv < HRV_VERY_LOW:
        return None

    evidence = ["heart_rate", "thermal_stress_gradient"]
    if hrv is not None:
        evidence.append("hrv_rmssd")

    return ClinicalFinding(
        finding_id="CV-STRESS-001",
        system="cardiovascular",
        title="Stress-Related Elevated Heart Rate",
        description=(
            "The screening detected an elevated resting heart rate alongside "
            "a high facial thermal stress gradient — a marker of sympathetic "
            "nervous system activation (the 'fight or flight' response). "
            "This pattern is consistent with anxiety, acute stress, or caffeine. "
            "If this is persistent, a doctor can help assess underlying causes."
        ),
        urgency=UrgencyLevel.MONITOR,
        triggering_biomarkers=_trigger(bs, *evidence),
        referrals=[
            SpecialistReferral(
                specialist="General Physician",
                reason=(
                    "Persistent stress-driven tachycardia warrants assessment "
                    "to rule out thyroid dysfunction, anaemia, or anxiety disorders."
                ),
                urgency=UrgencyLevel.ROUTINE,
            ),
            SpecialistReferral(
                specialist="Psychiatrist / Psychologist",
                reason="If stress and anxiety are the likely cause, mental health support can be highly effective.",
                urgency=UrgencyLevel.MONITOR,
            ),
        ],
        clinical_notes=(
            f"HR={round(hr)} bpm, thermal stress gradient={round(stress_grad, 2)}°C "
            f"(threshold >{THERMAL_STRESS_HIGH}°C). "
            f"HRV={round(hrv) if hrv else 'N/A'} ms."
        ),
    )


# ── Rule 6: Autonomic Dysfunction (low HRV alone) ────────────────────────────

def rule_autonomic_dysfunction(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Isolated low HRV without significant HR abnormality.

    Low HRV in isolation is a predictor of cardiovascular events, diabetes
    complications, and chronic stress. Not immediately actionable but warrants
    monitoring and lifestyle review.

    Reference: Tsuji et al. (1994, 1996) ARIC and Framingham studies.
    """
    hr, c1  = _get(bs, "heart_rate")
    hrv, c2 = _get(bs, "hrv_rmssd")

    if hrv is None:
        return None
    if hrv >= HRV_LOW:
        return None

    # If HR is also abnormal, other rules already cover it
    if hr is not None and (hr > HR_TACHY_MILD or hr < HR_BRADY_MILD):
        return None

    urgency = UrgencyLevel.ROUTINE if hrv < HRV_VERY_LOW else UrgencyLevel.MONITOR

    return ClinicalFinding(
        finding_id="CV-HRV-001",
        system="cardiovascular",
        title="Low Heart Rate Variability — Autonomic Imbalance",
        description=(
            "The screening found that your heart rate variability (HRV) is "
            "below the normal range. HRV measures the variation between "
            "heartbeats and reflects how well your nervous system is regulating "
            "your heart. Low HRV is associated with chronic stress, poor "
            "recovery, and, over time, higher cardiovascular risk. "
            "Lifestyle factors like sleep quality, exercise, and stress management "
            "can significantly improve HRV."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, "hrv_rmssd"),
        referrals=[
            SpecialistReferral(
                specialist="Cardiologist",
                reason="Chronically suppressed HRV is a cardiovascular risk marker — cardiac evaluation is advised.",
                urgency=UrgencyLevel.ROUTINE,
            ),
            SpecialistReferral(
                specialist="Endocrinologist",
                reason="Low HRV can be associated with autonomic neuropathy from diabetes or thyroid conditions.",
                urgency=UrgencyLevel.MONITOR,
            ),
        ],
        clinical_notes=(
            f"HRV RMSSD={round(hrv)} ms (normal: {HRV_LOW}–80 ms). "
            f"HR={round(hr) if hr else 'N/A'} bpm (within normal range). "
            f"Framingham-validated risk marker for cardiovascular events."
        ),
    )


# ── Public interface ───────────────────────────────────────────────────────────

CV_RULES = [
    rule_hypertensive_crisis,    # Severity 1 — most urgent
    rule_hypertension,           # Severity 2
    rule_tachycardia_low_hrv,    # Severity 3
    rule_bradycardia_low_hrv,    # Severity 4
    rule_stress_tachycardia,     # Severity 5
    rule_autonomic_dysfunction,  # Severity 6
]


def evaluate_cardiovascular(biomarker_set: "BiomarkerSet") -> List[ClinicalFinding]:
    """
    Evaluate all cardiovascular rules against a BiomarkerSet.

    Returns:
        List of ClinicalFindings (may be empty for a healthy profile).
    """
    findings = []
    for rule in CV_RULES:
        try:
            finding = rule(biomarker_set)
            if finding is not None:
                findings.append(finding)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"CV rule {rule.__name__} raised an exception: {exc}"
            )
    return findings

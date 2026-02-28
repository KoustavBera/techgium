"""
CNS Clinical Decision Rules — Phase 1

Evaluates CNS biomarkers from `cns.py` and produces ClinicalFindings.

Pattern logic is derived from the clinical pathway analysis and peer-reviewed
thresholds from the following sources:
  - Zeni et al. (2008): Gait variability for neurological screening
  - Elble & McNames (2016): Tremor frequency-band classification
  - Berg Balance Scale clinical correlations (normal > 75/100)

Design principles:
  - Each rule function is pure: (BiomarkerSet) → Optional[ClinicalFinding]
  - Rules are evaluated independently and collected by the engine.
  - Thresholds are module-level constants so they can be reviewed / tuned
    without hunting through logic.
  - Rules are ordered from highest clinical severity to lowest.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .base import ClinicalFinding, UrgencyLevel, SpecialistReferral

if TYPE_CHECKING:
    from app.core.extraction.base import BiomarkerSet

# ── Thresholds (mirror cns.py normal_ranges but with clinical severity bands) ──
# Gait variability CV
GAIT_VAR_MILD     = 0.06   # > 6 %  mild abnormality
GAIT_VAR_MODERATE = 0.10   # > 10% moderate
GAIT_VAR_SEVERE   = 0.14   # > 14% severe

# Tremor power (normalized PSD)
TREMOR_MILD       = 0.05   # upper edge of normal
TREMOR_MODERATE   = 0.12
TREMOR_SEVERE     = 0.20

# CNS stability score (0-100)
STABILITY_MILD    = 75     # lower edge of normal
STABILITY_MODERATE= 60
STABILITY_SEVERE  = 45

# Postural sway amplitudes
SWAY_MILD         = 0.05   # upper edge of normal
SWAY_HIGH         = 0.10

# Posture entropy SampEn (< 0.5 is too rigid, > 2.5 is chaotic)
ENTROPY_LOW_RIGID = 0.40   # pathologically rigid
ENTROPY_HIGH      = 2.60   # pathologically chaotic

# Thermal stress gradient (°C)
STRESS_GRADIENT_HIGH = 2.0

# Minimum confidence to trust a biomarker reading
MIN_CONFIDENCE = 0.50


# ── Helper ────────────────────────────────────────────────────────────────────

def _get(biomarker_set: "BiomarkerSet", name: str):
    """
    Return (value, confidence) for a named biomarker, or (None, 0) if absent.
    Skips biomarkers whose confidence is below MIN_CONFIDENCE.
    """
    bm = biomarker_set.get(name)
    if bm is None:
        return None, 0.0
    if bm.confidence < MIN_CONFIDENCE:
        return None, bm.confidence
    return bm.value, bm.confidence


def _trigger(biomarker_set: "BiomarkerSet", *names: str) -> List[dict]:
    """Build the `triggering_biomarkers` list for named biomarkers."""
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


# ── Rule 1: Parkinson's Risk Pattern ─────────────────────────────────────────

def rule_parkinsonian_pattern(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Parkinson's Disease screening flag.

    Classic triad: resting tremor (4–6 Hz) elevated + rigid posture (low entropy)
    + abnormal gait variability.  Requires at least 2 of 3 criteria.

    References:
      - Elble & McNames (2016): resting tremor PSD threshold
      - Zeni et al. (2008): gait variability CV > 8% in PD patients
    """
    tremor_rest, c1   = _get(bs, "tremor_resting")
    entropy, c2       = _get(bs, "posture_entropy")
    gait_var, c3      = _get(bs, "gait_variability")
    stability, c4     = _get(bs, "cns_stability_score")

    if tremor_rest is None and entropy is None:
        return None

    criteria_met = 0
    evidence_names = []

    # Criterion 1: Elevated resting tremor
    if tremor_rest is not None and tremor_rest > TREMOR_MILD:
        criteria_met += 1
        evidence_names.append("tremor_resting")

    # Criterion 2: Rigid posture (low entropy — too regular)
    if entropy is not None and entropy < ENTROPY_LOW_RIGID:
        criteria_met += 1
        evidence_names.append("posture_entropy")

    # Criterion 3: High gait variability (shuffling / uneven stride timing)
    if gait_var is not None and gait_var > GAIT_VAR_MILD:
        criteria_met += 1
        evidence_names.append("gait_variability")

    if criteria_met < 2:
        return None

    # Severity
    if tremor_rest is not None and tremor_rest > TREMOR_SEVERE:
        urgency = UrgencyLevel.URGENT
    elif criteria_met == 3:
        urgency = UrgencyLevel.URGENT
    else:
        urgency = UrgencyLevel.ROUTINE

    return ClinicalFinding(
        finding_id="CNS-PARK-001",
        system="central_nervous_system",
        title="Parkinson's Disease Risk Pattern Detected",
        description=(
            "The screening detected a combination of elevated resting tremor, "
            "reduced postural complexity, and irregular gait timing. "
            "These three features — when present together — are associated with "
            "early-stage Parkinson's Disease. This is a screening observation, "
            "not a diagnosis. A neurologist can assess this further with a "
            "formal clinical examination."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, *evidence_names),
        referrals=[
            SpecialistReferral(
                specialist="Neurologist (Movement Disorders)",
                reason=(
                    "Resting tremor combined with rigid posture and gait irregularity "
                    "warrants a formal neurological examination for movement disorders."
                ),
                urgency=urgency,
            )
        ],
        clinical_notes=(
            f"Criteria met: {criteria_met}/3. "
            f"Resting tremor PSD={round(tremor_rest,3) if tremor_rest is not None else 'N/A'}, "
            f"SampEn={round(entropy,3) if entropy is not None else 'N/A'}, "
            f"Gait CV={round(gait_var,3) if gait_var is not None else 'N/A'}."
        ),
    )


# ── Rule 2: Essential Tremor ──────────────────────────────────────────────────

def rule_essential_tremor(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Essential Tremor screening flag.

    Pattern: postural tremor (6–12 Hz) elevated while resting tremor is normal
    and gait variability is within normal limits.  This distinguishes ET from PD.
    """
    tremor_post, c1   = _get(bs, "tremor_postural")
    tremor_rest, c2   = _get(bs, "tremor_resting")
    gait_var, c3      = _get(bs, "gait_variability")

    if tremor_post is None:
        return None

    # Must have elevated postural tremor
    if tremor_post <= TREMOR_MILD:
        return None

    # If resting tremor is also very high, Parkinson rule likely fires — skip here
    if tremor_rest is not None and tremor_rest > TREMOR_MODERATE:
        return None

    # Gait should be relatively normal (distinguishes from PD)
    if gait_var is not None and gait_var > GAIT_VAR_MODERATE:
        return None  # Abnormal gait → let Parkinson rule handle

    urgency = (
        UrgencyLevel.URGENT if tremor_post > TREMOR_SEVERE
        else UrgencyLevel.ROUTINE
    )

    return ClinicalFinding(
        finding_id="CNS-ET-001",
        system="central_nervous_system",
        title="Essential Tremor Pattern Detected",
        description=(
            "The screening detected elevated postural tremor (the type that occurs "
            "when holding an object or posture), while resting tremor and gait "
            "timing appear relatively normal. This pattern is more consistent with "
            "Essential Tremor than Parkinson's Disease. A neurologist can confirm "
            "this and discuss management options."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, "tremor_postural", "tremor_resting"),
        referrals=[
            SpecialistReferral(
                specialist="Neurologist",
                reason=(
                    "Elevated postural tremor with a spared resting component "
                    "is the hallmark of Essential Tremor — a manageable condition "
                    "that a neurologist can confirm and treat."
                ),
                urgency=urgency,
            )
        ],
        clinical_notes=(
            f"Postural PSD={round(tremor_post,3)}, "
            f"Resting PSD={round(tremor_rest,3) if tremor_rest is not None else 'N/A'}, "
            f"Gait CV={round(gait_var,3) if gait_var is not None else 'N/A'}."
        ),
    )


# ── Rule 3: Cerebellar Dysfunction ────────────────────────────────────────────

def rule_cerebellar_pattern(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Cerebellar dysfunction screening flag.

    Pattern: intention tremor (3–5 Hz) elevated + high gait variability +
    high postural entropy (chaotic sway, the opposite of Parkinson's rigidity).
    """
    tremor_int, c1    = _get(bs, "tremor_intention")
    gait_var, c2      = _get(bs, "gait_variability")
    entropy, c3       = _get(bs, "posture_entropy")

    if tremor_int is None:
        return None

    if tremor_int <= TREMOR_MILD:
        return None

    criteria_met = 1  # intention tremor already elevated
    evidence_names = ["tremor_intention"]

    if gait_var is not None and gait_var > GAIT_VAR_MODERATE:
        criteria_met += 1
        evidence_names.append("gait_variability")

    if entropy is not None and entropy > ENTROPY_HIGH:
        criteria_met += 1
        evidence_names.append("posture_entropy")

    if criteria_met < 2:
        return None

    urgency = UrgencyLevel.URGENT if criteria_met == 3 else UrgencyLevel.ROUTINE

    return ClinicalFinding(
        finding_id="CNS-CEREB-001",
        system="central_nervous_system",
        title="Cerebellar Dysfunction Pattern Detected",
        description=(
            "The screening detected elevated intention tremor (trembling during "
            "movement) combined with irregular gait and chaotic postural sway. "
            "This combination can indicate cerebellar dysfunction, which affects "
            "coordination. A neurologist should evaluate this further."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, *evidence_names),
        referrals=[
            SpecialistReferral(
                specialist="Neurologist",
                reason=(
                    "Intention tremor with high gait variability and chaotic "
                    "postural entropy suggests cerebellar involvement — "
                    "a neurological evaluation is warranted."
                ),
                urgency=urgency,
            )
        ],
        clinical_notes=(
            f"Criteria met: {criteria_met}/3. "
            f"Intention PSD={round(tremor_int,3)}, "
            f"Gait CV={round(gait_var,3) if gait_var is not None else 'N/A'}, "
            f"SampEn={round(entropy,3) if entropy is not None else 'N/A'}."
        ),
    )


# ── Rule 4: Fall Risk / Balance Disorder ──────────────────────────────────────

def rule_fall_risk(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Fall risk screening flag.

    Pattern: low CNS stability score + high sway (AP or ML) + high gait variability,
    WITHOUT significant tremor.  This profile is typical of balance/vestibular
    disorders and is a major fall-risk indicator in older adults.
    """
    stability, c1     = _get(bs, "cns_stability_score")
    sway_ap, c2       = _get(bs, "sway_amplitude_ap")
    sway_ml, c3       = _get(bs, "sway_amplitude_ml")
    gait_var, c4      = _get(bs, "gait_variability")
    tremor_rest, c5   = _get(bs, "tremor_resting")
    tremor_post, c6   = _get(bs, "tremor_postural")

    if stability is None and sway_ap is None:
        return None

    # Tremor rules take priority — don't double-flag
    max_tremor = max(
        (tremor_rest or 0.0),
        (tremor_post or 0.0),
    )
    if max_tremor > TREMOR_MODERATE:
        return None  # Tremor rules already cover this

    criteria_met = 0
    evidence_names = []

    if stability is not None and stability < STABILITY_MILD:
        criteria_met += 1
        evidence_names.append("cns_stability_score")

    if sway_ap is not None and sway_ap > SWAY_MILD:
        criteria_met += 1
        evidence_names.append("sway_amplitude_ap")

    if sway_ml is not None and sway_ml > SWAY_MILD:
        criteria_met += 1
        evidence_names.append("sway_amplitude_ml")

    if gait_var is not None and gait_var > GAIT_VAR_MILD:
        criteria_met += 1
        evidence_names.append("gait_variability")

    if criteria_met < 2:
        return None

    urgency = (
        UrgencyLevel.URGENT   if stability is not None and stability < STABILITY_SEVERE
        else UrgencyLevel.ROUTINE if stability is not None and stability < STABILITY_MODERATE
        else UrgencyLevel.MONITOR
    )

    return ClinicalFinding(
        finding_id="CNS-FALL-001",
        system="central_nervous_system",
        title="Elevated Fall Risk — Balance and Gait Instability",
        description=(
            "The screening detected increased postural sway and irregular gait "
            "timing. This combination is a known predictor of fall risk. "
            "While this does not mean you will fall, it suggests that your "
            "balance system may benefit from professional evaluation and "
            "targeted exercise. A physiotherapist or specialist can design "
            "a personalised balance training programme."
        ),
        urgency=urgency,
        triggering_biomarkers=_trigger(bs, *evidence_names),
        referrals=[
            SpecialistReferral(
                specialist="Physiotherapist / Physical Therapist",
                reason=(
                    "Elevated postural sway and gait variability are addressable "
                    "with targeted balance and strengthening exercises."
                ),
                urgency=UrgencyLevel.ROUTINE,
            ),
            SpecialistReferral(
                specialist="Geriatrician (if age > 60)",
                reason=(
                    "In older adults, this pattern warrants a comprehensive fall-risk "
                    "assessment including medication review and home hazard evaluation."
                ),
                urgency=urgency,
            ),
        ],
        clinical_notes=(
            f"Criteria met: {criteria_met}/4. "
            f"Stability={round(stability,1) if stability is not None else 'N/A'}/100, "
            f"Sway AP={round(sway_ap,4) if sway_ap is not None else 'N/A'}, "
            f"Sway ML={round(sway_ml,4) if sway_ml is not None else 'N/A'}, "
            f"Gait CV={round(gait_var,3) if gait_var is not None else 'N/A'}."
        ),
    )


# ── Rule 5: Autonomic Stress / Anxiety-Driven Tremor ─────────────────────────

def rule_autonomic_stress(bs: "BiomarkerSet") -> Optional[ClinicalFinding]:
    """
    Autonomic stress pattern.

    High thermal stress gradient (forehead/nose temperature delta) combined
    with elevated tremor can indicate sympathetic nervous system overactivation
    (anxiety, chronic stress, panic disorder).
    """
    stress_grad, c1   = _get(bs, "thermal_stress_gradient")
    tremor_post, c2   = _get(bs, "tremor_postural")
    tremor_rest, c3   = _get(bs, "tremor_resting")

    if stress_grad is None:
        return None

    if stress_grad <= STRESS_GRADIENT_HIGH:
        return None

    max_tremor = max((tremor_post or 0.0), (tremor_rest or 0.0))
    if max_tremor < TREMOR_MILD:
        return None  # Stress gradient alone is not actionable

    evidence_names = ["thermal_stress_gradient"]
    if tremor_post is not None and tremor_post > TREMOR_MILD:
        evidence_names.append("tremor_postural")
    if tremor_rest is not None and tremor_rest > TREMOR_MILD:
        evidence_names.append("tremor_resting")

    return ClinicalFinding(
        finding_id="CNS-STRESS-001",
        system="central_nervous_system",
        title="Autonomic Stress Response Pattern",
        description=(
            "The screening detected an elevated facial thermal stress gradient "
            "combined with tremor. This pattern can occur during episodes of "
            "high stress or anxiety, which activate the sympathetic nervous system "
            "and produce measurable physical symptoms. If this is persistent, "
            "a doctor can help rule out anxiety-related conditions."
        ),
        urgency=UrgencyLevel.MONITOR,
        triggering_biomarkers=_trigger(bs, *evidence_names),
        referrals=[
            SpecialistReferral(
                specialist="General Physician",
                reason=(
                    "A thermal stress gradient with tremor can reflect sustained "
                    "sympathetic activation — a GP can screen for anxiety disorders "
                    "or other causes."
                ),
                urgency=UrgencyLevel.ROUTINE,
            ),
        ],
        clinical_notes=(
            f"Thermal stress gradient={stress_grad:.2f}°C (threshold {STRESS_GRADIENT_HIGH}°C), "
            f"Max tremor PSD={max_tremor:.3f}."
        ),
    )


# ── Public interface ───────────────────────────────────────────────────────────

# All rules in evaluation order (highest severity first).
CNS_RULES = [
    rule_parkinsonian_pattern,
    rule_cerebellar_pattern,
    rule_essential_tremor,
    rule_fall_risk,
    rule_autonomic_stress,
]


def evaluate_cns(biomarker_set: "BiomarkerSet") -> List[ClinicalFinding]:
    """
    Evaluate all CNS rules against a CNS BiomarkerSet.

    Returns:
        List of ClinicalFindings (may be empty if no patterns detected).
        Filters out None results from rules that did not fire.
    """
    findings = []
    for rule in CNS_RULES:
        try:
            finding = rule(biomarker_set)
            if finding is not None:
                findings.append(finding)
        except Exception as exc:
            # Never let a rule crash the pipeline
            import logging
            logging.getLogger(__name__).warning(
                f"CNS rule {rule.__name__} raised an exception: {exc}"
            )
    return findings

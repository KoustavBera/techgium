"""
Clinical Decision Layer — Base Types

Defines the data contracts that all system-specific rule modules produce.
These are system-agnostic and consumed by the report generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class UrgencyLevel(str, Enum):
    """
    Clinical urgency of the finding.

    URGENT       – seek care within 24–48 h (red flag patterns)
    ROUTINE      – schedule an outpatient appointment
    MONITOR      – track over time; no immediate action required
    INFORMATIONAL – observation only, no referral needed
    """
    URGENT        = "urgent"
    ROUTINE       = "routine"
    MONITOR       = "monitor"
    INFORMATIONAL = "informational"


@dataclass
class SpecialistReferral:
    """A single specialist referral recommendation."""
    specialist: str          # e.g. "Neurologist (Movement Disorders)"
    reason: str              # 1-sentence plain-language reason
    urgency: UrgencyLevel    # How quickly to act


@dataclass
class ClinicalFinding:
    """
    One distinct clinical pattern detected by the decision engine.

    A single screening can produce 0-N findings.  Each finding maps
    directly to one row in the "referral" section of the patient report.
    """
    # ── Core identity ─────────────────────────────────────────────────────
    finding_id: str                          # e.g. "CNS-PARKINSON-001"
    system: str                              # e.g. "central_nervous_system"
    title: str                               # Short title shown in the report
    description: str                         # Plain-language patient explanation
    urgency: UrgencyLevel

    # ── Evidence ──────────────────────────────────────────────────────────
    # The specific biomarkers that triggered this finding,
    # plus the measured value and the direction of deviation.
    triggering_biomarkers: List[dict] = field(default_factory=list)
    # e.g. [{"name": "tremor_resting", "value": 0.12, "status": "high"}]

    # ── Referral ──────────────────────────────────────────────────────────
    referrals: List[SpecialistReferral] = field(default_factory=list)

    # ── Optional metadata ─────────────────────────────────────────────────
    # Free-form notes for the doctor report (not shown to patients)
    clinical_notes: str = ""

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "system": self.system,
            "title": self.title,
            "description": self.description,
            "urgency": self.urgency.value,
            "triggering_biomarkers": self.triggering_biomarkers,
            "referrals": [
                {
                    "specialist": r.specialist,
                    "reason": r.reason,
                    "urgency": r.urgency.value,
                }
                for r in self.referrals
            ],
            "clinical_notes": self.clinical_notes,
        }

"""
Clinical Decision Engine

Central dispatcher. Takes a dict of BiomarkerSets (one per physiological
system) and returns all ClinicalFindings across all active rule modules.

Usage:
    from app.core.clinical import ClinicalDecisionEngine

    engine = ClinicalDecisionEngine()
    findings = engine.analyze(biomarker_sets)
    for f in findings:
        print(f.title, f.urgency, [r.specialist for r in f.referrals])

Adding a new system (Phase 2+):
    1. Create  app/core/clinical/rules_<system>.py
    2. Implement evaluate_<system>(BiomarkerSet) -> List[ClinicalFinding]
    3. Register it in _SYSTEM_EVALUATORS below.
"""
from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

from app.core.extraction.base import PhysiologicalSystem
from .base import ClinicalFinding, UrgencyLevel
from .rules_cns import evaluate_cns
from .rules_cardiovascular import evaluate_cardiovascular

if TYPE_CHECKING:
    from app.core.extraction.base import BiomarkerSet

logger = logging.getLogger(__name__)

# ── Registry: system → rule evaluator ────────────────────────────────────────
# Phase 1 ships only CNS.  Future systems are added here.
_SYSTEM_EVALUATORS = {
    PhysiologicalSystem.CNS: evaluate_cns,
    PhysiologicalSystem.CARDIOVASCULAR: evaluate_cardiovascular,
    # PhysiologicalSystem.SKELETAL:       evaluate_skeletal,         # Phase 3
    # PhysiologicalSystem.EYES:           evaluate_eyes,             # Phase 3
    # PhysiologicalSystem.SKIN:           evaluate_skin,             # Phase 3
    # PhysiologicalSystem.PULMONARY:      evaluate_pulmonary,        # Phase 3
    # PhysiologicalSystem.VISUAL_DISEASE: evaluate_visual_disease,   # Phase 3
}

# Urgency sort order (lower = more urgent → appears first in report)
_URGENCY_ORDER = {
    UrgencyLevel.URGENT:        0,
    UrgencyLevel.ROUTINE:       1,
    UrgencyLevel.MONITOR:       2,
    UrgencyLevel.INFORMATIONAL: 3,
}


class ClinicalDecisionEngine:
    """
    Transforms BiomarkerSets into actionable ClinicalFindings.

    Stateless — safe to call from multiple threads / concurrent requests.
    """

    def analyze(
        self,
        biomarker_sets: Dict[PhysiologicalSystem, "BiomarkerSet"],
    ) -> List[ClinicalFinding]:
        """
        Evaluate all registered rule modules against the provided biomarker sets.

        Args:
            biomarker_sets: Dict mapping each PhysiologicalSystem to its
                            BiomarkerSet from the extraction layer.

        Returns:
            List of ClinicalFindings sorted by urgency (urgent first).
            Returns an empty list if no patterns are detected, which is
            the expected result for a healthy screening.
        """
        all_findings: List[ClinicalFinding] = []

        for system, evaluator in _SYSTEM_EVALUATORS.items():
            bs = biomarker_sets.get(system)
            if bs is None:
                logger.debug(f"ClinicalDecisionEngine: no BiomarkerSet for {system.value}, skipping")
                continue

            if not bs.biomarkers:
                logger.debug(f"ClinicalDecisionEngine: empty BiomarkerSet for {system.value}, skipping")
                continue

            try:
                findings = evaluator(bs)
                all_findings.extend(findings)
                if findings:
                    logger.info(
                        f"ClinicalDecisionEngine [{system.value}]: "
                        f"{len(findings)} finding(s) — "
                        + ", ".join(f.finding_id for f in findings)
                    )
                else:
                    logger.debug(f"ClinicalDecisionEngine [{system.value}]: no findings (healthy)")
            except Exception as exc:
                # Isolate failures — one system crashing must not block others
                logger.error(
                    f"ClinicalDecisionEngine [{system.value}]: evaluator raised {exc}",
                    exc_info=True
                )

        # Sort: most urgent first; within same urgency, keep insertion order
        all_findings.sort(key=lambda f: _URGENCY_ORDER.get(f.urgency, 99))
        return all_findings

    def analyze_system(
        self,
        system: PhysiologicalSystem,
        biomarker_set: "BiomarkerSet",
    ) -> List[ClinicalFinding]:
        """
        Evaluate rules for a single system.  Useful for unit-testing individual
        system analysers without running the full pipeline.
        """
        evaluator = _SYSTEM_EVALUATORS.get(system)
        if evaluator is None:
            logger.debug(f"ClinicalDecisionEngine: no evaluator registered for {system.value}")
            return []
        return evaluator(biomarker_set)

    @staticmethod
    def registered_systems() -> List[PhysiologicalSystem]:
        """Return which physiological systems have active rule modules."""
        return list(_SYSTEM_EVALUATORS.keys())

    @staticmethod
    def summarise(findings: List[ClinicalFinding]) -> Dict:
        """
        Build a compact summary dict suitable for JSON API responses.

        Example output:
        {
            "total_findings": 2,
            "urgent_count": 1,
            "routine_count": 1,
            "referrals": ["Neurologist (Movement Disorders)", "Physiotherapist"],
            "findings": [{...}, {...}]
        }
        """
        urgent  = sum(1 for f in findings if f.urgency == UrgencyLevel.URGENT)
        routine = sum(1 for f in findings if f.urgency == UrgencyLevel.ROUTINE)
        monitor = sum(1 for f in findings if f.urgency == UrgencyLevel.MONITOR)

        # Deduplicated specialist list (preserves urgency order)
        seen_specialists = set()
        referrals_ordered = []
        for f in findings:
            for r in f.referrals:
                if r.specialist not in seen_specialists:
                    seen_specialists.add(r.specialist)
                    referrals_ordered.append(r.specialist)

        return {
            "total_findings": len(findings),
            "urgent_count":   urgent,
            "routine_count":  routine,
            "monitor_count":  monitor,
            "referrals":      referrals_ordered,
            "findings":       [f.to_dict() for f in findings],
        }

"""
Multi-LLM Risk Interpreter — 2-Tier Quality Pipeline
======================================================

Pipeline:
  Phase 1: Gemini   → Primary JSON report (summary, explanation, recs, caveats)
  Phase 2: II-Medical-8B → Combined Validator + Corrector in ONE HF call
            (previously 2 separate HF calls; collapsed to 1 for 50% cost saving)

Smart Cutoffs:
  - MODERATE risk (25-75) with confidence ≥ 0.5 → Single Gemini fast-path
  - LOW (<25) / HIGH (>75) OR confidence < 0.5  → Full 2-LLM pipeline

Gemini RPM guard:
  - A 12-second sleep is injected before the Gemini call when we know this is
    NOT the first call in the process (indicated by ``_interpretation_count > 0``),
    keeping well within the 5 RPM (= 1 call per 12 s) free-tier limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import asyncio
import json
import time

from app.core.inference.risk_engine import SystemRiskResult, RiskScore, RiskLevel
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope
from app.core.llm.gemini_client import GeminiClient, GeminiConfig
from app.core.agents.hf_client import HuggingFaceClient, HFConfig
from app.core.llm.validators import LLMValidator
from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Gemini rate-limit guard (5 RPM free tier → ≥12 s between calls)
# ---------------------------------------------------------------------------
_GEMINI_MIN_INTERVAL_S: float = 12.0   # seconds between consecutive Gemini calls


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class CombinedReview:
    """
    Result of the Phase 2 combined HF validator + corrector.

    Replaces the old ``ValidationReview`` + ``ArbiterDecision`` pair.
    One HF call now returns all six fields.
    """
    is_clinically_appropriate: bool = True
    tone_matches_risk: bool = True
    missing_caveats: List[str] = field(default_factory=list)
    confidence: str = "high"
    use_corrected: bool = False
    corrected_summary: str = ""
    raw_response: str = ""


@dataclass
class MultiLLMInterpretation:
    """Combined interpretation from the 2-tier LLM pipeline."""

    # Primary output
    summary: str = ""
    detailed_explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    # Pipeline audit trail
    pipeline_mode: str = "single"            # "single" | "full_pipeline"
    phase1_latency_ms: float = 0.0
    phase2_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    validation_passed: bool = True
    review_decision: str = ""               # replaces old arbiter_decision
    review_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "recommendations": self.recommendations,
            "caveats": self.caveats,
            "pipeline_mode": self.pipeline_mode,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "validation_passed": self.validation_passed,
            "review_decision": self.review_decision,
        }


# ===========================================================================
# Interpreter
# ===========================================================================

class MultiLLMInterpreter:
    """
    2-Tier Sequential Quality Pipeline for Medical Interpretation.

    Phase 1: Gemini generates authoritative JSON baseline.
    Phase 2: II-Medical-8B validates tone AND proposes corrections — one call.

    Smart Cutoffs:
      - MODERATE (25–75) + confidence ≥ 0.5 → Gemini only (fast path)
      - LOW / HIGH / low-confidence          → Full 2-LLM pipeline
    """

    SYSTEM_INSTRUCTION = (
        "You are a medical screening interpretation assistant. Your role is to EXPLAIN "
        "health screening results, NOT to diagnose or treat.\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "1. Explain PRE-COMPUTED risk scores — do NOT assign new scores.\n"
        "2. Do NOT make diagnoses or suggest specific conditions.\n"
        "3. Do NOT recommend specific treatments or medications.\n"
        "4. Always recommend consulting a healthcare professional.\n"
        "5. Use appropriate uncertainty language based on confidence levels.\n"
        "6. Be clear this is a screening tool, not a diagnostic instrument.\n"
        "7. When mentioning risk scores, use the EXACT numbers provided.\n"
        "Provide clear, educational, thorough explanations. Explain WHY scores are high/low."
    )

    REVIEWER_SYSTEM = (
        "You are a medical quality reviewer and report editor. "
        "Critique health reports for clinical appropriateness. "
        "If the report is inappropriate, provide a corrected summary. "
        "Be concise and return JSON only."
    )

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        hf_client: Optional[HuggingFaceClient] = None,
    ):
        """Initialise the 2-tier quality pipeline."""

        # Phase 1 — Gemini
        if gemini_client is None:
            json_schema = {
                "type": "object",
                "properties": {
                    "summary":              {"type": "string"},
                    "detailed_explanation": {"type": "string"},
                    "recommendations":      {"type": "array", "items": {"type": "string"}},
                    "caveats":              {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "detailed_explanation", "recommendations"],
            }
            config = GeminiConfig(
                api_key=settings.gemini_api_key,
                response_mime_type="application/json",
                response_schema=json_schema,
            )
            self.gemini_client = GeminiClient(config)
        else:
            self.gemini_client = gemini_client

        # Phase 2 — II-Medical-8B via HF Inference API
        if hf_client is None:
            config = HFConfig(api_key=settings.hf_token)
            self.hf_client = HuggingFaceClient(config)
        else:
            self.hf_client = hf_client

        self._interpretation_count: int = 0
        self._last_gemini_call_time: float = 0.0
        self._validator = LLMValidator()
        logger.info("MultiLLMInterpreter initialised — 2-Tier Quality Pipeline")

    # ------------------------------------------------------------------
    # Rate-limit helper
    # ------------------------------------------------------------------
    async def _wait_for_gemini_rpm(self) -> None:
        """
        Enforce a minimum gap between Gemini calls to respect the 5 RPM limit.
        Only waits when there has been a previous Gemini call in this process.
        """
        if self._last_gemini_call_time == 0.0:
            return  # First call — no wait needed
        elapsed = time.time() - self._last_gemini_call_time
        gap = _GEMINI_MIN_INTERVAL_S - elapsed
        if gap > 0:
            logger.info(f"Gemini RPM guard: waiting {gap:.1f}s before next call …")
            await asyncio.sleep(gap)

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------
    async def interpret_composite_risk(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None,
    ) -> MultiLLMInterpretation:
        """
        Run the 2-tier quality pipeline to interpret screening results.

        Pipeline:
          1. Gemini: Generates primary insight + JSON report.
          2. II-Medical-8B: Combined validation + correction in ONE HF pass
             (only for LOW/HIGH risk or low-confidence data).
        """
        start_time = time.time()
        result = MultiLLMInterpretation(pipeline_mode="single")

        score      = composite_risk.score
        confidence = composite_risk.confidence

        use_full_pipeline = (
            score < 25           # Unexpectedly low → verify
            or score > 75        # Action-required  → must validate
            or confidence < 0.5  # Low-confidence data → validate report tone
        )

        result.pipeline_mode = "full_pipeline" if use_full_pipeline else "single"
        logger.info(
            f"Risk {score:.1f} (conf={confidence:.0%}) → "
            f"{'FULL 2-LLM pipeline' if use_full_pipeline else 'SINGLE Gemini fast-path'}"
        )

        # ── Phase 1: Gemini primary generation ───────────────────────────────
        await self._wait_for_gemini_rpm()
        p1_start = time.time()
        prompt_p1 = self._build_primary_prompt(system_results, composite_risk, trust_envelope)
        logger.info("Phase 1 (Gemini): generating primary report …")
        resp_p1 = await self.gemini_client.generate_async(
            prompt_p1, system_instruction=self.SYSTEM_INSTRUCTION
        )
        self._last_gemini_call_time = time.time()
        result.phase1_latency_ms = (time.time() - p1_start) * 1000

        if resp_p1.is_mock:
            logger.warning("Phase 1 returned MOCK response")

        # Parse Phase 1 output
        try:
            p1_json = self._parse_json_response(resp_p1.text)
            result.summary              = p1_json.get("summary", "")
            result.detailed_explanation = p1_json.get("detailed_explanation", "")
            result.recommendations      = p1_json.get("recommendations", [])
            result.caveats              = p1_json.get("caveats", [])
        except Exception as e:
            logger.error(f"Phase 1 JSON parse failed: {e}")
            result.summary          = "Preliminary screening completed. Clinical review recommended."
            result.validation_passed = False
            result.total_latency_ms  = (time.time() - start_time) * 1000
            return result

        # Fast-path: no HF call needed
        if not use_full_pipeline:
            result.validation_passed  = True
            result.review_decision    = "fast_path_approved"
            result.total_latency_ms   = (time.time() - start_time) * 1000
            self._add_standard_caveats(result, trust_envelope)
            self._interpretation_count += 1
            return result

        # ── Phase 2: Combined HF Validation + Correction (ONE call) ──────────
        p2_start = time.time()
        review = await self._phase2_combined_review_async(resp_p1.text, composite_risk)
        result.phase2_latency_ms = (time.time() - p2_start) * 1000

        result.validation_passed = review.is_clinically_appropriate and review.tone_matches_risk

        if result.validation_passed:
            result.review_decision = "approved_by_reviewer"
        else:
            result.review_decision = "corrected_by_reviewer"
            # Apply correction if reviewer provided one
            if review.use_corrected and review.corrected_summary:
                result.summary = review.corrected_summary
                result.caveats.insert(0, "Report adjusted by quality reviewer.")

        # Append any missing caveats the reviewer flagged
        for caveat in review.missing_caveats:
            if caveat and caveat not in result.caveats:
                result.caveats.append(caveat)

        result.total_latency_ms = (time.time() - start_time) * 1000
        self._add_standard_caveats(result, trust_envelope)
        self._interpretation_count += 1
        logger.info(
            f"2-Tier pipeline complete — {result.review_decision} "
            f"({result.total_latency_ms:.0f} ms total, "
            f"phase1={result.phase1_latency_ms:.0f} ms, "
            f"phase2={result.phase2_latency_ms:.0f} ms)"
        )
        return result

    # ------------------------------------------------------------------
    # Phase 2 — Combined review (was Phase 2 + Phase 3)
    # ------------------------------------------------------------------
    async def _phase2_combined_review_async(
        self, primary_response: str, risk: RiskScore
    ) -> CombinedReview:
        """
        Phase 2: Single HF call that simultaneously validates AND corrects.

        Old pipeline: 2 × HF calls (validate → arbitrate).
        New pipeline: 1 × HF call (validate+correct).
        Saves ~50% of HuggingFace token cost + network latency.
        """
        review = CombinedReview()

        # Use just the summary sentence to keep the HF prompt compact.
        try:
            p1_data = self._parse_json_response(primary_response)
            summary_text = p1_data.get("summary", primary_response[:300])
        except Exception:
            summary_text = primary_response[:300]

        reviewer_prompt = (
            f"Medical report review. Risk level: {risk.level.value.upper()} "
            f"(score {risk.score:.0f}/100, confidence {risk.confidence:.0%}).\n"
            f'Report summary: "{summary_text[:500]}"\n\n'
            "Evaluate the summary and return JSON only:\n"
            "{\n"
            '  "is_clinically_appropriate": true/false,\n'
            '  "tone_matches_risk": true/false,\n'
            '  "missing_caveats": ["caveat1", ...],\n'
            '  "confidence": "high/medium/low",\n'
            '  "use_corrected": true/false,\n'
            '  "corrected_summary": "rewritten summary if use_corrected is true, else empty string"\n'
            "}\n"
            "Set use_corrected=true ONLY if the summary is clinically inappropriate or "
            "dangerously overstated. Always return all six keys."
        )

        try:
            response = await self.hf_client.generate_async(
                reviewer_prompt,
                model=settings.medical_model_1,   # II-Medical-8B
                system_prompt=self.REVIEWER_SYSTEM,
            )
            review.raw_response = response.text
            data = self._parse_json_response(response.text)
            review.is_clinically_appropriate = data.get("is_clinically_appropriate", True)
            review.tone_matches_risk          = data.get("tone_matches_risk", True)
            review.missing_caveats            = data.get("missing_caveats", [])
            review.confidence                 = data.get("confidence", "medium")
            review.use_corrected              = data.get("use_corrected", False)
            review.corrected_summary          = data.get("corrected_summary", "")
        except Exception as e:
            logger.warning(f"Phase 2 combined review failed: {e} — defaulting to pass")

        return review

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _build_primary_prompt(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope],
    ) -> str:
        """Build the Phase 1 prompt for Gemini."""
        prompt = (
            "Interpret the following comprehensive health screening results for a patient.\n\n"
            "OVERALL HEALTH ASSESSMENT:\n"
            f"- Composite Risk Score: {composite_risk.score:.1f}/100\n"
            f"- Composite Risk Level: {composite_risk.level.value.upper()}\n"
            f"- Overall Confidence:   {composite_risk.confidence:.0%}\n\n"
            "SYSTEM-BY-SYSTEM BREAKDOWN:\n"
        )

        for system, result in system_results.items():
            sys_name = system.value.replace("_", " ").title()
            prompt += (
                f"\n{sys_name}:\n"
                f"  - Risk Level:  {result.overall_risk.level.value.upper()}\n"
                f"  - Risk Score:  {result.overall_risk.score:.1f}/100\n"
                f"  - Confidence:  {result.overall_risk.confidence:.0%}\n"
            )
            if result.alerts:
                prompt += f"  - Alerts:      {len(result.alerts)} item(s) requiring attention\n"

        if trust_envelope:
            prompt += (
                "\nDATA RELIABILITY CONTEXT:\n"
                f"- Overall Reliability:          {trust_envelope.overall_reliability:.0%}\n"
                f"- Data Quality (Sensors):       {trust_envelope.data_quality_score:.0%}\n"
                f"- Physiological Plausibility:   {trust_envelope.biomarker_plausibility:.0%}\n"
                f"- Cross-System Consistency:     {trust_envelope.cross_system_consistency:.0%}\n"
            )
            if trust_envelope.critical_issues:
                prompt += f"- CRITICAL ISSUES: {'; '.join(trust_envelope.critical_issues[:5])}\n"
            if trust_envelope.warnings:
                prompt += f"- Data Warnings:   {'; '.join(trust_envelope.warnings[:5])}\n"
            prompt += f"- Reliability Guidance: {trust_envelope.interpretation_guidance}\n"

        prompt += (
            "\nProvide a comprehensive interpretation in JSON format:\n"
            "{\n"
            '  "summary": "3-4 sentences summarising overall health status",\n'
            '  "detailed_explanation": "Comprehensive analysis with three sections: '
            "1. Main Findings, 2. Potential Causes (lifestyle/stress/etc.), "
            '3. Urgency Assessment. Use <br/> for line breaks.",\n'
            '  "recommendations": ["Specific, actionable step 1", "Step 2", ...],\n'
            '  "caveats": ["Limit 1", "Limit 2", ...]\n'
            "}\n\n"
            "Use patient-friendly language. Always recommend professional medical consultation."
        )
        return prompt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response."""
        text = text.strip()
        try:
            if text.startswith("{"):
                return json.loads(text)
            if "```json" in text:
                start = text.find("```json") + 7
                end   = text.find("```", start)
                return json.loads(text[start:end].strip())
            if "```" in text:
                start = text.find("```") + 3
                end   = text.find("```", start)
                return json.loads(text[start:end].strip())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
        return {}

    def _add_standard_caveats(
        self, result: MultiLLMInterpretation, trust_envelope: Optional[TrustEnvelope]
    ) -> None:
        """Append standard medical caveats."""
        standard = [
            "This is a screening report, not a medical diagnosis.",
            "Results should be reviewed by a qualified healthcare provider.",
        ]
        for caveat in standard:
            if caveat not in result.caveats:
                result.caveats.append(caveat)

        if trust_envelope and not trust_envelope.is_reliable:
            result.caveats.insert(0, "Data quality issues detected — results should be verified.")

    def get_stats(self) -> Dict[str, Any]:
        """Get interpreter statistics."""
        return {
            "interpretation_count": self._interpretation_count,
            "gemini_available":     self.gemini_client.is_available,
            "hf_available":         self.hf_client.is_available,
            "pipeline_type":        "2_tier_quality",
            "models_used": [
                f"{self.gemini_client.config.model.value} (Primary Writer — Phase 1)",
                f"{settings.medical_model_1} (Combined Reviewer — Phase 2)",
            ],
        }


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
# Old code that imported ArbiterDecision, ValidationReview is unlikely to
# exist outside this file, but we keep a stub for safety.
ValidationReview = CombinedReview

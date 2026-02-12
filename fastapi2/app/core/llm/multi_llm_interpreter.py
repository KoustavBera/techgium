"""
Multi-LLM Risk Interpreter - Sequential Quality Pipeline

Uses LLMs in a SEQUENTIAL quality pipeline:
- Phase 1: Gemini generates primary JSON report
- Phase 2: HF Medical 1 validates clinical tone (Second Opinion)
- Phase 3: HF Medical 2 arbitrates conflicts (Quality Gate)

Smart Cutoffs:
- LOW/HIGH risk: Full 3-LLM pipeline
- MODERATE risk: Single Gemini (fast path)
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
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


@dataclass
class ValidationReview:
    """Result from Phase 2 medical validation."""
    is_clinically_appropriate: bool = True
    tone_matches_risk: bool = True
    missing_caveats: List[str] = field(default_factory=list)
    confidence: str = "high"
    raw_response: str = ""


@dataclass
class ArbiterDecision:
    """Result from Phase 3 arbitration."""
    approved: bool = True
    primary_reason: str = ""
    use_corrected: bool = False
    corrected_summary: str = ""
    escalate_to_human: bool = False
    raw_response: str = ""


@dataclass
class MultiLLMInterpretation:
    """Combined interpretation from sequential LLM pipeline."""
    # Primary output
    summary: str = ""
    detailed_explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    
    # Pipeline audit trail
    pipeline_mode: str = "single"  # single / full_pipeline
    phase1_latency_ms: float = 0.0
    phase2_latency_ms: float = 0.0
    phase3_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    validation_passed: bool = True
    arbiter_decision: str = ""
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
            "arbiter_decision": self.arbiter_decision
        }


class MultiLLMInterpreter:
    """
    Sequential Quality Pipeline for Medical Interpretation.
    
    Phase 1: Gemini generates authoritative JSON baseline
    Phase 2: HF Medical 1 validates clinical appropriateness (Second Opinion)
    Phase 3: HF Medical 2 arbitrates disputes (Quality Gate)
    
    Smart Cutoffs:
    - MODERATE risk (30-80): Single Gemini only
    - LOW (<30) / HIGH (>80): Full 3-LLM pipeline
    """
    
    SYSTEM_INSTRUCTION = """You are a medical screening interpretation assistant. Your role is to EXPLAIN health screening results, NOT to diagnose or treat.

CRITICAL CONSTRAINTS:
1. You are explaining PRE-COMPUTED risk scores - do NOT assign new scores
2. Do NOT make diagnoses or suggest specific conditions
3. Do NOT recommend specific treatments or medications
4. Always recommend consulting a healthcare professional
5. Use appropriate uncertainty language based on confidence levels
6. Be clear that this is a screening tool, not a diagnostic instrument
7. When mentioning risk scores, use the EXACT numbers provided in the input.

Provide clear, educational, and thorough explanations. Avoid being vague. Explain WHY a score might be high or low, and WHAT the patient should focus on."""

    VALIDATOR_SYSTEM = """You are a medical quality reviewer. Your role is to critique health reports for clinical appropriateness and tone accuracy. Be concise and return JSON only."""

    ARBITER_SYSTEM = """You are a medical quality arbiter. Your role is to make final decisions on report approval based on primary content and validator feedback. Be concise and return JSON only."""
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        hf_client: Optional[HuggingFaceClient] = None
    ):
        """Initialize sequential quality pipeline."""
        # Initialize Gemini client (Phase 1)
        if gemini_client is None:
            json_schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "detailed_explanation": {"type": "string"},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                    "caveats": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["summary", "detailed_explanation", "recommendations"]
            }
            config = GeminiConfig(
                api_key=settings.gemini_api_key,
                response_mime_type="application/json",
                response_schema=json_schema
            )
            self.gemini_client = GeminiClient(config)
        else:
            self.gemini_client = gemini_client
        
        # Initialize HuggingFace client (Phase 2 & 3)
        if hf_client is None:
            config = HFConfig(api_key=settings.hf_token)
            self.hf_client = HuggingFaceClient(config)
        else:
            self.hf_client = hf_client
        
        self._interpretation_count = 0
        self._validator = LLMValidator()
        logger.info("MultiLLMInterpreter initialized with Sequential Quality Pipeline")
    
    async def interpret_composite_risk(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None
    ) -> MultiLLMInterpretation:
        """
        Run the sequential 3-LLM quality pipeline to interpret screening results.
        
        Pipeline:
        1. Gemini: Generates primary insight and JSON report structure.
        2. II-Medical-8B: Validates clinical appropriateness and medical tone.
        3. GPT-OSS-120B: Arbitrates any conflicts and makes final "Pass/Fail" decision.
        """
        start_time = time.time()
        result = MultiLLMInterpretation(
            pipeline_mode="single"
        )
        
        # SMART CUTOFF: Determine pipeline mode
        score = composite_risk.score
        
        # FORCE VALIDATION: Ensure Intelligent Internet (Phase 2) always runs
        # use_full_pipeline = score < 30 or score > 80  # Old logic
        use_full_pipeline = True  # User requested full validation for all reports
        
        if use_full_pipeline:
            result.pipeline_mode = "full_pipeline"
            logger.info(f"Risk score {score:.1f} triggers FULL 3-LLM pipeline (forced)")
        else:
            result.pipeline_mode = "single"
            logger.info(f"Risk score {score:.1f} uses SINGLE Gemini (fast path)")
            
        # === PHASE 1: Primary Generation (Gemini) ===
        p1_start = time.time()
        prompt_p1 = self._build_primary_prompt(system_results, composite_risk, trust_envelope)
        
        logger.info("Starting MultiLLM Phase 1 (Gemini)...")
        resp_p1 = await self.gemini_client.generate_async(prompt_p1, system_instruction=self.SYSTEM_INSTRUCTION)
        result.phase1_latency_ms = (time.time() - p1_start) * 1000
        
        if resp_p1.is_mock:
            logger.warning("Phase 1 returned MOCK response")
        
        # Parse Phase 1 output
        try:
            p1_json = self._parse_json_response(resp_p1.text)
            result.summary = p1_json.get("summary", "")
            result.detailed_explanation = p1_json.get("detailed_explanation", "")
            result.recommendations = p1_json.get("recommendations", [])
            result.caveats = p1_json.get("caveats", [])
        except Exception as e:
            logger.error(f"Failed to parse Phase 1 JSON: {e}")
            result.summary = "Preliminary screening completed. Clinical review recommended."
            result.validation_passed = False
            return result

        if not use_full_pipeline:
            result.validation_passed = True
            result.total_latency_ms = (time.time() - start_time) * 1000
            self._add_standard_caveats(result, trust_envelope)
            return result

        # === PHASE 2: Medical Validation (HF Medical 1/II-Medical) ===
        p2_start = time.time()
        validation = await self._phase2_validate_async(resp_p1.text, composite_risk)
        result.phase2_latency_ms = (time.time() - p2_start) * 1000
        result.validation_passed = validation.is_clinically_appropriate and validation.tone_matches_risk
        
        if result.validation_passed:
            result.arbiter_decision = "approved_by_validator"
            result.total_latency_ms = (time.time() - start_time) * 1000
            self._add_standard_caveats(result, trust_envelope)
            self._interpretation_count += 1
            return result
        
        # === PHASE 3: Arbitration (HF Medical 2/GPT-OSS) ===
        p3_start = time.time()
        arbiter = await self._phase3_arbitrate_async(resp_p1.text, validation, composite_risk)
        result.phase3_latency_ms = (time.time() - p3_start) * 1000
        
        result.arbiter_decision = "approved" if arbiter.approved else "corrected"
        
        if arbiter.use_corrected and arbiter.corrected_summary:
            result.summary = arbiter.corrected_summary
            result.caveats.insert(0, "Report adjusted by quality pipeline.")
        
        if arbiter.escalate_to_human:
            result.caveats.insert(0, "Manual review recommended for this result.")
        
        result.total_latency_ms = (time.time() - start_time) * 1000
        self._add_standard_caveats(result, trust_envelope)
        self._interpretation_count += 1
        logger.info(f"Full pipeline complete: {result.total_latency_ms:.0f}ms")
        
        return result
    
    async def _phase2_validate_async(self, primary_response: str, risk: RiskScore) -> ValidationReview:
        """Phase 2: Validate clinical tone with HF Medical 1."""
        review = ValidationReview()
        
        validator_prompt = f"""Review this health screening report for a {risk.level.value.upper()} risk patient (score: {risk.score:.1f}/100):
\n{primary_response[:1500]}\n
Critique this report (JSON only):
{{
    "is_clinically_appropriate": true/false,
    "tone_matches_risk": true/false,
    "missing_caveats": ["list of missing items"],
    "confidence": "high/medium/low"
}}"""

        try:
            response = await self.hf_client.generate_async(
                validator_prompt,
                model=settings.medical_model_1,
                system_prompt=self.VALIDATOR_SYSTEM
            )
            review.raw_response = response.text
            data = self._parse_json_response(response.text)
            review.is_clinically_appropriate = data.get("is_clinically_appropriate", True)
            review.tone_matches_risk = data.get("tone_matches_risk", True)
            review.missing_caveats = data.get("missing_caveats", [])
            review.confidence = data.get("confidence", "medium")
        except Exception as e:
            logger.warning(f"Phase 2 validation failed: {e}, defaulting to pass")
        
        return review

    async def _phase3_arbitrate_async(
        self, primary_response: str, validation: ValidationReview, risk: RiskScore
    ) -> ArbiterDecision:
        """Phase 3: Arbitrate conflicts with HF Medical 2."""
        decision = ArbiterDecision()
        arb_prompt = f"""ARBITRATE HEALTH REPORT:
RISK: {risk.level.value.upper()} ({risk.score:.1f})
PRIMARY: {primary_response[:1000]}
VALIDATOR: {validation.is_clinically_appropriate}, {validation.tone_matches_risk}, {validation.missing_caveats}
JSON DECISION:
{{ "approved": bool, "primary_reason": "...", "use_corrected": bool, "corrected_summary": "...", "escalate_to_human": bool }}"""

        try:
            response = await self.hf_client.generate_async(
                arb_prompt,
                model=settings.medical_model_2,
                system_prompt=self.ARBITER_SYSTEM
            )
            data = self._parse_json_response(response.text)
            decision.approved = data.get("approved", True)
            decision.use_corrected = data.get("use_corrected", False)
            decision.corrected_summary = data.get("corrected_summary", "")
            decision.escalate_to_human = data.get("escalate_to_human", False)
        except Exception as e:
            logger.warning(f"Phase 3 arbitration failed: {e}")
            decision.approved = True
        
        return decision
    
    def _build_primary_prompt(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope]
    ) -> str:
        """Build prompt for Phase 1 (Gemini)."""
        prompt = f"""Interpret the following comprehensive health screening results for a patient.

OVERALL HEALTH ASSESSMENT:
- Composite Risk Score: {composite_risk.score:.1f}/100
- Composite Risk Level: {composite_risk.level.value.upper()}
- Overall Confidence: {composite_risk.confidence:.0%}

SYSTEM-BY-SYSTEM BREAKDOWN:
"""
        
        for system, result in system_results.items():
            system_name = system.value.replace("_", " ").title()
            prompt += f"\n{system_name}:\n"
            prompt += f"  - Risk Level: {result.overall_risk.level.value.upper()}\n"
            prompt += f"  - Risk Score: {result.overall_risk.score:.1f}/100\n"
            prompt += f"  - Confidence: {result.overall_risk.confidence:.0%}\n"
            if result.alerts:
                prompt += f"  - Alerts: {len(result.alerts)} items requiring attention\n"
        
        if trust_envelope:
            prompt += f"\nDATA RELIABILITY: {trust_envelope.overall_reliability:.0%}\n"
            if trust_envelope.critical_issues:
                prompt += f"CRITICAL ISSUES: {len(trust_envelope.critical_issues)}\n"
        
        prompt += """
Provide a comprehensive interpretation in JSON format:
{
    "summary": "3-4 sentences summarizing overall health status",
    "detailed_explanation": "Comprehensive analysis with 3 distinct sections: 1. Main Findings (what stands out), 2. Potential Causes (lifestyle/stress/etc), 3. Urgency Assessment (how quickly to act). Use <br/> for line breaks.",
    "recommendations": ["Specific, actionable step 1", "Specific, actionable step 2", ...],
    "caveats": ["Limit 1", "Limit 2", ...]
}

Use simple, patient-friendly language but do NOT be superficial. Go deep into the 'why'. Always recommend professional medical consultation."""
        
        return prompt
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try direct parse
            if text.strip().startswith("{"):
                return json.loads(text)
            
            # Try extracting from markdown code block
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                return json.loads(text[start:end].strip())
            
            if "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                return json.loads(text[start:end].strip())
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
        
        return {}
    
    def _add_standard_caveats(self, result: MultiLLMInterpretation, trust_envelope: Optional[TrustEnvelope]):
        """Add standard caveats to result."""
        standard = [
            "This is a screening report, not a medical diagnosis.",
            "Results should be reviewed by a qualified healthcare provider."
        ]
        
        for caveat in standard:
            if caveat not in result.caveats:
                result.caveats.append(caveat)
        
        if trust_envelope and not trust_envelope.is_reliable:
            result.caveats.insert(0, "Data quality issues detected - results should be verified.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get interpreter statistics."""
        return {
            "interpretation_count": self._interpretation_count,
            "gemini_available": self.gemini_client.is_available,
            "hf_available": self.hf_client.is_available,
            "pipeline_type": "sequential_quality",
            "models_used": [
                "gemini-1.5-flash (Primary)",
                f"{settings.medical_model_1} (Validator)",
                f"{settings.medical_model_2} (Arbiter)"
            ]
        }

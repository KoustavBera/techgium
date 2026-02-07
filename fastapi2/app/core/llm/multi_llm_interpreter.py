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
    
    def interpret_composite_risk(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None
    ) -> MultiLLMInterpretation:
        """
        Interpret overall health risk using sequential quality pipeline.
        
        Smart Cutoffs:
        - MODERATE (30-80): Single Gemini (fast path)
        - LOW/HIGH: Full 3-LLM pipeline
        """
        result = MultiLLMInterpretation()
        total_start = time.time()
        
        # Build primary prompt
        prompt = self._build_primary_prompt(system_results, composite_risk, trust_envelope)
        
        # SMART CUTOFF: Determine pipeline mode
        score = composite_risk.score
        use_full_pipeline = score < 30 or score > 80
        
        if use_full_pipeline:
            result.pipeline_mode = "full_pipeline"
            logger.info(f"Risk score {score:.1f} triggers FULL 3-LLM pipeline")
        else:
            result.pipeline_mode = "single"
            logger.info(f"Risk score {score:.1f} uses SINGLE Gemini (fast path)")
        
        # === PHASE 1: Primary Generation (Gemini) ===
        phase1_start = time.time()
        primary_response = self._phase1_generate_primary(prompt)
        result.phase1_latency_ms = (time.time() - phase1_start) * 1000
        logger.info(f"Phase 1 complete: {result.phase1_latency_ms:.0f}ms")
        
        # Parse primary response
        primary_data = self._parse_json_response(primary_response)
        result.summary = primary_data.get("summary", "")
        result.detailed_explanation = primary_data.get("detailed_explanation", "")
        result.recommendations = primary_data.get("recommendations", [])
        result.caveats = primary_data.get("caveats", [])
        
        # Fast path: MODERATE risk - return after Phase 1
        if not use_full_pipeline:
            result.total_latency_ms = (time.time() - total_start) * 1000
            self._add_standard_caveats(result, trust_envelope)
            self._interpretation_count += 1
            logger.info(f"Fast path complete: {result.total_latency_ms:.0f}ms total")
            return result
        
        # === PHASE 2: Medical Validation (HF Medical 1) ===
        phase2_start = time.time()
        validation = self._phase2_validate(primary_response, composite_risk)
        result.phase2_latency_ms = (time.time() - phase2_start) * 1000
        result.validation_passed = validation.is_clinically_appropriate and validation.tone_matches_risk
        logger.info(f"Phase 2 complete: {result.phase2_latency_ms:.0f}ms, passed={result.validation_passed}")
        
        if result.validation_passed:
            # Validation passed - use primary
            result.arbiter_decision = "approved_by_validator"
            result.total_latency_ms = (time.time() - total_start) * 1000
            self._add_standard_caveats(result, trust_envelope)
            self._interpretation_count += 1
            logger.info(f"Pipeline complete (2 phases): {result.total_latency_ms:.0f}ms")
            return result
        
        # Add validation notes
        result.review_notes = validation.missing_caveats
        
        # === PHASE 3: Arbitration (HF Medical 2) ===
        phase3_start = time.time()
        arbiter = self._phase3_arbitrate(primary_response, validation, composite_risk)
        result.phase3_latency_ms = (time.time() - phase3_start) * 1000
        logger.info(f"Phase 3 complete: {result.phase3_latency_ms:.0f}ms")
        
        result.arbiter_decision = "approved" if arbiter.approved else "corrected"
        
        if arbiter.use_corrected and arbiter.corrected_summary:
            result.summary = arbiter.corrected_summary
            result.caveats.insert(0, "Report adjusted by quality pipeline.")
        
        if arbiter.escalate_to_human:
            result.caveats.insert(0, "Manual review recommended for this result.")
        
        result.total_latency_ms = (time.time() - total_start) * 1000
        self._add_standard_caveats(result, trust_envelope)
        self._interpretation_count += 1
        logger.info(f"Full pipeline complete: {result.total_latency_ms:.0f}ms")
        
        return result
    
    def _phase1_generate_primary(self, prompt: str) -> str:
        """Phase 1: Generate primary report with Gemini."""
        try:
            response = self.gemini_client.generate(prompt, self.SYSTEM_INSTRUCTION)
            return response.text
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            return json.dumps({
                "summary": "Health screening analysis completed. Consult healthcare provider.",
                "detailed_explanation": "Unable to generate detailed analysis.",
                "recommendations": ["Consult a healthcare professional."],
                "caveats": ["Report generation experienced technical issues."]
            })
    
    def _phase2_validate(self, primary_response: str, risk: RiskScore) -> ValidationReview:
        """Phase 2: Validate clinical tone with HF Medical 1."""
        review = ValidationReview()
        
        validator_prompt = f"""Review this health screening report for a {risk.level.value.upper()} risk patient (score: {risk.score:.1f}/100):

---
{primary_response[:1500]}
---

Critique this report (JSON only):
{{
    "is_clinically_appropriate": true/false,
    "tone_matches_risk": true/false,
    "missing_caveats": ["list of missing items"],
    "confidence": "high/medium/low"
}}"""

        try:
            response = self.hf_client.generate(
                validator_prompt,
                model=settings.medical_model_1,
                system_prompt=self.VALIDATOR_SYSTEM
            )
            review.raw_response = response.text
            
            # Parse JSON response
            data = self._parse_json_response(response.text)
            review.is_clinically_appropriate = data.get("is_clinically_appropriate", True)
            review.tone_matches_risk = data.get("tone_matches_risk", True)
            review.missing_caveats = data.get("missing_caveats", [])
            review.confidence = data.get("confidence", "medium")
            
        except Exception as e:
            logger.warning(f"Phase 2 validation failed: {e}, defaulting to pass")
            review.is_clinically_appropriate = True
            review.tone_matches_risk = True
        
        return review
    
    def _phase3_arbitrate(
        self, primary_response: str, validation: ValidationReview, risk: RiskScore
    ) -> ArbiterDecision:
        """Phase 3: Arbitrate conflicts with HF Medical 2."""
        decision = ArbiterDecision()
        
        arbiter_prompt = f"""You are making a final quality decision on a health report.

RISK LEVEL: {risk.level.value.upper()} ({risk.score:.1f}/100)

PRIMARY REPORT:
{primary_response[:1000]}

VALIDATOR FEEDBACK:
- Clinically appropriate: {validation.is_clinically_appropriate}
- Tone matches risk: {validation.tone_matches_risk}
- Missing caveats: {validation.missing_caveats}

Make your decision (JSON only):
{{
    "approved": true/false,
    "primary_reason": "brief reason",
    "use_corrected": false,
    "corrected_summary": "only if use_corrected is true",
    "escalate_to_human": false
}}"""

        try:
            response = self.hf_client.generate(
                arbiter_prompt,
                model=settings.medical_model_2,
                system_prompt=self.ARBITER_SYSTEM
            )
            decision.raw_response = response.text
            
            data = self._parse_json_response(response.text)
            decision.approved = data.get("approved", True)
            decision.primary_reason = data.get("primary_reason", "")
            decision.use_corrected = data.get("use_corrected", False)
            decision.corrected_summary = data.get("corrected_summary", "")
            decision.escalate_to_human = data.get("escalate_to_human", False)
            
        except Exception as e:
            logger.warning(f"Phase 3 arbitration failed: {e}, approving primary")
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
                "gemini-2.5-flash (Primary)",
                f"{settings.medical_model_1} (Validator)",
                f"{settings.medical_model_2} (Arbiter)"
            ]
        }

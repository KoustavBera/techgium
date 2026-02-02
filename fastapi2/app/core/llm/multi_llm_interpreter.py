"""
Multi-LLM Risk Interpreter

Uses multiple LLMs (Gemini via LangChain + 2 HuggingFace medical models)
for comprehensive health screening interpretation.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio

from app.core.inference.risk_engine import SystemRiskResult, RiskScore, RiskLevel
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope
from app.core.llm.gemini_client import GeminiClient, GeminiConfig
from app.core.agents.hf_client import HuggingFaceClient, HFConfig
from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MultiLLMInterpretation:
    """Combined interpretation from multiple LLMs."""
    gemini_response: str = ""
    medical_model_1_response: str = ""
    medical_model_2_response: str = ""
    
    # Synthesized content
    summary: str = ""
    detailed_explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    
    # Metadata
    consensus_level: str = "high"  # high/medium/low
    total_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gemini_response": self.gemini_response[:200] + "...",
            "medical_model_1_response": self.medical_model_1_response[:200] + "...",
            "medical_model_2_response": self.medical_model_2_response[:200] + "...",
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "recommendations": self.recommendations,
            "caveats": self.caveats,
            "consensus_level": self.consensus_level,
            "total_latency_ms": round(self.total_latency_ms, 2)
        }


class MultiLLMInterpreter:
    """
    Multi-LLM interpreter using:
    - Gemini 2.5 Flash (via LangChain)
    - GPT-OSS-120B (HuggingFace)
    - II-Medical-8B (HuggingFace)
    
    All LLMs are NON-DECISIONAL - they explain pre-computed risks only.
    """
    
    SYSTEM_INSTRUCTION = """You are a medical screening interpretation assistant. Your role is to EXPLAIN health screening results, NOT to diagnose or treat.

CRITICAL CONSTRAINTS:
1. You are explaining PRE-COMPUTED risk scores - do NOT assign new scores
2. Do NOT make diagnoses or suggest specific conditions
3. Do NOT recommend specific treatments or medications
4. Always recommend consulting a healthcare professional
5. Use appropriate uncertainty language based on confidence levels
6. Be clear that this is a screening tool, not a diagnostic instrument

Provide clear, educational explanations suitable for patients."""
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        hf_client: Optional[HuggingFaceClient] = None
    ):
        """Initialize multi-LLM interpreter."""
        # Initialize Gemini client (LangChain)
        if gemini_client is None:
            config = GeminiConfig(api_key=settings.gemini_api_key)
            self.gemini_client = GeminiClient(config)
        else:
            self.gemini_client = gemini_client
        
        # Initialize HuggingFace client
        if hf_client is None:
            config = HFConfig(api_key=settings.hf_token)
            self.hf_client = HuggingFaceClient(config)
        else:
            self.hf_client = hf_client
        
        self._interpretation_count = 0
        logger.info("MultiLLMInterpreter initialized with 3 models")
    
    def interpret_composite_risk(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None
    ) -> MultiLLMInterpretation:
        """
        Interpret overall health risk using all 3 LLMs.
        
        Args:
            system_results: Risk results for all systems
            composite_risk: Overall composite risk score
            trust_envelope: Optional trust envelope
            
        Returns:
            MultiLLMInterpretation with combined responses
        """
        result = MultiLLMInterpretation()
        
        # Build comprehensive prompt
        prompt = self._build_composite_prompt(system_results, composite_risk, trust_envelope)
        
        total_start = __import__('time').time()
        
        # Query all 3 LLMs
        logger.info("Querying 3 LLMs for health screening interpretation...")
        
        # 1. Gemini via LangChain
        try:
            gemini_response = self.gemini_client.generate(prompt, self.SYSTEM_INSTRUCTION)
            result.gemini_response = gemini_response.text
            logger.info(f"Gemini response: {len(gemini_response.text)} chars, {gemini_response.latency_ms:.0f}ms")
        except Exception as e:
            logger.error(f"Gemini interpretation failed: {e}")
            result.gemini_response = "[Gemini unavailable]"
        
        # 2. GPT-OSS-120B (Medical Model 1)
        try:
            model1_response = self.hf_client.generate(
                prompt,
                model=settings.medical_model_1,
                system_prompt=self.SYSTEM_INSTRUCTION
            )
            result.medical_model_1_response = model1_response.text
            logger.info(f"Medical Model 1 response: {len(model1_response.text)} chars, {model1_response.latency_ms:.0f}ms")
        except Exception as e:
            logger.error(f"Medical Model 1 interpretation failed: {e}")
            result.medical_model_1_response = "[Medical Model 1 unavailable]"
        
        # 3. II-Medical-8B (Medical Model 2)
        try:
            model2_response = self.hf_client.generate(
                prompt,
                model=settings.medical_model_2,
                system_prompt=self.SYSTEM_INSTRUCTION
            )
            result.medical_model_2_response = model2_response.text
            logger.info(f"Medical Model 2 response: {len(model2_response.text)} chars, {model2_response.latency_ms:.0f}ms")
        except Exception as e:
            logger.error(f"Medical Model 2 interpretation failed: {e}")
            result.medical_model_2_response = "[Medical Model 2 unavailable]"
        
        result.total_latency_ms = (__import__('time').time() - total_start) * 1000
        
        # Synthesize responses
        self._synthesize_responses(result, trust_envelope)
        
        self._interpretation_count += 1
        logger.info(f"Multi-LLM interpretation complete: {result.total_latency_ms:.0f}ms total")
        
        return result
    
    def _build_composite_prompt(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope]
    ) -> str:
        """Build comprehensive prompt for all LLMs."""
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
        
        # Add trust context
        if trust_envelope:
            prompt += f"\nDATA RELIABILITY: {trust_envelope.overall_reliability:.0%}\n"
            if trust_envelope.critical_issues:
                prompt += f"CRITICAL ISSUES: {len(trust_envelope.critical_issues)}\n"
        
        prompt += """
Please provide a comprehensive interpretation including:
1. A clear SUMMARY (3-4 sentences) of the overall health status
2. KEY FINDINGS across all body systems
3. RECOMMENDATIONS for next steps (general health guidance)
4. IMPORTANT CAVEATS about screening limitations

Use simple, patient-friendly language. Always recommend professional medical consultation."""
        
        return prompt
    
    def _synthesize_responses(
        self,
        result: MultiLLMInterpretation,
        trust_envelope: Optional[TrustEnvelope]
    ):
        """Synthesize multiple LLM responses into final interpretation."""
        # Combine summaries (prioritize Gemini as primary, then medical models)
        summaries = []
        if result.gemini_response and len(result.gemini_response) > 50:
            summaries.append(self._extract_summary(result.gemini_response))
        if result.medical_model_1_response and len(result.medical_model_1_response) > 50:
            summaries.append(self._extract_summary(result.medical_model_1_response))
        if result.medical_model_2_response and len(result.medical_model_2_response) > 50:
            summaries.append(self._extract_summary(result.medical_model_2_response))
        
        # Use Gemini as primary summary, enrich with medical model insights
        if summaries:
            result.summary = summaries[0]  # Primary from Gemini
            if len(summaries) > 1:
                result.detailed_explanation = "\n\n".join(summaries)
            else:
                result.detailed_explanation = summaries[0]
        else:
            result.summary = "Health screening analysis completed. Consult healthcare provider for interpretation."
            result.detailed_explanation = result.summary
        
        # Combine recommendations from all models
        all_recommendations = []
        all_recommendations.extend(self._extract_recommendations(result.gemini_response))
        all_recommendations.extend(self._extract_recommendations(result.medical_model_1_response))
        all_recommendations.extend(self._extract_recommendations(result.medical_model_2_response))
        
        # Deduplicate and prioritize
        seen = set()
        unique_recs = []
        for rec in all_recommendations:
            rec_lower = rec.lower()
            if rec_lower not in seen:
                seen.add(rec_lower)
                unique_recs.append(rec)
        
        result.recommendations = unique_recs[:7] or [
            "Consult a healthcare professional for detailed evaluation.",
            "Maintain a healthy lifestyle with regular exercise.",
            "Schedule regular health checkups."
        ]
        
        # Standard caveats
        result.caveats = [
            "This is a screening report, not a medical diagnosis.",
            "Results should be reviewed by a qualified healthcare provider.",
            "Individual biomarkers may vary; professional interpretation recommended."
        ]
        
        # Add trust-based caveats
        if trust_envelope and not trust_envelope.is_reliable:
            result.caveats.insert(0, "Data quality issues detected - results should be verified.")
        
        # Determine consensus level
        response_count = sum([
            1 if result.gemini_response and len(result.gemini_response) > 50 else 0,
            1 if result.medical_model_1_response and len(result.medical_model_1_response) > 50 else 0,
            1 if result.medical_model_2_response and len(result.medical_model_2_response) > 50 else 0
        ])
        
        if response_count >= 3:
            result.consensus_level = "high"
        elif response_count >= 2:
            result.consensus_level = "medium"
        else:
            result.consensus_level = "low"
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary from LLM response."""
        # Look for summary section
        text_upper = text.upper()
        if "SUMMARY:" in text_upper:
            idx = text_upper.find("SUMMARY:")
            end_idx = text.find("\n\n", idx)
            if end_idx > idx:
                return text[idx + 8:end_idx].strip()
        
        # Fallback: use first 3 sentences
        sentences = text.split(". ")
        return ". ".join(sentences[:3]).strip() + ("." if not sentences[2].endswith(".") else "")
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from LLM response."""
        recommendations = []
        text_upper = text.upper()
        
        if "RECOMMENDATION" in text_upper:
            idx = text_upper.find("RECOMMENDATION")
            section = text[idx:idx + 1000]  # Next 1000 chars
            
            # Extract bullet points or numbered items
            lines = section.split("\n")
            for line in lines[1:]:  # Skip header
                clean = line.strip()
                if clean.startswith(("- ", "* ", "• ", "1.", "2.", "3.", "4.", "5.")):
                    # Remove bullet/number
                    if clean[0].isdigit():
                        clean = clean[2:].strip()
                    else:
                        clean = clean[2:].strip()
                    if clean and len(clean) > 10:
                        recommendations.append(clean)
        
        return recommendations[:5]  # Max 5 per model
    
    def get_stats(self) -> Dict[str, Any]:
        """Get interpreter statistics."""
        return {
            "interpretation_count": self._interpretation_count,
            "gemini_available": self.gemini_client.is_available,
            "hf_available": self.hf_client.is_available,
            "models_used": [
                "gemini-2.5-flash (LangChain)",
                settings.medical_model_1,
                settings.medical_model_2
            ]
        }

"""
Risk Interpreter Module

Interprets pre-computed risk scores using LLM.
LLM is NON-DECISIONAL - explains risks, does NOT assign or modify them.

Now uses MultiLLMInterpreter for comprehensive analysis with 3 models.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import json

from app.core.inference.risk_engine import SystemRiskResult, RiskScore, RiskLevel
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope, SafetyFlag
from app.core.llm.gemini_client import GeminiClient, GeminiConfig, GeminiResponse
from app.core.llm.multi_llm_interpreter import MultiLLMInterpreter, MultiLLMInterpretation
from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)


class InterpretationAudience(str, Enum):
    """Target audience for interpretation."""
    PATIENT = "patient"           # Simple, reassuring language
    CLINICAL = "clinical"         # Medical professional terminology
    TECHNICAL = "technical"       # Full technical details


@dataclass
class InterpretationResult:
    """Result of risk interpretation by LLM."""
    system: Optional[PhysiologicalSystem] = None
    audience: InterpretationAudience = InterpretationAudience.PATIENT
    
    # LLM-generated content
    summary: str = ""
    detailed_explanation: str = ""
    medical_context: str = ""
    recommendations: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    
    # Metadata
    confidence_statement: str = ""
    is_mock: bool = False
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system.value if self.system else None,
            "audience": self.audience.value,
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "medical_context": self.medical_context,
            "recommendations": self.recommendations,
            "caveats": self.caveats,
            "confidence_statement": self.confidence_statement,
            "is_mock": self.is_mock,
            "latency_ms": round(self.latency_ms, 2)
        }


class RiskInterpreter:
    """
    Interprets risk scores using multiple LLMs.
    
    NON-DECISIONAL: The LLMs explain pre-computed risks.
    Uses Gemini (LangChain) + 2 HuggingFace medical models for comprehensive analysis.
    """
    
    # System instruction for safe medical interpretation
    SYSTEM_INSTRUCTION = """You are a medical screening interpretation assistant. Your role is to EXPLAIN health screening results, NOT to diagnose or treat.

CRITICAL CONSTRAINTS:
1. You are explaining PRE-COMPUTED risk scores - do NOT assign new scores
2. Do NOT make diagnoses or suggest specific conditions
3. Do NOT recommend specific treatments or medications
4. Always recommend consulting a healthcare professional
5. Use appropriate uncertainty language based on confidence levels
6. Be clear that this is a screening tool, not a diagnostic instrument

Your output should:
- Explain what the biomarkers and risk scores mean
- Provide general educational context
- Use appropriate language for the target audience
- Include appropriate caveats about screening limitations
"""
    
    def __init__(self, client: Optional[GeminiClient] = None, multi_llm: bool = True):
        """
        Initialize risk interpreter.
        
        Args:
            client: Optional GeminiClient (for backward compatibility)
            multi_llm: Use multi-LLM interpreter (default True)
        """
        self.use_multi_llm = multi_llm
        
        if self.use_multi_llm:
            # Use new multi-LLM interpreter (3 models)
            self.multi_interpreter = MultiLLMInterpreter()
            logger.info("RiskInterpreter initialized with Multi-LLM mode (3 models)")
        else:
            # Legacy single-LLM mode
            if client is None:
                config = GeminiConfig(
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_model
                )
                self.client = GeminiClient(config)
            else:
                self.client = client
            logger.info("RiskInterpreter initialized with single Gemini client")
        
        self._interpretation_count = 0
    
    def interpret_system_risk(
        self,
        risk_result: SystemRiskResult,
        trust_envelope: Optional[TrustEnvelope] = None,
        audience: InterpretationAudience = InterpretationAudience.PATIENT
    ) -> InterpretationResult:
        """
        Interpret a single system's risk result.
        
        Args:
            risk_result: Pre-computed risk for one physiological system
            trust_envelope: Optional trust envelope for confidence context
            audience: Target audience for interpretation
            
        Returns:
            InterpretationResult with LLM-generated explanation
        """
        # Note: Single system interpretation uses legacy Gemini client
        # Multi-LLM is primarily for composite interpretations
        if not self.use_multi_llm and hasattr(self, 'client'):
            result = InterpretationResult(
                system=risk_result.system,
                audience=audience
            )
            
            # Build prompt
            prompt = self._build_system_prompt(risk_result, trust_envelope, audience)
            
            # Generate interpretation
            response = self.client.generate(prompt, self.SYSTEM_INSTRUCTION)
            
            # Parse response
            result = self._parse_response(response, result)
            result.latency_ms = response.latency_ms
            result.is_mock = response.is_mock
            
            # Add confidence statement based on trust envelope
            if trust_envelope:
                result.confidence_statement = self._generate_confidence_statement(trust_envelope)
                if trust_envelope.requires_caveats:
                    result.caveats.extend(self._get_trust_caveats(trust_envelope))
            
            self._interpretation_count += 1
            return result
        else:
            # Fallback: create basic interpretation
            result = InterpretationResult(
                system=risk_result.system,
                audience=audience
            )
            system_name = risk_result.system.value.replace("_", " ").title()
            result.summary = f"{system_name} screening completed with risk level: {risk_result.overall_risk.level.value}"
            result.recommendations = ["Consult a healthcare professional for detailed evaluation."]
            result.caveats = ["This is a screening result, not a diagnosis."]
            if trust_envelope:
                result.confidence_statement = self._generate_confidence_statement(trust_envelope)
            self._interpretation_count += 1
            return result
    
    def interpret_composite_risk(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None,
        audience: InterpretationAudience = InterpretationAudience.PATIENT
    ) -> InterpretationResult:
        """
        Interpret the overall composite health risk.
        
        Args:
            system_results: Risk results for all systems
            composite_risk: Overall composite risk score
            trust_envelope: Trust envelope for confidence
            audience: Target audience
            
        Returns:
            InterpretationResult for overall health
        """
        if self.use_multi_llm:
            # Use multi-LLM interpreter
            multi_result = self.multi_interpreter.interpret_composite_risk(
                system_results, composite_risk, trust_envelope
            )
            
            # Convert to InterpretationResult format
            result = InterpretationResult(
                system=None,
                audience=audience,
                summary=multi_result.summary,
                detailed_explanation=multi_result.detailed_explanation,
                recommendations=multi_result.recommendations,
                caveats=multi_result.caveats,
                latency_ms=multi_result.total_latency_ms,
                is_mock=False
            )
            
            # Add confidence statement
            if trust_envelope:
                result.confidence_statement = self._generate_confidence_statement(trust_envelope)
            
            self._interpretation_count += 1
            return result
        
        # Legacy single-LLM path
        result = InterpretationResult(
            system=None,  # Composite
            audience=audience
        )
        
        # Build composite prompt
        prompt = self._build_composite_prompt(
            system_results, composite_risk, trust_envelope, audience
        )
        
        # Generate interpretation
        response = self.client.generate(prompt, self.SYSTEM_INSTRUCTION)
        
        # Parse response
        result = self._parse_response(response, result)
        result.latency_ms = response.latency_ms
        result.is_mock = response.is_mock
        
        # Add confidence statement
        if trust_envelope:
            result.confidence_statement = self._generate_confidence_statement(trust_envelope)
            if trust_envelope.requires_caveats:
                result.caveats.extend(self._get_trust_caveats(trust_envelope))
        
        self._interpretation_count += 1
        return result
    
    def _build_system_prompt(
        self,
        risk_result: SystemRiskResult,
        trust_envelope: Optional[TrustEnvelope],
        audience: InterpretationAudience
    ) -> str:
        """Build prompt for single system interpretation."""
        system_name = risk_result.system.value.replace("_", " ").title()
        risk_level = risk_result.overall_risk.level.value.upper()
        risk_score = risk_result.overall_risk.score
        confidence = risk_result.overall_risk.confidence
        
        # Audience-specific instructions
        if audience == InterpretationAudience.PATIENT:
            tone = "Use simple, reassuring language. Avoid medical jargon."
        elif audience == InterpretationAudience.CLINICAL:
            tone = "Use appropriate medical terminology for healthcare professionals."
        else:
            tone = "Include full technical details and biomarker specifics."
        
        prompt = f"""Interpret the following health screening result for the {system_name} system.

TARGET AUDIENCE: {audience.value}
TONE INSTRUCTION: {tone}

PRE-COMPUTED RISK ASSESSMENT:
- System: {system_name}
- Risk Level: {risk_level}
- Risk Score: {risk_score:.1f}/100
- Confidence: {confidence:.0%}

BIOMARKER SUMMARY:
"""
        # Add biomarker details
        for name, info in risk_result.biomarker_summary.items():
            if isinstance(info, dict):
                value = info.get("value", "N/A")
                status = info.get("status", "unknown")
                prompt += f"- {name}: {value} ({status})\n"
        
        if risk_result.alerts:
            prompt += f"\nALERTS:\n"
            for alert in risk_result.alerts:
                prompt += f"- {alert}\n"
        
        # Add trust context
        if trust_envelope:
            prompt += f"\nDATA RELIABILITY: {trust_envelope.overall_reliability:.0%}"
            if not trust_envelope.is_reliable:
                prompt += " (LOW - add strong caveats)"
        
        prompt += """

Please provide:
1. A brief SUMMARY (2-3 sentences)
2. DETAILED EXPLANATION of what these results mean
3. MEDICAL CONTEXT (general educational information)
4. RECOMMENDATIONS (general health guidance, always include "consult healthcare professional")
5. CAVEATS (screening limitations, confidence considerations)

Remember: This is a screening result, NOT a diagnosis."""
        
        return prompt
    
    def _build_composite_prompt(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope],
        audience: InterpretationAudience
    ) -> str:
        """Build prompt for composite interpretation."""
        # Audience-specific instructions
        if audience == InterpretationAudience.PATIENT:
            tone = "Use simple, reassuring language suitable for patients."
        elif audience == InterpretationAudience.CLINICAL:
            tone = "Use medical terminology appropriate for healthcare professionals."
        else:
            tone = "Include comprehensive technical details."
        
        prompt = f"""Interpret the following comprehensive health screening results.

TARGET AUDIENCE: {audience.value}
TONE INSTRUCTION: {tone}

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
            if result.alerts:
                prompt += f"  - Alerts: {len(result.alerts)} items requiring attention\n"
        
        # Add trust context
        if trust_envelope:
            prompt += f"\nDATA RELIABILITY: {trust_envelope.overall_reliability:.0%}"
            if trust_envelope.critical_issues:
                prompt += f"\nCRITICAL ISSUES: {len(trust_envelope.critical_issues)}"
        
        prompt += """

Please provide:
1. SUMMARY: Overall health screening summary (3-4 sentences)
2. DETAILED EXPLANATION: What these results collectively indicate
3. KEY FINDINGS: Most important observations across systems
4. RECOMMENDATIONS: Prioritized general health guidance
5. CAVEATS: Important limitations and next steps

Remember: This is a screening tool. Always recommend professional medical consultation."""
        
        return prompt
    
    def _parse_response(
        self,
        response: GeminiResponse,
        result: InterpretationResult
    ) -> InterpretationResult:
        """Parse LLM response into structured result."""
        text = response.text
        
        # Simple section extraction
        sections = {
            "summary": "",
            "detailed": "",
            "context": "",
            "recommendations": [],
            "caveats": []
        }
        
        current_section = "summary"
        lines = text.split("\n")
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if "summary" in line_lower and (":" in line or line_lower.startswith("#")):
                current_section = "summary"
                continue
            elif "detailed" in line_lower or "explanation" in line_lower:
                current_section = "detailed"
                continue
            elif "context" in line_lower or "medical" in line_lower:
                current_section = "context"
                continue
            elif "recommendation" in line_lower:
                current_section = "recommendations"
                continue
            elif "caveat" in line_lower or "limitation" in line_lower:
                current_section = "caveats"
                continue
            elif "finding" in line_lower or "key" in line_lower:
                current_section = "detailed"
                continue
            
            # Clean line
            clean_line = line.strip()
            if not clean_line:
                continue
            
            # Remove bullet points for list items
            if clean_line.startswith(("- ", "* ", "• ")):
                clean_line = clean_line[2:].strip()
            elif clean_line.startswith(("1.", "2.", "3.", "4.", "5.")):
                clean_line = clean_line[2:].strip()
            
            # Add to appropriate section
            if current_section in ["summary", "detailed", "context"]:
                if sections[current_section]:
                    sections[current_section] += " " + clean_line
                else:
                    sections[current_section] = clean_line
            elif current_section == "recommendations" and clean_line:
                sections["recommendations"].append(clean_line)
            elif current_section == "caveats" and clean_line:
                sections["caveats"].append(clean_line)
        
        # If no structured sections found, use full text as summary
        if not sections["summary"] and not sections["detailed"]:
            sections["summary"] = text[:500] if len(text) > 500 else text
        
        result.summary = sections["summary"]
        result.detailed_explanation = sections["detailed"]
        result.medical_context = sections["context"]
        result.recommendations = sections["recommendations"] or ["Consult a healthcare professional for a complete evaluation."]
        result.caveats = sections["caveats"] or ["This is a screening result, not a diagnosis."]
        
        return result
    
    def _generate_confidence_statement(self, trust_envelope: TrustEnvelope) -> str:
        """Generate human-readable confidence statement."""
        reliability = trust_envelope.overall_reliability
        
        if reliability >= 0.9:
            return "High confidence: Data quality and validation checks passed."
        elif reliability >= 0.7:
            return "Moderate confidence: Results should be interpreted with some caution."
        elif reliability >= 0.5:
            return "Low confidence: Results have significant uncertainty. Verification recommended."
        else:
            return "Very low confidence: Data quality issues detected. Results may be unreliable."
    
    def _get_trust_caveats(self, trust_envelope: TrustEnvelope) -> List[str]:
        """Get caveats based on trust envelope issues."""
        caveats = []
        
        if SafetyFlag.DATA_QUALITY_ISSUE in trust_envelope.safety_flags:
            caveats.append("Some sensor data quality issues were detected.")
        
        if SafetyFlag.PHYSIOLOGICAL_ANOMALY in trust_envelope.safety_flags:
            caveats.append("Some biomarker values are at physiological extremes.")
        
        if SafetyFlag.INTERNAL_INCONSISTENCY in trust_envelope.safety_flags:
            caveats.append("Some inconsistencies between body systems were noted.")
        
        if trust_envelope.confidence_penalty > 0.2:
            caveats.append(f"Results have a {trust_envelope.confidence_penalty:.0%} confidence reduction due to data issues.")
        
        return caveats

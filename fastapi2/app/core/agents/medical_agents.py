"""
Medical Validation Agents Module

Specialized agents using medical LLMs for validation.
All agents are NON-DECISIONAL - they flag issues, not diagnose.

Agents:
- MedGemmaAgent: Biomarker plausibility validation
- OpenBioLLMAgent: Cross-system consistency checking
- AgentConsensus: Multi-agent agreement aggregation
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
from datetime import datetime

from app.core.inference.risk_engine import SystemRiskResult, RiskLevel
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope, SafetyFlag
from app.core.agents.hf_client import HuggingFaceClient, HFConfig, HFModel, HFResponse
from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)


class ValidationStatus(str, Enum):
    """Validation status from agents."""
    VALID = "valid"
    PLAUSIBLE = "plausible"
    UNCERTAIN = "uncertain"
    FLAGGED = "flagged"
    INVALID = "invalid"


class FlagSeverity(str, Enum):
    """Severity of validation flags."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationFlag:
    """A flag raised by a validation agent."""
    agent: str
    severity: FlagSeverity
    category: str
    message: str
    biomarker: Optional[str] = None
    system: Optional[PhysiologicalSystem] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "biomarker": self.biomarker,
            "system": self.system.value if self.system else None
        }


@dataclass
class ValidationResult:
    """Result from a medical validation agent."""
    agent_name: str
    model_used: str
    status: ValidationStatus = ValidationStatus.PLAUSIBLE
    confidence: float = 0.5
    flags: List[ValidationFlag] = field(default_factory=list)
    explanation: str = ""
    raw_response: str = ""
    is_mock: bool = False
    latency_ms: float = 0.0
    
    @property
    def has_critical_flags(self) -> bool:
        return any(f.severity == FlagSeverity.CRITICAL for f in self.flags)
    
    @property
    def has_warnings(self) -> bool:
        return any(f.severity == FlagSeverity.WARNING for f in self.flags)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "model_used": self.model_used,
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "flags": [f.to_dict() for f in self.flags],
            "explanation": self.explanation,
            "has_critical_flags": self.has_critical_flags,
            "is_mock": self.is_mock,
            "latency_ms": round(self.latency_ms, 2)
        }


class MedGemmaAgent:
    """
    Medical validation agent using MedGemma.
    
    Validates biomarker plausibility from a medical perspective.
    NON-DECISIONAL: Flags issues, does NOT diagnose.
    """
    
    SYSTEM_PROMPT = """You are a medical validation assistant using MedGemma. Your role is to validate health screening biomarkers for PLAUSIBILITY only.

CRITICAL CONSTRAINTS:
1. You are checking if biomarker VALUES are PLAUSIBLE, not diagnosing conditions
2. Do NOT make diagnoses or suggest specific diseases
3. Do NOT recommend treatments or medications
4. Flag values that are physiologically impossible or highly unlikely
5. Consider normal physiological ranges and common variations
6. Output must be structured with VALIDATION STATUS, FLAGS, and EXPLANATION

Your validation should check:
- Are values within physiologically possible ranges?
- Are there any obvious measurement errors?
- Are biomarker combinations physiologically coherent?

Output format:
VALIDATION STATUS: [VALID/PLAUSIBLE/UNCERTAIN/FLAGGED/INVALID]
CONFIDENCE: [HIGH/MODERATE/LOW]
FLAGS: [List any concerns]
EXPLANATION: [Brief explanation]"""
    
    def __init__(self, client: Optional[HuggingFaceClient] = None):
        """Initialize MedGemma agent."""
        if client is None:
            # Use settings for API configuration
            config = HFConfig(api_key=settings.hf_token)
            self.client = HuggingFaceClient(config)
        else:
            self.client = client
        # Use first medical model from settings
        self.model = settings.medical_model_1
        self._validation_count = 0
        logger.info(f"MedGemmaAgent initialized with model: {self.model}")
    
    def validate_biomarkers(
        self,
        biomarker_summary: Dict[str, Any],
        system: PhysiologicalSystem
    ) -> ValidationResult:
        """
        Validate biomarkers for physiological plausibility.
        
        Args:
            biomarker_summary: Dict of biomarker names to values/status
            system: The physiological system being validated
            
        Returns:
            ValidationResult with status and flags
        """
        # Build validation prompt
        prompt = self._build_prompt(biomarker_summary, system)
        
        # Get response from MedGemma
        response = self.client.generate(
            prompt,
            model=self.model,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Parse response into structured result
        result = self._parse_response(response, system)
        result.latency_ms = response.latency_ms
        result.is_mock = response.is_mock
        result.raw_response = response.text
        
        self._validation_count += 1
        return result
    
    def _build_prompt(
        self,
        biomarker_summary: Dict[str, Any],
        system: PhysiologicalSystem
    ) -> str:
        """Build validation prompt for MedGemma."""
        system_name = system.value.replace("_", " ").title()
        
        prompt = f"""Validate the following biomarker data from a {system_name} health screening for PLAUSIBILITY.

BIOMARKERS TO VALIDATE:
"""
        for name, info in biomarker_summary.items():
            if isinstance(info, dict):
                value = info.get("value", "N/A")
                unit = info.get("unit", "")
                status = info.get("status", "unknown")
                prompt += f"- {name}: {value} {unit} (status: {status})\n"
            else:
                prompt += f"- {name}: {info}\n"
        
        prompt += """
Check if these values are:
1. Within physiologically POSSIBLE ranges
2. Internally consistent with each other
3. Free from obvious measurement errors

Provide your validation assessment."""
        
        return prompt
    
    def _parse_response(
        self,
        response: HFResponse,
        system: PhysiologicalSystem
    ) -> ValidationResult:
        """Parse MedGemma response into ValidationResult."""
        result = ValidationResult(
            agent_name="MedGemma",
            model_used=self.model  # Now a string from config
        )
        
        text = response.text.upper()
        
        # Parse validation status
        if "INVALID" in text:
            result.status = ValidationStatus.INVALID
            result.confidence = 0.3
        elif "FLAGGED" in text:
            result.status = ValidationStatus.FLAGGED
            result.confidence = 0.5
        elif "UNCERTAIN" in text:
            result.status = ValidationStatus.UNCERTAIN
            result.confidence = 0.4
        elif "PLAUSIBLE" in text or "VALID" in text:
            result.status = ValidationStatus.PLAUSIBLE
            result.confidence = 0.8
        
        # Parse confidence
        if "HIGH" in text:
            result.confidence = max(result.confidence, 0.85)
        elif "LOW" in text:
            result.confidence = min(result.confidence, 0.4)
        
        # Extract flags from response
        result.flags = self._extract_flags(response.text, system)
        
        # Extract explanation
        if "EXPLANATION:" in response.text.upper():
            idx = response.text.upper().find("EXPLANATION:")
            result.explanation = response.text[idx + 12:].strip()[:500]
        else:
            result.explanation = response.text[:500]
        
        return result
    
    def _extract_flags(
        self,
        text: str,
        system: PhysiologicalSystem
    ) -> List[ValidationFlag]:
        """Extract validation flags from response text."""
        flags = []
        text_lower = text.lower()
        
        # Look for concerning keywords
        if "impossible" in text_lower or "implausible" in text_lower:
            flags.append(ValidationFlag(
                agent="MedGemma",
                severity=FlagSeverity.CRITICAL,
                category="plausibility",
                message="One or more biomarker values flagged as implausible",
                system=system
            ))
        
        if "inconsistent" in text_lower or "contradiction" in text_lower:
            flags.append(ValidationFlag(
                agent="MedGemma",
                severity=FlagSeverity.WARNING,
                category="consistency",
                message="Internal inconsistency detected in biomarker values",
                system=system
            ))
        
        if "error" in text_lower or "measurement" in text_lower:
            flags.append(ValidationFlag(
                agent="MedGemma",
                severity=FlagSeverity.WARNING,
                category="measurement",
                message="Possible measurement error flagged",
                system=system
            ))
        
        return flags


class OpenBioLLMAgent:
    """
    Biomedical validation agent using OpenBioLLM.
    
    Validates cross-system consistency and physiological coherence.
    NON-DECISIONAL: Flags issues, does NOT diagnose.
    """
    
    SYSTEM_PROMPT = """You are a biomedical validation assistant using OpenBioLLM. Your role is to check CONSISTENCY across health screening results.

CRITICAL CONSTRAINTS:
1. You are checking CONSISTENCY between body systems, not diagnosing
2. Do NOT make diagnoses or suggest specific diseases
3. Do NOT recommend treatments
4. Flag contradictory or incoherent patterns
5. Consider known physiological relationships between systems

Check for:
- Contradictions between different body systems
- Patterns that don't make physiological sense together
- Anomalies in overall health profile coherence

Output format:
CONSISTENCY STATUS: [CONSISTENT/PARTIALLY_CONSISTENT/INCONSISTENT]
CONFIDENCE: [HIGH/MODERATE/LOW]
INCONSISTENCIES: [List any found]
EXPLANATION: [Brief explanation]"""
    
    def __init__(self, client: Optional[HuggingFaceClient] = None):
        """Initialize OpenBioLLM agent."""
        if client is None:
            # Use settings for API configuration
            config = HFConfig(api_key=settings.hf_token)
            self.client = HuggingFaceClient(config)
        else:
            self.client = client
        # Use second medical model from settings
        self.model = settings.medical_model_2
        self._validation_count = 0
        logger.info(f"OpenBioLLMAgent initialized with model: {self.model}")
    
    def validate_consistency(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        trust_envelope: Optional[TrustEnvelope] = None
    ) -> ValidationResult:
        """
        Validate consistency across multiple body systems.
        
        Args:
            system_results: Risk results for each physiological system
            trust_envelope: Optional trust envelope for context
            
        Returns:
            ValidationResult with consistency assessment
        """
        # Build consistency prompt
        prompt = self._build_prompt(system_results, trust_envelope)
        
        # Get response from OpenBioLLM
        response = self.client.generate(
            prompt,
            model=self.model,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Parse response
        result = self._parse_response(response, system_results)
        result.latency_ms = response.latency_ms
        result.is_mock = response.is_mock
        result.raw_response = response.text
        
        self._validation_count += 1
        return result
    
    def _build_prompt(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        trust_envelope: Optional[TrustEnvelope]
    ) -> str:
        """Build consistency check prompt."""
        prompt = """Check the CONSISTENCY of the following multi-system health screening results.

SYSTEM-BY-SYSTEM RESULTS:
"""
        for system, result in system_results.items():
            system_name = system.value.replace("_", " ").title()
            risk_level = result.overall_risk.level.value.upper()
            prompt += f"\n{system_name}:\n"
            prompt += f"  - Risk Level: {risk_level}\n"
            prompt += f"  - Key Biomarkers: "
            
            biomarkers = list(result.biomarker_summary.keys())[:3]
            prompt += ", ".join(biomarkers) + "\n"
        
        if trust_envelope:
            prompt += f"\nOVERALL DATA QUALITY: {trust_envelope.overall_reliability:.0%}\n"
        
        prompt += """
Check if these results are:
1. Internally consistent across systems
2. Physiologically coherent
3. Free from contradictory patterns

Provide your consistency assessment."""
        
        return prompt
    
    def _parse_response(
        self,
        response: HFResponse,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult]
    ) -> ValidationResult:
        """Parse OpenBioLLM response into ValidationResult."""
        result = ValidationResult(
            agent_name="OpenBioLLM",
            model_used=self.model  # Now a string from config
        )
        
        text = response.text.upper()
        
        # Parse consistency status
        if "INCONSISTENT" in text and "PARTIALLY" not in text:
            result.status = ValidationStatus.FLAGGED
            result.confidence = 0.4
        elif "PARTIALLY" in text:
            result.status = ValidationStatus.UNCERTAIN
            result.confidence = 0.6
        elif "CONSISTENT" in text:
            result.status = ValidationStatus.VALID
            result.confidence = 0.8
        
        # Parse confidence
        if "HIGH" in text:
            result.confidence = max(result.confidence, 0.85)
        elif "LOW" in text:
            result.confidence = min(result.confidence, 0.4)
        
        # Extract flags
        result.flags = self._extract_flags(response.text, system_results)
        
        # Extract explanation
        if "EXPLANATION:" in response.text.upper():
            idx = response.text.upper().find("EXPLANATION:")
            result.explanation = response.text[idx + 12:].strip()[:500]
        else:
            result.explanation = response.text[:500]
        
        return result
    
    def _extract_flags(
        self,
        text: str,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult]
    ) -> List[ValidationFlag]:
        """Extract consistency flags from response."""
        flags = []
        text_lower = text.lower()
        
        if "contradiction" in text_lower or "contradictory" in text_lower:
            flags.append(ValidationFlag(
                agent="OpenBioLLM",
                severity=FlagSeverity.WARNING,
                category="cross_system",
                message="Cross-system contradiction detected"
            ))
        
        if "incoherent" in text_lower or "doesn't match" in text_lower:
            flags.append(ValidationFlag(
                agent="OpenBioLLM",
                severity=FlagSeverity.WARNING,
                category="coherence",
                message="Physiological incoherence flagged"
            ))
        
        if "anomaly" in text_lower or "unusual" in text_lower:
            flags.append(ValidationFlag(
                agent="OpenBioLLM",
                severity=FlagSeverity.INFO,
                category="pattern",
                message="Unusual pattern detected - may require review"
            ))
        
        return flags


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus."""
    overall_status: ValidationStatus
    overall_confidence: float
    agent_agreement: float  # 0-1, how much agents agree
    combined_flags: List[ValidationFlag]
    agent_results: Dict[str, ValidationResult]
    recommendation: str
    
    @property
    def requires_human_review(self) -> bool:
        """Check if results need human review."""
        return (
            self.overall_status in [ValidationStatus.FLAGGED, ValidationStatus.INVALID] or
            self.agent_agreement < 0.5 or
            any(f.severity == FlagSeverity.CRITICAL for f in self.combined_flags)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "overall_confidence": round(self.overall_confidence, 3),
            "agent_agreement": round(self.agent_agreement, 3),
            "combined_flags": [f.to_dict() for f in self.combined_flags],
            "agent_results": {k: v.to_dict() for k, v in self.agent_results.items()},
            "recommendation": self.recommendation,
            "requires_human_review": self.requires_human_review
        }


class AgentConsensus:
    """
    Aggregates results from multiple medical validation agents.
    
    Computes consensus status and combines flags.
    NON-DECISIONAL: Summarizes agent opinions, does not diagnose.
    """
    
    def __init__(self):
        """Initialize consensus aggregator."""
        self._consensus_count = 0
        logger.info("AgentConsensus initialized")
    
    def compute_consensus(
        self,
        agent_results: Dict[str, ValidationResult]
    ) -> ConsensusResult:
        """
        Compute consensus from multiple agent validation results.
        
        Args:
            agent_results: Dict of agent name to ValidationResult
            
        Returns:
            ConsensusResult with aggregated assessment
        """
        if not agent_results:
            return ConsensusResult(
                overall_status=ValidationStatus.UNCERTAIN,
                overall_confidence=0.0,
                agent_agreement=0.0,
                combined_flags=[],
                agent_results={},
                recommendation="No validation data available."
            )
        
        # Calculate agreement
        statuses = [r.status for r in agent_results.values()]
        status_counts = {}
        for s in statuses:
            status_counts[s] = status_counts.get(s, 0) + 1
        
        most_common = max(status_counts.values())
        agreement = most_common / len(statuses)
        
        # Determine overall status
        overall_status = self._determine_overall_status(agent_results)
        
        # Calculate overall confidence
        confidences = [r.confidence for r in agent_results.values()]
        overall_confidence = sum(confidences) / len(confidences)
        
        # Combine all flags
        combined_flags = []
        for result in agent_results.values():
            combined_flags.extend(result.flags)
        
        # Remove duplicates based on message
        seen_messages = set()
        unique_flags = []
        for flag in combined_flags:
            if flag.message not in seen_messages:
                seen_messages.add(flag.message)
                unique_flags.append(flag)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_status, agreement, unique_flags
        )
        
        self._consensus_count += 1
        
        return ConsensusResult(
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            agent_agreement=agreement,
            combined_flags=unique_flags,
            agent_results=agent_results,
            recommendation=recommendation
        )
    
    def _determine_overall_status(
        self,
        agent_results: Dict[str, ValidationResult]
    ) -> ValidationStatus:
        """Determine overall status from agent results."""
        # Check for critical flags first
        for result in agent_results.values():
            if result.has_critical_flags:
                return ValidationStatus.FLAGGED
            if result.status == ValidationStatus.INVALID:
                return ValidationStatus.INVALID
        
        # Check for warnings
        has_warnings = any(r.has_warnings for r in agent_results.values())
        has_uncertain = any(
            r.status == ValidationStatus.UNCERTAIN 
            for r in agent_results.values()
        )
        
        if has_warnings:
            return ValidationStatus.UNCERTAIN
        
        if has_uncertain:
            return ValidationStatus.UNCERTAIN
        
        # All agents agree it's valid/plausible
        all_valid = all(
            r.status in [ValidationStatus.VALID, ValidationStatus.PLAUSIBLE]
            for r in agent_results.values()
        )
        
        if all_valid:
            return ValidationStatus.VALID
        
        return ValidationStatus.PLAUSIBLE
    
    def _generate_recommendation(
        self,
        status: ValidationStatus,
        agreement: float,
        flags: List[ValidationFlag]
    ) -> str:
        """Generate human-readable recommendation."""
        if status == ValidationStatus.VALID and agreement > 0.8:
            return (
                "Validation passed. All agents agree the health screening data "
                "is plausible and internally consistent. Results may be used "
                "for preliminary assessment. Professional medical review recommended."
            )
        elif status == ValidationStatus.PLAUSIBLE:
            return (
                "Validation indicates plausible data with minor uncertainties. "
                "Results can be used with appropriate caveats. "
                "Consult a healthcare professional for interpretation."
            )
        elif status == ValidationStatus.UNCERTAIN:
            return (
                "Validation shows some uncertainty. Results should be interpreted "
                "with caution. Professional review is recommended before any "
                "health decisions."
            )
        elif status == ValidationStatus.FLAGGED:
            critical_count = sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL)
            return (
                f"Validation flagged {critical_count} critical issue(s). "
                "Results require professional medical review. "
                "Do not use for clinical decisions without further validation."
            )
        else:  # INVALID
            return (
                "Validation detected significant data issues. "
                "Results should NOT be used for any health assessment. "
                "Data re-collection or professional review required."
            )

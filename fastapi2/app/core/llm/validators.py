"""
LLM Response Validators

Validates LLM outputs for structural correctness and semantic consistency
with the pre-computed risk scores.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from app.core.inference.risk_engine import RiskLevel
from app.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of LLM response validation."""
    is_valid: bool
    structural_ok: bool
    semantic_ok: bool
    errors: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


class LLMValidator:
    """
    Validates LLM-generated interpretations.
    
    Checks:
    1. Structural: Does the response contain required sections?
    2. Semantic: Does the language match the risk level?
    """
    
    # Keywords that indicate urgency/concern (expected for HIGH/CRITICAL risk)
    URGENT_KEYWORDS = [
        "urgent", "immediately", "critical", "serious", "significant concern",
        "requires attention", "elevated", "high risk", "consult", "seek medical"
    ]
    
    # Keywords that indicate reassurance (expected for LOW risk)
    REASSURING_KEYWORDS = [
        "normal", "healthy", "within range", "no concern", "reassuring",
        "low risk", "stable", "good", "optimal"
    ]
    
    # Negation words that invert keyword meaning
    NEGATION_WORDS = ["no", "not", "without", "lack of", "absence of", "isn't", "aren't", "don't", "doesn't"]
    
    # Required sections in the response
    REQUIRED_SECTIONS = ["summary", "recommendation"]
    
    def validate(
        self,
        text: str,
        risk_level: RiskLevel,
        risk_score: float,
        alerts: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate an LLM response against the computed risk.
        
        Args:
            text: The LLM-generated text
            risk_level: Pre-computed risk level
            risk_score: Pre-computed risk score (0-100)
            alerts: List of alerts that should be mentioned
            
        Returns:
            ValidationResult with validation status and errors
        """
        errors = []
        
        # 1. Structural validation
        structural_ok = self._validate_structure(text, errors)
        
        # 2. Semantic validation
        semantic_ok = self._validate_semantics(text, risk_level, risk_score, errors)
        
        # 3. Score Fidelity
        score_ok = self._validate_score_fidelity(text, risk_score, errors)
        
        # 3. Alert coverage (optional but logged)
        if alerts:
            self._check_alert_coverage(text, alerts, errors)
        
        is_valid = structural_ok and semantic_ok and score_ok
        
        if not is_valid:
            logger.warning(f"LLM validation failed: {errors}")
        
        return ValidationResult(
            is_valid=is_valid,
            structural_ok=structural_ok,
            semantic_ok=semantic_ok,
            errors=errors
        )
    
    def _validate_score_fidelity(self, text: str, expected_score: float, errors: List[str]) -> bool:
        """Ensure LLM doesn't contradict math score."""
        # Stricter regex: only match explicit "risk score" or "composite score" mentions
        # Avoids false positives from "age: 45", "100% confidence", etc.
        import re
        
        # Pattern 1: "risk score: X", "composite score of X", "score: X/100"
        strict_patterns = [
            r'(?:risk|composite|overall)\s*score[^\d]*(\d+(?:\.\d+)?)',
            r'score\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*/\s*100',
        ]
        
        score_mentions = []
        for pattern in strict_patterns:
            score_mentions.extend(re.findall(pattern, text.lower()))
        
        for mention in score_mentions:
            try:
                val = float(mention)
                # Only check reasonable range (0-100)
                if 0 <= val <= 100:
                    # Tolerance: allow ±5 for rounding
                    if abs(val - expected_score) > 5.0:
                        errors.append(f"Score fidelity error: Text mentions {val} but calculated score is {expected_score:.1f}")
                        return False
            except ValueError:
                continue
                
        return True
    
    def _validate_structure(self, text: str, errors: List[str]) -> bool:
        """Check if required sections are present."""
        text_lower = text.lower()
        
        missing = []
        for section in self.REQUIRED_SECTIONS:
            # Check for section header or content indicators
            if section not in text_lower:
                missing.append(section)
        
        if missing:
            errors.append(f"Missing sections: {', '.join(missing)}")
            return False
        
        # Check minimum length (avoid empty/truncated responses)
        if len(text) < 100:
            errors.append("Response too short (< 100 chars)")
            return False
        
        return True
    
    def _count_keywords_with_negation(self, text: str, keywords: list) -> int:
        """
        Count keyword occurrences, excluding those preceded by negation words.
        
        Example: "no urgent care needed" won't count "urgent" as positive.
        """
        count = 0
        text_lower = text.lower()
        
        for kw in keywords:
            # Find all occurrences of the keyword
            start = 0
            while True:
                idx = text_lower.find(kw, start)
                if idx == -1:
                    break
                
                # Check if preceded by negation (within 20 chars before)
                prefix = text_lower[max(0, idx - 20):idx]
                is_negated = any(neg in prefix for neg in self.NEGATION_WORDS)
                
                if not is_negated:
                    count += 1
                
                start = idx + len(kw)
        
        return count
    
    def _validate_semantics(
        self,
        text: str,
        risk_level: RiskLevel,
        risk_score: float,
        errors: List[str]
    ) -> bool:
        """Check if language matches risk level (negation-aware)."""
        text_lower = text.lower()
        
        # Count keyword matches with negation awareness
        urgent_count = self._count_keywords_with_negation(text_lower, self.URGENT_KEYWORDS)
        reassuring_count = self._count_keywords_with_negation(text_lower, self.REASSURING_KEYWORDS)
        
        # High/Critical risk should have urgent language
        if risk_level in [RiskLevel.HIGH, RiskLevel.ACTION_REQUIRED]:
            if urgent_count == 0:
                errors.append(
                    f"HIGH/CRITICAL risk ({risk_score:.0f}) but no urgent language found"
                )
                return False
            if reassuring_count > urgent_count:
                errors.append(
                    f"HIGH/CRITICAL risk but more reassuring ({reassuring_count}) "
                    f"than urgent ({urgent_count}) language"
                )
                return False
        
        # Low risk should have reassuring language
        elif risk_level == RiskLevel.LOW:
            if reassuring_count == 0 and urgent_count > 2:
                errors.append(
                    f"LOW risk ({risk_score:.0f}) but excessive urgent language"
                )
                return False
        
        return True
    
    def _check_alert_coverage(
        self,
        text: str,
        alerts: List[str],
        errors: List[str]
    ) -> None:
        """Check if critical alerts are mentioned (warning only, not a failure)."""
        text_lower = text.lower()
        
        for alert in alerts[:3]:  # Check top 3 alerts
            # Extract key terms from alert
            alert_terms = re.findall(r'\b\w{4,}\b', alert.lower())
            
            # Check if any key term appears in text
            found = any(term in text_lower for term in alert_terms)
            
            if not found:
                logger.debug(f"Alert not explicitly covered: {alert[:50]}...")


def generate_correction_prompt(
    original_prompt: str,
    original_response: str,
    validation_result: ValidationResult,
    risk_level: RiskLevel,
    risk_score: float
) -> str:
    """
    Generate a correction prompt for the LLM to fix its response.
    
    Args:
        original_prompt: The original prompt sent to the LLM
        original_response: The LLM's original response
        validation_result: The validation result with errors
        risk_level: The pre-computed risk level
        risk_score: The pre-computed risk score
        
    Returns:
        A correction prompt for the LLM
    """
    error_list = "\n".join(f"- {e}" for e in validation_result.errors)
    
    return f"""Your previous response had issues that need correction.

ERRORS FOUND:
{error_list}

REMINDER:
- The pre-computed risk level is: {risk_level.value.upper()}
- The pre-computed risk score is: {risk_score:.1f}/100
- Your language and tone MUST match this risk level.
- You must include a SUMMARY and RECOMMENDATIONS section.

Please regenerate your response, fixing these issues:

{original_prompt}"""

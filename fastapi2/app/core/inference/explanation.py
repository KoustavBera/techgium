"""
Explanation Generator Module

Generates human-readable explanations for risk assessments.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem, Biomarker
from app.core.inference.risk_engine import (
    SystemRiskResult, RiskScore, RiskLevel, CompositeRiskCalculator,
    TrustedRiskResult
)
from app.utils import get_logger

logger = get_logger(__name__)


class ExplanationType(str, Enum):
    """Types of explanations."""
    SUMMARY = "summary"           # Brief overview
    DETAILED = "detailed"         # Full analysis
    CLINICAL = "clinical"         # Medical professional format
    PATIENT = "patient_friendly"  # Lay person format


@dataclass
class RiskExplanation:
    """Structured explanation for a risk assessment."""
    system: PhysiologicalSystem
    risk_level: RiskLevel
    summary: str
    detailed_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_statement: str = ""
    caveats: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "system": self.system.value,
            "risk_level": self.risk_level.value,
            "summary": self.summary,
            "detailed_findings": self.detailed_findings,
            "recommendations": self.recommendations,
            "confidence_statement": self.confidence_statement,
            "caveats": self.caveats,
        }
    
    def to_text(self, format_type: ExplanationType = ExplanationType.SUMMARY) -> str:
        """Convert to formatted text."""
        if format_type == ExplanationType.SUMMARY:
            return self.summary
        elif format_type == ExplanationType.PATIENT:
            return self._format_patient_friendly()
        elif format_type == ExplanationType.CLINICAL:
            return self._format_clinical()
        else:
            return self._format_detailed()
    
    def _format_patient_friendly(self) -> str:
        """Format for lay person understanding."""
        lines = [
            f"**{self.system.value.replace('_', ' ').title()} Assessment**",
            "",
            f"Overall Status: {self.risk_level.value.upper()}",
            "",
            self.summary,
            ""
        ]
        
        if self.recommendations:
            lines.append("What you can do:")
            for rec in self.recommendations:
                lines.append(f"• {rec}")
        
        return "\n".join(lines)
    
    def _format_clinical(self) -> str:
        """Format for medical professionals."""
        lines = [
            f"SYSTEM: {self.system.value}",
            f"RISK LEVEL: {self.risk_level.value}",
            f"CONFIDENCE: {self.confidence_statement}",
            "",
            "FINDINGS:",
        ]
        
        for finding in self.detailed_findings:
            lines.append(f"  - {finding}")
        
        if self.caveats:
            lines.append("")
            lines.append("CAVEATS:")
            for caveat in self.caveats:
                lines.append(f"  - {caveat}")
        
        return "\n".join(lines)
    
    def _format_detailed(self) -> str:
        """Full detailed format."""
        lines = [
            f"=== {self.system.value.replace('_', ' ').title()} Risk Assessment ===",
            "",
            f"Risk Level: {self.risk_level.value}",
            "",
            "Summary:",
            self.summary,
            "",
            "Detailed Findings:"
        ]
        
        for finding in self.detailed_findings:
            lines.append(f"  • {finding}")
        
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  → {rec}")
        
        lines.append("")
        lines.append(f"Confidence: {self.confidence_statement}")
        
        if self.caveats:
            lines.append("")
            lines.append("Important Notes:")
            for caveat in self.caveats:
                lines.append(f"  ⚠ {caveat}")
        
        return "\n".join(lines)


class ExplanationGenerator:
    """
    Generates human-readable explanations for risk assessments.
    
    Creates structured explanations with findings, recommendations,
    and appropriate caveats.
    """
    
    def __init__(self):
        """Initialize explanation generator."""
        self._generation_count = 0
        self._recommendation_templates = self._load_recommendation_templates()
        logger.info("ExplanationGenerator initialized")
    
    def _load_recommendation_templates(self) -> Dict[PhysiologicalSystem, Dict[RiskLevel, List[str]]]:
        """Load recommendation templates by system and risk level."""
        return {
            PhysiologicalSystem.CNS: {
                RiskLevel.LOW: [
                    "Maintain regular physical activity",
                    "Continue current lifestyle habits"
                ],
                RiskLevel.MODERATE: [
                    "Consider neurological check-up if symptoms persist",
                    "Monitor for changes in balance or coordination"
                ],
                RiskLevel.HIGH: [
                    "Recommend neurological evaluation",
                    "Document any tremor or gait changes"
                ],
                RiskLevel.CRITICAL: [
                    "Urgent neurological consultation recommended",
                    "Avoid activities requiring fine motor control until evaluated"
                ]
            },
            PhysiologicalSystem.CARDIOVASCULAR: {
                RiskLevel.LOW: [
                    "Continue heart-healthy lifestyle",
                    "Maintain regular exercise routine"
                ],
                RiskLevel.MODERATE: [
                    "Consider cardiovascular screening",
                    "Monitor blood pressure regularly"
                ],
                RiskLevel.HIGH: [
                    "Schedule cardiovascular evaluation",
                    "Review diet and exercise habits"
                ],
                RiskLevel.CRITICAL: [
                    "Seek immediate cardiovascular assessment",
                    "Monitor for chest pain or shortness of breath"
                ]
            },
            PhysiologicalSystem.RENAL: {
                RiskLevel.LOW: ["Stay well hydrated", "Maintain balanced diet"],
                RiskLevel.MODERATE: ["Monitor fluid intake", "Consider renal function testing"],
                RiskLevel.HIGH: ["Schedule kidney function tests", "Review medications"],
                RiskLevel.CRITICAL: ["Urgent nephrology consultation", "Monitor for swelling"]
            },
            PhysiologicalSystem.GASTROINTESTINAL: {
                RiskLevel.LOW: ["Maintain fiber-rich diet", "Stay hydrated"],
                RiskLevel.MODERATE: ["Monitor digestive symptoms", "Consider dietary adjustments"],
                RiskLevel.HIGH: ["Consult gastroenterologist", "Keep food diary"],
                RiskLevel.CRITICAL: ["Urgent GI evaluation needed", "Monitor for severe symptoms"]
            },
            PhysiologicalSystem.SKELETAL: {
                RiskLevel.LOW: ["Continue regular physical activity", "Maintain good posture"],
                RiskLevel.MODERATE: ["Consider physical therapy assessment", "Focus on balance exercises"],
                RiskLevel.HIGH: ["Orthopedic evaluation recommended", "Use assistive devices if needed"],
                RiskLevel.CRITICAL: ["Urgent orthopedic consultation", "Fall prevention measures"]
            },
            PhysiologicalSystem.SKIN: {
                RiskLevel.LOW: ["Maintain good skincare routine", "Use sun protection"],
                RiskLevel.MODERATE: ["Monitor skin changes", "Consider dermatology consult"],
                RiskLevel.HIGH: ["Schedule dermatological examination", "Document any lesions"],
                RiskLevel.CRITICAL: ["Urgent dermatology referral", "Biopsy may be needed"]
            },
            PhysiologicalSystem.EYES: {
                RiskLevel.LOW: ["Regular eye exams", "Reduce screen time"],
                RiskLevel.MODERATE: ["Consider ophthalmology visit", "Monitor vision changes"],
                RiskLevel.HIGH: ["Eye examination recommended", "Test for underlying conditions"],
                RiskLevel.CRITICAL: ["Urgent eye examination", "Watch for vision deterioration"]
            },
            PhysiologicalSystem.NASAL: {
                RiskLevel.LOW: ["Practice breathing exercises", "Maintain indoor air quality"],
                RiskLevel.MODERATE: ["Monitor breathing patterns", "Consider allergy testing"],
                RiskLevel.HIGH: ["Pulmonary function testing", "ENT evaluation"],
                RiskLevel.CRITICAL: ["Urgent respiratory assessment", "Monitor for breathing difficulty"]
            },
            PhysiologicalSystem.REPRODUCTIVE: {
                RiskLevel.LOW: ["Maintain healthy lifestyle", "Regular health screenings"],
                RiskLevel.MODERATE: ["Monitor stress levels", "Consider hormonal evaluation"],
                RiskLevel.HIGH: ["Endocrine evaluation recommended", "Lifestyle modifications"],
                RiskLevel.CRITICAL: ["Specialist consultation needed", "Comprehensive hormonal workup"]
            }
        }
    
    def generate_explanation(
        self,
        risk_result: SystemRiskResult,
        biomarker_set: Optional[BiomarkerSet] = None
    ) -> RiskExplanation:
        """
        Generate explanation for a system risk result.
        
        Args:
            risk_result: The risk assessment result
            biomarker_set: Optional biomarker data for detailed findings
            
        Returns:
            RiskExplanation with structured explanation
        """
        system = risk_result.system
        risk_level = risk_result.overall_risk.level
        confidence = risk_result.overall_risk.confidence
        
        # Generate summary
        summary = self._generate_summary(risk_result)
        
        # Generate detailed findings
        findings = self._generate_findings(risk_result, biomarker_set)
        
        # Get recommendations
        recommendations = self._get_recommendations(system, risk_level)
        
        # Generate confidence statement
        confidence_stmt = self._generate_confidence_statement(confidence)
        
        # Generate caveats
        caveats = self._generate_caveats(system, confidence)
        
        self._generation_count += 1
        
        return RiskExplanation(
            system=system,
            risk_level=risk_level,
            summary=summary,
            detailed_findings=findings,
            recommendations=recommendations,
            confidence_statement=confidence_stmt,
            caveats=caveats
        )
    
    def generate_trusted_explanation(
        self,
        trusted_result: TrustedRiskResult,
        biomarker_set: Optional[BiomarkerSet] = None
    ) -> RiskExplanation:
        """
        Generate explanation for a trusted/gated risk result.
        
        Handles rejected data and incorporates trust caveats.
        
        Args:
            trusted_result: The trusted risk result
            biomarker_set: Optional biomarker data
            
        Returns:
            RiskExplanation with trust context
        """
        # Handle rejected data
        if trusted_result.was_rejected:
            # Determine system from biomarker set if available, otherwise unknown
            system = biomarker_set.system if biomarker_set else PhysiologicalSystem.CNS
            
            return RiskExplanation(
                system=system,
                risk_level=RiskLevel.LOW,  # Default safe level
                summary=(
                    f"Assessment for {system.value.replace('_', ' ').title()} "
                    "could not be completed due to data quality issues."
                ),
                detailed_findings=[
                    f"Analysis Rejected: {trusted_result.rejection_reason}"
                ],
                recommendations=[
                    "Repeat assessment ensuring proper sensor placement",
                    "Check environmental conditions (lighting, stability)"
                ],
                confidence_statement="No Confidence - Assessment Rejected",
                caveats=trusted_result.caveats or ["Results invalid due to failed validation"]
            )
        
        # Handle valid data
        if trusted_result.risk_result is None:
            # Should not happen if was_rejected is False, but safe fallback
            return self.generate_trusted_explanation(
                TrustedRiskResult(was_rejected=True, rejection_reason="Internal error: Missing risk result")
            )
            
        # Generate base explanation
        explanation = self.generate_explanation(trusted_result.risk_result, biomarker_set)
        
        # Incorporate trust adjustments
        if trusted_result.caveats:
            explanation.caveats.extend(trusted_result.caveats)
        
        # Modify confidence statement if penalized
        if trusted_result.trust_adjusted_confidence < trusted_result.risk_result.overall_risk.confidence:
            explanation.confidence_statement += (
                f" (Penalized from {trusted_result.risk_result.overall_risk.confidence:.0%} "
                f"due to data quality)"
            )
            
        return explanation
    
    def generate_composite_explanation(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore
    ) -> str:
        """
        Generate overall health summary explanation.
        
        Args:
            system_results: All system risk results
            composite_risk: Composite risk score
            
        Returns:
            Formatted overall health explanation
        """
        lines = [
            "=== COMPREHENSIVE HEALTH SCREENING SUMMARY ===",
            "",
            f"Overall Health Risk: {composite_risk.level.value.upper()}",
            f"Composite Score: {composite_risk.score:.1f}/100",
            f"Confidence: {composite_risk.confidence:.0%}",
            "",
            "--- System-by-System Summary ---",
            ""
        ]
        
        # Sort by risk level (critical first)
        sorted_results = sorted(
            system_results.items(),
            key=lambda x: x[1].overall_risk.score,
            reverse=True
        )
        
        for system, result in sorted_results:
            level = result.overall_risk.level
            icon = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}
            lines.append(
                f"{icon.get(level.value, '⚪')} {system.value.replace('_', ' ').title()}: "
                f"{level.value} ({result.overall_risk.score:.0f}/100)"
            )
        
        # Key alerts
        all_alerts = []
        for result in system_results.values():
            all_alerts.extend(result.alerts)
        
        if all_alerts:
            lines.extend([
                "",
                "--- Key Alerts ---",
                ""
            ])
            for alert in all_alerts[:5]:
                lines.append(f"⚠ {alert}")
            if len(all_alerts) > 5:
                lines.append(f"   (+{len(all_alerts) - 5} additional alerts)")
        
        lines.extend([
            "",
            "--- Important Notice ---",
            "This is a preliminary screening assessment only.",
            "Results should be reviewed by qualified healthcare professionals.",
            "This is not a medical diagnosis."
        ])
        
        return "\n".join(lines)
    
    def _generate_summary(self, risk_result: SystemRiskResult) -> str:
        """Generate summary text for risk result."""
        system_name = risk_result.system.value.replace('_', ' ').title()
        level = risk_result.overall_risk.level
        score = risk_result.overall_risk.score
        
        if level == RiskLevel.LOW:
            summary = f"{system_name} indicators are within normal parameters. "
            summary += "No significant concerns identified from this assessment."
        elif level == RiskLevel.MODERATE:
            summary = f"{system_name} assessment shows some values requiring attention. "
            if risk_result.alerts:
                summary += f"Notable finding: {risk_result.alerts[0]}"
        elif level == RiskLevel.HIGH:
            summary = f"{system_name} assessment indicates elevated risk markers. "
            summary += "Professional evaluation is recommended."
        else:  # CRITICAL
            summary = f"{system_name} assessment shows concerning indicators. "
            summary += "Prompt professional evaluation is strongly recommended."
        
        return summary
    
    def _generate_findings(
        self,
        risk_result: SystemRiskResult,
        biomarker_set: Optional[BiomarkerSet]
    ) -> List[str]:
        """Generate detailed findings list."""
        findings = []
        
        # Add sub-risk findings
        for sub_risk in risk_result.sub_risks:
            if sub_risk.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                findings.append(sub_risk.explanation)
            elif sub_risk.level == RiskLevel.MODERATE:
                findings.append(f"{sub_risk.name}: borderline value noted")
        
        # Add from biomarker summary
        for name, info in risk_result.biomarker_summary.items():
            if info.get("is_abnormal") is True:
                value = info.get("value", "N/A")
                unit = info.get("unit", "")
                normal = info.get("normal_range", ("?", "?"))
                findings.append(
                    f"{name}: {value} {unit} (normal: {normal[0]}-{normal[1]})"
                )
        
        # Limit to top findings
        return findings[:8]
    
    def _get_recommendations(
        self,
        system: PhysiologicalSystem,
        level: RiskLevel
    ) -> List[str]:
        """Get recommendations for system and risk level."""
        templates = self._recommendation_templates.get(system, {})
        return templates.get(level, ["Consult healthcare provider for personalized advice"])
    
    def _generate_confidence_statement(self, confidence: float) -> str:
        """Generate statement about confidence level."""
        if confidence >= 0.8:
            return f"High confidence ({confidence:.0%}) - Based on comprehensive data"
        elif confidence >= 0.6:
            return f"Moderate confidence ({confidence:.0%}) - Some data limitations"
        elif confidence >= 0.4:
            return f"Limited confidence ({confidence:.0%}) - Interpret with caution"
        else:
            return f"Low confidence ({confidence:.0%}) - Preliminary assessment only"
    
    def _generate_caveats(
        self,
        system: PhysiologicalSystem,
        confidence: float
    ) -> List[str]:
        """Generate appropriate caveats."""
        caveats = [
            "This is a screening assessment, not a clinical diagnosis",
            "Results should be confirmed by qualified healthcare professionals"
        ]
        
        if confidence < 0.6:
            caveats.append("Limited data quality may affect accuracy")
        
        if system == PhysiologicalSystem.REPRODUCTIVE:
            caveats.append("Based on autonomic proxies only - direct assessment recommended")
        
        if system in [PhysiologicalSystem.RENAL, PhysiologicalSystem.GASTROINTESTINAL]:
            caveats.append("Non-invasive assessment has inherent limitations for this system")
        
        return caveats

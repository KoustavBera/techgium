"""
Doctor Report Generator

Generates detailed PDF reports for healthcare professionals.
Includes biomarker tables, trust envelope, validation results.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import io
import os

from app.core.inference.risk_engine import RiskLevel, SystemRiskResult, RiskScore
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope, SafetyFlag
from app.core.llm.risk_interpreter import InterpretationResult
from app.core.agents.medical_agents import ConsensusResult, ValidationStatus
from app.utils import get_logger

logger = get_logger(__name__)

# Import reportlab
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.colors import HexColor, black, white, lightgrey
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed - PDF generation unavailable")


# Risk level colors for tables
# Risk level colors for tables (Pastel)
RISK_BG_COLORS = {
    RiskLevel.LOW: HexColor("#D1FAE5") if REPORTLAB_AVAILABLE else "#D1FAE5",       # Mint Green
    RiskLevel.MODERATE: HexColor("#FEF3C7") if REPORTLAB_AVAILABLE else "#FEF3C7",  # Pale Amber
    RiskLevel.HIGH: HexColor("#FEE2E2") if REPORTLAB_AVAILABLE else "#FEE2E2",      # Pale Rose
    RiskLevel.CRITICAL: HexColor("#FEF2F2") if REPORTLAB_AVAILABLE else "#FEF2F2",  # Very Pale Red
    RiskLevel.UNKNOWN: HexColor("#F3F4F6") if REPORTLAB_AVAILABLE else "#F3F4F6",   # Light Gray
}

RISK_TEXT_COLORS = {
    RiskLevel.LOW: HexColor("#065F46") if REPORTLAB_AVAILABLE else "#065F46",
    RiskLevel.MODERATE: HexColor("#92400E") if REPORTLAB_AVAILABLE else "#92400E",
    RiskLevel.HIGH: HexColor("#B91C1C") if REPORTLAB_AVAILABLE else "#B91C1C",
    RiskLevel.CRITICAL: HexColor("#7F1D1D") if REPORTLAB_AVAILABLE else "#7F1D1D",
    RiskLevel.UNKNOWN: HexColor("#6B7280") if REPORTLAB_AVAILABLE else "#6B7280",   # Gray
}


@dataclass
class DoctorReport:
    """Data container for doctor/clinical report."""
    report_id: str
    generated_at: datetime
    
    # Patient info
    patient_id: str = "ANONYMOUS"
    
    # Overall assessment
    overall_risk_level: RiskLevel = RiskLevel.LOW
    overall_risk_score: float = 0.0
    overall_confidence: float = 0.0
    
    # Detailed system data
    system_details: Dict[PhysiologicalSystem, Dict[str, Any]] = field(default_factory=dict)
    
    # Trust envelope data
    trust_envelope_data: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    
    # All biomarkers
    all_biomarkers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Alerts
    all_alerts: List[str] = field(default_factory=list)
    
    # Output path
    pdf_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "patient_id": self.patient_id,
            "overall_risk_level": self.overall_risk_level.value,
            "overall_risk_score": round(self.overall_risk_score, 1),
            "overall_confidence": round(self.overall_confidence, 3),
            "system_count": len(self.system_details),
            "biomarker_count": len(self.all_biomarkers),
            "alert_count": len(self.all_alerts),
            "pdf_path": self.pdf_path
        }


class DoctorReportGenerator:
    """
    Generates detailed PDF reports for healthcare professionals.
    
    Features:
    - Comprehensive biomarker tables
    - Trust envelope visualization
    - Validation agent results
    - Statistical confidence intervals
    - Technical alerts and flags
    """
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize generator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if REPORTLAB_AVAILABLE:
            self._styles = getSampleStyleSheet()
            self._create_custom_styles()
        
        logger.info(f"DoctorReportGenerator initialized, output: {output_dir}")
    
    def _create_custom_styles(self):
        """Create custom paragraph styles with minimalist clinical design."""
        if not REPORTLAB_AVAILABLE:
            return
        
        # Check if styles already exist to avoid KeyError
        if 'ClinicalTitle' not in self._styles:
            # Clinical title
            self._styles.add(ParagraphStyle(
                name='ClinicalTitle',
                parent=self._styles['Title'],
                fontSize=20,
                spaceAfter=20,
                textColor=HexColor("#111827"),
                alignment=TA_LEFT,
                fontName='Helvetica-Bold'
            ))
        
        if 'ClinicalSection' not in self._styles:
            # Section header - minimalist, no background box
            self._styles.add(ParagraphStyle(
                name='ClinicalSection',
                parent=self._styles['Heading2'],
                fontSize=12,
                spaceBefore=15,
                spaceAfter=8,
                textColor=HexColor("#374151"),
                fontName='Helvetica-Bold',
                borderPadding=0,
                borderWidth=0,
                textTransform='uppercase' # All caps for section headers
            ))
        
        if 'Subsection' not in self._styles:
            # Subsection
            self._styles.add(ParagraphStyle(
                name='Subsection',
                parent=self._styles['Heading3'],
                fontSize=11,
                spaceBefore=10,
                spaceAfter=5,
                textColor=HexColor("#4B5563"),
                fontName='Helvetica-Bold'
            ))
        
        if 'ClinicalBody' not in self._styles:
            # Body text
            self._styles.add(ParagraphStyle(
                name='ClinicalBody',
                parent=self._styles['Normal'],
                fontSize=9,
                spaceAfter=6,
                leading=12,
                fontName='Helvetica'
            ))
        
        if 'SmallText' not in self._styles:
            # Small text
            self._styles.add(ParagraphStyle(
                name='SmallText',
                parent=self._styles['Normal'],
                fontSize=8,
                textColor=HexColor("#6B7280"),
                spaceAfter=4,
                fontName='Helvetica'
            ))
        
        if 'AlertText' not in self._styles:
            # Alert text
            self._styles.add(ParagraphStyle(
                name='AlertText',
                parent=self._styles['Normal'],
                fontSize=9,
                textColor=HexColor("#B91C1C"),
                spaceBefore=2,
                spaceAfter=2,
                fontName='Helvetica'
        ))
    
    def generate(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None,
        interpretation: Optional[InterpretationResult] = None,
        validation_result: Optional[ConsensusResult] = None,
        patient_id: str = "ANONYMOUS"
    ) -> DoctorReport:
        """
        Generate a detailed clinical PDF report.
        
        Args:
            system_results: Risk results for each system
            composite_risk: Overall composite risk
            trust_envelope: Data quality trust envelope
            interpretation: LLM interpretation results
            validation_result: Agentic validation consensus
            patient_id: Patient identifier
            
        Returns:
            DoctorReport with PDF path
        """
        # Create report data
        report_id = f"DR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        report = DoctorReport(
            report_id=report_id,
            generated_at=datetime.now(),
            patient_id=patient_id,
            overall_risk_level=composite_risk.level,
            overall_risk_score=composite_risk.score,
            overall_confidence=composite_risk.confidence
        )
        
        # Build detailed system data
        all_biomarkers = []
        all_alerts = []
        
        for system, result in system_results.items():
            report.system_details[system] = {
                "risk": result.overall_risk,
                "biomarker_summary": result.biomarker_summary,
                "alerts": result.alerts,
                "raw_scores": result.raw_scores if hasattr(result, 'raw_scores') else {}
            }
            
            # Collect all biomarkers
            for name, info in result.biomarker_summary.items():
                biomarker_entry = {
                    "system": system.value,
                    "name": name,
                }
                if isinstance(info, dict):
                    biomarker_entry.update(info)
                else:
                    biomarker_entry["value"] = info
                all_biomarkers.append(biomarker_entry)
            
            # Collect alerts
            all_alerts.extend([f"[{system.value}] {a}" for a in result.alerts])
        
        report.all_biomarkers = all_biomarkers
        report.all_alerts = all_alerts
        
        # Add trust envelope data
        if trust_envelope:
            report.trust_envelope_data = {
                "overall_reliability": trust_envelope.overall_reliability,
                "data_quality_score": trust_envelope.data_quality_score,
                "biomarker_plausibility": trust_envelope.biomarker_plausibility,
                "cross_system_consistency": trust_envelope.cross_system_consistency,
                "confidence_penalty": trust_envelope.confidence_penalty,
                "safety_flags": [f.value for f in trust_envelope.safety_flags],
                "is_reliable": trust_envelope.is_reliable
            }
        
        # Add validation summary
        if validation_result:
            report.validation_summary = {
                "overall_status": validation_result.overall_status.value,
                "agent_agreement": validation_result.agent_agreement,
                "overall_confidence": validation_result.overall_confidence,
                "flag_count": len(validation_result.combined_flags),
                "requires_review": validation_result.requires_human_review,
                "recommendation": validation_result.recommendation
            }
        
        # Generate PDF
        if REPORTLAB_AVAILABLE:
            pdf_path = self._generate_pdf(report, trust_envelope, validation_result)
            report.pdf_path = pdf_path
        else:
            logger.warning("PDF generation skipped - reportlab not available")
            report.pdf_path = None
        
        return report
    
    def _generate_pdf(
        self,
        report: DoctorReport,
        trust_envelope: Optional[TrustEnvelope],
        validation_result: Optional[ConsensusResult]
    ) -> str:
        """Generate the actual PDF file."""
        filename = f"{report.report_id}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        story = []
        
        # Header
        story.append(Paragraph(
            "Clinical Health Screening Report",
            self._styles['ClinicalTitle']
        ))
        
        # Metadata table
        meta_data = [
            ["Report ID", report.report_id, "Generated", report.generated_at.strftime('%Y-%m-%d %H:%M')],
            ["Patient ID", report.patient_id, "Report Type", "Comprehensive Clinical"]
        ]
        meta_table = Table(meta_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
        meta_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor("#6B7280")),
            ('TEXTCOLOR', (2, 0), (2, -1), HexColor("#6B7280")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", self._styles['ClinicalSection']))
        
        summary_data = [
            ["Metric", "Value", "Interpretation"],
            ["Overall Risk Score", f"{report.overall_risk_score:.1f}/100", report.overall_risk_level.value.upper()],
            ["Confidence Level", f"{report.overall_confidence:.1%}", self._confidence_interpretation(report.overall_confidence)],
            ["Systems Assessed", str(len(report.system_details)), ""],
            ["Biomarkers Measured", str(len(report.all_biomarkers)), ""],
            ["Active Alerts", str(len(report.all_alerts)), "Review recommended" if report.all_alerts else "None"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            # Minimalist header
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#374151")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('LINEBELOW', (0, 0), (-1, 0), 1, HexColor("#E5E7EB")),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            
            # Content
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, HexColor("#F9FAFB")),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        # Trust Envelope Section
        if trust_envelope:
            story.append(Paragraph("DATA QUALITY & TRUST ENVELOPE", self._styles['ClinicalSection']))
            
            trust_data = [
                ["Component", "Score", "Status"],
                ["Overall Reliability", f"{trust_envelope.overall_reliability:.1%}", 
                 "✓ Reliable" if trust_envelope.is_reliable else "⚠ Low Reliability"],
                ["Data Quality", f"{trust_envelope.data_quality_score:.1%}", 
                 self._score_status(trust_envelope.data_quality_score)],
                ["Biomarker Plausibility", f"{trust_envelope.biomarker_plausibility:.1%}",
                 self._score_status(trust_envelope.biomarker_plausibility)],
                ["Cross-System Consistency", f"{trust_envelope.cross_system_consistency:.1%}",
                 self._score_status(trust_envelope.cross_system_consistency)],
                ["Confidence Penalty", f"-{trust_envelope.confidence_penalty:.1%}",
                 "Applied" if trust_envelope.confidence_penalty > 0 else "None"]
            ]
            
            trust_table = Table(trust_data, colWidths=[2.5*inch, 1.5*inch, 2.5*inch])
            trust_table.setStyle(TableStyle([
                # Minimalist header
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#374151")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('LINEBELOW', (0, 0), (-1, 0), 1, HexColor("#E5E7EB")),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                
                # Content
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('LINEBELOW', (0, 1), (-1, -1), 0.5, HexColor("#F9FAFB")),
            ]))
            story.append(trust_table)
            
            if trust_envelope.safety_flags:
                story.append(Spacer(1, 5))
                flags_text = ", ".join([f.value for f in trust_envelope.safety_flags])
                story.append(Paragraph(f"Safety Flags: {flags_text}", self._styles['SmallText']))
            
            story.append(Spacer(1, 15))
        
        # Validation Results
        if validation_result:
            story.append(Paragraph("AGENTIC VALIDATION RESULTS", self._styles['ClinicalSection']))
            
            val_data = [
                ["Agent Consensus", "Value"],
                ["Overall Status", validation_result.overall_status.value.upper()],
                ["Agent Agreement", f"{validation_result.agent_agreement:.0%}"],
                ["Combined Confidence", f"{validation_result.overall_confidence:.1%}"],
                ["Flags Raised", str(len(validation_result.combined_flags))],
                ["Human Review Required", "Yes" if validation_result.requires_human_review else "No"]
            ]
            
            val_table = Table(val_data, colWidths=[2.5*inch, 4*inch])
            val_table.setStyle(TableStyle([
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#374151")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('LINEBELOW', (0, 0), (-1, 0), 1, HexColor("#E5E7EB")),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('LINEBELOW', (0, 1), (-1, -1), 0.5, HexColor("#F9FAFB")),
            ]))
            story.append(val_table)
            story.append(Spacer(1, 10))
            
            if validation_result.recommendation:
                story.append(Paragraph(
                    f"Recommendation: {validation_result.recommendation[:300]}",
                    self._styles['ClinicalBody']
                ))
            story.append(Spacer(1, 15))
        
        # System-by-System Details
        story.append(Paragraph("SYSTEM-BY-SYSTEM ANALYSIS", self._styles['ClinicalSection']))
        
        for system, details in report.system_details.items():
            system_name = system.value.replace("_", " ").title()
            risk = details["risk"]
            
            story.append(Paragraph(system_name, self._styles['Subsection']))
            
            # System summary row - Pastel background
            risk_color = RISK_BG_COLORS.get(risk.level, lightgrey)
            risk_text_color = RISK_TEXT_COLORS.get(risk.level, black)
            
            sys_data = [
                ["Risk Score", "Risk Level", "Confidence", "Alerts"],
                [f"{risk.score:.1f}/100", risk.level.value.upper(), 
                 f"{risk.confidence:.0%}", str(len(details.get("alerts", [])))]
            ]
            
            sys_table = Table(sys_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            sys_table.setStyle(TableStyle([
                # Header with pastel background
                ('BACKGROUND', (0, 0), (-1, 1), risk_color), # Specific background for this block
                ('TEXTCOLOR', (0, 0), (-1, -1), risk_text_color),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0, white), # No grid
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(sys_table)
            
            # Biomarkers for this system
            biomarkers = details.get("biomarker_summary", {})
            if biomarkers:
                bm_data = [["Biomarker", "Value", "Status"]]
                for name, info in list(biomarkers.items())[:8]:  # Limit to 8
                    if isinstance(info, dict):
                        value = str(info.get("value", "N/A"))
                        unit = info.get("unit", "")
                        status = info.get("status", "unknown")
                        bm_data.append([name, f"{value} {unit}".strip(), status])
                    else:
                        bm_data.append([name, str(info), ""])
                
                if len(bm_data) > 1:
                    bm_table = Table(bm_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
                    bm_table.setStyle(TableStyle([
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#374151")),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('LINEBELOW', (0, 0), (-1, 0), 0.5, HexColor("#E5E7EB")),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                        ('GRID', (0, 0), (-1, -1), 0, white),
                        ('LINEBELOW', (0, 1), (-1, -1), 0.25, HexColor("#F9FAFB")),
                    ]))
                    story.append(bm_table)
            
            story.append(Spacer(1, 10))
        
        # Alerts Section
        if report.all_alerts:
            story.append(Paragraph("ALERTS & FLAGS", self._styles['ClinicalSection']))
            for alert in report.all_alerts[:10]:  # Limit to 10
                story.append(Paragraph(f"⚠ {alert}", self._styles['AlertText']))
            story.append(Spacer(1, 15))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "This report is intended for healthcare professionals. Data should be interpreted "
            "in the context of clinical examination and patient history. This automated screening "
            "tool does not replace clinical judgment.",
            self._styles['SmallText']
        ))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Doctor report generated: {filepath}")
        
        return filepath
    
    def _confidence_interpretation(self, confidence: float) -> str:
        """Interpret confidence level."""
        if confidence >= 0.9:
            return "High - Results highly reliable"
        elif confidence >= 0.7:
            return "Moderate - Interpret with standard caution"
        elif confidence >= 0.5:
            return "Low - Additional verification recommended"
        else:
            return "Very Low - Results require confirmation"
    
    def _score_status(self, score: float) -> str:
        """Convert score to status text."""
        if score < 0:
            return "∅ No Data"
        if score >= 0.9:
            return "✓ Excellent"
        elif score >= 0.7:
            return "✓ Good"
        elif score >= 0.5:
            return "⚠ Fair"
        else:
            return "⚠ Poor"
    
    def generate_bytes(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None,
        validation_result: Optional[ConsensusResult] = None
    ) -> bytes:
        """Generate PDF and return as bytes for download."""
        if not REPORTLAB_AVAILABLE:
            return b""
        
        report = self.generate(
            system_results, composite_risk, trust_envelope,
            validation_result=validation_result
        )
        
        if report.pdf_path and os.path.exists(report.pdf_path):
            with open(report.pdf_path, 'rb') as f:
                return f.read()
        
        return b""

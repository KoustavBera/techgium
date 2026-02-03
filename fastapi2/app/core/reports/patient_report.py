"""
Enhanced Patient Report Generator

Generates detailed, informative PDF reports with:
- Individual biomarker breakdowns
- Color-coded status indicators
- Simple, patient-friendly explanations
- AI-generated insights
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import io
import os

from app.core.inference.risk_engine import RiskLevel, SystemRiskResult, RiskScore
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope
from app.core.llm.risk_interpreter import InterpretationResult
from app.core.agents.medical_agents import ConsensusResult
from app.core.llm.gemini_client import GeminiClient, GeminiConfig
from app.utils import get_logger

logger = get_logger(__name__)

# Import reportlab
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.colors import (
        HexColor, green, red, orange, yellow, 
        black, white, lightgrey, darkgrey
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, Flowable, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect, Circle, String
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed - PDF generation unavailable")


# Risk level colors
RISK_COLORS = {
    RiskLevel.LOW: HexColor("#22C55E") if REPORTLAB_AVAILABLE else "#22C55E",       # Green
    RiskLevel.MODERATE: HexColor("#F59E0B") if REPORTLAB_AVAILABLE else "#F59E0B",  # Amber
    RiskLevel.HIGH: HexColor("#EF4444") if REPORTLAB_AVAILABLE else "#EF4444",      # Red
    RiskLevel.CRITICAL: HexColor("#991B1B") if REPORTLAB_AVAILABLE else "#991B1B", # Dark Red
}

# Biomarker status colors (lighter backgrounds)
STATUS_COLORS = {
    "normal": HexColor("#ECFDF5") if REPORTLAB_AVAILABLE else "#ECFDF5",      # Light green
    "low": HexColor("#FEF3C7") if REPORTLAB_AVAILABLE else "#FEF3C7",         # Light amber
    "high": HexColor("#FEF3C7") if REPORTLAB_AVAILABLE else "#FEF3C7",        # Light amber
    "not_assessed": HexColor("#F9FAFB") if REPORTLAB_AVAILABLE else "#F9FAFB" # Light gray
}

RISK_LABELS = {
    RiskLevel.LOW: "Low Risk - Good Health",
    RiskLevel.MODERATE: "Moderate Risk - Attention Recommended",
    RiskLevel.HIGH: "High Risk - Consult Doctor Soon",
    RiskLevel.CRITICAL: "Critical - Seek Immediate Medical Care",
}

# Simplified biomarker names
BIOMARKER_NAMES = {
    "heart_rate": "Heart Rate",
    "hrv_rmssd": "Heart Rate Variability",
    "hrv": "Heart Rate Variability",
    "blood_pressure_systolic": "Blood Pressure (Systolic)",
    "blood_pressure_diastolic": "Blood Pressure (Diastolic)",
    "systolic_bp": "Blood Pressure (Systolic)",
    "diastolic_bp": "Blood Pressure (Diastolic)",
    "spo2": "Blood Oxygen Level",
    "respiratory_rate": "Breathing Rate",
    "breath_depth": "Breath Depth",
    "gait_variability": "Walking Pattern Stability",
    "balance_score": "Balance Score",
    "tremor": "Hand Steadiness",
    "reaction_time": "Reaction Time",
    "glucose": "Blood Sugar",
    "cholesterol": "Cholesterol Level",
}


@dataclass
class PatientReport:
    """Data container for patient report."""
    report_id: str
    generated_at: datetime
    
    # Patient info (anonymized)
    patient_id: str = "ANONYMOUS"
    
    # Overall assessment
    overall_risk_level: RiskLevel = RiskLevel.LOW
    overall_risk_score: float = 0.0
    overall_confidence: float = 0.0
    
    # System summaries
    system_summaries: Dict[PhysiologicalSystem, Dict[str, Any]] = field(default_factory=dict)
    
    # LLM interpretation
    interpretation_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Caveats
    caveats: List[str] = field(default_factory=list)
    
    # Output path
    pdf_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "patient_id": self.patient_id,
            "overall_risk_level": self.overall_risk_level.value,
            "overall_risk_score": round(self.overall_risk_score, 1),
            "overall_confidence": round(self.overall_confidence, 2),
            "system_count": len(self.system_summaries),
            "pdf_path": self.pdf_path
        }


# Define RiskIndicator only if reportlab available
if REPORTLAB_AVAILABLE:
    class RiskIndicator(Flowable):
        """Custom flowable for risk level visual indicator."""
        
        def __init__(self, risk_level: RiskLevel, width: float = 100, height: float = 40):
            Flowable.__init__(self)
            self.risk_level = risk_level
            self.width = width
            self.height = height
        
        def draw(self):
            color = RISK_COLORS.get(self.risk_level, lightgrey)
            label = RISK_LABELS.get(self.risk_level, "Unknown")
            
            # Draw rounded rectangle background
            self.canv.setFillColor(color)
            self.canv.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)
            
            # Draw text
            self.canv.setFillColor(white)
            self.canv.setFont("Helvetica-Bold", 13)
            text_width = self.canv.stringWidth(label, "Helvetica-Bold", 13)
            self.canv.drawString((self.width - text_width) / 2, self.height / 2.5, label)
else:
    # Dummy class when reportlab not available
    class RiskIndicator:
        def __init__(self, *args, **kwargs):
            pass


class EnhancedPatientReportGenerator:
    """
    Generates detailed, patient-friendly PDF health screening reports.
    
    New Features:
    - Individual biomarker breakdowns with values
    - Color-coded status indicators
    - Simple explanations for each measurement
    - AI-generated system-specific insights
    - Visual progress indicators
    """
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize generator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if REPORTLAB_AVAILABLE:
            self._styles = getSampleStyleSheet()
            self._create_custom_styles()
        
        # Initialize Gemini client for AI explanations
        try:
            self.gemini_client = GeminiClient()
            logger.info("GeminiClient initialized for AI explanations")
        except Exception as e:
            logger.warning(f"GeminiClient initialization failed: {e}. Will use fallback explanations.")
            self.gemini_client = None
        
        logger.info(f"EnhancedPatientReportGenerator initialized, output: {output_dir}")
    
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        if not REPORTLAB_AVAILABLE:
            return
        
        # Check if styles already exist to avoid KeyError
        if 'CustomTitle' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self._styles['Title'],
                fontSize=26,
                spaceAfter=25,
                textColor=HexColor("#1E40AF"),
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        if 'SectionHeader' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self._styles['Heading2'],
                fontSize=16,
                spaceBefore=25,
                spaceAfter=12,
                textColor=HexColor("#1F2937"),
                fontName='Helvetica-Bold'
            ))
        
        if 'SubHeader' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='SubHeader',
                parent=self._styles['Heading3'],
                fontSize=13,
                spaceBefore=15,
                spaceAfter=8,
                textColor=HexColor("#374151"),
                fontName='Helvetica-Bold'
            ))
        
        if 'BodyText' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='BodyText',
                parent=self._styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                leading=15,
                alignment=TA_JUSTIFY
            ))
        
        if 'BiomarkerExplanation' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='BiomarkerExplanation',
                parent=self._styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                leading=13,
                textColor=HexColor("#4B5563"),
                leftIndent=15
            ))
        
        if 'Caveat' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='Caveat',
                parent=self._styles['Normal'],
                fontSize=9,
                textColor=HexColor("#6B7280"),
                spaceBefore=5,
                spaceAfter=5
            ))
    
    def generate(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        interpretation: Optional[InterpretationResult] = None,
        trust_envelope: Optional[TrustEnvelope] = None,
        patient_id: str = "ANONYMOUS"
    ) -> PatientReport:
        """
        Generate an enhanced patient PDF report.
        
        Args:
            system_results: Risk results for each system
            composite_risk: Overall composite risk score
            interpretation: Optional LLM interpretation
            trust_envelope: Optional trust envelope
            patient_id: Patient identifier (anonymized)
            
        Returns:
            PatientReport with PDF path
        """
        # Create report data
        report_id = f"PR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        report = PatientReport(
            report_id=report_id,
            generated_at=datetime.now(),
            patient_id=patient_id,
            overall_risk_level=composite_risk.level,
            overall_risk_score=composite_risk.score,
            overall_confidence=composite_risk.confidence
        )
        
        # Build system summaries WITH biomarker details
        for system, result in system_results.items():
            report.system_summaries[system] = {
                "risk_level": result.overall_risk.level,
                "risk_score": result.overall_risk.score,
                "status": self._get_simple_status(result.overall_risk.level),
                "alerts": result.alerts,
                "biomarkers": result.biomarker_summary,  # Include full biomarker data
                "explanation": result.overall_risk.explanation
            }
        
        # Add interpretation
        if interpretation:
            report.interpretation_summary = interpretation.summary
            report.recommendations = interpretation.recommendations
            report.caveats = interpretation.caveats
        else:
            report.recommendations = self._generate_default_recommendations(system_results)
            report.caveats = [
                "This is a screening report, not a medical diagnosis.",
                "Results should be reviewed by a qualified healthcare provider.",
                "Individual results may vary based on age, gender, and other factors."
            ]
        
        # Generate PDF
        if REPORTLAB_AVAILABLE:
            pdf_path = self._generate_pdf(report, system_results, trust_envelope)
            report.pdf_path = pdf_path
        else:
            logger.warning("PDF generation skipped - reportlab not available")
            report.pdf_path = None
        
        return report
    
    def _generate_default_recommendations(self, system_results: Dict) -> List[str]:
        """Generate personalized recommendations based on findings."""
        recs = []
        
        # Check for cardiovascular issues
        if PhysiologicalSystem.CARDIOVASCULAR in system_results:
            cv_result = system_results[PhysiologicalSystem.CARDIOVASCULAR]
            if cv_result.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
                recs.append("Monitor your blood pressure regularly and reduce salt intake.")
                recs.append("Engage in 30 minutes of moderate exercise daily.")
        
        # Check for pulmonary issues
        if PhysiologicalSystem.PULMONARY in system_results:
            pulm_result = system_results[PhysiologicalSystem.PULMONARY]
            if pulm_result.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
                recs.append("Practice breathing exercises and avoid air pollutants.")
        
        # Check for CNS issues
        if PhysiologicalSystem.CNS in system_results:
            cns_result = system_results[PhysiologicalSystem.CNS]
            if cns_result.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
                recs.append("Work on balance exercises and ensure adequate sleep.")
        
        # General recommendations
        recs.extend([
            "Consult a healthcare professional for comprehensive evaluation.",
            "Maintain a balanced diet rich in fruits and vegetables.",
            "Stay hydrated with 8 glasses of water daily.",
            "Schedule regular health checkups."
        ])
        
        return recs[:6]  # Return top 6 recommendations
    
    def _get_simple_status(self, level: RiskLevel) -> str:
        """Convert risk level to simple patient-friendly status."""
        statuses = {
            RiskLevel.LOW: "✓ Good",
            RiskLevel.MODERATE: "⚠ Attention Recommended",
            RiskLevel.HIGH: "⚠ Consult Doctor",
            RiskLevel.CRITICAL: "🚨 Urgent Care Needed"
        }
        return statuses.get(level, "Unknown")
    
    def _simplify_biomarker_name(self, name: str) -> str:
        """Convert technical biomarker names to patient-friendly terms."""
        return BIOMARKER_NAMES.get(name, name.replace("_", " ").title())
    
    def _get_biomarker_status_icon(self, status: str) -> str:
        """Get icon for biomarker status."""
        if status == "normal":
            return "✓ Normal"
        elif status == "low":
            return "⚠ Below Normal"
        elif status == "high":
            return "⚠ Above Normal"
        else:
            return "— Not Assessed"
    
    def _format_normal_range(self, normal_range: Optional[tuple]) -> str:
        """Format normal range for display."""
        if not normal_range:
            return "—"
        low, high = normal_range
        return f"{low}-{high}"
    
    def _abbreviate_unit(self, unit: str) -> str:
        """Abbreviate long unit names to prevent table overlap."""
        abbreviations = {
            "power_spectral_density": "PSD",
            "coefficient_of_variation": "CV",
            "normalized_amplitude": "norm.",
            "normalized_units_per_frame": "units/frame",
            "breaths_per_min": "brpm",
            "blinks_per_min": "bpm",
            "saccades_per_sec": "sacc/s",
            "score_0_100": "score",
            "score_0_1": "score",
            "variance_score": "var",
            "normalized_intensity": "norm",
            "normalized": "norm",
        }
        return abbreviations.get(unit, unit)
    
    def _get_biomarker_explanation(self, biomarker_name: str, value: float, status: str) -> str:
        """Generate simple explanation for biomarker."""
        name = self._simplify_biomarker_name(biomarker_name)
        
        explanations = {
            "heart_rate": {
                "normal": "Your heart is beating at a healthy, steady pace.",
                "low": "Your heart is beating slower than usual, which may indicate good fitness or a need for evaluation.",
                "high": "Your heart is beating faster than usual. This could be due to stress, exercise, or other factors."
            },
            "spo2": {
                "normal": "Your blood oxygen levels are in the healthy range.",
                "low": "Your blood oxygen is lower than ideal. This should be checked by a doctor.",
                "high": "Your blood oxygen levels are good."
            },
            "respiratory_rate": {
                "normal": "You're breathing at a healthy rate.",
                "low": "Your breathing rate is slower than typical.",
                "high": "Your breathing rate is faster than typical. This could indicate various conditions."
            },
            "gait_variability": {
                "normal": "Your walking pattern is stable and consistent.",
                "high": "Your walking pattern shows some variation, which may warrant attention."
            },
            "balance_score": {
                "normal": "Your balance is good.",
                "low": "Your balance could be improved. Consider balance exercises."
            }
        }
        
        # Get explanation or provide default
        if biomarker_name in explanations and status in explanations[biomarker_name]:
            return explanations[biomarker_name][status]
        
        # Default explanation
        if status == "normal":
            return f"Your {name.lower()} is in the normal range."
        elif status == "low":
            return f"Your {name.lower()} is below the normal range."
        elif status == "high":
            return f"Your {name.lower()} is above the normal range."
        else:
            return f"{name} was measured during your screening."
    
    def _generate_ai_explanation(self, biomarker_name: str, value: float, status: str, unit: str = "") -> str:
        """
        Generate AI explanation for biomarker using GeminiClient.
        Falls back to hardcoded explanation if AI unavailable.
        
        Args:
            biomarker_name: Technical biomarker name
            value: Measured value
            status: Status (normal, low, high, not_assessed)
            unit: Unit of measurement
            
        Returns:
            Patient-friendly explanation string
        """
        # Try AI explanation first
        if self.gemini_client and self.gemini_client.is_available and status != 'not_assessed':
            try:
                friendly_name = self._simplify_biomarker_name(biomarker_name)
                
                prompt = f"""You are explaining a health screening result to a patient in simple, non-technical language.

Biomarker: {friendly_name}
Measured Value: {value} {unit}
Status: {status}

Explain what this result means to the patient in 1-2 simple sentences. Be reassuring if normal, and suggest consulting a doctor if abnormal. Do not use medical jargon."""
                
                response = self.gemini_client.generate(
                    prompt=prompt,
                    system_instruction="You are a helpful health assistant explaining screening results to patients in simple terms."
                )
                
                if response and response.text and not response.is_mock:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"AI explanation failed for {biomarker_name}: {e}. Using fallback.")
        
        # Fallback to hardcoded explanation
        return self._get_biomarker_explanation(biomarker_name, value, status)
    
    def _generate_pdf(
        self,
        report: PatientReport,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        trust_envelope: Optional[TrustEnvelope]
    ) -> str:
        """Generate the actual enhanced PDF file."""
        filename = f"{report.report_id}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title
        story.append(Paragraph(
            "Your Health Screening Report",
            self._styles['CustomTitle']
        ))
        
        # Report info
        story.append(Paragraph(
            f"Report ID: <b>{report.report_id}</b> | Generated: {report.generated_at.strftime('%B %d, %Y at %I:%M %p')}",
            self._styles['Caveat']
        ))
        story.append(Spacer(1, 25))
        
        # ===== OVERALL RISK SECTION =====
        story.append(Paragraph("📊 Your Overall Health Assessment", self._styles['SectionHeader']))
        story.append(Spacer(1, 10))
        story.append(RiskIndicator(report.overall_risk_level, width=300, height=45))
        story.append(Spacer(1, 12))
        
        confidence_text = f"Assessment Confidence: <b>{report.overall_confidence:.0%}</b>"
        story.append(Paragraph(confidence_text, self._styles['BodyText']))
        story.append(Spacer(1, 10))
        
        # Overall summary
        story.append(Paragraph(
            f"We assessed <b>{len(report.system_summaries)} body system(s)</b> during your screening. "
            f"Below you'll find detailed results for each system, including the specific measurements we took.",
            self._styles['BodyText']
        ))
        story.append(Spacer(1, 30))
        
        # ===== DETAILED SYSTEM RESULTS =====
        story.append(Paragraph("🔍 Your Results in Detail", self._styles['SectionHeader']))
        story.append(Spacer(1, 15))
        
        for system, summary in report.system_summaries.items():
            system_name = system.value.replace("_", " ").title()
            risk_level = summary["risk_level"]
            biomarkers = summary.get("biomarkers", {})
            
            # System header with colored box
            system_elements = []
            
            # System name and overall status
            system_header = Paragraph(
                f"<b>{system_name}</b> — {self._get_simple_status(risk_level)}",
                self._styles['SubHeader']
            )
            system_elements.append(system_header)
            system_elements.append(Spacer(1, 8))
            
            # Biomarker details table
            if biomarkers:
                table_data = [["What We Measured", "Your Value", "Normal Range", "Status"]]
                
                for bm_name, bm_data in biomarkers.items():
                    friendly_name = self._simplify_biomarker_name(bm_name)
                    # Round value to 2 decimal places
                    value = bm_data['value']
                    if isinstance(value, (int, float)):
                        value = round(value, 2)
                    # Abbreviate long unit names to prevent overlap
                    unit = bm_data.get('unit', '')
                    unit = self._abbreviate_unit(unit)
                    value_str = f"{value} {unit}"
                    normal_range = self._format_normal_range(bm_data.get('normal_range'))
                    status = bm_data.get('status', 'not_assessed')
                    status_icon = self._get_biomarker_status_icon(status)
                    
                    table_data.append([friendly_name, value_str, normal_range, status_icon])
                
                # Create table with adjusted column widths for better fit
                biomarker_table = Table(table_data, colWidths=[2.0*inch, 1.5*inch, 1.2*inch, 1.8*inch])
                
                # Base style
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1E40AF")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('TOPPADDING', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#D1D5DB")),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('TOPPADDING', (0, 1), (-1, -1), 7),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
                ]
                
                # Add color coding for each row
                for i, row in enumerate(table_data[1:], start=1):
                    status_text = row[3]
                    if "✓" in status_text:
                        bg_color = HexColor("#ECFDF5")  # Light green
                    elif "⚠" in status_text:
                        bg_color = HexColor("#FEF3C7")  # Light amber
                    else:
                        bg_color = white
                    
                    table_style.append(('BACKGROUND', (0, i), (-1, i), bg_color))
                
                biomarker_table.setStyle(TableStyle(table_style))
                system_elements.append(biomarker_table)
                system_elements.append(Spacer(1, 12))
                
                # Add simple explanations for each biomarker - FIXED: now appending to system_elements
                system_elements.append(Paragraph(
                    "<b>What This Means:</b>",
                    self._styles['BodyText']
                ))
                system_elements.append(Spacer(1, 6))
                
                for bm_name, bm_data in biomarkers.items():
                    status = bm_data.get('status', 'not_assessed')
                    if status != 'not_assessed':
                        # Use AI-generated explanation with fallback
                        explanation = self._generate_ai_explanation(
                            bm_name, 
                            bm_data['value'], 
                            status,
                            bm_data.get('unit', '')
                        )
                        friendly_name = self._simplify_biomarker_name(bm_name)
                        system_elements.append(Paragraph(
                            f"• <b>{friendly_name}:</b> {explanation}",
                            self._styles['BiomarkerExplanation']
                        ))
            
            # Add alerts if any
            if summary.get("alerts"):
                system_elements.append(Spacer(1, 10))
                system_elements.append(Paragraph(
                    "<b>⚠ Important Notes:</b>",
                    self._styles['SubHeader']
                ))
                for alert in summary["alerts"][:3]:
                    system_elements.append(Paragraph(
                        f"• {alert}",
                        self._styles['BiomarkerExplanation']
                    ))
            
            # Keep system together on same page
            story.append(KeepTogether(system_elements))
            story.append(Spacer(1, 25))
        
        # ===== RECOMMENDATIONS =====
        story.append(Paragraph("💡 What You Should Do Next", self._styles['SectionHeader']))
        story.append(Spacer(1, 10))
        
        for i, rec in enumerate(report.recommendations[:6], 1):
            story.append(Paragraph(f"{i}. {rec}", self._styles['BodyText']))
        story.append(Spacer(1, 25))
        
        # ===== IMPORTANT NOTES =====
        story.append(Paragraph("⚕️ Important Information", self._styles['SectionHeader']))
        story.append(Spacer(1, 10))
        
        for caveat in report.caveats:
            story.append(Paragraph(f"• {caveat}", self._styles['BodyText']))
        
        # Footer disclaimer
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> This health screening report is for informational purposes only and does not "
            "constitute medical advice, diagnosis, or treatment. Always consult with a qualified "
            "healthcare provider for proper medical evaluation and personalized medical advice. "
            "Do not disregard professional medical advice or delay seeking it based on this report.",
            self._styles['Caveat']
        ))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Enhanced patient report generated: {filepath}")
        
        return filepath


# For backward compatibility, create alias
PatientReportGenerator = EnhancedPatientReportGenerator
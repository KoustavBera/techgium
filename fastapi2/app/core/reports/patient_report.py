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
    from reportlab.graphics.charts.legends import Legend
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    REPORTLAB_ERROR = str(e)
    logger.warning(f"reportlab not installed - PDF generation unavailable. Error: {e}")


# Risk level colors
# Risk level colors (Pastel Backgrounds)
RISK_COLORS = {
    RiskLevel.LOW: HexColor("#D1FAE5") if REPORTLAB_AVAILABLE else "#D1FAE5",       # Mint Green (Light)
    RiskLevel.MODERATE: HexColor("#FEF3C7") if REPORTLAB_AVAILABLE else "#FEF3C7",  # Pale Amber
    RiskLevel.HIGH: HexColor("#FEE2E2") if REPORTLAB_AVAILABLE else "#FEE2E2",      # Pale Rose
    RiskLevel.CRITICAL: HexColor("#FEF2F2") if REPORTLAB_AVAILABLE else "#FEF2F2", # Very Pale Red
}

# Chart specific solid colors for the Pie slices (Visual Pop)
CHART_COLORS = {
    RiskLevel.LOW: HexColor("#10B981"),       # Emerald 500
    RiskLevel.MODERATE: HexColor("#F59E0B"),  # Amber 500
    RiskLevel.HIGH: HexColor("#EF4444"),      # Red 500
    RiskLevel.CRITICAL: HexColor("#B91C1C"),  # Red 700
}

# Risk text colors (Darker for contrast)
RISK_TEXT_COLORS = {
    RiskLevel.LOW: HexColor("#065F46") if REPORTLAB_AVAILABLE else "#065F46",       # Dark Emerald
    RiskLevel.MODERATE: HexColor("#92400E") if REPORTLAB_AVAILABLE else "#92400E",  # Dark Amber
    RiskLevel.HIGH: HexColor("#B91C1C") if REPORTLAB_AVAILABLE else "#B91C1C",      # Dark Red
    RiskLevel.CRITICAL: HexColor("#7F1D1D") if REPORTLAB_AVAILABLE else "#7F1D1D",  # Deep Red
}

STATUS_COLORS = {
    "normal": HexColor("#ECFDF5") if REPORTLAB_AVAILABLE else "#ECFDF5",      # Mint
    "low": HexColor("#EFF6FF") if REPORTLAB_AVAILABLE else "#EFF6FF",         # Pale Blue
    "high": HexColor("#FFFBEB") if REPORTLAB_AVAILABLE else "#FFFBEB",        # Pale Amber
    "not_assessed": HexColor("#F9FAFB") if REPORTLAB_AVAILABLE else "#F9FAFB" # Light Gray
}

RISK_LABELS = {
    RiskLevel.LOW: "Low Risk • Healthy",
    RiskLevel.MODERATE: "Moderate Risk • Monitor",
    RiskLevel.HIGH: "High Risk • Consult Doctor",
    RiskLevel.CRITICAL: "Critical • Immediate Care",
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
    "spo2": "Blood Oxygen (SpO2)",
    "respiratory_rate": "Breathing Rate",
    "breath_depth": "Breath Depth",
    "gait_variability": "Walking Stability",
    "balance_score": "Balance Score",
    "tremor": "Hand Steadiness",
    "reaction_time": "Reaction Time",
    "glucose": "Blood Sugar Estimate",
    "cholesterol": "Lipid Profile Estimate",
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
        """Custom flowable for risk level visual indicator (Pill Shape)."""
        
        def __init__(self, risk_level: RiskLevel, width: float = 120, height: float = 30):
            Flowable.__init__(self)
            self.risk_level = risk_level
            self.width = width
            self.height = height
        
        def draw(self):
            bg_color = RISK_COLORS.get(self.risk_level, lightgrey)
            text_color = RISK_TEXT_COLORS.get(self.risk_level, black)
            label = RISK_LABELS.get(self.risk_level, "Unknown")
            
            # Draw rounded pill background
            self.canv.setFillColor(bg_color)
            self.canv.setStrokeColor(bg_color)
            self.canv.roundRect(0, 0, self.width, self.height, self.height/2, fill=1, stroke=0)
            
            # Draw text
            self.canv.setFillColor(text_color)
            self.canv.setFont("Helvetica-Bold", 10)
            text_width = self.canv.stringWidth(label, "Helvetica-Bold", 10)
            # Center text vertically and horizontally
            self.canv.drawString((self.width - text_width) / 2, (self.height - 8) / 2 + 2, label)

    class HealthStatsChart(Flowable):
        """
        Beautiful Donut/Pie Chart showing breakdown of system health.
        """
        def __init__(self, system_summaries: Dict, overall_risk: RiskLevel, width: float = 400, height: float = 150):
            Flowable.__init__(self)
            self.width = width
            self.height = height
            self.system_summaries = system_summaries
            self.overall_risk = overall_risk
            
        def draw(self):
            # Calculate stats
            stats = {
                RiskLevel.LOW: 0,
                RiskLevel.MODERATE: 0,
                RiskLevel.HIGH: 0,
                RiskLevel.CRITICAL: 0
            }
            
            for s in self.system_summaries.values():
                lvl = s.get("risk_level", RiskLevel.LOW)
                if lvl in stats:
                    stats[lvl] += 1
            
            # Prepare data for pie chart
            data = []
            labels = []
            colors = []
            
            # Order: Low, Moderate, High, Critical
            order = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            for lvl in order:
                count = stats[lvl]
                if count > 0:
                    data.append(count)
                    labels.append(f"{RISK_LABELS[lvl].split('•')[0].strip()} ({count})")
                    # Use CHART_COLORS if available, else fallback
                    if 'CHART_COLORS' in globals():
                        colors.append(CHART_COLORS.get(lvl, darkgrey))
                    else:
                         # Fallback if CHART_COLORS not yet defined
                        colors.append(RISK_TEXT_COLORS.get(lvl, darkgrey))
            
            # If no data (empty report?), fallback
            if not data:
                data = [1]
                labels = ["No Data"]
                colors = [lightgrey]

            # Create Drawing
            d = Drawing(self.width, self.height)
            
            # Pie Chart
            pie = Pie()
            pie.x = 20
            pie.y = 10
            pie.width = 130
            pie.height = 130
            pie.data = data
            pie.labels = None # We use legend
            
            # Pie styling
            pie.slices.strokeWidth = 1
            pie.slices.strokeColor = white
            pie.simpleLabels = 0
            
            # Assign colors
            for i, col in enumerate(colors):
                pie.slices[i].fillColor = col
                
            d.add(pie)
            
            # Legend
            legend = Legend()
            legend.x = 180
            legend.y = 100
            legend.dx = 8
            legend.dy = 8
            legend.fontName = 'Helvetica'
            legend.fontSize = 10
            legend.boxAnchor = 'w'
            legend.columnMaximum = 10
            legend.strokeWidth = 1
            legend.strokeColor = white
            legend.subCols.dx = 0
            legend.alignment = 'right'
            
            legend.colorNamePairs = list(zip(colors, labels))
            d.add(legend)
            
            # Add "Health Score" or "Total Systems" text overlay or on side
            # Let's add a summary text summary on the right
            d.add(String(180, 50, "System Health Breakdown", fontName="Helvetica-Bold", fontSize=12, fillColor=HexColor("#374151")))
            d.add(String(180, 35, f"Total Body Systems Assessed: {sum(data)}", fontName="Helvetica", fontSize=10, fillColor=HexColor("#6B7280")))
            
            # Render drawing onto canvas
            d.drawOn(self.canv, 0, 0)
else:
    # Dummy class when reportlab not available
    class RiskIndicator:
        def __init__(self, *args, **kwargs):
            pass

    class HealthStatsChart:
        def __init__(self, *args, **kwargs):
            pass


class EnhancedPatientReportGenerator:
    """
    Generates detailed, patient-friendly PDF health screening reports.
    
    New Features for Redesign:
    - Pastel color palette
    - Minimalist typography (Helvetica)
    - Cleaner layouts (less borders)
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
        """Create custom paragraph styles with minimalist design."""
        if not REPORTLAB_AVAILABLE:
            return
        
        # Check if styles already exist to avoid KeyError
        if 'CustomTitle' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self._styles['Title'],
                fontSize=24,
                leading=30,
                spaceAfter=10,
                textColor=HexColor("#111827"), # Dark gray
                alignment=TA_LEFT,
                fontName='Helvetica-Bold'
            ))
        
        if 'SectionHeader' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self._styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10,
                textColor=HexColor("#374151"),
                fontName='Helvetica-Bold',
                borderWidth=0, # Removed boxy feel
                bulletFontName='Helvetica'
            ))
        
        if 'SubHeader' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='SubHeader',
                parent=self._styles['Heading3'],
                fontSize=12,
                spaceBefore=12,
                spaceAfter=6,
                textColor=HexColor("#4B5563"),
                fontName='Helvetica-Bold'
            ))
        
        if 'BodyText' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='BodyText',
                parent=self._styles['Normal'],
                fontSize=10,
                spaceAfter=8,
                leading=16, # More breathing room
                textColor=HexColor("#374151"),
                alignment=TA_LEFT,
                fontName='Helvetica'
            ))
        
        if 'BiomarkerExplanation' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='BiomarkerExplanation',
                parent=self._styles['BodyText'],
                fontSize=9,
                leading=14,
                textColor=HexColor("#6B7280"),
                leftIndent=10
            ))
        
        if 'Caveat' not in self._styles:
            self._styles.add(ParagraphStyle(
                name='Caveat',
                parent=self._styles['Normal'],
                fontSize=8,
                textColor=HexColor("#9CA3AF"),
                spaceBefore=4,
                spaceAfter=4,
                alignment=TA_LEFT
            ))
    
    def generate(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        interpretation: Optional[InterpretationResult] = None,
        trust_envelope: Optional[TrustEnvelope] = None,
        patient_id: str = "ANONYMOUS",
        trusted_results: Optional[Dict[PhysiologicalSystem, Any]] = None,
        rejected_systems: Optional[List[str]] = None
    ) -> PatientReport:
        """
        Generate an enhanced patient PDF report.
        
        Args:
            system_results: Risk results for each system (valid only)
            composite_risk: Overall composite risk score
            interpretation: Optional LLM interpretation
            trust_envelope: Optional trust envelope
            patient_id: Patient identifier (anonymized)
            trusted_results: Optional dict of TrustedRiskResult per system
            rejected_systems: Optional list of rejected system names
            
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
        
        # Build system summaries WITH biomarker details for valid systems
        for system, result in system_results.items():
            report.system_summaries[system] = {
                "risk_level": result.overall_risk.level,
                "risk_score": result.overall_risk.score,
                "status": self._get_simple_status(result.overall_risk.level),
                "alerts": result.alerts,
                "biomarkers": result.biomarker_summary,  # Include full biomarker data
                "explanation": result.overall_risk.explanation,
                "was_rejected": False
            }
        
        # Add rejected systems to summaries with special status
        if trusted_results:
            for system, trusted in trusted_results.items():
                if trusted.was_rejected and system not in report.system_summaries:
                    report.system_summaries[system] = {
                        "risk_level": RiskLevel.LOW,  # Default
                        "risk_score": 0.0,
                        "status": "⚠ Assessment Incomplete",
                        "alerts": [trusted.rejection_reason] if trusted.rejection_reason else ["Data quality insufficient"],
                        "biomarkers": {},
                        "explanation": f"Assessment could not be completed: {trusted.rejection_reason}",
                        "was_rejected": True,
                        "caveats": trusted.caveats
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
        
        # Add rejection note to caveats
        if rejected_systems:
            report.caveats.insert(0, 
                f"Note: {len(rejected_systems)} system(s) could not be assessed due to data quality issues: "
                f"{', '.join(rejected_systems)}."
            )
        
        # Generate PDF
        if REPORTLAB_AVAILABLE:
            pdf_path = self._generate_pdf(report, system_results, trust_envelope)
            report.pdf_path = pdf_path
        else:
            msg = f"PDF generation skipped - reportlab not available. Details: {REPORTLAB_ERROR}"
            logger.warning(msg)
            report.pdf_path = msg
        
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
                "normal": "<b>Meaning:</b> Your heart rate is within the healthy range (60-100 bpm).<br/><b>Details:</b> This indicates efficient heart function and good cardiovascular fitness.<br/><b>Guidance:</b> Maintain this with regular cardio exercise like walking or swimming.",
                "low": "<b>Meaning:</b> Your heart rate is lower than average (<60 bpm).<br/><b>Potential Causes:</b> This is common in athletes (a sign of efficiency) but can also be due to medications or electrical issues.<br/><b>Guidance:</b> If you feel dizzy or faint, consult a doctor. Otherwise, keep monitoring.",
                "high": "<b>Meaning:</b> Your heart rate is elevated (>100 bpm).<br/><b>Potential Causes:</b> Stress, caffeine, dehydration, anxiety, or underlying conditions.<br/><b>Guidance:</b> Try deep breathing, reduce caffeine, and hydrate. If it persists at rest, see a doctor."
            },
            "spo2": {
                "normal": "<b>Meaning:</b> Blood oxygen allows your body to function properly (95-100%).<br/><b>Details:</b> Your lungs are effectively transferring oxygen to your blood.<br/><b>Guidance:</b> No action needed. Continue deep breathing exercises.",
                "low": "<b>Meaning:</b> Oxygen saturation is below optimal levels (<95%).<br/><b>Potential Causes:</b> Respiratory issues, high altitude, or shallow breathing.<br/><b>Guidance:</b> Sit upright, take deep controlled breaths. If it stays low or you feel short of breath, seek medical attention.",
                "high": "<b>Meaning:</b> Your blood oxygen levels are excellent.<br/><b>Details:</b> Your body is well-oxygenated.<br/><b>Guidance:</b> Keep up your current healthy lifestyle."
            },
            "respiratory_rate": {
                "normal": "<b>Meaning:</b> Your breathing rate is normal (12-20 breaths/min).<br/><b>Details:</b> This suggests healthy lung function and calmness.<br/><b>Guidance:</b> Practice mindfulness to maintain this balance.",
                "low": "<b>Meaning:</b> You are breathing slowly.<br/><b>Potential Causes:</b> Deep relaxation, sleepiness, or potential central nervous system effects.<br/><b>Guidance:</b> If you feel alert and fine, this is likely healthy. If confused or groggy, seek help.",
                "high": "<b>Meaning:</b> Your breathing is rapid.<br/><b>Potential Causes:</b> Anxiety, exertion, fever, or respiratory distress.<br/><b>Guidance:</b> Rest and try 'box breathing' (inhale 4s, hold 4s, exhale 4s). If it persists, consult a doctor."
            },
            "gait_variability": {
                "normal": "<b>Meaning:</b> Your walking pattern is steady and rhythmic.<br/><b>Details:</b> This indicates good balance and neurological control.<br/><b>Guidance:</b> Maintain activity levels to preserve this mobility.",
                "high": "<b>Meaning:</b> Your steps vary significantly in timing or length.<br/><b>Potential Causes:</b> Fatigue, joint pain, muscle weakness, or potential neurological concerns.<br/><b>Guidance:</b> Focus on strength training for legs/core. Wear supportive shoes. Consider a gait analysis if falls are a concern."
            },
            "balance_score": {
                "normal": "<b>Meaning:</b> You have good stability.<br/><b>Details:</b> Your body effectively maintains posture against gravity.<br/><b>Guidance:</b> Yoga or Tai Chi are great for maintaining this.",
                "low": "<b>Meaning:</b> Your stability is reduced.<br/><b>Potential Causes:</b> Inner ear issues, muscle weakness, vision problems, or medication side effects.<br/><b>Guidance:</b> Clear walking paths at home. Incorporate balance exercises (e.g., standing on one leg with support)."
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
                
                prompt = f"""You are explaining a health screening result to a patient.
                
Biomarker: {friendly_name}
Measured Value: {value} {unit}
Status: {status}

Provide a structured 3-part explanation (use HTML <b> tags for bolding):
1. <b>Meaning:</b> What this result indicates about their body function.
2. <b>Potential Causes:</b> Why it might be {status} (lifestyle, temporary factors).
3. <b>Guidance:</b> Simple actionable advice or next steps.

Format as a single paragraph with line breaks (<br/>) between sections. Keep it encouraging but realistic. Avoid medical jargon."""
                
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
        story.append(Spacer(1, 15))
        
        # Add the new stats chart
        if report.system_summaries:
            # Use the new Pie Chart if we have system data
            story.append(HealthStatsChart(report.system_summaries, report.overall_risk_level, width=450, height=140))
        else:
            # Fallback if no systems
            story.append(RiskIndicator(report.overall_risk_level, width=300, height=45))
        
        story.append(Spacer(1, 15))
        
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
            
            # 1. Header and Table Group (KeepTogether)
            header_group = []
            
            # System name and overall status
            system_header = Paragraph(
                f"<b>{system_name}</b> — {self._get_simple_status(risk_level)}",
                self._styles['SubHeader']
            )
            header_group.append(system_header)
            header_group.append(Spacer(1, 8))
            
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
                biomarker_table = Table(table_data, colWidths=[2.2*inch, 1.3*inch, 1.2*inch, 1.8*inch])
                
                # Minimalist Table Style
                table_style = [
                    # Header: Simple, clean text
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#374151")),
                    ('FontName', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('TOPPADDING', (0, 0), (-1, 0), 12),
                    
                    # Remove all vertical grids
                    ('GRID', (0, 0), (-1, -1), 0, white), 
                    
                    # Bottom line for header
                    ('LINEBELOW', (0, 0), (-1, 0), 1, HexColor("#E5E7EB")),
                    
                    # Content formatting
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('TOPPADDING', (0, 1), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
                    
                    # Horizontal dividers for rows
                    ('LINEBELOW', (0, 1), (-1, -1), 0.5, HexColor("#F3F4F6")),
                ]
                
                # Add status pill colors for the status column
                for i, row in enumerate(table_data[1:], start=1):
                    status_text = row[3]
                    # We won't color the whole row, just the status text or cell if possible.
                    # For minimalist look, let's keep the row white but maybe distinct text color?
                    # actually, the STATUS_COLORS are pastel backgrounds. 
                    # Let's apply a subtle background to the Status cell ONLY.
                    
                    if "Normal" in status_text or "Good" in status_text:
                        bg_color = STATUS_COLORS["normal"]
                    elif "Below" in status_text or "Low" in status_text:
                        bg_color = STATUS_COLORS["low"]
                    elif "Above" in status_text or "High" in status_text:
                        bg_color = STATUS_COLORS["high"]
                    else:
                        bg_color = STATUS_COLORS["not_assessed"]
                        
                    # Apply background to the Status column (index 3)
                    table_style.append(('BACKGROUND', (3, i), (3, i), bg_color))
                    # Add rounded corner feel by not drawing borders? ReportLab tables are rectangular.
                    # We can't easily do rounded cells in simple Table, but the pastel block looks good.
                
                biomarker_table.setStyle(TableStyle(table_style))
                header_group.append(biomarker_table)
                header_group.append(Spacer(1, 16))
                
            # Add KeepTogether group to story
            story.append(KeepTogether(header_group))
                
            # 2. Add simple explanations - (Allowed to break across pages)
            if biomarkers:
                story.append(Paragraph(
                    "<b>What This Means:</b>",
                    self._styles['BodyText']
                ))
                story.append(Spacer(1, 6))
                
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
                        story.append(Paragraph(
                            f"• <b>{friendly_name}:</b> {explanation}",
                            self._styles['BiomarkerExplanation']
                        ))
            
            # Add alerts if any
            if summary.get("alerts"):
                story.append(Spacer(1, 10))
                story.append(Paragraph(
                    "<b>⚠ Important Notes:</b>",
                    self._styles['SubHeader']
                ))
                for alert in summary["alerts"][:3]:
                    story.append(Paragraph(
                        f"• {alert}",
                        self._styles['BiomarkerExplanation']
                    ))
            
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
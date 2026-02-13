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

# Optional explanation generator (for offline fallback)
try:
    from app.core.inference.explanation import ExplanationGenerator
    EXPLANATION_GENERATOR_AVAILABLE = True
except ImportError:
    EXPLANATION_GENERATOR_AVAILABLE = False
    ExplanationGenerator = None

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
    RiskLevel.ACTION_REQUIRED: HexColor("#FFEDD5") if REPORTLAB_AVAILABLE else "#FFEDD5", # Alert Amber (Pastel)
}

# Chart specific solid colors for the Pie slices (Visual Pop)
CHART_COLORS = {
    RiskLevel.LOW: HexColor("#10B981"),       # Emerald 500
    RiskLevel.MODERATE: HexColor("#F59E0B"),  # Amber 500
    RiskLevel.HIGH: HexColor("#EF4444"),      # Red 500
    RiskLevel.ACTION_REQUIRED: HexColor("#D97706"),  # Amber 600
}

# Risk text colors (Darker for contrast)
RISK_TEXT_COLORS = {
    RiskLevel.LOW: HexColor("#065F46") if REPORTLAB_AVAILABLE else "#065F46",       # Dark Emerald
    RiskLevel.MODERATE: HexColor("#92400E") if REPORTLAB_AVAILABLE else "#92400E",  # Dark Amber
    RiskLevel.HIGH: HexColor("#B91C1C") if REPORTLAB_AVAILABLE else "#B91C1C",      # Dark Red
    RiskLevel.ACTION_REQUIRED: HexColor("#92400E") if REPORTLAB_AVAILABLE else "#92400E", # Dark Amber
}

STATUS_COLORS = {
    "normal": HexColor("#ECFDF5") if REPORTLAB_AVAILABLE else "#ECFDF5",      # Mint (Good)
    "low": HexColor("#FFECD2") if REPORTLAB_AVAILABLE else "#FFECD2",         # Pastel Orange-Yellow (Abnormal)
    "high": HexColor("#FFECD2") if REPORTLAB_AVAILABLE else "#FFECD2",        # Pastel Orange-Yellow (Abnormal)
    "not_assessed": HexColor("#F9FAFB") if REPORTLAB_AVAILABLE else "#F9FAFB" # Light Gray
}

RISK_LABELS = {
    RiskLevel.LOW: "Low Risk • Healthy",
    RiskLevel.MODERATE: "Moderate Risk • Monitor",
    RiskLevel.HIGH: "High Risk • Consult Doctor",
    RiskLevel.ACTION_REQUIRED: "Action Required • Consult Provider",
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
    # Skin biomarkers (Thermal: ESP32 thermal camera, Visual: texture/color analysis)
    "skin_temperature": "Body Temperature",
    "skin_temperature_max": "Max Facial Temp (Fever Check)",
    "inflammation_index": "Inflammation Level",
    "face_mean_temperature": "Facial Temperature",
    "thermal_stability": "Temperature Stability",
    "texture_roughness": "Skin Texture",
    "skin_redness": "Skin Redness",
    "skin_yellowness": "Skin Yellowness",
    "color_uniformity": "Skin Tone Uniformity",
    "lesion_count": "Skin Lesions",
    "blink_rate": "Eye Blink Rate",
    "blink_count": "Total Blinks",
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
                RiskLevel.ACTION_REQUIRED: 0
            }
            
            for s in self.system_summaries.values():
                lvl = s.get("risk_level", RiskLevel.LOW)
                if lvl in stats:
                    stats[lvl] += 1
            
            # Prepare data for pie chart
            data = []
            labels = []
            colors = []
            
            # Order: Low, Moderate, High, Action Required
            order = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.ACTION_REQUIRED]
            
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
            
            # Render drawing onto canvas
            d.drawOn(self.canv, 0, 0)

    class ConfidenceMeter(Flowable):
        """Visual bar for assessment confidence."""
        def __init__(self, confidence: float, width: float = 250, height: float = 12):
            Flowable.__init__(self)
            self.confidence = confidence
            self.width = width
            self.height = height
            
        def draw(self):
            # Draw outline/bg
            self.canv.setStrokeColor(HexColor("#E5E7EB"))
            self.canv.setFillColor(HexColor("#F3F4F6"))
            self.canv.roundRect(0, 0, self.width, self.height, self.height/2, fill=1, stroke=1)
            
            # Draw fill based on confidence percentage
            if self.confidence > 0.0:
                # Color logic: Green for high confidence, Amber for low
                fill_color = HexColor("#10B981") if self.confidence >= 0.8 else HexColor("#F59E0B")
                self.canv.setFillColor(fill_color)
                self.canv.roundRect(0, 0, self.width * self.confidence, self.height, self.height/2, fill=1, stroke=0)
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
            # PRE-COMPUTE: Generate ALL biomarker explanations in ONE API call
            global_explanations = self._generate_all_biomarkers_explanations_global(report.system_summaries)
            
            pdf_path = self._generate_pdf(report, system_results, trust_envelope, global_explanations)
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
            RiskLevel.LOW: "Good",
            RiskLevel.MODERATE: "Attention Recommended",
            RiskLevel.HIGH: "Consult Doctor",
            RiskLevel.ACTION_REQUIRED: "Action Required",
            RiskLevel.UNKNOWN: "Device Required"
        }
        return statuses.get(level, "Unknown")
    
    def _simplify_biomarker_name(self, name: str) -> str:
        """Convert technical biomarker names to patient-friendly terms."""
        return BIOMARKER_NAMES.get(name, name.replace("_", " ").title())
    
    def _get_biomarker_status_icon(self, status: str) -> str:
        """Get icon for biomarker status."""
        if status == "normal":
            return "Normal"
        elif status == "low":
            return "Below Normal"
        elif status == "high":
            return "Above Normal"
        else:
            return "Not Assessed"
    
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
            "blinks_per_min": "blinks/min",
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
            "blink_rate": {
                "normal": "<b>Meaning:</b> Your blinking frequency is normal (12-20 blinks/min).<br/><b>Details:</b> Regular blinking keeps the eyes lubricated and protects them from strain.<br/><b>Guidance:</b> If you use screens often, follow the 20-20-20 rule (every 20 mins, look 20 feet away for 20 secs).",
                "low": "<b>Meaning:</b> You are blinking less frequently than average.<br/><b>Potential Causes:</b> Intense focus (screen use), dry eyes, or a short scan duration.<br/><b>Guidance:</b> Try to blink consciously when using digital devices to prevent eye fatigue.",
                "high": "<b>Meaning:</b> Your blink rate is higher than average (>20 blinks/min).<br/><b>Potential Causes:</b> Eye irritation, stress, fatigue, or dry air.<br/><b>Guidance:</b> Use lubricating eye drops if eyes feel dry, and ensure you are getting enough sleep."
            },
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
                "high": "<b>Meaning:</b> Your steps vary significantly in timing or length.<br/><b>Potential Causes:</b> Fatigue, joint pain, muscle weakness, or potential neurological concerns.<br/><b>Guidance:</b> Focus on strength training for legs/core. Wear supportive shoes. Consider a gait analysis if falls are a concern.",
                "not_assessed": "<b>Meaning:</b> Gait was not assessed because you were stationary during the scan.<br/><b>Details:</b> You appeared to be sitting or standing still. Walking analysis requires movement across the camera frame.<br/><b>Guidance:</b> For future screenings, ensure you walk naturally in front of the camera for at least 10 seconds to enable gait assessment."
            },
            "balance_score": {
                "normal": "<b>Meaning:</b> You have good stability.<br/><b>Details:</b> Your body effectively maintains posture against gravity.<br/><b>Guidance:</b> Yoga or Tai Chi are great for maintaining this.",
                "low": "<b>Meaning:</b> Your stability is reduced.<br/><b>Potential Causes:</b> Inner ear issues, muscle weakness, vision problems, or medication side effects.<br/><b>Guidance:</b> Clear walking paths at home. Incorporate balance exercises (e.g., standing on one leg with support)."
            },
            # Skin system biomarker explanations (thermal & visual)
            "skin_temperature": {
                "normal": "<b>Meaning:</b> Your body temperature is in the normal range (36-37.5°C).<br/><b>Details:</b> This indicates healthy thermoregulation and no signs of fever.<br/><b>Guidance:</b> No action needed. Stay hydrated.",
                "low": "<b>Meaning:</b> Your skin temperature is lower than average.<br/><b>Potential Causes:</b> Cold environment, poor circulation, or measurement error.<br/><b>Guidance:</b> Ensure room temperature is comfortable. If you feel cold or have pale skin, consult a doctor.",
                "high": "<b>Meaning:</b> Your temperature is elevated, which may indicate fever.<br/><b>Potential Causes:</b> Infection, inflammation, recent physical activity, or warm environment.<br/><b>Guidance:</b> Rest, hydrate, and monitor. If fever persists over 38°C with symptoms, see a doctor."
            },
            "skin_temperature_max": {
                 "normal": "<b>Meaning:</b> Your inner eye temperature (core proxy) is normal.<br/><b>Details:</b> This is a reliable indicator of core body temperature.<br/><b>Guidance:</b> Maintain hydration.",
                 "low": "<b>Meaning:</b> Inner eye temperature reading is low.<br/><b>Potential Causes:</b> Sensor positioning, cold exposure.<br/><b>Guidance:</b> Ensure the sensor had a clear view of your face.",
                 "high": "<b>Meaning:</b> Possible fever detected at the inner eye.<br/><b>Potential Causes:</b> Infection or inflammation.<br/><b>Guidance:</b> Monitor temperature with a thermometer."
            },
            "inflammation_index": {
                "normal": "<b>Meaning:</b> No significant inflammation detected in the facial region.<br/><b>Details:</b> Normal thermal distribution across your face suggests healthy blood flow.<br/><b>Guidance:</b> Maintain a healthy lifestyle with anti-inflammatory foods.",
                "high": "<b>Meaning:</b> Elevated inflammation markers detected via thermal imaging.<br/><b>Potential Causes:</b> Localized inflammation, allergies, skin conditions, or early signs of infection.<br/><b>Guidance:</b> If you notice swelling or pain, consult a doctor. Consider reducing processed foods."
            },
            "face_mean_temperature": {
                "normal": "<b>Meaning:</b> Your average facial temperature is within normal range.<br/><b>Details:</b> This indicates healthy blood circulation and no localized hot spots.<br/><b>Guidance:</b> No action needed.",
                "low": "<b>Meaning:</b> Your facial temperature is lower than average.<br/><b>Potential Causes:</b> Cold exposure, reduced circulation.<br/><b>Guidance:</b> Ensure warm environment during future screenings.",
                "high": "<b>Meaning:</b> Your facial temperature is elevated.<br/><b>Potential Causes:</b> Warm environment, recent activity, or inflammation.<br/><b>Guidance:</b> Rest in a cool environment and recheck if concerned."
            },
            "thermal_stability": {
                "normal": "<b>Meaning:</b> Your skin temperature is stable over time.<br/><b>Details:</b> Consistent temperature readings indicate reliable measurement and stable physiology.<br/><b>Guidance:</b> No action needed.",
                "high": "<b>Meaning:</b> Temperature fluctuated during the scan.<br/><b>Potential Causes:</b> Movement, changing environment, or vascular instability.<br/><b>Guidance:</b> Try to remain still during future screenings."
            },
            "texture_roughness": {
                "normal": "<b>Meaning:</b> Your skin texture appears smooth and healthy.<br/><b>Details:</b> Normal texture suggests good skin hydration and minimal sun damage.<br/><b>Guidance:</b> Continue your skincare routine and use sun protection.",
                "high": "<b>Meaning:</b> Your skin texture shows increased roughness.<br/><b>Potential Causes:</b> Dehydration, sun damage, aging, or dry skin conditions.<br/><b>Guidance:</b> Moisturize regularly and consider using gentle exfoliants. Consult a dermatologist if concerned."
            },
            "skin_redness": {
                "normal": "<b>Meaning:</b> Your skin redness is within normal range.<br/><b>Details:</b> This suggests normal blood flow and no significant inflammation.<br/><b>Guidance:</b> Continue normal skincare.",
                "high": "<b>Meaning:</b> Increased skin redness detected.<br/><b>Potential Causes:</b> Rosacea, sunburn, allergic reaction, or inflammation.<br/><b>Guidance:</b> Use gentle skincare products. If persistent or accompanied by itching, see a dermatologist."
            },
            "skin_yellowness": {
                "normal": "<b>Meaning:</b> Your skin tone balance is normal.<br/><b>Details:</b> No signs of jaundice or pigmentation issues.<br/><b>Guidance:</b> No action needed.",
                "high": "<b>Meaning:</b> Increased skin yellowness detected.<br/><b>Potential Causes:</b> Carotenoid-rich diet, jaundice (liver issues), or natural skin tone variation.<br/><b>Guidance:</b> If eyes also appear yellow or you have abdominal symptoms, consult a doctor for liver function tests."
            },
            "color_uniformity": {
                "normal": "<b>Meaning:</b> Your skin tone is uniform and consistent.<br/><b>Details:</b> This suggests healthy melanin distribution and no significant pigmentation issues.<br/><b>Guidance:</b> Continue sun protection to maintain uniformity.",
                "low": "<b>Meaning:</b> Your skin shows uneven pigmentation.<br/><b>Potential Causes:</b> Sun damage, post-inflammatory hyperpigmentation, melasma, or aging.<br/><b>Guidance:</b> Use broad-spectrum sunscreen daily. Consider vitamin C serums or consult a dermatologist."
            },
            "lesion_count": {
                "normal": "<b>Meaning:</b> No concerning skin lesions detected.<br/><b>Details:</b> Regular skin checks are still important for early detection.<br/><b>Guidance:</b> Perform monthly self-exams and annual dermatology visits.",
                "high": "<b>Meaning:</b> Multiple skin lesions detected.<br/><b>Potential Causes:</b> Moles, age spots, or other benign lesions. Requires professional evaluation.<br/><b>Guidance:</b> Schedule a dermatology appointment for professional skin examination."
            },
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
    
    def _generate_batched_ai_explanations(self, system_name: str, biomarkers: Dict) -> Dict[str, str]:
        """
        Generate AI explanations for ALL biomarkers in one system with a SINGLE API call.
        
        This reduces API calls from N (one per biomarker) to 1 (one per system).
        Falls back to hardcoded explanations if AI unavailable.
        
        Args:
            system_name: Name of the body system (e.g., "Cardiovascular")
            biomarkers: Dict of biomarker data with 'value', 'status', 'unit' keys
            
        Returns:
            Dict mapping biomarker names to their explanations
        """
        explanations = {}
        
        # Filter to only abnormal biomarkers (normal ones get simple fallback)
        abnormal_biomarkers = {}
        for bm_name, bm_data in biomarkers.items():
            status = bm_data.get('status', 'not_assessed')
            if status == 'normal':
                # Use simple fallback for normal biomarkers - no AI needed
                explanations[bm_name] = self._get_biomarker_explanation(bm_name, bm_data.get('value', 0), status)
            elif status != 'not_assessed':
                abnormal_biomarkers[bm_name] = bm_data
        
        # If no abnormal biomarkers or AI unavailable, return what we have
        if not abnormal_biomarkers or not self.gemini_client or not self.gemini_client.is_available:
            for bm_name, bm_data in abnormal_biomarkers.items():
                explanations[bm_name] = self._get_biomarker_explanation(
                    bm_name, bm_data.get('value', 0), bm_data.get('status', 'unknown')
                )
            return explanations
        
        # Build single batched prompt for all abnormal biomarkers
        try:
            biomarker_list = []
            for i, (bm_name, bm_data) in enumerate(abnormal_biomarkers.items(), 1):
                friendly_name = self._simplify_biomarker_name(bm_name)
                biomarker_list.append(
                    f"{i}. {friendly_name}: {bm_data.get('value', 0)} {bm_data.get('unit', '')} ({bm_data.get('status', 'unknown')})"
                )
            
            prompt = f"""You are explaining {system_name} health screening results to a patient.

The following biomarkers need explanation:
{chr(10).join(biomarker_list)}

For EACH biomarker, provide a brief 2-3 sentence explanation covering:
- What it means for their health
- Simple advice or next steps

Format your response as a JSON object where keys are the biomarker names and values are the explanations.
Example: {{"Heart Rate": "Your heart rate is elevated...", "Blood Pressure": "Your blood pressure..."}}

Keep explanations encouraging but realistic. Avoid medical jargon."""

            response = self.gemini_client.generate(
                prompt=prompt,
                system_instruction="You are a helpful health assistant. Respond with valid JSON only."
            )
            
            if response and response.text and not response.is_mock:
                import json
                try:
                    # Parse JSON response
                    ai_explanations = json.loads(response.text.strip())
                    
                    # Map AI explanations back to original biomarker names
                    for bm_name, bm_data in abnormal_biomarkers.items():
                        friendly_name = self._simplify_biomarker_name(bm_name)
                        if friendly_name in ai_explanations:
                            explanations[bm_name] = ai_explanations[friendly_name]
                        else:
                            # Fallback if not found in response
                            explanations[bm_name] = self._get_biomarker_explanation(
                                bm_name, bm_data.get('value', 0), bm_data.get('status', 'unknown')
                            )
                    return explanations
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse batched AI response for {system_name}")
                    
        except Exception as e:
            logger.warning(f"Batched AI explanation failed for {system_name}: {e}")
        
        # Fallback: Use hardcoded explanations for all abnormal biomarkers
        for bm_name, bm_data in abnormal_biomarkers.items():
            explanations[bm_name] = self._get_biomarker_explanation(
                bm_name, bm_data.get('value', 0), bm_data.get('status', 'unknown')
            )
        
        return explanations
    
    def _generate_all_biomarkers_explanations_global(
        self,
        system_summaries: Dict[PhysiologicalSystem, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate AI explanations for ALL abnormal biomarkers across ALL systems in ONE API call.
        
        This replaces the old pattern of calling _generate_batched_ai_explanations per system.
        Returns a flat dict: {"biomarker_name": "explanation text"}
        """
        # Collect all abnormal biomarkers from all systems
        all_abnormal = {}
        for system, summary in system_summaries.items():
            biomarkers = summary.get("biomarkers", {})
            for bm_name, bm_data in biomarkers.items():
                status = bm_data.get("status", "not_assessed")
                # Only use AI for abnormal readings (high/low)
                if status in ["high", "low"]:
                    all_abnormal[bm_name] = bm_data
        
        # If no abnormal biomarkers, return empty dict
        if not all_abnormal or not self.gemini_client or not self.gemini_client.is_available:
            logger.info("Skipping global AI explanations (no abnormal biomarkers or Gemini unavailable)")
            return {}
        
        # Build a single consolidated prompt
        prompt = """Generate patient-friendly explanations for the following abnormal health screening biomarkers.

For each biomarker, provide a concise explanation in this format:
<b>Meaning:</b> What this result means<br/><b>Potential Causes:</b> Common reasons for this reading<br/><b>Guidance:</b> What the patient should do

Return JSON only:
{
  "biomarker_name_1": "explanation text",
  "biomarker_name_2": "explanation text",
  ...
}

BIOMARKERS:
"""
        
        for bm_name, bm_data in all_abnormal.items():
            friendly_name = self._simplify_biomarker_name(bm_name)
            value = bm_data.get("value", 0)
            unit = bm_data.get("unit", "")
            status = bm_data.get("status", "unknown")
            prompt += f"\n- {friendly_name}: {value} {unit} (Status: {status.upper()})"
        
        prompt += "\n\nProvide explanations in JSON format as described above."
        
        try:
            logger.info(f"Generating global AI explanations for {len(all_abnormal)} abnormal biomarkers in ONE call")
            response = self.gemini_client.generate(
                prompt,
                system_instruction="You are a medical explanation assistant. Provide clear, patient-friendly explanations."
            )
            
            # Parse JSON response
            import json
            text = response.text.strip()
            
            # Try direct parse
            if text.startswith("{"):
                explanations = json.loads(text)
                logger.info(f"Successfully parsed {len(explanations)} AI explanations")
                return explanations
            
            # Try extracting from code block
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                explanations = json.loads(text[start:end].strip())
                logger.info(f"Successfully parsed {len(explanations)} AI explanations from code block")
                return explanations
            
            if "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                explanations = json.loads(text[start:end].strip())
                logger.info(f"Successfully parsed {len(explanations)} AI explanations from generic code block")
                return explanations
            
            logger.warning("Failed to parse global AI response as JSON")
            return {}
            
        except Exception as e:
            logger.warning(f"Global AI explanation failed: {e}")
            return {}
    
    def _generate_pdf(
        self,
        report: PatientReport,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        trust_envelope: Optional[TrustEnvelope],
        global_explanations: Dict[str, str]
    ) -> str:
        """Generate the actual enhanced PDF file."""
        filename = f"{report.report_id}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=1.0*inch,
            leftMargin=1.0*inch,
            topMargin=1.0*inch,
            bottomMargin=1.0*inch
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
        story.append(Spacer(1, 4))
        story.append(ConfidenceMeter(report.overall_confidence, width=300))
        story.append(Spacer(1, 15))
        
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
            
            # Phase 2: Add Experimental tag for specific systems
            is_experimental = system in [PhysiologicalSystem.NASAL, PhysiologicalSystem.RENAL]
            if is_experimental:
                system_name += " (Experimental)"
            
            risk_level = summary["risk_level"]
            biomarkers = summary.get("biomarkers", {})
            try: 
                logger.info(f"Generating PDF report for system: {system_name} with biomarkers: {list(biomarkers.keys())}")
            except Exception:
                pass

            
            # 1. Header and Table Group (KeepTogether)
            header_group = []
            
            # System name (clean header, no status text)
            system_header = Paragraph(
                f"<b>{system_name}</b>",
                self._styles['SubHeader']
            )
            header_group.append(system_header)
            header_group.append(Spacer(1, 6))

            # Visual risk indicator (pill)
            header_group.append(
                RiskIndicator(
                    risk_level=risk_level,
                    width=200,
                    height=28
                )
            )

            header_group.append(Spacer(1, 10))

            
            # Biomarker details table
            if biomarkers:
                table_data = [["What We Measured", "Your Value", "Normal Range", "Status"]]
                
                for bm_name, bm_data in biomarkers.items():
                    # CONTEXT-AWARE: Check for stationary gait BEFORE simplification
                    is_stationary_gait = (bm_name == "gait_variability" and bm_data.get('status') == 'not_assessed')
                    
                    friendly_name = self._simplify_biomarker_name(bm_name)
                    
                    # Add (Stationary) suffix for stationary gait
                    if is_stationary_gait:
                        friendly_name += " (Stationary)"
                    
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
                    
                    # Update status label for stationary gait
                    if is_stationary_gait:
                        status_icon = "Not Assessed (Stationary)"
                    
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
                    # Determine background color based on status
                    # Check order matters: check "Above/Below" before "Normal" to avoid false matches
                    
                    if "Below" in status_text:
                        bg_color = STATUS_COLORS["low"]  # Pastel orange
                    elif "Above" in status_text:
                        bg_color = STATUS_COLORS["high"]  # Pastel orange
                    elif "✓" in status_text or status_text == "✓ Normal":
                        bg_color = STATUS_COLORS["normal"]  # Mint green (only for truly normal)
                    else:
                        bg_color = STATUS_COLORS["not_assessed"]  # Light gray
                        
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
                
                # Use PRE-COMPUTED global explanations instead of per-system batched calls
                for bm_name, bm_data in biomarkers.items():
                    status = bm_data.get('status', 'not_assessed')
                    
                    # CONTEXT-AWARE: Include explanation for stationary gait
                    if status != 'not_assessed' or bm_name == 'gait_variability':
                        # Try global AI explanation first, fallback to hardcoded
                        explanation = global_explanations.get(
                            bm_name,
                            self._get_biomarker_explanation(bm_name, bm_data['value'], status)
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
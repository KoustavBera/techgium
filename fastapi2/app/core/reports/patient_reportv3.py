"""
Enhanced Patient Report Generator V3 (Playwright)
Generates detailed, informative PDF reports using HTML templates and Playwright for rendering.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import jinja2

# Check for Playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from app.core.inference.risk_engine import RiskLevel, SystemRiskResult, RiskScore
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope
from app.core.llm.risk_interpreter import InterpretationResult
from app.core.llm.gemini_client import GeminiClient
from app.utils import get_logger

logger = get_logger(__name__)

@dataclass
class PatientReportV3:
    """Data container for patient report."""
    report_id: str
    generated_at: datetime
    patient_id: str = "ANONYMOUS"
    overall_risk_level: str = "LOW"
    overall_risk_score: float = 0.0
    overall_confidence: float = 0.0
    system_summaries: Dict[PhysiologicalSystem, Dict[str, Any]] = field(default_factory=dict)
    interpretation_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    pdf_path: Optional[str] = None

# Simplified biomarker names
BIOMARKER_NAMES = {
    # Cardiovascular
    "heart_rate": "Heart Rate",
    "hrv_rmssd": "Heart Rate Variability",
    "hrv": "Heart Rate Variability",
    "blood_pressure_systolic": "Blood Pressure (Systolic)",
    "blood_pressure_diastolic": "Blood Pressure (Diastolic)",
    "systolic_bp": "Blood Pressure (Systolic)",
    "diastolic_bp": "Blood Pressure (Diastolic)",
    "thoracic_impedance": "Chest Fluid Level",
    "chest_micro_motion": "Chest Wall Movement",
    
    # Pulmonary
    "spo2": "Blood Oxygen (SpO2)",
    "respiratory_rate": "Breathing Rate",
    "respiration_rate": "Breathing Rate",
    "breath_depth": "Breath Depth",
    "breathing_depth": "Breath Depth",
    "breath_depth_index": "Breathing Depth Index",
    
    # Skin / Thermal
    "surface_temperature_avg": "Skin Temperature",
    "skin_temperature": "Skin Temperature",
    "skin_temperature_max": "Peak Skin Temperature",
    "thermal_asymmetry": "Thermal Asymmetry",
    "nostril_thermal_asymmetry": "Nasal Temperature Difference",
    "texture_roughness": "Skin Texture",
    "skin_redness": "Skin Redness",
    "skin_yellowness": "Skin Yellowness",
    "color_uniformity": "Skin Color Evenness",
    "lesion_count": "Skin Marks Count",
    
    # CNS / Neurological
    "gait_variability": "Walking Stability",
    "balance_score": "Balance Score",
    "tremor": "Hand Steadiness",
    "tremor_resting": "Resting Tremor",
    "tremor_postural": "Postural Tremor",
    "tremor_kinetic": "Movement Tremor",
    "tremor_intentional": "Action Tremor",
    "reaction_time": "Reaction Time",
    "posture_entropy": "Posture Variability",
    "cns_stability_score": "Nervous System Stability",
    "sway_amplitude_ap": "Forward/Backward Sway",
    "sway_amplitude_ml": "Side-to-Side Sway",
    
    # Skeletal / Musculoskeletal
    "posture_score": "Posture Score",
    "symmetry_index": "Symmetry Index",
    "gait_symmetry_ratio": "Walking Symmetry",
    "step_length_symmetry": "Step Balance",
    "stance_stability_score": "Standing Stability",
    "sway_velocity": "Sway Speed",
    "average_joint_rom": "Joint Mobility",
    
    # Nasal / Respiratory
    "nostril_occlusion_score": "Nasal Clearance",
    "respiratory_effort_index": "Breathing Effort",
    "nasal_cycle_balance": "Nasal Airflow Balance",
    
    # Gastrointestinal
    "abdominal_rhythm_score": "Digestive Rhythm",
    "visceral_motion_variance": "Internal Movement",
    "abdominal_respiratory_rate": "Belly Breathing Rate",
    
    # Renal / Fluid Balance
    "fluid_asymmetry_index": "Fluid Balance",
    "total_body_water_proxy": "Body Water Level",
    "extracellular_fluid_ratio": "Tissue Fluid Ratio",
    "fluid_overload_index": "Fluid Retention",
    
    # Reproductive / Hormonal
    "autonomic_imbalance_index": "Stress Balance",
    "stress_response_proxy": "Stress Response",
    "regional_flow_variability": "Blood Flow Pattern",
    "thermoregulation_proxy": "Temperature Control",
    
    # Metabolic
    "glucose": "Blood Sugar Estimate",
    "cholesterol": "Lipid Profile Estimate",
}

# Short definitions for "What We Measured"
BIOMARKER_DEFINITIONS = {
    # Cardiovascular
    "heart_rate": "The speed of the heartbeat measured in beats per minute.",
    "hrv_rmssd": "A measure of the variation in time between each heartbeat.",
    "blood_pressure_systolic": "The pressure in your arteries when your heart beats.",
    "blood_pressure_diastolic": "The pressure in your arteries when your heart rests between beats.",
    "systolic_bp": "The pressure in your arteries when your heart beats.",
    "diastolic_bp": "The pressure in your arteries when your heart rests between beats.",
    "thoracic_impedance": "Electrical resistance in the chest, indicating fluid levels.",
    "chest_micro_motion": "Tiny movements of the chest wall from heartbeat and breathing.",
    
    # Pulmonary
    "spo2": "Percentage of oxygen in your blood.",
    "respiratory_rate": "The number of breaths you take per minute.",
    "respiration_rate": "The number of breaths you take per minute.",
    "breath_depth": "The relative depth of your breathing movements.",
    "breathing_depth": "The relative depth of your breathing movements.",
    "breath_depth_index": "Overall measure of how deeply you breathe.",
    
    # Skin / Thermal
    "surface_temperature_avg": "Average temperature of your skin surface.",
    "skin_temperature": "Temperature of your skin surface.",
    "skin_temperature_max": "Highest temperature detected on your skin.",
    "thermal_asymmetry": "Temperature difference between left and right sides of the body.",
    "nostril_thermal_asymmetry": "Temperature difference between left and right nostrils.",
    "texture_roughness": "Smoothness or roughness of skin surface.",
    "skin_redness": "Amount of redness in skin tone.",
    "skin_yellowness": "Amount of yellow pigmentation in skin.",
    "color_uniformity": "Evenness of skin color across areas.",
    "lesion_count": "Number of unusual marks or spots detected.",
    
    # CNS / Neurological
    "gait_variability": "Consistency of your walking pattern.",
    "balance_score": "Your ability to maintain stability while standing.",
    "tremor_resting": "Involuntary shaking when muscles are relaxed.",
    "tremor_postural": "Involuntary shaking when holding a position.",
    "tremor_kinetic": "Involuntary shaking during movement.",
    "tremor_intentional": "Involuntary shaking when reaching for objects.",
    "reaction_time": "Time taken to respond to a stimulus.",
    "posture_entropy": "Variability and complexity of your posture patterns.",
    "cns_stability_score": "Overall nervous system balance and stability.",
    "sway_amplitude_ap": "Amount you sway forward and backward.",
    "sway_amplitude_ml": "Amount you sway side to side.",
    
    # Skeletal / Musculoskeletal
    "posture_score": "Alignment of your body segments (Head/Shoulder/Spine).",
    "symmetry_index": "Balance of movement/alignment between left and right sides.",
    "gait_symmetry_ratio": "How evenly you walk on both sides of your body.",
    "step_length_symmetry": "Similarity of step size between left and right legs.",
    "stance_stability_score": "Your steadiness when standing still.",
    "sway_velocity": "Speed of your body's swaying movements.",
    "average_joint_rom": "Average range of motion across your joints.",
    
    # Nasal / Respiratory
    "nostril_occlusion_score": "How clear or blocked your nasal passages are.",
    "respiratory_effort_index": "How hard you're working to breathe.",
    "nasal_cycle_balance": "Natural alternation of airflow between nostrils.",
    
    # Gastrointestinal
    "abdominal_rhythm_score": "Regularity of digestive system movements.",
    "visceral_motion_variance": "Variability in internal organ movements.",
    "abdominal_respiratory_rate": "Breathing rate measured from belly movement.",
    
    # Renal / Fluid Balance
    "fluid_asymmetry_index": "Difference in fluid distribution between body sides.",
    "total_body_water_proxy": "Estimated total water content in your body.",
    "extracellular_fluid_ratio": "Proportion of fluid outside your cells.",
    "fluid_overload_index": "Indicator of excess fluid retention.",
    
    # Reproductive / Hormonal
    "autonomic_imbalance_index": "Balance between relaxation and stress responses.",
    "stress_response_proxy": "Indicator of how your body responds to stress.",
    "regional_flow_variability": "Blood flow variation in different body regions.",
    "thermoregulation_proxy": "Your body's ability to maintain temperature.",
}

class EnhancedPatientReportGeneratorV3:
    """
    Generates detailed, patient-friendly PDF health screening reports using Playwright.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize generator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup Jinja2 Environment
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Initialize Gemini client (optional, used for dynamic explanations)
        try:
            self.gemini_client = GeminiClient()
        except Exception:
            self.gemini_client = None
            
        logger.info(f"EnhancedPatientReportGeneratorV3 initialized, output: {output_dir}")
    
    def _generate_executive_summary(self, system_results: Dict, composite_risk: RiskScore, interpretation: Optional[InterpretationResult]) -> str:
        """Generate AI-powered executive summary for report header."""
        if not self.gemini_client or not self.gemini_client.is_available:
            return self._get_default_executive_summary(system_results, composite_risk)
        
        try:
            # Collect key findings
            findings = []
            for system, result in system_results.items():
                if result.overall_risk.level not in [RiskLevel.LOW]:
                    findings.append(f"{system.value.replace('_', ' ').title()}: {result.overall_risk.level.value}")
            
            findings_text = ", ".join(findings) if findings else "All systems within normal range"
            
            prompt = f'''You are a healthcare AI assistant creating an executive summary for a patient health screening report.

Overall Risk Score: {composite_risk.score:.1f}/10
Overall Risk Level: {composite_risk.level.value}
Systems Assessed: {len(system_results)}
Key Findings: {findings_text}

Create a concise, reassuring 2-3 sentence executive summary for the patient that:
1. Highlights their overall health status in encouraging language
2. Mentions any areas that need attention (if any) without causing alarm
3. Emphasizes that this is a screening tool, not a diagnosis

Keep it warm, professional, and under 60 words. Use simple language a general audience can understand.'''
            
            response = self.gemini_client.generate(
                prompt=prompt,
                system_instruction="You are a compassionate healthcare communication specialist."
            )
            
            if response and response.text and not response.is_mock:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}. Using fallback.")
        
        return self._get_default_executive_summary(system_results, composite_risk)
    
    def _get_default_executive_summary(self, system_results: Dict, composite_risk: RiskScore) -> str:
        """Fallback executive summary when AI is unavailable."""
        if composite_risk.level == RiskLevel.LOW:
            return "Your health screening shows positive results across all monitored systems. Continue maintaining your healthy habits and schedule regular check-ups."
        elif composite_risk.level == RiskLevel.MODERATE:
            return "Your screening shows mostly healthy results with a few areas that may benefit from attention. Review the detailed findings below and consider discussing them with your healthcare provider."
        elif composite_risk.level == RiskLevel.HIGH:
            return "Your screening has identified several areas that require medical attention. Please consult with a healthcare professional to discuss these findings and create an appropriate care plan."
        else:
            return "Your screening has identified critical health indicators that need immediate medical evaluation. Please seek professional healthcare guidance as soon as possible."

    def generate(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        interpretation: Optional[InterpretationResult] = None,
        trust_envelope: Optional[TrustEnvelope] = None,
        patient_id: str = "ANONYMOUS",
        trusted_results: Optional[Dict[PhysiologicalSystem, Any]] = None,
        rejected_systems: Optional[List[str]] = None
    ) -> PatientReportV3:
        """
        Generate an enhanced patient PDF report V3.
        """
        report_id = f"PR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        report = PatientReportV3(
            report_id=report_id,
            generated_at=datetime.now(),
            patient_id=patient_id,
            overall_risk_level=composite_risk.level.value,
            overall_risk_score=composite_risk.score,
            overall_confidence=composite_risk.confidence
        )
        
        # Build System Summaries
        for system, result in system_results.items():
            # Process biomarkers using V1 logic (formatting + explanation)
            processed_biomarkers = {}
            for name, data in result.biomarker_summary.items():
                # Extract basic data
                if isinstance(data, dict):
                    value = data.get("value")
                    unit = data.get("unit", "")
                    status = data.get("status", "unknown")
                    normal_range = data.get("normal_range")
                else:
                    value = data
                    unit = ""
                    status = "unknown"
                    normal_range = None

                # Format values
                formatted_value = round(value, 2) if isinstance(value, (int, float)) else value
                formatted_unit = self._abbreviate_unit(unit)
                formatted_range = self._format_normal_range(normal_range)

                # Generate detailed explanation (AI or Fallback)
                explanation_html = self._generate_ai_explanation(name, value, status, unit)
                
                # Get definition (with fallback)
                definition = self._get_biomarker_definition(name)

                processed_biomarkers[name] = {
                    "value": formatted_value,
                    "unit": formatted_unit,
                    "status": status,
                    "normal_range": formatted_range,
                    "explanation": explanation_html,
                    "definition": definition
                }

            report.system_summaries[system] = {
                "risk_level": result.overall_risk.level.value,
                "risk_score": result.overall_risk.score,
                "status": self._get_status_text(result.overall_risk.level),
                "alerts": result.alerts,
                "biomarkers": processed_biomarkers,
                "explanation": result.overall_risk.explanation or self._get_default_system_explanation(system, result.overall_risk.level),
                "was_rejected": False
            }
            
        # Handle Rejected Systems (V1 Logic)
        if trusted_results:
            for system, trusted in trusted_results.items():
                if getattr(trusted, "was_rejected", False) and system not in report.system_summaries:
                     report.system_summaries[system] = {
                        "risk_level": "LOW",  # Default string value
                        "risk_score": 0.0,
                        "status": "⚠ Assessment Incomplete",
                        "alerts": [trusted.rejection_reason] if trusted.rejection_reason else ["Data quality insufficient"],
                        "biomarkers": {},
                        "explanation": f"Assessment could not be completed: {trusted.rejection_reason}",
                        "was_rejected": True,
                    }

        # Add Interpretation
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

        # Add rejection note to caveats if needed (V1 Logic)
        if rejected_systems:
            report.caveats.insert(0, 
                f"Note: {len(rejected_systems)} system(s) could not be assessed due to data quality issues: "
                f"{', '.join(rejected_systems)}."
            )

        # Generate executive summary using AI
        executive_summary = self._generate_executive_summary(system_results, composite_risk, interpretation)
        
        # Generate PDF
        if PLAYWRIGHT_AVAILABLE:
            pdf_path = self._render_pdf(report, executive_summary)
            report.pdf_path = pdf_path
        else:
            logger.error("Playwright not available.")
            report.pdf_path = None
            
        return report

    def _render_pdf(self, report: PatientReportV3, executive_summary: str) -> str:
        """Render HTML and convert to PDF using Playwright (sync)."""
        template = self.jinja_env.get_template("patient_report.html")
        html_content = template.render(
            report=report,
            biomarker_names=BIOMARKER_NAMES,
            executive_summary=executive_summary,
        )
        
        filename = f"{report.report_id}_v3.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Use a helper to isolate sync playwright in its own thread/event loop
        self._render_pdf_sync(html_content, filepath)
        logger.info(f"PDF generated successfully at {filepath}")
        return filepath
    
    def _render_pdf_sync(self, html_content: str, filepath: str):
        """Run Playwright synchronously in its own event loop context."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def _run_in_thread():
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content(html_content)
                page.pdf(
                    path=filepath,
                    format="A4",
                    print_background=True,
                    margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"}
                )
                browser.close()
        
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We ARE in an async context, so we run in a thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_in_thread)
                future.result()  # Wait for completion
        except RuntimeError:
            # No running loop, safe to run directly
            _run_in_thread()

    def _get_status_text(self, level: RiskLevel) -> str:
        """Convert risk level to simple patient-friendly status (V1 Logic)."""
        statuses = {
            RiskLevel.LOW: "✓ Good",
            RiskLevel.MODERATE: "⚠ Attention Recommended",
            RiskLevel.HIGH: "⚠ Consult Doctor",
            RiskLevel.CRITICAL: "🚨 Urgent Care Needed"
        }
        return statuses.get(level, "Unknown")
        
    def _generate_default_recommendations(self, system_results: Dict) -> List[str]:
        """Generate personalized recommendations based on findings (V1 Logic)."""
        recs = []
        
        # Check for cardiovascular issues
        if PhysiologicalSystem.CARDIOVASCULAR in system_results:
            cv_result = system_results[PhysiologicalSystem.CARDIOVASCULAR]
            if cv_result.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recs.append("Monitor your blood pressure regularly and reduce salt intake.")
                recs.append("Engage in 30 minutes of moderate exercise daily.")
        
        # Check for pulmonary issues
        if PhysiologicalSystem.PULMONARY in system_results:
            pulm_result = system_results[PhysiologicalSystem.PULMONARY]
            if pulm_result.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recs.append("Practice breathing exercises and avoid air pollutants.")
        
        # Check for CNS issues
        if PhysiologicalSystem.CNS in system_results:
            cns_result = system_results[PhysiologicalSystem.CNS]
            if cns_result.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recs.append("Work on balance exercises and ensure adequate sleep.")
        
        # General recommendations
        recs.extend([
            "Consult a healthcare professional for comprehensive evaluation.",
            "Maintain a balanced diet rich in fruits and vegetables.",
            "Stay hydrated with 8 glasses of water daily.",
            "Schedule regular health checkups."
        ])
        
        return recs[:6]

    def _abbreviate_unit(self, unit: str) -> str:
        """Abbreviate long unit names (V1 Logic)."""
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

    def _format_normal_range(self, normal_range: Optional[tuple]) -> str:
        """Format normal range for display (V1 Logic)."""
        if not normal_range:
            return "—"
        try:
            low, high = normal_range
            return f"{low}-{high}"
        except:
            return str(normal_range)
            
    def _get_default_system_explanation(self, system: PhysiologicalSystem, level: RiskLevel) -> str:
        """Fallback explanation if RiskScore has none."""
        if level == RiskLevel.LOW:
            return f"Your {system.value.replace('_', ' ')} metrics are within normal range."
        return f"Some {system.value.replace('_', ' ')} metrics require attention."

    def _simplify_biomarker_name(self, name: str) -> str:
        """Convert technical biomarker names to patient-friendly terms."""
        return BIOMARKER_NAMES.get(name, name.replace("_", " ").title())
    
    def _get_biomarker_definition(self, name: str) -> str:
        """Get biomarker definition with AI fallback."""
        # Try dictionary first
        if name in BIOMARKER_DEFINITIONS:
            return BIOMARKER_DEFINITIONS[name]
        
        # Try AI generation as fallback
        if self.gemini_client and self.gemini_client.is_available:
            try:
                friendly_name = self._simplify_biomarker_name(name)
                prompt = f'''Define the health biomarker "{friendly_name}" in one simple sentence (under 15 words) that a patient can understand. Focus on what it measures, not medical jargon.'''
                
                response = self.gemini_client.generate(
                    prompt=prompt,
                    system_instruction="You are a health education specialist explaining medical terms simply."
                )
                
                if response and response.text and not response.is_mock:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"AI definition failed for {name}: {e}")
        
        # Final fallback: generic description
        friendly_name = self._simplify_biomarker_name(name)
        return f"A health indicator related to {friendly_name.lower()}."

    def _get_biomarker_explanation(self, biomarker_name: str, value: Any, status: str) -> str:
        """Generate simple explanation for biomarker (Comprehensive)."""
        name = self._simplify_biomarker_name(biomarker_name)
        
        # Helper for default guidance
        def get_guidance(s):
            if s == "normal": return "Continue your current healthy habits."
            if s == "low": return "Monitor this metric. Ensure adequate rest and nutrition."
            if s == "high": return "Monitor this metric. Reduce stress and consult a professional if persistent."
            return "No specific action needed."

        explanations = {
            # CARDIOVASCULAR
            "heart_rate": {
                "normal": "<b>Meaning:</b> Your heart rate is within the healthy range (60-100 bpm).<br/><b>Details:</b> This indicates efficient heart function and good cardiovascular fitness.<br/><b>Guidance:</b> Maintain this with regular cardio exercise like walking or swimming.",
                "low": "<b>Meaning:</b> Your heart rate is lower than average (<60 bpm).<br/><b>Potential Causes:</b> Common in athletes, but check for dizziness.<br/><b>Guidance:</b> If you feel well, this is good. If faint, see a doctor.",
                "high": "<b>Meaning:</b> Your heart rate is elevated (>100 bpm).<br/><b>Potential Causes:</b> Stress, caffeine, anxiety, or recent exertion.<br/><b>Guidance:</b> Practice deep breathing and reduce caffeine intake."
            },
            "blood_pressure_systolic": {
                "normal": "<b>Meaning:</b> Systolic pressure is healthy (<120 mmHg).<br/><b>Details:</b> Your heart is pumping effectively without excessive force.<br/><b>Guidance:</b> Keep a low-sodium diet and stay active.",
                "high": "<b>Meaning:</b> Systolic pressure is elevated.<br/><b>Potential Causes:</b> Stress, diet (salt), or stiffness in arteries.<br/><b>Guidance:</b> Limit salt, manage stress, and check BP regularly.",
                "low": "<b>Meaning:</b> Systolic pressure is low.<br/><b>Potential Causes:</b> Dehydration or standing up too quickly.<br/><b>Guidance:</b> Drink more water and stand up slowly."
            },
             "blood_pressure_diastolic": {
                "normal": "<b>Meaning:</b> Diastolic pressure is healthy (<80 mmHg).<br/><b>Details:</b> Blood vessels are relaxing properly between beats.<br/><b>Guidance:</b> Maintain a heart-healthy diet.",
                "high": "<b>Meaning:</b> Diastolic pressure is elevated.<br/><b>Potential Causes:</b> Underlying vascular tension or kidney issues.<br/><b>Guidance:</b> Consult a doctor for a check-up.",
                "low": "<b>Meaning:</b> Diastolic pressure is low.<br/><b>Potential Causes:</b> Dehydration or medication.<br/><b>Guidance:</b> Stay hydrated."
            },
            "hrv_rmssd": {
                "normal": "<b>Meaning:</b> Your Heart Rate Variability (HRV) is good.<br/><b>Details:</b> This indicates a flexible and responsive autonomic nervous system (good recovery).<br/><b>Guidance:</b> Continue prioritizing sleep and stress management.",
                "low": "<b>Meaning:</b> Your HRV is lower than average.<br/><b>Potential Causes:</b> Stress, fatigue, overtraining, or illness.<br/><b>Guidance:</b> Prioritize rest, sleep, and relaxation techniques (meditation)."
            },

            # PULMONARY
            "spo2": {
                "normal": "<b>Meaning:</b> Blood oxygen is excellent (95-100%).<br/><b>Details:</b> Your body is well-oxygenated.<br/><b>Guidance:</b> Continue your current healthy habits.",
                "low": "<b>Meaning:</b> Oxygen saturation is below optimal levels (<95%).<br/><b>Potential Causes:</b> Shallow breathing, altitude, or respiratory issues.<br/><b>Guidance:</b> Sit upright, take deep controlled breaths. If short of breath, seek help.",
            },
            "respiratory_rate": {
                "normal": "<b>Meaning:</b> Breathing rate is normal (12-20 bpm).<br/><b>Details:</b> Indicates healthy lung function.<br/><b>Guidance:</b> Practice mindfulness.",
                "low": "<b>Meaning:</b> Breathing is slow.<br/><b>Potential Causes:</b> Deep relaxation or sleepiness.<br/><b>Guidance:</b> If alert, this is fine.",
                "high": "<b>Meaning:</b> Breathing is rapid.<br/><b>Potential Causes:</b> Anxiety, fever, or exertion.<br/><b>Guidance:</b> Rest and try 'box breathing' to calm down."
            },
            "breath_depth": {
                 "normal": "<b>Meaning:</b> Breathing depth is normal.<br/><b>Details:</b> Good tidal volume.<br/><b>Guidance:</b> Continue healthy breathing.",
                 "low": "<b>Meaning:</b> Breathing is shallow.<br/><b>Potential Causes:</b> Stress, posture, or restrictive clothing.<br/><b>Guidance:</b> Focus on diaphragmatic breathing (belly breathing).",
            },

            # SKIN / THERMAL
            "surface_temperature_avg": {
                "normal": "<b>Meaning:</b> Skin temperature is within normal range.<br/><b>Details:</b> Indicates good thermoregulation.<br/><b>Guidance:</b> Stay hydrated.",
                "high": "<b>Meaning:</b> Skin temperature is elevated.<br/><b>Potential Causes:</b> Fever, inflammation, or hot environment.<br/><b>Guidance:</b> Check for fever symptoms, hydrate, and rest.",
                "low": "<b>Meaning:</b> Skin temperature is cool.<br/><b>Potential Causes:</b> Cold environment or poor circulation.<br/><b>Guidance:</b> Keep warm and monitor."
            },
            "thermal_asymmetry": {
                "normal": "<b>Meaning:</b> Temperature is balanced on both sides.<br/><b>Details:</b> No signs of localized inflammation or blood flow issues.<br/><b>Guidance:</b> Maintain good posture.",
                "high": "<b>Meaning:</b> Significant temperature difference detected.<br/><b>Potential Causes:</b> Nerve impingement, inflammation, or localized injury.<br/><b>Guidance:</b> If you have pain, consult a specialist."
            },

            # CNS
            "gait_variability": {
                "normal": "<b>Meaning:</b> Walking pattern is steady.<br/><b>Details:</b> Good motor control and balance.<br/><b>Guidance:</b> Stay active.",
                "high": "<b>Meaning:</b> Steps are irregular.<br/><b>Potential Causes:</b> Fatigue, joint pain, or neurological issues.<br/><b>Guidance:</b> Wear supportive shoes and consider strength training."
            },
            "balance_score": {
                "normal": "<b>Meaning:</b> Good stability.<br/><b>Details:</b> Postural control is effective.<br/><b>Guidance:</b> Yoga/Tai Chi helps maintain this.",
                "low": "<b>Meaning:</b> Stability is reduced.<br/><b>Potential Causes:</b> Weakness, inner ear issues, or vision.<br/><b>Guidance:</b> Use support when needed; clear tripping hazards."
            },
            "tremor_resting": {
                "normal": "<b>Meaning:</b> No significant resting tremor.<br/><b>Details:</b> Hands are steady at rest.<br/><b>Guidance:</b> Continue to monitor.",
                "high": "<b>Meaning:</b> Tremor detected while resting.<br/><b>Potential Causes:</b> Stress, caffeine, or neurological conditions.<br/><b>Guidance:</b> Reduce caffeine. If persistent, consult a neurologist."
            },
            "tremor_postural": {
                "normal": "<b>Meaning:</b> No significant postural tremor.<br/><b>Details:</b> Hands are steady when held up.<br/><b>Guidance:</b> Continue to monitor.",
                "high": "<b>Meaning:</b> Tremor detected when holding a position.<br/><b>Potential Causes:</b> Fatigue, anxiety, or essential tremor.<br/><b>Guidance:</b> Monitor if it affects daily tasks."
            },
            "reaction_time": {
                "normal": "<b>Meaning:</b> Reaction speed is normal.<br/><b>Details:</b> Good neural processing speed.<br/><b>Guidance:</b> Stay mentally active.",
                "slow": "<b>Meaning:</b> Reaction speed is slower than average.<br/><b>Potential Causes:</b> Fatigue, distraction, or age-related changes.<br/><b>Guidance:</b> Ensure adequate sleep."
            },
            
            # SKELETAL
            "posture_score": {
                "normal": "<b>Meaning:</b> Good alignment.<br/><b>Details:</b> Head and shoulders are aligned with the spine.<br/><b>Guidance:</b> Maintain ergonomic seating.",
                "low": "<b>Meaning:</b> Poor posture detected.<br/><b>Potential Causes:</b> Slouching, desk work, or phone use.<br/><b>Guidance:</b> Correction exercises, ergonomic chair, take breaks."
            },
            "symmetry_index": {
                "normal": "<b>Meaning:</b> Movement is symmetrical.<br/><b>Details:</b> Both sides of the body move equally.<br/><b>Guidance:</b> Keep balanced workouts.",
                "low": "<b>Meaning:</b> Asymmetry detected.<br/><b>Potential Causes:</b> Injury favor, muscle imbalance.<br/><b>Guidance:</b> Unilateral exercises (lunges, etc.) to correct imbalance."
            }
        }
        
        key = biomarker_name
        # Check specific dictionary first
        if key in explanations and status in explanations[key]:
            return explanations[key][status]
        
        # Generic fallbacks if specific explanation missing but key exists (e.g., "high" missing)
        if key in explanations:
            # Try to infer generic advice from status
             return f"<b>Meaning:</b> Your {name.lower()} is {status}.<br/><b>Potential Causes:</b> A factor might be influencing this result.<br/><b>Guidance:</b> {get_guidance(status)}"

        # Default fallback for unknown biomarkers
        return f"<b>Meaning:</b> Your {name.lower()} is {status}.<br/><b>Potential Causes:</b> A factor might be influencing this result.<br/><b>Guidance:</b> {get_guidance(status)}"

    def _generate_ai_explanation(self, biomarker_name: str, value: Any, status: str, unit: str = "") -> str:
        """
        Generate AI explanation for biomarker using GeminiClient.
        Falls back to hardcoded explanation if AI unavailable.
        """
        # Try AI explanation first
        if self.gemini_client and self.gemini_client.is_available and status != 'not_assessed':
            try:
                friendly_name = self._simplify_biomarker_name(biomarker_name)
                
                prompt = f'''You are explaining a health screening result to a patient.
                
Biomarker: {friendly_name}
Measured Value: {value} {unit}
Status: {status}

Provide a structured 3-part explanation (use HTML <b> tags for bolding):
1. <b>Meaning:</b> What this result indicates about their body function.
2. <b>Potential Causes:</b> (If abnormal) Why it might be {status} / (If normal) <b>Details:</b> Why this is good.
3. <b>Guidance:</b> Simple actionable advice or next steps.

Format as a single paragraph with line breaks (<br/>) between sections. Keep it encouraging but realistic. Avoid medical jargon.'''
                
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

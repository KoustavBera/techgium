"""
Enhanced Patient Report Generator — Optimised Version

Key improvements over patient_report.py:
  1. Parallel LLM batching  (ThreadPoolExecutor, smaller chunks)
  2. Strict JSON prompts     (no conversational fluff)
  3. No emojis in headers    (clean professional look)
  4. Proper table grids      (visible borders, zebra-striped rows)
  5. "What It Means" rendered as a structured 4-column table
     (Biomarker | Meaning | Potential Causes | Guidance)
"""
from __future__ import annotations

import io
import json
import os
import re
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.core.inference.risk_engine import RiskLevel, SystemRiskResult, RiskScore
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope
from app.core.llm.risk_interpreter import InterpretationResult
from app.core.agents.medical_agents import ConsensusResult
from app.core.llm.gemini_client import GeminiClient, GeminiConfig
from app.utils import get_logger

# Optional explanation generator (offline fallback)
try:
    from app.core.inference.explanation import ExplanationGenerator
    EXPLANATION_GENERATOR_AVAILABLE = True
except ImportError:
    EXPLANATION_GENERATOR_AVAILABLE = False
    ExplanationGenerator = None

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ReportLab import
# ---------------------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.colors import (
        HexColor, green, red, orange, yellow,
        black, white, lightgrey, darkgrey,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, Flowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect, Circle, String
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.legends import Legend
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    REPORTLAB_ERROR = str(e)
    logger.warning(f"reportlab not installed — PDF generation unavailable. Error: {e}")


# ---------------------------------------------------------------------------
# Colour Palettes
# ---------------------------------------------------------------------------
if REPORTLAB_AVAILABLE:
    RISK_COLORS = {
        RiskLevel.LOW:             HexColor("#D1FAE5"),
        RiskLevel.MODERATE:        HexColor("#FEF3C7"),
        RiskLevel.HIGH:            HexColor("#FEE2E2"),
        RiskLevel.ACTION_REQUIRED: HexColor("#FFEDD5"),
        RiskLevel.UNKNOWN:         HexColor("#F3F4F6"),
    }
    CHART_COLORS = {
        RiskLevel.LOW:             HexColor("#10B981"),
        RiskLevel.MODERATE:        HexColor("#F59E0B"),
        RiskLevel.HIGH:            HexColor("#EF4444"),
        RiskLevel.ACTION_REQUIRED: HexColor("#D97706"),
        RiskLevel.UNKNOWN:         HexColor("#9CA3AF"),
    }
    RISK_TEXT_COLORS = {
        RiskLevel.LOW:             HexColor("#065F46"),
        RiskLevel.MODERATE:        HexColor("#92400E"),
        RiskLevel.HIGH:            HexColor("#B91C1C"),
        RiskLevel.ACTION_REQUIRED: HexColor("#92400E"),
        RiskLevel.UNKNOWN:         HexColor("#374151"),
    }
    STATUS_COLORS = {
        "normal":       HexColor("#ECFDF5"),
        "low":          HexColor("#FFECD2"),
        "high":         HexColor("#FFECD2"),
        "not_assessed": HexColor("#F9FAFB"),
    }
    # Zebra-stripe tint for alternating data rows
    ZEBRA_TINT = HexColor("#F9FAFB")
else:
    RISK_COLORS        = {}
    CHART_COLORS       = {}
    RISK_TEXT_COLORS   = {}
    STATUS_COLORS      = {}
    ZEBRA_TINT         = "#F9FAFB"

RISK_LABELS = {
    RiskLevel.LOW:             "Low Risk — Healthy",
    RiskLevel.MODERATE:        "Moderate Risk — Monitor",
    RiskLevel.HIGH:            "High Risk — Consult Doctor",
    RiskLevel.ACTION_REQUIRED: "Action Required — Consult Provider",
    RiskLevel.UNKNOWN:         "Not Assessed — Device Required",
}

# ---------------------------------------------------------------------------
# Biomarker name mapping
# ---------------------------------------------------------------------------
BIOMARKER_NAMES: Dict[str, str] = {
    "heart_rate":                    "Heart Rate",
    "hrv_rmssd":                     "Heart Rate Variability",
    "hrv":                           "Heart Rate Variability",
    "blood_pressure_systolic":       "Blood Pressure (Systolic)",
    "blood_pressure_diastolic":      "Blood Pressure (Diastolic)",
    "systolic_bp":                   "Blood Pressure (Systolic)",
    "diastolic_bp":                  "Blood Pressure (Diastolic)",
    "spo2":                          "Blood Oxygen (SpO2)",
    "respiratory_rate":              "Breathing Rate",
    "breath_depth":                  "Breath Depth",
    "respiratory_regularity_index":  "Breathing Stability",
    "nasal_surface_temp_elevation":  "Nasal Temperature Check",
    "airflow_thermal_symmetry_index":"Airflow Symmetry",
    "gait_variability":              "Walking Stability",
    "balance_score":                 "Balance Score",
    "tremor":                        "Hand Steadiness",
    "reaction_time":                 "Reaction Time",
    "glucose":                       "Blood Sugar Estimate",
    "cholesterol":                   "Lipid Profile Estimate",
    "skin_temperature":              "Body Temperature",
    "skin_temperature_max":          "Max Facial Temp (Fever Check)",
    "inflammation_index":            "Inflammation Level",
    "face_mean_temperature":         "Facial Temperature",
    "thermal_stability":             "Temperature Stability",
    "texture_roughness":             "Skin Texture",
    "skin_redness":                  "Skin Redness",
    "skin_yellowness":               "Skin Yellowness",
    "color_uniformity":              "Skin Tone Uniformity",
    "lesion_count":                  "Skin Lesions",
    "blink_rate":                    "Eye Blink Rate",
    "blink_count":                   "Total Blinks",
}

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class PatientReport:
    """Data container for patient report."""
    report_id:           str
    generated_at:        datetime
    patient_id:          str = "ANONYMOUS"
    overall_risk_level:  RiskLevel = RiskLevel.LOW
    overall_risk_score:  float = 0.0
    overall_confidence:  float = 0.0
    system_summaries:    Dict[PhysiologicalSystem, Dict[str, Any]] = field(default_factory=dict)
    clinical_findings:   List[Any] = field(default_factory=list)
    interpretation_summary: str = ""
    recommendations:     List[str] = field(default_factory=list)
    caveats:             List[str] = field(default_factory=list)
    pdf_path:            Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id":        self.report_id,
            "generated_at":     self.generated_at.isoformat(),
            "patient_id":       self.patient_id,
            "overall_risk_level":  self.overall_risk_level.value,
            "overall_risk_score":  round(self.overall_risk_score, 1),
            "overall_confidence":  round(self.overall_confidence, 2),
            "system_count":     len(self.system_summaries),
            "pdf_path":         self.pdf_path,
        }


# ---------------------------------------------------------------------------
# Custom ReportLab Flowables
# ---------------------------------------------------------------------------
if REPORTLAB_AVAILABLE:
    class RiskIndicator(Flowable):
        """Pill-shaped risk level badge (no emojis)."""

        def __init__(self, risk_level: RiskLevel, width: float = 120, height: float = 30):
            Flowable.__init__(self)
            self.risk_level = risk_level
            self.width = width
            self.height = height

        def draw(self):
            bg_color   = RISK_COLORS.get(self.risk_level, lightgrey)
            text_color = RISK_TEXT_COLORS.get(self.risk_level, black)
            label      = RISK_LABELS.get(self.risk_level, "Unknown")
            self.canv.setFillColor(bg_color)
            self.canv.setStrokeColor(bg_color)
            self.canv.roundRect(0, 0, self.width, self.height, self.height / 2, fill=1, stroke=0)
            self.canv.setFillColor(text_color)
            self.canv.setFont("Helvetica-Bold", 10)
            tw = self.canv.stringWidth(label, "Helvetica-Bold", 10)
            self.canv.drawString((self.width - tw) / 2, (self.height - 8) / 2 + 2, label)

    class HealthStatsChart(Flowable):
        """Donut/Pie chart — system health breakdown."""

        def __init__(self, system_summaries: Dict, overall_risk: RiskLevel,
                     width: float = 400, height: float = 150):
            Flowable.__init__(self)
            self.width   = width
            self.height  = height
            self.system_summaries = system_summaries
            self.overall_risk     = overall_risk

        def draw(self):
            stats = {lvl: 0 for lvl in [RiskLevel.LOW, RiskLevel.MODERATE,
                                         RiskLevel.HIGH, RiskLevel.ACTION_REQUIRED]}
            for s in self.system_summaries.values():
                lvl = s.get("risk_level", RiskLevel.LOW)
                if lvl in stats:
                    stats[lvl] += 1

            data, labels, colors = [], [], []
            for lvl in [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.ACTION_REQUIRED]:
                count = stats[lvl]
                if count > 0:
                    data.append(count)
                    labels.append(f"{RISK_LABELS[lvl].split('—')[0].strip()} ({count})")
                    colors.append(CHART_COLORS.get(lvl, darkgrey))
            if not data:
                data, labels, colors = [1], ["No Data"], [lightgrey]

            d   = Drawing(self.width, self.height)
            pie = Pie()
            pie.x, pie.y   = 20, 10
            pie.width      = 130
            pie.height     = 130
            pie.data       = data
            pie.labels     = None
            pie.slices.strokeWidth  = 1
            pie.slices.strokeColor  = white
            pie.simpleLabels        = 0
            for i, col in enumerate(colors):
                pie.slices[i].fillColor = col
            d.add(pie)

            legend = Legend()
            legend.x, legend.y = 180, 100
            legend.dx, legend.dy = 8, 8
            legend.fontName       = "Helvetica"
            legend.fontSize       = 10
            legend.boxAnchor      = "w"
            legend.columnMaximum  = 10
            legend.strokeWidth    = 1
            legend.strokeColor    = white
            legend.subCols.dx     = 0
            legend.alignment      = "right"
            legend.colorNamePairs = list(zip(colors, labels))
            d.add(legend)
            d.drawOn(self.canv, 0, 0)

    class ConfidenceMeter(Flowable):
        """Horizontal progress bar for assessment confidence."""

        def __init__(self, confidence: float, width: float = 250, height: float = 12):
            Flowable.__init__(self)
            self.confidence = confidence
            self.width  = width
            self.height = height

        def draw(self):
            self.canv.setStrokeColor(HexColor("#E5E7EB"))
            self.canv.setFillColor(HexColor("#F3F4F6"))
            self.canv.roundRect(0, 0, self.width, self.height, self.height / 2, fill=1, stroke=1)
            if self.confidence > 0.0:
                fill_color = HexColor("#10B981") if self.confidence >= 0.8 else HexColor("#F59E0B")
                self.canv.setFillColor(fill_color)
                self.canv.roundRect(0, 0, self.width * self.confidence,
                                    self.height, self.height / 2, fill=1, stroke=0)

    class SectionDivider(Flowable):
        """Full-width horizontal rule between major sections."""

        def __init__(self, width: float = 6.5 * inch, color: str = "#D1D5DB"):
            Flowable.__init__(self)
            self.width  = width
            self._color = HexColor(color)

        def draw(self):
            self.canv.setStrokeColor(self._color)
            self.canv.setLineWidth(0.75)
            self.canv.line(0, 0, self.width, 0)

else:
    class RiskIndicator:
        def __init__(self, *a, **kw): pass
    class HealthStatsChart:
        def __init__(self, *a, **kw): pass
    class ConfidenceMeter:
        def __init__(self, *a, **kw): pass
    class SectionDivider:
        def __init__(self, *a, **kw): pass

# ===========================================================================
# PHASE 2 — Generator Class: __init__, styles, helpers, biomarker dict
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper: parse structured explanation tuple
# ---------------------------------------------------------------------------
def _parse_explanation_parts(html_blob: str) -> Tuple[str, str, str]:
    """
    Convert the legacy HTML blob format into (meaning, causes, guidance) tuple.

    Legacy format (from hardcoded dict or old AI responses):
        "<b>Meaning:</b> ... <br/><b>Potential Causes:</b> ... <br/><b>Guidance:</b> ..."

    Returns plain-text strings for each column.  Falls back gracefully for
    any format that does not match (e.g. plain AI text).
    """
    # Strip all HTML tags to get plain text
    plain = re.sub(r"<[^>]+>", "", html_blob).strip()

    meaning_match = re.search(r"Meaning[:\s]+(.*?)(?:Potential Causes|Causes|$)", plain, re.DOTALL | re.IGNORECASE)
    causes_match  = re.search(r"(?:Potential )?Causes[:\s]+(.*?)(?:Guidance|$)", plain, re.DOTALL | re.IGNORECASE)
    guidance_match= re.search(r"Guidance[:\s]+(.*?)$", plain, re.DOTALL | re.IGNORECASE)

    meaning  = meaning_match.group(1).strip()  if meaning_match  else plain
    causes   = causes_match.group(1).strip()   if causes_match   else ""
    guidance = guidance_match.group(1).strip() if guidance_match else ""

    return meaning, causes, guidance


class EnhancedPatientReportGenerator:
    """
    Generates detailed, patient-friendly PDF health screening reports.

    Optimisations vs original:
    - Parallel LLM calls via ThreadPoolExecutor (small batches of ≤5)
    - Strict JSON prompts — no conversational fluff
    - All emojis removed throughout the PDF
    - Proper table grids + zebra-striped rows
    - 'What It Means' rendered as a proper 4-column table
    """

    # Maximum biomarkers per LLM batch (controls API call frequency)
    # Increased to 20 to strictly avoid the 5 Requests-Per-Minute (RPM) rate limit on free tier.
    _LLM_BATCH_SIZE = 20
    # Process sequentially or with max 2 workers to avoid burst limits.
    _LLM_MAX_WORKERS = 1

    def __init__(self, output_dir: str = "reports"):
        """Initialise generator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if REPORTLAB_AVAILABLE:
            self._styles = getSampleStyleSheet()
            self._create_custom_styles()

        try:
            self.gemini_client = GeminiClient()
            logger.info("GeminiClient initialised for AI explanations")
        except Exception as e:
            logger.warning(f"GeminiClient init failed: {e}. Will use fallback explanations.")
            self.gemini_client = None

        logger.info(f"EnhancedPatientReportGenerator (optimised) initialised, output: {output_dir}")

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------
    def _create_custom_styles(self):
        """Register custom paragraph styles."""
        if not REPORTLAB_AVAILABLE:
            return

        defs = [
            ("CustomTitle", {
                "parent": "Title", "fontSize": 24, "leading": 30, "spaceAfter": 10,
                "textColor": HexColor("#111827"), "alignment": TA_LEFT,
                "fontName": "Helvetica-Bold",
            }),
            ("SectionHeader", {
                "parent": "Heading2", "fontSize": 14, "spaceBefore": 20,
                "spaceAfter": 10, "textColor": HexColor("#374151"),
                "fontName": "Helvetica-Bold", "borderWidth": 0,
            }),
            ("SubHeader", {
                "parent": "Heading3", "fontSize": 12, "spaceBefore": 12,
                "spaceAfter": 6, "textColor": HexColor("#4B5563"),
                "fontName": "Helvetica-Bold",
            }),
            ("BodyText", {
                "parent": "Normal", "fontSize": 10, "spaceAfter": 8,
                "leading": 16, "textColor": HexColor("#374151"),
                "alignment": TA_LEFT, "fontName": "Helvetica",
            }),
            ("BiomarkerExplanation", {
                "parent": "BodyText", "fontSize": 9, "leading": 14,
                "textColor": HexColor("#4B5563"), "leftIndent": 4,
            }),
            ("TableCell", {
                "parent": "Normal", "fontSize": 8, "leading": 12,
                "textColor": HexColor("#374151"), "fontName": "Helvetica",
            }),
            ("Caveat", {
                "parent": "Normal", "fontSize": 8, "textColor": HexColor("#9CA3AF"),
                "spaceBefore": 4, "spaceAfter": 4, "alignment": TA_LEFT,
            }),
        ]

        for name, props in defs:
            if name not in self._styles:
                parent_name = props.pop("parent", "Normal")
                self._styles.add(ParagraphStyle(
                    name=name,
                    parent=self._styles[parent_name],
                    **props
                ))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _get_simple_status(self, level: RiskLevel) -> str:
        return {
            RiskLevel.LOW:             "Good",
            RiskLevel.MODERATE:        "Attention Recommended",
            RiskLevel.HIGH:            "Consult Doctor",
            RiskLevel.ACTION_REQUIRED: "Action Required",
            RiskLevel.UNKNOWN:         "Device Required",
        }.get(level, "Unknown")

    def _simplify_biomarker_name(self, name: str) -> str:
        return BIOMARKER_NAMES.get(name, name.replace("_", " ").title())

    def _abbreviate_unit(self, unit: str) -> str:
        abbr = {
            "power_spectral_density":       "PSD",
            "coefficient_of_variation":     "CV",
            "normalized_amplitude":         "norm.",
            "normalized_units_per_frame":   "units/frame",
            "breaths_per_min":              "brpm",
            "blinks_per_min":               "blinks/min",
            "saccades_per_sec":             "sacc/s",
            "score_0_100":                  "score",
            "score_0_1":                    "score",
            "variance_score":               "var",
            "normalized_intensity":         "norm",
            "normalized":                   "norm",
        }
        return abbr.get(unit, unit)

    def _format_normal_range(self, normal_range: Optional[tuple]) -> str:
        if not normal_range:
            return "—"
        low, high = normal_range
        return f"{low}–{high}"

    def _get_biomarker_status_icon(self, status: str) -> str:
        """Plain-text status label (no emojis)."""
        return {
            "normal": "Normal",
            "low":    "Below Normal",
            "high":   "Above Normal",
        }.get(status, "Not Assessed")

    @staticmethod
    def _status_meta(status_text: str):
        """Single source of truth: status text -> (label, bg_color)."""
        if "Below" in status_text or status_text == "low":
            return ("Below Normal", STATUS_COLORS["low"])
        elif "Above" in status_text or status_text == "high":
            return ("Above Normal", STATUS_COLORS["high"])
        elif status_text in ("Normal", "normal"):
            return ("Normal", STATUS_COLORS["normal"])
        else:
            return (status_text, STATUS_COLORS["not_assessed"])

    # ------------------------------------------------------------------
    # Hardcoded biomarker explanations (structured as 3-tuple)
    # Each value: (meaning, potential_causes, guidance)
    # ------------------------------------------------------------------
    _BIOMARKER_EXPLANATIONS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
        "blink_rate": {
            "normal": (
                "Your blinking frequency is normal (12–20 blinks/min).",
                "Regular blinking keeps eyes lubricated and prevents strain.",
                "If you use screens often, follow the 20-20-20 rule.",
            ),
            "low": (
                "You are blinking less frequently than average.",
                "Intense focus (screen use), dry eyes, or short scan duration.",
                "Blink consciously when using digital devices.",
            ),
            "high": (
                "Your blink rate is higher than average (>20 blinks/min).",
                "Eye irritation, stress, fatigue, or dry air.",
                "Use lubricating eye drops and ensure adequate sleep.",
            ),
        },
        "heart_rate": {
            "normal": (
                "Your heart rate is within the healthy range (60–100 bpm).",
                "Indicates efficient heart function.",
                "Maintain with regular cardio exercise.",
            ),
            "low": (
                "Your heart rate is lower than average (<60 bpm).",
                "Common in athletes; may also be due to medications.",
                "See a doctor if you feel dizzy or faint.",
            ),
            "high": (
                "Your heart rate is elevated (>100 bpm).",
                "Stress, caffeine, dehydration, anxiety, or underlying conditions.",
                "Try deep breathing, reduce caffeine, hydrate. Consult a doctor if persistent.",
            ),
        },
        "spo2": {
            "normal": (
                "Blood oxygen is within a healthy range (95–100%).",
                "Lungs are effectively transferring oxygen to blood.",
                "No action needed. Continue deep breathing exercises.",
            ),
            "low": (
                "Oxygen saturation is below optimal levels (<95%).",
                "Respiratory issues, high altitude, or shallow breathing.",
                "Sit upright, take deep controlled breaths. Seek help if it persists.",
            ),
            "high": (
                "Your blood oxygen levels are excellent.",
                "Body is well-oxygenated.",
                "Keep up your current healthy lifestyle.",
            ),
        },
        "respiratory_rate": {
            "normal": (
                "Your breathing rate is normal (12–20 breaths/min).",
                "Healthy lung function and calmness.",
                "Practice mindfulness to maintain this balance.",
            ),
            "low": (
                "You are breathing slowly.",
                "Deep relaxation, sleepiness, or potential CNS effects.",
                "If alert and fine, this is healthy. If confused, seek help.",
            ),
            "high": (
                "Your breathing is rapid.",
                "Anxiety, exertion, fever, or respiratory distress.",
                "Rest and try box breathing (inhale 4s, hold 4s, exhale 4s).",
            ),
        },
        "respiratory_regularity_index": {
            "normal": (
                "Your autonomic nervous system appears stable.",
                "Measured by breath-to-breath variability (CV 0.02–0.25).",
                "Good sign of stress resilience.",
            ),
            "low": (
                "Your breathing is extremely metronomic.",
                "Forced breathing control or rigidity.",
                "Relax and breathe naturally.",
            ),
            "high": (
                "High breathing variability detected.",
                "Stress, anxiety, or irregular breathing patterns.",
                "Try 5 minutes of guided rhythmic breathing.",
            ),
        },
        "nasal_surface_temp_elevation": {
            "normal": (
                "No significant thermal inflammation detected.",
                "Temperature difference between nostril and cheek is normal (−0.2 to 1.0 °C).",
                "Maintain nasal hygiene.",
            ),
            "high": (
                "Elevated local temperature detected.",
                "Local inflammation, sinusitis, or congestion.",
                "If accompanied by pain or congestion, consider a check-up.",
            ),
        },
        "airflow_thermal_symmetry_index": {
            "normal": (
                "Airflow appears balanced between nostrils.",
                "Both sides contribute equally to breathing.",
                "Healthy nasal function.",
            ),
            "high": (
                "Significant asymmetry in airflow detected.",
                "Deviated septum, unilateral congestion, or nasal cycle peak.",
                "Common and often benign; consult an ENT if breathing is difficult.",
            ),
        },
        "gait_variability": {
            "normal": (
                "Your walking pattern is steady and rhythmic.",
                "Indicates good balance and neurological control.",
                "Maintain activity levels to preserve this mobility.",
            ),
            "high": (
                "Your steps vary significantly in timing or length.",
                "Fatigue, joint pain, muscle weakness, or neurological concerns.",
                "Focus on strength training and wear supportive shoes.",
            ),
            "not_assessed": (
                "Gait was not assessed because you were stationary during the scan.",
                "You appeared to be sitting or standing still during measurement.",
                "For future screenings, walk naturally in front of the camera for at least 10 seconds.",
            ),
        },
        "balance_score": {
            "normal": (
                "You have good stability.",
                "Body effectively maintains posture against gravity.",
                "Yoga or Tai Chi are great for maintaining this.",
            ),
            "low": (
                "Your stability is reduced.",
                "Inner ear issues, muscle weakness, vision problems, or medication side effects.",
                "Clear walking paths at home. Incorporate balance exercises.",
            ),
        },
        "skin_temperature": {
            "normal": (
                "Body temperature is in the normal range (36–37.5 °C).",
                "Healthy thermoregulation, no signs of fever.",
                "No action needed. Stay hydrated.",
            ),
            "low": (
                "Your skin temperature is lower than average.",
                "Cold environment, poor circulation, or measurement error.",
                "Ensure room temperature is comfortable. Consult doctor if pale or cold.",
            ),
            "high": (
                "Temperature is elevated — may indicate fever.",
                "Infection, inflammation, recent physical activity, or warm environment.",
                "Rest, hydrate, and monitor. See a doctor if fever persists above 38 °C.",
            ),
        },
        "skin_temperature_max": {
            "normal": (
                "Inner eye temperature (core proxy) is normal.",
                "Reliable indicator of core body temperature.",
                "Maintain hydration.",
            ),
            "low": (
                "Inner eye temperature reading is low.",
                "Sensor positioning or cold exposure.",
                "Ensure the sensor had a clear view of your face.",
            ),
            "high": (
                "Possible fever detected at the inner eye.",
                "Infection or inflammation.",
                "Monitor temperature with a thermometer.",
            ),
        },
        "inflammation_index": {
            "normal": (
                "No significant inflammation detected in the facial region.",
                "Normal thermal distribution indicates healthy blood flow.",
                "Maintain healthy lifestyle with anti-inflammatory foods.",
            ),
            "high": (
                "Elevated inflammation markers detected via thermal imaging.",
                "Localised inflammation, allergies, skin conditions, or early infection.",
                "If you notice swelling or pain, consult a doctor.",
            ),
        },
        "face_mean_temperature": {
            "normal": (
                "Average facial temperature is within normal range.",
                "Indicates healthy blood circulation and no localised hot spots.",
                "No action needed.",
            ),
            "low": (
                "Facial temperature is lower than average.",
                "Cold exposure or reduced circulation.",
                "Ensure warm environment during future screenings.",
            ),
            "high": (
                "Facial temperature is elevated.",
                "Warm environment, recent activity, or inflammation.",
                "Rest in a cool environment and recheck if concerned.",
            ),
        },
        "thermal_stability": {
            "normal": (
                "Skin temperature is stable over time.",
                "Consistent readings indicate reliable measurement and stable physiology.",
                "No action needed.",
            ),
            "high": (
                "Temperature fluctuated during the scan.",
                "Movement, changing environment, or vascular instability.",
                "Try to remain still during future screenings.",
            ),
        },
        "texture_roughness": {
            "normal": (
                "Skin texture appears smooth and healthy.",
                "Normal texture suggests good hydration and minimal sun damage.",
                "Continue your skincare routine and use sun protection.",
            ),
            "high": (
                "Skin texture shows increased roughness.",
                "Dehydration, sun damage, aging, or dry skin conditions.",
                "Moisturise regularly and consider seeing a dermatologist.",
            ),
        },
        "skin_redness": {
            "normal": (
                "Skin redness is within normal range.",
                "Normal blood flow and no significant inflammation.",
                "Continue normal skincare.",
            ),
            "high": (
                "Increased skin redness detected.",
                "Rosacea, sunburn, allergic reaction, or inflammation.",
                "Use gentle skincare products. See a dermatologist if persistent.",
            ),
        },
        "skin_yellowness": {
            "normal": (
                "Skin tone balance is normal.",
                "No signs of jaundice or pigmentation issues.",
                "No action needed.",
            ),
            "high": (
                "Increased skin yellowness detected.",
                "Carotenoid-rich diet, jaundice (liver issues), or natural skin tone variation.",
                "If eyes also appear yellow or you have abdominal symptoms, consult a doctor.",
            ),
        },
        "color_uniformity": {
            "normal": (
                "Skin tone is uniform and consistent.",
                "Healthy melanin distribution and no significant pigmentation issues.",
                "Continue sun protection to maintain uniformity.",
            ),
            "low": (
                "Skin shows uneven pigmentation.",
                "Sun damage, post-inflammatory hyperpigmentation, melasma, or aging.",
                "Use broad-spectrum sunscreen daily. Consider vitamin C serums.",
            ),
        },
        "lesion_count": {
            "normal": (
                "No concerning skin lesions detected.",
                "Regular skin checks are still important for early detection.",
                "Perform monthly self-exams and annual dermatology visits.",
            ),
            "high": (
                "Multiple skin lesions detected.",
                "Moles, age spots, or other lesions requiring professional evaluation.",
                "Schedule a dermatology appointment for professional skin examination.",
            ),
        },
    }

    def _get_explanation_tuple(
        self, biomarker_name: str, value: float, status: str
    ) -> Tuple[str, str, str]:
        """
        Return a (meaning, causes, guidance) tuple from the hardcoded dict,
        or build a generic one for unknown biomarkers.
        """
        bm_data = self._BIOMARKER_EXPLANATIONS.get(biomarker_name, {})
        if status in bm_data:
            return bm_data[status]

        name = self._simplify_biomarker_name(biomarker_name)
        if status == "normal":
            return (f"Your {name.lower()} is in the normal range.", "", "No action needed.")
        elif status == "low":
            return (f"Your {name.lower()} is below the normal range.", "", "Monitor and consult if needed.")
        elif status == "high":
            return (f"Your {name.lower()} is above the normal range.", "", "Monitor and consult if needed.")
        else:
            return (f"{name} was measured during your screening.", "", "")

    def _generate_default_recommendations(self, system_results: Dict) -> List[str]:
        """Personalised recommendations based on findings."""
        recs = []
        if PhysiologicalSystem.CARDIOVASCULAR in system_results:
            cv = system_results[PhysiologicalSystem.CARDIOVASCULAR]
            if cv.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
                recs.append("Monitor your blood pressure regularly and reduce salt intake.")
                recs.append("Engage in 30 minutes of moderate exercise daily.")
        if PhysiologicalSystem.PULMONARY in system_results:
            pulm = system_results[PhysiologicalSystem.PULMONARY]
            if pulm.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
                recs.append("Practice breathing exercises and avoid air pollutants.")
        if PhysiologicalSystem.CNS in system_results:
            cns = system_results[PhysiologicalSystem.CNS]
            if cns.overall_risk.level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
                recs.append("Work on balance exercises and ensure adequate sleep.")
        recs.extend([
            "Consult a healthcare professional for comprehensive evaluation.",
            "Maintain a balanced diet rich in fruits and vegetables.",
            "Stay hydrated with at least 8 glasses of water daily.",
            "Schedule regular health checkups.",
        ])
        return recs[:6]

    # ===========================================================================
    # PHASE 3 — Parallel LLM Engine + Public API (generate / generate_bytes)
    # ===========================================================================

    # ------------------------------------------------------------------
    # Parallel LLM explanation engine
    # ------------------------------------------------------------------

    def _call_gemini_batch(self, batch: List[Tuple[str, Any]]) -> Dict[str, Tuple[str, str, str]]:
        """
        Call Gemini for one batch of <= _LLM_BATCH_SIZE abnormal biomarkers.

        Returns a dict: { "biomarker_key": (meaning, causes, guidance) }

        The prompt is a tight JSON-first instruction so the model outputs
        minimal tokens and we spend almost no time on regex cleanup.
        """
        if not batch:
            return {}

        lines = []
        for key, bm_data in batch:
            friendly = self._simplify_biomarker_name(key)
            val      = bm_data.get("value", 0)
            unit     = self._abbreviate_unit(bm_data.get("unit", ""))
            status   = bm_data.get("status", "unknown").upper()
            lines.append(f'  "{key}": {{"name":"{friendly}","value":{val},"unit":"{unit}","status":"{status}"}}')

        prompt = (
            "You are a medical screening assistant. Return ONLY valid JSON. "
            "For each biomarker in the input object produce a concise patient-friendly explanation "
            "split into three plain-text fields: meaning, causes, guidance.\n\n"
            "INPUT:\n{\n" + ",\n".join(lines) + "\n}\n\n"
            "OUTPUT format (no markdown, no extra keys):\n"
            '{\n  "<biomarker_key>": {"meaning":"...","causes":"...","guidance":"..."},\n  ...\n}'
        )

        try:
            response = self.gemini_client.generate(
                prompt,
                system_instruction=(
                    "You are a medical explanation assistant. "
                    "Respond with valid JSON only. No markdown fences."
                ),
            )
            if not response or not response.text or response.is_mock:
                return {}

            text = response.text.strip()
            # Strip accidental ```json … ``` fences
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)

            raw: Dict = json.loads(text)
            result: Dict[str, Tuple[str, str, str]] = {}
            for key, v in raw.items():
                if isinstance(v, dict):
                    result[key] = (
                        v.get("meaning", ""),
                        v.get("causes", ""),
                        v.get("guidance", ""),
                    )
            return result

        except Exception as exc:
            logger.warning(f"Gemini batch call failed: {exc}")
            return {}

    def _generate_all_biomarkers_explanations_global(
        self,
        system_summaries: Dict[PhysiologicalSystem, Dict[str, Any]],
    ) -> Dict[str, Tuple[str, str, str]]:
        """
        Generate AI explanations for ALL abnormal biomarkers across ALL systems.

        OPTIMISATION vs. original:
        - Splits work into small batches of _LLM_BATCH_SIZE biomarkers each.
        - Dispatches batches concurrently via ThreadPoolExecutor(_LLM_MAX_WORKERS).
        - Falls back silently to hardcoded tuples for any failed batch.

        Returns a flat dict: { "biomarker_key": (meaning, causes, guidance) }
        """
        # Collect all abnormal biomarkers (high / low only)
        all_abnormal: Dict[str, Any] = {}
        for summary in system_summaries.values():
            for bm_name, bm_data in summary.get("biomarkers", {}).items():
                if bm_data.get("status") in ("high", "low"):
                    all_abnormal[bm_name] = bm_data

        if not all_abnormal:
            logger.info("No abnormal biomarkers — skipping AI explanation generation.")
            return {}

        if not self.gemini_client or not self.gemini_client.is_available:
            logger.info("Gemini unavailable — using hardcoded explanations.")
            return {}

        # Chunk the abnormal biomarkers into batches
        items  = list(all_abnormal.items())
        batches = [
            items[i: i + self._LLM_BATCH_SIZE]
            for i in range(0, len(items), self._LLM_BATCH_SIZE)
        ]

        logger.info(
            f"AI explanation engine: {len(items)} abnormal biomarkers "
            f"split into {len(batches)} batch(es) "
            f"with up to {self._LLM_MAX_WORKERS} parallel workers to respect rate limits."
        )

        merged: Dict[str, Tuple[str, str, str]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._LLM_MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(self._call_gemini_batch, batch): batch
                for batch in batches
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    merged.update(future.result())
                except Exception as exc:
                    logger.warning(f"A parallel LLM batch raised: {exc}")

        logger.info(f"AI engine returned explanations for {len(merged)}/{len(items)} biomarkers.")
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        system_results:   Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk:   RiskScore,
        interpretation:   Optional[InterpretationResult] = None,
        trust_envelope:   Optional[TrustEnvelope]        = None,
        patient_id:       str                            = "ANONYMOUS",
        trusted_results:  Optional[Dict[PhysiologicalSystem, Any]] = None,
        rejected_systems: Optional[List[str]]            = None,
        clinical_findings:Optional[List[Any]]            = None,
    ) -> PatientReport:
        """
        Generate an enhanced patient PDF report.

        Args:
            system_results:   Risk results per physiological system (valid systems only).
            composite_risk:   Overall composite risk score.
            interpretation:   Optional LLM interpretation.
            trust_envelope:   Optional trust envelope.
            patient_id:       Anonymised patient identifier.
            trusted_results:  Optional TrustedRiskResult per system (for rejected-system info).
            rejected_systems: Optional list of rejected system names.
            clinical_findings:Optional Clinical Decision Layer findings.

        Returns:
            PatientReport dataclass with pdf_path populated.
        """
        report_id = f"PR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        report = PatientReport(
            report_id          = report_id,
            generated_at       = datetime.now(),
            patient_id         = patient_id,
            overall_risk_level = composite_risk.level,
            overall_risk_score = composite_risk.score,
            overall_confidence = composite_risk.confidence,
        )

        # Build system summaries for valid systems
        for system, result in system_results.items():
            if system == "renal":
                continue
            report.system_summaries[system] = {
                "risk_level":  result.overall_risk.level,
                "risk_score":  result.overall_risk.score,
                "status":      self._get_simple_status(result.overall_risk.level),
                "alerts":      result.alerts,
                "biomarkers":  result.biomarker_summary,
                "explanation": result.overall_risk.explanation,
                "was_rejected":False,
            }

        # Inject rejected systems (marked as incomplete)
        if trusted_results:
            for system, trusted in trusted_results.items():
                if trusted.was_rejected and system not in report.system_summaries:
                    report.system_summaries[system] = {
                        "risk_level":  RiskLevel.LOW,
                        "risk_score":  0.0,
                        "status":      "Assessment Incomplete",
                        "alerts":      [trusted.rejection_reason] if trusted.rejection_reason
                                        else ["Data quality insufficient"],
                        "biomarkers":  {},
                        "explanation": f"Assessment could not be completed: {trusted.rejection_reason}",
                        "was_rejected":True,
                        "caveats":     trusted.caveats,
                    }

        # Interpretation / recommendations / caveats
        if interpretation:
            report.interpretation_summary = interpretation.summary
            report.recommendations        = interpretation.recommendations
            report.caveats                = interpretation.caveats
        else:
            report.recommendations = self._generate_default_recommendations(system_results)
            report.caveats = [
                "This is a screening report, not a medical diagnosis.",
                "Results should be reviewed by a qualified healthcare provider.",
                "Individual results may vary based on age, gender, and other factors.",
            ]

        if rejected_systems:
            report.caveats.insert(
                0,
                f"Note: {len(rejected_systems)} system(s) could not be assessed due to "
                f"data quality issues: {', '.join(rejected_systems)}.",
            )

        if clinical_findings:
            report.clinical_findings = clinical_findings

        # Generate PDF
        if REPORTLAB_AVAILABLE:
            # *** KEY OPTIMISATION: one parallel AI call before PDF build ***
            global_explanations = self._generate_all_biomarkers_explanations_global(
                report.system_summaries
            )
            pdf_path = self._generate_pdf(report, system_results, trust_envelope, global_explanations)
            report.pdf_path = pdf_path
        else:
            msg = f"PDF generation skipped — reportlab unavailable. Details: {REPORTLAB_ERROR}"
            logger.warning(msg)
            report.pdf_path = msg

        return report

    def generate_bytes(
        self,
        system_results:   Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk:   RiskScore,
        interpretation:   Optional[InterpretationResult] = None,
        trust_envelope:   Optional[TrustEnvelope]        = None,
        patient_id:       str                            = "ANONYMOUS",
        trusted_results:  Optional[Dict[PhysiologicalSystem, Any]] = None,
        rejected_systems: Optional[List[str]]            = None,
        clinical_findings:Optional[List[Any]]            = None,
    ) -> bytes:
        """
        Same as generate() but returns raw PDF bytes (for HTTP streaming).
        Writes to a temporary directory so nothing persists on disk.
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab is not installed; PDF generation unavailable.")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            orig = self.output_dir
            self.output_dir = tmp_dir
            try:
                report = self.generate(
                    system_results   = system_results,
                    composite_risk   = composite_risk,
                    interpretation   = interpretation,
                    trust_envelope   = trust_envelope,
                    patient_id       = patient_id,
                    trusted_results  = trusted_results,
                    rejected_systems = rejected_systems,
                    clinical_findings= clinical_findings,
                )
                if report.pdf_path and os.path.isfile(report.pdf_path):
                    with open(report.pdf_path, "rb") as f:
                        return f.read()
                raise RuntimeError("PDF was not generated correctly.")
            finally:
                self.output_dir = orig

    # ===========================================================================
    # PHASE 4 — PDF Rendering Helpers (grids, zebra rows, zero emojis)
    # ===========================================================================

    # ------------------------------------------------------------------
    # Biomarker measurement table style (the per-system "What We Measured" table)
    # ------------------------------------------------------------------
    def _build_biomarker_table_style(self, table_data: list) -> list:
        """
        TableStyle commands for the per-system biomarker measurement table.

        Changes vs original:
        - GRID lines enabled (0.5pt #E5E7EB) so column boundaries are visible.
        - Zebra-stripe applied to alternating data rows.
        - Status cell still gets its own colour; abnormal row gets an amber tint.
        """
        ROW_TINT  = HexColor("#FFFDF5")   # very light amber for abnormal rows
        GRID_COL  = HexColor("#E5E7EB")

        style = [
            # ── Header ──────────────────────────────────────────────────────────
            ("BACKGROUND",    (0, 0), (-1, 0),  HexColor("#F3F4F6")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  HexColor("#374151")),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  9),
            ("TOPPADDING",    (0, 0), (-1, 0),  10),
            ("BOTTOMPADDING", (0, 0), (-1, 0),  10),
            # ── Grid (all cells) — the key change for visible column lines ─────
            ("GRID",          (0, 0), (-1, -1), 0.5, GRID_COL),
            # ── Data rows ────────────────────────────────────────────────────────
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
            # ── Right-align numeric columns (Value=1, Normal Range=2) ─────────
            ("ALIGN",         (1, 1), (2, -1),  "RIGHT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]

        # Per-row: zebra stripe + status cell colour + abnormal row amber tint
        for i, row in enumerate(table_data[1:], start=1):
            status_text = row[3] if len(row) > 3 else ""
            _, bg_color = self._status_meta(status_text)
            # Zebra stripe on even data rows (i=1,3,5…)
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (2, i), ZEBRA_TINT))
            # Status column pill colour
            style.append(("BACKGROUND", (3, i), (3, i), bg_color))
            # Amber tint on the full row for abnormal readings
            if "Below" in status_text or "Above" in status_text:
                style.append(("BACKGROUND", (0, i), (2, i), ROW_TINT))

        return style

    # ------------------------------------------------------------------
    # Trust envelope section
    # ------------------------------------------------------------------
    def _draw_trust_envelope(self, story: List[Any], trust_envelope: Optional[TrustEnvelope]):
        """Render Data Reliability & Quality section — no emojis."""
        if not trust_envelope or not REPORTLAB_AVAILABLE:
            return

        story.append(Paragraph("Data Reliability & Quality", self._styles["SectionHeader"]))

        score       = trust_envelope.overall_reliability
        status_text = (
            "HIGH RELIABILITY"     if score >= 0.8 else
            "MODERATE RELIABILITY" if score >= 0.5 else
            "LOW RELIABILITY"
        )

        trust_data = [[
            Paragraph(
                f"<b>System Confidence: {score:.0%}</b><br/>"
                f"<font size='8' color='#6B7280'>{status_text}</font>",
                self._styles["BodyText"],
            ),
            ConfidenceMeter(score, width=200),
        ]]
        t = Table(trust_data, colWidths=[3 * inch, 3 * inch])
        t.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",   (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        if trust_envelope.safety_flags:
            flags_text = " | ".join(
                [f.value.replace("_", " ").upper() for f in trust_envelope.safety_flags]
            )
            story.append(Paragraph(f"<b>Safety Observations:</b> {flags_text}", self._styles["Caveat"]))
            story.append(Spacer(1, 5))

        if trust_envelope.interpretation_guidance:
            story.append(Paragraph(
                f"<b>Guidance:</b> {trust_envelope.interpretation_guidance}",
                self._styles["BiomarkerExplanation"],
            ))

        story.append(Spacer(1, 15))

    # ------------------------------------------------------------------
    # Screening Analysis executive summary table
    # ------------------------------------------------------------------
    def _draw_analysis_section(self, story: List[Any], report: "PatientReport") -> None:
        """Render Screening Analysis summary table — no emojis, zebra rows, visible grid."""
        if not REPORTLAB_AVAILABLE or not report.system_summaries:
            return

        story.append(Paragraph("Screening Analysis", self._styles["SectionHeader"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Summary of all assessed body systems, colour-coded by risk level.",
            self._styles["Caveat"],
        ))
        story.append(Spacer(1, 10))

        # Risk level text abbreviations (no emojis)
        RISK_SHORT = {
            RiskLevel.LOW:             "Low",
            RiskLevel.MODERATE:        "Moderate",
            RiskLevel.HIGH:            "High",
            RiskLevel.ACTION_REQUIRED: "Action Req.",
            RiskLevel.UNKNOWN:         "N/A",
        }

        header_row = ["Body System", "Risk Level", "Score", "Key Finding"]
        rows       = [header_row]
        row_bgs    = []

        for system, summary in report.system_summaries.items():
            if system == PhysiologicalSystem.VISUAL_DISEASE:
                continue
            sys_name   = system.value.replace("_", " ").title()
            risk_level = summary.get("risk_level", RiskLevel.LOW)
            risk_score = summary.get("risk_score", 0.0)
            alerts     = summary.get("alerts", [])

            if alerts:
                key_alert = alerts[0]
            elif risk_level == RiskLevel.UNKNOWN:
                key_alert = "Not assessed"
            elif risk_level == RiskLevel.LOW:
                key_alert = "All metrics normal"
            else:
                key_alert = "No anomalies detected"

            label = RISK_SHORT.get(risk_level, "Unknown")
            rows.append([
                Paragraph(f"<b>{sys_name}</b>", self._styles["BodyText"]),
                Paragraph(label,                 self._styles["BodyText"]),
                Paragraph(f"{risk_score:.0f}",   self._styles["BodyText"]),
                Paragraph(key_alert,              self._styles["BiomarkerExplanation"]),
            ])
            row_bgs.append(RISK_COLORS.get(risk_level, HexColor("#F9FAFB")))

        tbl = Table(rows, colWidths=[1.6 * inch, 1.5 * inch, 0.7 * inch, 2.7 * inch])

        tbl_style = TableStyle([
            # Header
            ("BACKGROUND",    (0, 0), (-1, 0), HexColor("#F3F4F6")),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 9),
            ("TOPPADDING",    (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            # Grid — full visible borders
            ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#D1D5DB")),
            # Data rows
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 1), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("ALIGN",         (2, 1), (2, -1),  "RIGHT"),
        ])
        # Per-row background from risk level; zebra-stripe where colour is the same
        for i, color in enumerate(row_bgs, start=1):
            tbl_style.add("BACKGROUND", (0, i), (-1, i), color)

        tbl.setStyle(tbl_style)
        story.append(tbl)
        story.append(Spacer(1, 20))

    # ------------------------------------------------------------------
    # Clinical findings / specialist referral section
    # ------------------------------------------------------------------
    def _draw_clinical_findings_section(
        self, story: List[Any], clinical_findings: List[Any]
    ) -> None:
        """
        Render Specialist Referral Recommendations — clean text badges, no emojis.

        Urgency badge is rendered as plain bold text inside a coloured cell
        (e.g. 'URGENT', 'ROUTINE') rather than a circle emoji.
        """
        if not REPORTLAB_AVAILABLE or not clinical_findings:
            return

        # (badge_text, card_bg, badge_text_color)
        URGENCY_BADGE = {
            "urgent":        ("URGENT",       HexColor("#FEE2E2"), HexColor("#B91C1C")),
            "routine":       ("ROUTINE",      HexColor("#FEF9C3"), HexColor("#92400E")),
            "monitor":       ("MONITOR",      HexColor("#ECFDF5"), HexColor("#065F46")),
            "informational": ("INFO",         HexColor("#EFF6FF"), HexColor("#1D4ED8")),
        }

        story.append(Paragraph("Specialist Referral Recommendations", self._styles["SectionHeader"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "The following patterns were detected during your screening. "
            "These are <b>not diagnoses</b> — they indicate that a specialist review "
            "may be beneficial. Please share this report with your doctor.",
            self._styles["Caveat"],
        ))
        story.append(Spacer(1, 12))

        for finding in clinical_findings:
            urgency_val                        = getattr(finding, "urgency", None)
            urgency_val                        = urgency_val.value if urgency_val else "informational"
            badge_text, card_bg, badge_color   = URGENCY_BADGE.get(
                urgency_val, URGENCY_BADGE["informational"]
            )
            card_elements = []

            # Header row: badge + title
            title_text  = getattr(finding, "title", "Finding")
            system_text = getattr(finding, "system", "")
            if system_text:
                system_text = system_text.replace("_", " ").title()

            badge_style = ParagraphStyle(
                "Badge",
                parent    = self._styles["BodyText"],
                textColor = badge_color,
                fontSize  = 8,
                leading   = 10,
                alignment = TA_CENTER,
            )
            header_data = [[
                Paragraph(f"<b>{badge_text}</b>", badge_style),
                Paragraph(
                    f"<b>{title_text}</b><br/>"
                    f"<font size='8' color='#6B7280'>{system_text}</font>",
                    self._styles["BodyText"],
                ),
            ]]
            header_tbl = Table(header_data, colWidths=[1.0 * inch, 5.5 * inch])
            header_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (0, 0), card_bg),
                ("BACKGROUND",    (1, 0), (1, 0), HexColor("#F9FAFB")),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
            ]))
            card_elements.append(header_tbl)

            # Description
            description = getattr(finding, "description", "")
            if description:
                desc_style = ParagraphStyle(
                    "FindingDesc", parent=self._styles["BodyText"],
                    fontSize=9, leading=13, leftIndent=0,
                )
                desc_data = [[Paragraph(description, desc_style)]]
                desc_tbl  = Table(desc_data, colWidths=[6.5 * inch])
                desc_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#FAFAFA")),
                    ("TOPPADDING",    (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                    ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
                ]))
                card_elements.append(desc_tbl)

            # Referral table
            referrals = getattr(finding, "referrals", [])
            if referrals:
                ref_rows       = [["Specialist", "Reason", "Urgency"]]
                ref_row_colors = []
                for ref in referrals:
                    ref_urgency         = getattr(ref.urgency, "value", "routine") if hasattr(ref, "urgency") else "routine"
                    ref_badge, ref_bg, _= URGENCY_BADGE.get(ref_urgency, URGENCY_BADGE["informational"])
                    spec_style = ParagraphStyle("RefSpec",   parent=self._styles["BodyText"], fontSize=8, leading=11)
                    rsn_style  = ParagraphStyle("RefReason", parent=self._styles["BodyText"], fontSize=8, leading=11)
                    ref_rows.append([
                        Paragraph(f"<b>{ref.specialist}</b>", spec_style),
                        Paragraph(ref.reason,                  rsn_style),
                        ref_badge,
                    ])
                    ref_row_colors.append(ref_bg)

                ref_tbl   = Table(ref_rows, colWidths=[1.8 * inch, 3.8 * inch, 0.9 * inch])
                ref_style = TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0), HexColor("#F3F4F6")),
                    ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",      (0, 0), (-1, 0), 8),
                    ("TOPPADDING",    (0, 0), (-1, 0), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                    ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE",      (0, 1), (-1, -1), 8),
                    ("TOPPADDING",    (0, 1), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                    ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ])
                for i, color in enumerate(ref_row_colors, start=1):
                    ref_style.add("BACKGROUND", (2, i), (2, i), color)
                ref_tbl.setStyle(ref_style)

                wrapper = Table([[ref_tbl]], colWidths=[6.5 * inch])
                wrapper.setStyle(TableStyle([
                    ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                    ("TOPPADDING",    (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#FAFAFA")),
                ]))
                card_elements.append(wrapper)

            story.append(KeepTogether(card_elements))
            story.append(Spacer(1, 12))

        story.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # Visual disease probability section
    # ------------------------------------------------------------------
    def _draw_disease_probability_section(
        self, story: List[Any], system_summaries: Dict
    ) -> None:
        """
        Render Visual Disease Probability section — no emojis, proper grid.
        """
        if not REPORTLAB_AVAILABLE:
            return

        PREFIXES = {
            "skin_lesion_":    "Skin Lesion Detection",
            "eye_disease_":    "Eye Disease Detection",
            "conjunctivitis_": "Conjunctivitis Detection",
            "measles_":        "Measles Detection",
        }

        visual_summary    = system_summaries.get(PhysiologicalSystem.VISUAL_DISEASE, {})
        visual_biomarkers = visual_summary.get("biomarkers", {})
        if not visual_biomarkers:
            return

        grouped: Dict[str, list] = {title: [] for title in PREFIXES.values()}
        try:
            from app.core.extraction.visual_classification import CLASS_DISPLAY_NAMES, HEALTHY_CLASSES as _HEALTHY_CLASSES
        except ImportError:
            CLASS_DISPLAY_NAMES = {}
            _HEALTHY_CLASSES = {"normal_eye", "not_affected", "normal", "nv"}

        for bm_name, bm_data in visual_biomarkers.items():
            for prefix, title in PREFIXES.items():
                if bm_name.startswith(prefix):
                    raw_class = bm_name[len(prefix):]
                    label     = CLASS_DISPLAY_NAMES.get(raw_class, raw_class.replace("_", " ").title())
                    grouped[title].append({
                        "label":     label,
                        "raw_class": raw_class,
                        "value":     float(bm_data.get("value", 0.0)),
                        "status":    bm_data.get("status", "normal"),
                    })

        if not any(grouped.values()):
            return

        story.append(Paragraph("Disease Prediction", self._styles["SectionHeader"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "These results are generated by visual AI models analysing your webcam image. "
            "They are <b>not a clinical diagnosis</b> and should be reviewed by a qualified "
            "healthcare professional.",
            self._styles["Caveat"],
        ))
        story.append(Spacer(1, 12))

        for group_title, items in grouped.items():
            if not items:
                continue
            story.append(Paragraph(f"<b>{group_title}</b>", self._styles["SubHeader"]))
            story.append(Spacer(1, 6))

            items_sorted = sorted(items, key=lambda x: x["value"], reverse=True)
            table_data   = [["Condition", "AI Confidence", "Assessment"]]
            row_colors   = []

            for item in items_sorted:
                pct       = item["value"]
                label     = item["label"]
                raw_class = item.get("raw_class", "")
                pct_str   = f"{pct * 100:.1f}%"
                is_healthy = raw_class in _HEALTHY_CLASSES

                if is_healthy:
                    # Healthy class — high confidence = good
                    if pct > 0.6:
                        assessment = f"{pct_str} — Healthy"
                        row_colors.append(HexColor("#ECFDF5"))
                    elif pct > 0.4:
                        assessment = f"{pct_str} — Likely Healthy"
                        row_colors.append(HexColor("#ECFDF5"))
                    else:
                        assessment = f"{pct_str} — Inconclusive"
                        row_colors.append(HexColor("#FEF9C3"))
                else:
                    # Disease class — high confidence = concern
                    if pct > 0.6:
                        assessment = f"{pct_str} — High, Consult Doctor"
                        row_colors.append(HexColor("#FEE2E2"))
                    elif pct > 0.4:
                        assessment = f"{pct_str} — Moderate, Monitor"
                        row_colors.append(HexColor("#FEF9C3"))
                    else:
                        assessment = f"{pct_str} — Low, Likely Normal"
                        row_colors.append(HexColor("#ECFDF5"))

                table_data.append([label, pct_str, assessment])

            tbl   = Table(table_data, colWidths=[2.8 * inch, 1.1 * inch, 2.6 * inch])
            style = TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), HexColor("#F3F4F6")),
                ("TEXTCOLOR",     (0, 0), (-1, 0), HexColor("#374151")),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0), 9),
                ("TOPPADDING",    (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#D1D5DB")),
                ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",      (0, 1), (-1, -1), 9),
                ("TOPPADDING",    (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
            ])
            for i, color in enumerate(row_colors, start=1):
                style.add("BACKGROUND", (2, i), (2, i), color)
                if i % 2 == 0:
                    style.add("BACKGROUND", (0, i), (1, i), ZEBRA_TINT)

            tbl.setStyle(style)
            story.append(tbl)
            story.append(Spacer(1, 14))

        story.append(Spacer(1, 10))

    # ===========================================================================
    # PHASE 5 — Main PDF Assembly (_generate_pdf)
    # ===========================================================================

    def _generate_pdf(
        self,
        report:              PatientReport,
        system_results:      Dict[PhysiologicalSystem, SystemRiskResult],
        trust_envelope:      Optional[TrustEnvelope],
        global_explanations: Dict[str, Tuple[str, str, str]],
    ) -> str:
        """
        Build and save the PDF report.

        Key improvements vs original:
        - No emojis in any heading or section title.
        - "What It Means" rendered as a proper 4-column table
          (Biomarker | Meaning | Potential Causes | Guidance) for scannability.
        - AI explanations used directly as (meaning, causes, guidance) tuples.
        - Hardcoded fallback returns same tuple format — zero HTML cleanup.
        """
        filename = f"{report.report_id}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize     = letter,
            rightMargin  = 1.0 * inch,
            leftMargin   = 1.0 * inch,
            topMargin    = 1.0 * inch,
            bottomMargin = 1.0 * inch,
        )

        # Shared paragraph style for the explanation table cells
        cell_style = ParagraphStyle(
            "ExplCell",
            parent   = self._styles.get("TableCell", self._styles["Normal"]),
            fontSize = 8,
            leading  = 12,
            textColor= HexColor("#374151"),
        )
        hdr_cell_style = ParagraphStyle(
            "ExplHdrCell",
            parent    = cell_style,
            fontName  = "Helvetica-Bold",
            fontSize  = 8,
            textColor = HexColor("#374151"),
        )

        story = []

        # ── Title ────────────────────────────────────────────────────────────────
        story.append(Paragraph("Your Health Screening Report", self._styles["CustomTitle"]))
        story.append(Paragraph(
            f"Report ID: <b>{report.report_id}</b> | "
            f"Generated: {report.generated_at.strftime('%B %d, %Y at %I:%M %p')}",
            self._styles["Caveat"],
        ))
        story.append(Spacer(1, 25))

        # ── Overall Assessment ────────────────────────────────────────────────────
        story.append(Paragraph("Your Overall Health Assessment", self._styles["SectionHeader"]))
        story.append(Spacer(1, 15))

        if report.system_summaries:
            story.append(HealthStatsChart(
                report.system_summaries, report.overall_risk_level, width=450, height=140
            ))
        else:
            story.append(RiskIndicator(report.overall_risk_level, width=300, height=45))

        story.append(Spacer(1, 15))
        story.append(Paragraph(
            f"Assessment Confidence: <b>{report.overall_confidence:.0%}</b>",
            self._styles["BodyText"],
        ))
        story.append(Spacer(1, 4))
        story.append(ConfidenceMeter(report.overall_confidence, width=300))
        story.append(Spacer(1, 15))

        if trust_envelope:
            self._draw_trust_envelope(story, trust_envelope)

        story.append(Paragraph(
            f"We assessed <b>{len(report.system_summaries)} body system(s)</b> during your screening. "
            "Below you will find detailed results for each system, "
            "including the specific measurements taken.",
            self._styles["BodyText"],
        ))
        story.append(Spacer(1, 30))

        # ── Detailed per-system results ───────────────────────────────────────────
        story.append(Paragraph("Your Results in Detail", self._styles["SectionHeader"]))
        story.append(Spacer(1, 15))

        for system, summary in report.system_summaries.items():
            if system == PhysiologicalSystem.VISUAL_DISEASE:
                continue

            system_name = system.value.replace("_", " ").title()
            if system in [PhysiologicalSystem.NASAL]:
                system_name += " (Experimental)"

            risk_level = summary["risk_level"]
            biomarkers = summary.get("biomarkers", {})

            try:
                logger.info(f"Building PDF section: {system_name} ({len(biomarkers)} biomarkers)")
            except Exception:
                pass

            # ── Header + measurement table (KeepTogether so it doesn't split) ───
            header_group = []
            header_group.append(Paragraph(f"<b>{system_name}</b>", self._styles["SubHeader"]))
            header_group.append(Spacer(1, 6))
            header_group.append(RiskIndicator(risk_level=risk_level, width=200, height=28))
            header_group.append(Spacer(1, 10))

            if biomarkers:
                table_data = [["What We Measured", "Your Value", "Normal Range", "Status"]]
                for bm_name, bm_data in biomarkers.items():
                    is_stationary_gait = (
                        bm_name == "gait_variability"
                        and bm_data.get("status") == "not_assessed"
                    )
                    friendly_name = self._simplify_biomarker_name(bm_name)
                    if is_stationary_gait:
                        friendly_name += " (Stationary)"

                    value = bm_data["value"]
                    if isinstance(value, (int, float)):
                        value = round(value, 2)
                    unit         = self._abbreviate_unit(bm_data.get("unit", ""))
                    value_str    = f"{value} {unit}".strip()
                    normal_range = self._format_normal_range(bm_data.get("normal_range"))
                    status       = bm_data.get("status", "not_assessed")
                    status_icon  = self._get_biomarker_status_icon(status)
                    if is_stationary_gait:
                        status_icon = "Not Assessed (Stationary)"

                    table_data.append([friendly_name, value_str, normal_range, status_icon])

                bm_table = Table(table_data, colWidths=[2.2 * inch, 1.3 * inch, 1.2 * inch, 1.8 * inch])
                bm_table.setStyle(TableStyle(self._build_biomarker_table_style(table_data)))
                header_group.append(bm_table)
                header_group.append(Spacer(1, 16))

            story.append(KeepTogether(header_group))

            # ── "What It Means" — 4-column table ─────────────────────────────────
            if biomarkers:
                story.append(Paragraph(
                    "<b>What It Means:</b>",
                    self._styles["BodyText"],
                ))
                story.append(Spacer(1, 6))

                # Header row for the explanation table
                expl_rows = [[
                    Paragraph("<b>Biomarker</b>",       hdr_cell_style),
                    Paragraph("<b>Meaning</b>",         hdr_cell_style),
                    Paragraph("<b>Potential Causes</b>",hdr_cell_style),
                    Paragraph("<b>Guidance</b>",        hdr_cell_style),
                ]]

                for bm_name, bm_data in biomarkers.items():
                    status = bm_data.get("status", "not_assessed")
                    # Include all except truly not_assessed (except gait special case)
                    if status == "not_assessed" and bm_name != "gait_variability":
                        continue

                    friendly_name = self._simplify_biomarker_name(bm_name)
                    value         = bm_data.get("value", 0)

                    # Try AI-generated tuple first, then fall back to hardcoded
                    ai_tuple = global_explanations.get(bm_name)
                    if ai_tuple and isinstance(ai_tuple, dict):
                        # AI returned a dict; normalise to tuple
                        meaning  = ai_tuple.get("meaning", "")
                        causes   = ai_tuple.get("causes", "")
                        guidance = ai_tuple.get("guidance", "")
                    elif ai_tuple and isinstance(ai_tuple, tuple) and len(ai_tuple) == 3:
                        meaning, causes, guidance = ai_tuple
                    elif ai_tuple and isinstance(ai_tuple, str):
                        # Legacy HTML blob from old code path — parse it
                        meaning, causes, guidance = _parse_explanation_parts(ai_tuple)
                    else:
                        meaning, causes, guidance = self._get_explanation_tuple(bm_name, value, status)

                    expl_rows.append([
                        Paragraph(friendly_name, cell_style),
                        Paragraph(meaning or "—",  cell_style),
                        Paragraph(causes  or "—",  cell_style),
                        Paragraph(guidance or "—", cell_style),
                    ])

                if len(expl_rows) > 1:          # at least one data row
                    expl_tbl = Table(
                        expl_rows,
                        colWidths=[1.4 * inch, 1.85 * inch, 1.65 * inch, 1.6 * inch],
                    )
                    expl_style = TableStyle([
                        # Header
                        ("BACKGROUND",    (0, 0), (-1, 0), HexColor("#F3F4F6")),
                        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE",      (0, 0), (-1, 0), 8),
                        ("TOPPADDING",    (0, 0), (-1, 0), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        # Full grid
                        ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#D1D5DB")),
                        # Data rows
                        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE",      (0, 1), (-1, -1), 8),
                        ("TOPPADDING",    (0, 1), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
                        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                    ])
                    # Zebra-stripe data rows
                    for i in range(1, len(expl_rows)):
                        if i % 2 == 0:
                            expl_style.add("BACKGROUND", (0, i), (-1, i), ZEBRA_TINT)
                    expl_tbl.setStyle(expl_style)
                    story.append(expl_tbl)

            # ── Alerts ────────────────────────────────────────────────────────────
            if summary.get("alerts"):
                story.append(Spacer(1, 10))
                story.append(Paragraph("<b>Important Notes:</b>", self._styles["SubHeader"]))
                for alert in summary["alerts"][:3]:
                    story.append(Paragraph(
                        f"• {alert}", self._styles["BiomarkerExplanation"]
                    ))

            story.append(Spacer(1, 25))

        # ── Screening Analysis (executive summary) ────────────────────────────────
        self._draw_analysis_section(story, report)
        story.append(SectionDivider())
        story.append(Spacer(1, 20))

        # ── Visual Disease Probability ─────────────────────────────────────────────
        self._draw_disease_probability_section(story, report.system_summaries)

        # ── Specialist Referral Recommendations ───────────────────────────────────
        if report.clinical_findings:
            story.append(SectionDivider())
            story.append(Spacer(1, 10))
            self._draw_clinical_findings_section(story, report.clinical_findings)

        # ── Recommendations ───────────────────────────────────────────────────────
        story.append(SectionDivider())
        story.append(Spacer(1, 10))
        story.append(Paragraph("What You Should Do Next", self._styles["SectionHeader"]))
        story.append(Spacer(1, 10))
        for i, rec in enumerate(report.recommendations[:6], 1):
            story.append(Paragraph(f"{i}. {rec}", self._styles["BodyText"]))
        story.append(Spacer(1, 25))

        # ── Important Notes ───────────────────────────────────────────────────────
        story.append(SectionDivider())
        story.append(Spacer(1, 10))
        story.append(Paragraph("Important Information", self._styles["SectionHeader"]))
        story.append(Spacer(1, 10))
        for caveat in report.caveats:
            story.append(Paragraph(f"• {caveat}", self._styles["BodyText"]))

        # ── Footer disclaimer ─────────────────────────────────────────────────────
        story.append(Spacer(1, 30))
        story.append(SectionDivider())
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> This health screening report is for informational purposes only "
            "and does not constitute medical advice, diagnosis, or treatment. Always consult with "
            "a qualified healthcare provider for proper medical evaluation and personalised advice. "
            "Do not disregard professional medical advice or delay seeking it based on this report.",
            self._styles["Caveat"],
        ))

        doc.build(story)
        logger.info(f"Optimised patient report generated: {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
PatientReportGenerator = EnhancedPatientReportGenerator

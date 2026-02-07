"""
Enhanced Doctor Report Generator V3 (Playwright)
Generates detailed clinical PDF reports using HTML templates and Playwright for rendering.
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
from app.core.agents.medical_agents import ConsensusResult
from app.utils import get_logger

logger = get_logger(__name__)

@dataclass
class DoctorReportV3:
    """Data container for clinical report."""
    report_id: str
    generated_at: datetime
    patient_id: str = "ANONYMOUS"
    overall_risk_level: str = "LOW"
    overall_risk_score: float = 0.0
    overall_confidence: float = 0.0
    system_details: Dict[PhysiologicalSystem, Dict[str, Any]] = field(default_factory=dict)
    trust_envelope_data: Dict[str, Any] = field(default_factory=dict)
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    all_alerts: List[str] = field(default_factory=list)
    pdf_path: Optional[str] = None

class DoctorReportGeneratorV3:
    """
    Generates detailed, clinical PDF health screening reports using Playwright.
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
        
        logger.info(f"DoctorReportGeneratorV3 initialized, output: {output_dir}")

    def generate(
        self,
        system_results: Dict[PhysiologicalSystem, SystemRiskResult],
        composite_risk: RiskScore,
        trust_envelope: Optional[TrustEnvelope] = None,
        interpretation: Optional[InterpretationResult] = None,
        validation_result: Optional[ConsensusResult] = None,
        patient_id: str = "ANONYMOUS"
    ) -> DoctorReportV3:
        """
        Generate an enhanced clinical PDF report V3.
        """
        report_id = f"DR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        report = DoctorReportV3(
            report_id=report_id,
            generated_at=datetime.now(),
            patient_id=patient_id,
            overall_risk_level=composite_risk.level.value,
            overall_risk_score=composite_risk.score,
            overall_confidence=composite_risk.confidence
        )
        
        # Build System Details
        for system, result in system_results.items():
            # Format biomarkers for display if needed, but Doctor report usually shows raw data.
            # V1 just passed them through via 'biomarker_summary'.
            report.system_details[system] = {
                "risk_level": result.overall_risk.level.value,
                "risk_score": result.overall_risk.score,
                "biomarker_summary": result.biomarker_summary,
                "alerts": result.alerts
            }
            # Collect alerts
            report.all_alerts.extend([f"[{system.value}] {a}" for a in result.alerts])
            
        # Trust Envelope Data (V1 Logic - Enhanced fields)
        if trust_envelope:
            report.trust_envelope_data = {
                "overall_reliability": trust_envelope.overall_reliability,
                "data_quality_score": trust_envelope.data_quality_score,
                "biomarker_plausibility": trust_envelope.biomarker_plausibility,
                "cross_system_consistency": trust_envelope.cross_system_consistency,
                "confidence_penalty": getattr(trust_envelope, 'confidence_penalty', 0.0), # Ensure safety if attribute missing in some versions
                "safety_flags": [f.value for f in trust_envelope.safety_flags],
                "is_reliable": getattr(trust_envelope, 'is_reliable', True)
            }
            
        # Validation Summary (V1 Logic)
        if validation_result:
            report.validation_summary = {
                "overall_status": validation_result.overall_status.value,
                "agent_agreement": validation_result.agent_agreement,
                "overall_confidence": getattr(validation_result, 'overall_confidence', 0.0),
                "flag_count": len(validation_result.combined_flags) if hasattr(validation_result, 'combined_flags') else 0,
                "requires_review": getattr(validation_result, 'requires_human_review', False),
                "recommendation": validation_result.recommendation
            }

        # Generate PDF
        if PLAYWRIGHT_AVAILABLE:
            pdf_path = self._render_pdf(report)
            report.pdf_path = pdf_path
        else:
            logger.error("Playwright not available.")
            report.pdf_path = None
            
        return report

    def _render_pdf(self, report: DoctorReportV3) -> str:
        """Render HTML and convert to PDF using Playwright (sync)."""
        template = self.jinja_env.get_template("doctor_report.html")
        html_content = template.render(report=report)
        
        filename = f"{report.report_id}_v3.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Use a helper to isolate sync playwright in its own thread/event loop
        self._render_pdf_sync(html_content, filepath)
        logger.info(f"Clinical PDF generated successfully at {filepath}")
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
    
    def _confidence_interpretation(self, confidence: float) -> str:
        """Interpret confidence level (V1 Logic)."""
        if confidence >= 0.9:
            return "High - Results highly reliable"
        elif confidence >= 0.7:
            return "Moderate - Interpret with standard caution"
        elif confidence >= 0.5:
            return "Low - Additional verification recommended"
        else:
            return "Very Low - Results require confirmation"
    
    def _score_status(self, score: float) -> str:
        """Convert score to status text (V1 Logic)."""
        if score >= 0.9:
            return "✓ Excellent"
        elif score >= 0.7:
            return "✓ Good"
        elif score >= 0.5:
            return "⚠ Fair"
        else:
            return "⚠ Poor"

"""
Unit Tests for Report Generation (Phase 8)

Tests for patient and doctor PDF report generators.
"""
import pytest
import os
import tempfile
from typing import Dict, Any
from datetime import datetime

from app.core.reports import (
    PatientReportGenerator, PatientReport,
    DoctorReportGenerator, DoctorReport
)
from app.core.inference.risk_engine import SystemRiskResult, RiskScore, RiskLevel
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope, SafetyFlag
from app.core.llm.risk_interpreter import InterpretationResult
from app.core.agents.medical_agents import (
    ConsensusResult, ValidationStatus, ValidationFlag, FlagSeverity
)


# Check if reportlab is available
try:
    import reportlab
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# Fixtures
@pytest.fixture
def mock_risk_score() -> RiskScore:
    """Create mock composite risk score."""
    return RiskScore(
        name="composite_overall",
        score=42.0,
        confidence=0.85,
        explanation="Moderate overall risk detected."
    )


@pytest.fixture
def mock_system_results() -> Dict[PhysiologicalSystem, SystemRiskResult]:
    """Create mock system results."""
    results = {}
    
    # Cardiovascular
    cv_risk = RiskScore(
        name="cardiovascular_overall",
        score=35.0,
        confidence=0.9,
        explanation="Low cardiovascular risk."
    )
    results[PhysiologicalSystem.CARDIOVASCULAR] = SystemRiskResult(
        system=PhysiologicalSystem.CARDIOVASCULAR,
        overall_risk=cv_risk,
        biomarker_summary={
            "heart_rate": {"value": 72, "unit": "bpm", "status": "normal"},
            "hrv_rmssd": {"value": 45, "unit": "ms", "status": "normal"}
        },
        alerts=[]
    )
    
    # CNS
    cns_risk = RiskScore(
        name="cns_overall",
        score=50.0,
        confidence=0.8,
        explanation="Moderate CNS indicators."
    )
    results[PhysiologicalSystem.CNS] = SystemRiskResult(
        system=PhysiologicalSystem.CNS,
        overall_risk=cns_risk,
        biomarker_summary={
            "gait_variability": {"value": 0.08, "status": "borderline"},
            "posture_entropy": {"value": 1.5, "status": "normal"}
        },
        alerts=["Gait variability slightly elevated"]
    )
    
    return results


@pytest.fixture
def mock_trust_envelope() -> TrustEnvelope:
    """Create mock trust envelope."""
    return TrustEnvelope(
        overall_reliability=0.85,
        data_quality_score=0.9,
        biomarker_plausibility=0.88,
        cross_system_consistency=0.82,
        confidence_penalty=0.1,
        safety_flags=[SafetyFlag.LOW_CONFIDENCE]
    )


@pytest.fixture
def mock_interpretation() -> InterpretationResult:
    """Create mock interpretation result."""
    return InterpretationResult(
        system=None,
        summary="Your health screening shows mostly positive results.",
        detailed_explanation="Overall assessment indicates low to moderate risk.",
        recommendations=[
            "Maintain regular exercise routine.",
            "Schedule annual health checkups.",
            "Consult a healthcare professional for detailed evaluation."
        ],
        caveats=[
            "This is a screening result, not a diagnosis.",
            "Results should be verified by a healthcare provider."
        ]
    )


@pytest.fixture
def mock_validation_result() -> ConsensusResult:
    """Create mock validation consensus result."""
    return ConsensusResult(
        overall_status=ValidationStatus.PLAUSIBLE,
        overall_confidence=0.82,
        agent_agreement=0.9,
        combined_flags=[],
        agent_results={},
        recommendation="Validation passed. Results appear plausible."
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPatientReportGenerator:
    """Tests for PatientReportGenerator."""
    
    def test_init(self, temp_output_dir):
        """Test generator initialization."""
        generator = PatientReportGenerator(output_dir=temp_output_dir)
        assert os.path.exists(temp_output_dir)
    
    def test_generate_report_data(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test patient report data generation."""
        generator = PatientReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        assert isinstance(report, PatientReport)
        assert report.report_id.startswith("PR-")
        assert report.overall_risk_level == mock_risk_score.level
        assert len(report.system_summaries) == 2
    
    def test_generate_with_interpretation(
        self, temp_output_dir, mock_system_results, mock_risk_score, mock_interpretation
    ):
        """Test report with LLM interpretation."""
        generator = PatientReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score,
            interpretation=mock_interpretation
        )
        
        assert report.interpretation_summary == mock_interpretation.summary
        assert len(report.recommendations) > 0
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_generate_pdf(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test PDF file generation."""
        generator = PatientReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        assert report.pdf_path is not None
        assert os.path.exists(report.pdf_path)
        assert report.pdf_path.endswith(".pdf")
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_pdf_file_size(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test PDF file has reasonable size."""
        generator = PatientReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        if report.pdf_path:
            file_size = os.path.getsize(report.pdf_path)
            assert file_size > 1000  # At least 1KB
            assert file_size < 5_000_000  # Less than 5MB
    
    def test_report_serialization(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test report can be serialized."""
        generator = PatientReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        report_dict = report.to_dict()
        
        assert "report_id" in report_dict
        assert "overall_risk_level" in report_dict
        assert "system_count" in report_dict


class TestDoctorReportGenerator:
    """Tests for DoctorReportGenerator."""
    
    def test_init(self, temp_output_dir):
        """Test generator initialization."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        assert os.path.exists(temp_output_dir)
    
    def test_generate_report_data(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test doctor report data generation."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        assert isinstance(report, DoctorReport)
        assert report.report_id.startswith("DR-")
        assert len(report.system_details) == 2
        assert len(report.all_biomarkers) > 0
    
    def test_generate_with_trust_envelope(
        self, temp_output_dir, mock_system_results, mock_risk_score, mock_trust_envelope
    ):
        """Test report with trust envelope."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score,
            trust_envelope=mock_trust_envelope
        )
        
        assert "overall_reliability" in report.trust_envelope_data
        assert report.trust_envelope_data["overall_reliability"] == 0.85
    
    def test_generate_with_validation(
        self, temp_output_dir, mock_system_results, mock_risk_score, mock_validation_result
    ):
        """Test report with validation results."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score,
            validation_result=mock_validation_result
        )
        
        assert "overall_status" in report.validation_summary
        assert report.validation_summary["agent_agreement"] == 0.9
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_generate_pdf(
        self, temp_output_dir, mock_system_results, mock_risk_score, 
        mock_trust_envelope, mock_validation_result
    ):
        """Test PDF file generation with all data."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score,
            trust_envelope=mock_trust_envelope,
            validation_result=mock_validation_result
        )
        
        assert report.pdf_path is not None
        assert os.path.exists(report.pdf_path)
        assert report.pdf_path.endswith(".pdf")
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_pdf_file_size(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test doctor PDF file has reasonable size."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        if report.pdf_path:
            file_size = os.path.getsize(report.pdf_path)
            assert file_size > 1000  # At least 1KB
            assert file_size < 10_000_000  # Less than 10MB
    
    def test_report_serialization(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test report can be serialized."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        report_dict = report.to_dict()
        
        assert "report_id" in report_dict
        assert "biomarker_count" in report_dict
        assert "alert_count" in report_dict
    
    def test_alerts_collection(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test alerts are collected from all systems."""
        generator = DoctorReportGenerator(output_dir=temp_output_dir)
        report = generator.generate(
            mock_system_results,
            mock_risk_score
        )
        
        # Should have collected alerts from systems that have them
        # CNS has one alert about gait variability
        assert len(report.all_alerts) >= 1
        # At least one alert should mention gait
        assert any("gait" in alert.lower() for alert in report.all_alerts)


class TestRiskLevelColors:
    """Test risk level color mapping."""
    
    def test_all_risk_levels_have_colors(self):
        """Verify all risk levels have assigned colors."""
        from app.core.reports.patient_report import RISK_COLORS
        
        for level in RiskLevel:
            assert level in RISK_COLORS
    
    def test_all_risk_levels_have_labels(self):
        """Verify all risk levels have assigned labels."""
        from app.core.reports.patient_report import RISK_LABELS
        
        for level in RiskLevel:
            assert level in RISK_LABELS


class TestIntegration:
    """Integration tests for report generation."""
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_generate_both_reports(
        self, temp_output_dir, mock_system_results, mock_risk_score,
        mock_trust_envelope, mock_interpretation, mock_validation_result
    ):
        """Test generating both patient and doctor reports."""
        # Patient report
        patient_gen = PatientReportGenerator(output_dir=temp_output_dir)
        patient_report = patient_gen.generate(
            mock_system_results,
            mock_risk_score,
            interpretation=mock_interpretation,
            trust_envelope=mock_trust_envelope
        )
        
        # Doctor report
        doctor_gen = DoctorReportGenerator(output_dir=temp_output_dir)
        doctor_report = doctor_gen.generate(
            mock_system_results,
            mock_risk_score,
            trust_envelope=mock_trust_envelope,
            validation_result=mock_validation_result
        )
        
        # Both should generate successfully
        assert patient_report.pdf_path is not None
        assert doctor_report.pdf_path is not None
        
        # Doctor report should be more detailed (larger)
        if patient_report.pdf_path and doctor_report.pdf_path:
            patient_size = os.path.getsize(patient_report.pdf_path)
            doctor_size = os.path.getsize(doctor_report.pdf_path)
            # Doctor report typically larger due to more data
            assert doctor_size >= patient_size * 0.5  # At least half the size
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_generate_bytes(
        self, temp_output_dir, mock_system_results, mock_risk_score
    ):
        """Test generating PDF as bytes for download."""
        generator = PatientReportGenerator(output_dir=temp_output_dir)
        pdf_bytes = generator.generate_bytes(
            mock_system_results,
            mock_risk_score
        )
        
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        # PDF magic number check
        assert pdf_bytes[:4] == b'%PDF'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

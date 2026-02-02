"""
Unit Tests for LLM Interpretation Module (Phase 6)

Tests for Gemini client, risk interpreter, and context generator.
Verifies non-decisional constraints are enforced.
"""
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from app.core.llm import (
    GeminiClient, GeminiConfig,
    RiskInterpreter, InterpretationResult,
    MedicalContextGenerator, MedicalContext
)
from app.core.llm.gemini_client import GeminiResponse, GeminiModel
from app.core.llm.risk_interpreter import InterpretationAudience
from app.core.inference.risk_engine import SystemRiskResult, RiskScore, RiskLevel
from app.core.extraction.base import PhysiologicalSystem, BiomarkerSet, Biomarker
from app.core.validation.trust_envelope import TrustEnvelope, SafetyFlag


# Fixtures
@pytest.fixture
def mock_risk_result() -> SystemRiskResult:
    """Create a mock system risk result."""
    risk = RiskScore(
        name="cardiovascular_overall",
        score=45.0,
        confidence=0.85,
        explanation="Moderate cardiovascular activity detected."
    )
    return SystemRiskResult(
        system=PhysiologicalSystem.CARDIOVASCULAR,
        overall_risk=risk,
        biomarker_summary={
            "heart_rate": {"value": 78, "status": "normal"},
            "hrv_rmssd": {"value": 42, "status": "borderline"}
        },
        alerts=["HRV slightly below optimal range"]
    )


@pytest.fixture
def mock_trust_envelope() -> TrustEnvelope:
    """Create a mock trust envelope."""
    return TrustEnvelope(
        overall_reliability=0.85,
        data_quality_score=0.9,
        biomarker_plausibility=0.88,
        cross_system_consistency=0.82,
        confidence_penalty=0.1,
        safety_flags=[SafetyFlag.LOW_CONFIDENCE],
        interpretation_guidance="Results are reliable with minor caveats."
    )


@pytest.fixture
def gemini_config() -> GeminiConfig:
    """Create test Gemini config."""
    return GeminiConfig(
        api_key=None,  # Will use mock mode
        model=GeminiModel.FLASH_1_5,
        temperature=0.3
    )


class TestGeminiClient:
    """Tests for GeminiClient."""
    
    def test_init_without_api_key(self, gemini_config):
        """Test client initialization without API key (mock mode)."""
        client = GeminiClient(gemini_config)
        
        # Should be in mock mode
        assert not client.is_available
    
    def test_mock_response_generation(self, gemini_config):
        """Test mock response when API unavailable."""
        client = GeminiClient(gemini_config)
        response = client.generate("Explain the risk assessment.")
        
        assert isinstance(response, GeminiResponse)
        assert response.is_mock
        assert "MOCK" in response.text or "mock" in response.text.lower() or len(response.text) > 0
    
    def test_response_structure(self, gemini_config):
        """Test response has correct structure."""
        client = GeminiClient(gemini_config)
        response = client.generate("Test prompt")
        
        response_dict = response.to_dict()
        
        assert "text" in response_dict
        assert "model" in response_dict
        assert "is_mock" in response_dict
        assert "latency_ms" in response_dict
    
    def test_mock_response_contains_context(self, gemini_config):
        """Test that mock responses contain contextual content."""
        client = GeminiClient(gemini_config)
        
        risk_response = client.generate("Explain the risk level.")
        assert len(risk_response.text) > 50
        
        explain_response = client.generate("Explain the biomarkers.")
        assert len(explain_response.text) > 50


class TestRiskInterpreter:
    """Tests for RiskInterpreter."""
    
    def test_interpret_system_risk(self, mock_risk_result, mock_trust_envelope):
        """Test system risk interpretation."""
        interpreter = RiskInterpreter()
        result = interpreter.interpret_system_risk(
            mock_risk_result,
            mock_trust_envelope,
            InterpretationAudience.PATIENT
        )
        
        assert isinstance(result, InterpretationResult)
        assert result.system == PhysiologicalSystem.CARDIOVASCULAR
        assert result.audience == InterpretationAudience.PATIENT
        assert len(result.summary) > 0 or len(result.detailed_explanation) > 0
    
    def test_interpretation_for_different_audiences(self, mock_risk_result):
        """Test interpretation adapts to different audiences."""
        interpreter = RiskInterpreter()
        
        patient_result = interpreter.interpret_system_risk(
            mock_risk_result,
            audience=InterpretationAudience.PATIENT
        )
        
        clinical_result = interpreter.interpret_system_risk(
            mock_risk_result,
            audience=InterpretationAudience.CLINICAL
        )
        
        # Both should produce valid results
        assert patient_result.audience == InterpretationAudience.PATIENT
        assert clinical_result.audience == InterpretationAudience.CLINICAL
    
    def test_interpretation_includes_caveats(self, mock_risk_result, mock_trust_envelope):
        """Test that interpretation includes appropriate caveats."""
        interpreter = RiskInterpreter()
        result = interpreter.interpret_system_risk(
            mock_risk_result,
            mock_trust_envelope
        )
        
        # Should have caveats due to LOW_CONFIDENCE flag
        assert len(result.caveats) > 0
    
    def test_interpretation_includes_recommendations(self, mock_risk_result):
        """Test that interpretation always includes recommendations."""
        interpreter = RiskInterpreter()
        result = interpreter.interpret_system_risk(mock_risk_result)
        
        # Should always have at least one recommendation
        assert len(result.recommendations) >= 1
    
    def test_non_decisional_constraint(self, mock_risk_result):
        """Verify interpreter does not output diagnoses or scores."""
        interpreter = RiskInterpreter()
        result = interpreter.interpret_system_risk(mock_risk_result)
        
        result_dict = result.to_dict()
        
        # Should NOT contain diagnosis fields
        assert "diagnosis" not in result_dict
        assert "disease" not in result_dict
        assert "treatment" not in result_dict
        
        # Should NOT assign new risk scores
        assert "new_risk_score" not in result_dict
        assert "revised_score" not in result_dict
    
    def test_confidence_statement_generation(self, mock_risk_result, mock_trust_envelope):
        """Test confidence statement is generated properly."""
        interpreter = RiskInterpreter()
        result = interpreter.interpret_system_risk(
            mock_risk_result,
            mock_trust_envelope
        )
        
        assert len(result.confidence_statement) > 0
    
    def test_interpret_composite_risk(self, mock_risk_result, mock_trust_envelope):
        """Test composite risk interpretation."""
        interpreter = RiskInterpreter()
        
        system_results = {
            PhysiologicalSystem.CARDIOVASCULAR: mock_risk_result
        }
        composite_risk = RiskScore(
            name="composite_overall",
            score=42.0,
            confidence=0.80
        )
        
        result = interpreter.interpret_composite_risk(
            system_results,
            composite_risk,
            mock_trust_envelope
        )
        
        assert result.system is None  # Composite
        assert len(result.summary) > 0 or len(result.detailed_explanation) > 0


class TestMedicalContextGenerator:
    """Tests for MedicalContextGenerator."""
    
    def test_generate_context(self):
        """Test medical context generation."""
        generator = MedicalContextGenerator()
        context = generator.generate_context(
            PhysiologicalSystem.CARDIOVASCULAR,
            RiskLevel.MODERATE,
            biomarkers=["heart_rate", "hrv_rmssd"]
        )
        
        assert isinstance(context, MedicalContext)
        assert context.system == PhysiologicalSystem.CARDIOVASCULAR
        assert len(context.system_overview) > 0
    
    def test_context_includes_lifestyle_factors(self):
        """Test that context includes lifestyle factors."""
        generator = MedicalContextGenerator()
        context = generator.generate_context(
            PhysiologicalSystem.CNS,
            RiskLevel.LOW
        )
        
        assert len(context.lifestyle_factors) > 0
    
    def test_context_includes_warning_signs(self):
        """Test that context includes warning signs for elevated risk."""
        generator = MedicalContextGenerator()
        
        high_risk_context = generator.generate_context(
            PhysiologicalSystem.CARDIOVASCULAR,
            RiskLevel.HIGH
        )
        
        assert len(high_risk_context.warning_signs) > 0
    
    def test_context_for_all_systems(self):
        """Test context generation for all physiological systems."""
        generator = MedicalContextGenerator()
        
        for system in PhysiologicalSystem:
            context = generator.generate_context(system, RiskLevel.MODERATE)
            assert context.system == system
            assert len(context.system_overview) > 0
    
    def test_risk_level_meanings(self):
        """Test that all risk levels have defined meanings."""
        generator = MedicalContextGenerator()
        
        for level in RiskLevel:
            context = generator.generate_context(
                PhysiologicalSystem.CNS,
                level
            )
            assert len(context.risk_level_meaning) > 0
    
    def test_biomarker_explanations(self):
        """Test biomarker explanations are provided."""
        generator = MedicalContextGenerator()
        context = generator.generate_context(
            PhysiologicalSystem.CARDIOVASCULAR,
            RiskLevel.MODERATE,
            biomarkers=["heart_rate"],
            use_llm=False  # Use cached explanations
        )
        
        assert "heart_rate" in context.biomarker_explanations


class TestNonDecisionalConstraints:
    """Tests verifying all LLM components are non-decisional."""
    
    def test_interpretation_result_structure(self):
        """Verify InterpretationResult does not have decisional fields."""
        result = InterpretationResult()
        
        # Should have explanation fields
        assert hasattr(result, 'summary')
        assert hasattr(result, 'detailed_explanation')
        assert hasattr(result, 'recommendations')
        assert hasattr(result, 'caveats')
        
        # Should NOT have diagnosis fields
        assert not hasattr(result, 'diagnosis')
        assert not hasattr(result, 'disease')
        assert not hasattr(result, 'condition')
        assert not hasattr(result, 'treatment')
        assert not hasattr(result, 'medication')
    
    def test_medical_context_structure(self):
        """Verify MedicalContext is educational only."""
        context = MedicalContext(system=PhysiologicalSystem.CNS)
        
        # Should have educational fields
        assert hasattr(context, 'system_overview')
        assert hasattr(context, 'general_health_info')
        assert hasattr(context, 'lifestyle_factors')
        
        # Should NOT have clinical decision fields
        assert not hasattr(context, 'diagnosis')
        assert not hasattr(context, 'prescription')
        assert not hasattr(context, 'treatment_plan')
    
    def test_system_instruction_contains_constraints(self):
        """Verify system instruction includes non-decisional constraints."""
        interpreter = RiskInterpreter()
        instruction = interpreter.SYSTEM_INSTRUCTION.lower()
        
        # Should mention constraints
        assert "not" in instruction or "do not" in instruction
        assert "diagnos" in instruction  # diagnosis/diagnose
        assert "explain" in instruction or "interpretation" in instruction


class TestIntegration:
    """Integration tests for LLM module."""
    
    def test_full_interpretation_pipeline(self, mock_risk_result, mock_trust_envelope):
        """Test complete interpretation pipeline."""
        # 1. Create client
        client = GeminiClient()
        
        # 2. Interpret risk
        interpreter = RiskInterpreter(client)
        interpretation = interpreter.interpret_system_risk(
            mock_risk_result,
            mock_trust_envelope,
            InterpretationAudience.PATIENT
        )
        
        # 3. Generate context
        context_gen = MedicalContextGenerator(client)
        context = context_gen.generate_context(
            mock_risk_result.system,
            mock_risk_result.overall_risk.level,
            biomarkers=list(mock_risk_result.biomarker_summary.keys())
        )
        
        # Verify pipeline produces valid outputs
        assert interpretation.system == mock_risk_result.system
        assert context.system == mock_risk_result.system
        assert len(interpretation.summary) > 0 or interpretation.is_mock
        assert len(context.system_overview) > 0
    
    def test_serialization(self, mock_risk_result):
        """Test all results can be serialized."""
        interpreter = RiskInterpreter()
        interpretation = interpreter.interpret_system_risk(mock_risk_result)
        
        context_gen = MedicalContextGenerator()
        context = context_gen.generate_context(
            PhysiologicalSystem.CNS,
            RiskLevel.MODERATE
        )
        
        # Should serialize to dict without errors
        interp_dict = interpretation.to_dict()
        context_dict = context.to_dict()
        
        assert isinstance(interp_dict, dict)
        assert isinstance(context_dict, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

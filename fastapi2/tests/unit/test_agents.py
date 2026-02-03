"""
Unit Tests for Agentic Medical Validation (Phase 7)

Tests for Hugging Face client, MedGemma, OpenBioLLM agents,
and multi-agent consensus. Verifies non-decisional constraints.
"""
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from app.core.agents import (
    HuggingFaceClient, HFConfig,
    MedGemmaAgent, OpenBioLLMAgent,
    AgentConsensus, ValidationResult
)
from app.core.agents.hf_client import HFModel, HFResponse
from app.core.agents.medical_agents import (
    ValidationStatus, FlagSeverity, ValidationFlag, ConsensusResult
)
from app.core.inference.risk_engine import SystemRiskResult, RiskScore, RiskLevel
from app.core.extraction.base import PhysiologicalSystem
from app.core.validation.trust_envelope import TrustEnvelope


# Fixtures
@pytest.fixture
def mock_biomarker_summary() -> Dict[str, Any]:
    """Create mock biomarker summary."""
    return {
        "heart_rate": {"value": 72, "unit": "bpm", "status": "normal"},
        "hrv_rmssd": {"value": 45, "unit": "ms", "status": "normal"},
        "blood_pressure_systolic": {"value": 120, "unit": "mmHg", "status": "normal"}
    }


@pytest.fixture
def mock_risk_result() -> SystemRiskResult:
    """Create mock system risk result."""
    risk = RiskScore(
        name="cardiovascular_overall",
        score=35.0,
        confidence=0.85,
        explanation="Low cardiovascular risk detected."
    )
    return SystemRiskResult(
        system=PhysiologicalSystem.CARDIOVASCULAR,
        overall_risk=risk,
        biomarker_summary={
            "heart_rate": {"value": 72, "status": "normal"},
            "hrv_rmssd": {"value": 45, "status": "normal"}
        },
        alerts=[]
    )


@pytest.fixture
def mock_system_results() -> Dict[PhysiologicalSystem, SystemRiskResult]:
    """Create mock system results for multiple systems."""
    results = {}
    
    for system in [PhysiologicalSystem.CARDIOVASCULAR, PhysiologicalSystem.CNS]:
        risk = RiskScore(
            name=f"{system.value}_overall",
            score=30.0,
            confidence=0.80
        )
        results[system] = SystemRiskResult(
            system=system,
            overall_risk=risk,
            biomarker_summary={"test_biomarker": {"value": 50, "status": "normal"}},
            alerts=[]
        )
    
    return results


@pytest.fixture
def hf_config() -> HFConfig:
    """Create test HF config."""
    return HFConfig(api_key=None)  # Mock mode


class TestHuggingFaceClient:
    """Tests for HuggingFaceClient."""
    
    def test_init_without_api_key(self, hf_config):
        """Test client initialization without API key (mock mode)."""
        client = HuggingFaceClient(hf_config)
        assert not client.is_available
    
    def test_mock_response_generation(self, hf_config):
        """Test mock response when API unavailable."""
        client = HuggingFaceClient(hf_config)
        response = client.generate("Validate these biomarkers.")
        
        assert isinstance(response, HFResponse)
        assert response.is_mock
        assert len(response.text) > 0
    
    def test_response_structure(self, hf_config):
        """Test response has correct structure."""
        client = HuggingFaceClient(hf_config)
        response = client.generate("Test prompt")
        
        response_dict = response.to_dict()
        
        assert "text" in response_dict
        assert "model" in response_dict
        assert "is_mock" in response_dict
        assert "latency_ms" in response_dict
    
    def test_plausibility_mock_response(self, hf_config):
        """Test mock response for plausibility check."""
        client = HuggingFaceClient(hf_config)
        response = client.generate("Check plausibility of biomarkers.")
        
        assert "PLAUSIBLE" in response.text.upper() or "VALID" in response.text.upper()
    
    def test_consistency_mock_response(self, hf_config):
        """Test mock response for consistency check."""
        client = HuggingFaceClient(hf_config)
        response = client.generate("Check consistency across systems.")
        
        assert "CONSIST" in response.text.upper()


class TestMedGemmaAgent:
    """Tests for MedGemmaAgent."""
    
    def test_validate_biomarkers(self, mock_biomarker_summary):
        """Test biomarker validation."""
        agent = MedGemmaAgent()
        result = agent.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        
        assert isinstance(result, ValidationResult)
        assert result.agent_name == "MedGemma"
        assert result.status in ValidationStatus
    
    def test_validation_result_structure(self, mock_biomarker_summary):
        """Test validation result has correct structure."""
        agent = MedGemmaAgent()
        result = agent.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        
        result_dict = result.to_dict()
        
        assert "agent_name" in result_dict
        assert "model_used" in result_dict
        assert "status" in result_dict
        assert "confidence" in result_dict
        assert "flags" in result_dict
        assert "explanation" in result_dict
    
    def test_non_decisional_constraint(self, mock_biomarker_summary):
        """Verify MedGemma does not diagnose."""
        agent = MedGemmaAgent()
        result = agent.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        
        result_dict = result.to_dict()
        
        # Should NOT have diagnosis fields
        assert "diagnosis" not in result_dict
        assert "disease" not in result_dict
        assert "treatment" not in result_dict
        assert "prescription" not in result_dict
    
    def test_system_prompt_constraints(self):
        """Verify system prompt includes non-decisional constraints."""
        agent = MedGemmaAgent()
        prompt = agent.SYSTEM_PROMPT.lower()
        
        assert "not diagnosing" in prompt or "do not" in prompt
        assert "plausibility" in prompt or "plausible" in prompt


class TestOpenBioLLMAgent:
    """Tests for OpenBioLLMAgent."""
    
    def test_validate_consistency(self, mock_system_results):
        """Test consistency validation across systems."""
        agent = OpenBioLLMAgent()
        result = agent.validate_consistency(mock_system_results)
        
        assert isinstance(result, ValidationResult)
        assert result.agent_name == "OpenBioLLM"
        assert result.status in ValidationStatus
    
    def test_validation_with_trust_envelope(self, mock_system_results):
        """Test validation with trust envelope context."""
        agent = OpenBioLLMAgent()
        trust = TrustEnvelope(overall_reliability=0.85)
        
        result = agent.validate_consistency(mock_system_results, trust)
        
        assert isinstance(result, ValidationResult)
    
    def test_non_decisional_constraint(self, mock_system_results):
        """Verify OpenBioLLM does not diagnose."""
        agent = OpenBioLLMAgent()
        result = agent.validate_consistency(mock_system_results)
        
        result_dict = result.to_dict()
        
        # Should NOT have diagnosis fields
        assert "diagnosis" not in result_dict
        assert "condition" not in result_dict
        assert "treatment" not in result_dict
    
    def test_system_prompt_constraints(self):
        """Verify system prompt includes non-decisional constraints."""
        agent = OpenBioLLMAgent()
        prompt = agent.SYSTEM_PROMPT.lower()
        
        assert "not diagnosing" in prompt or "do not" in prompt
        assert "consistency" in prompt


class TestAgentConsensus:
    """Tests for AgentConsensus."""
    
    def test_compute_consensus(self, mock_biomarker_summary, mock_system_results):
        """Test consensus computation from multiple agents."""
        # Get results from both agents
        medgemma = MedGemmaAgent()
        openbio = OpenBioLLMAgent()
        
        medgemma_result = medgemma.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        openbio_result = openbio.validate_consistency(mock_system_results)
        
        # Compute consensus
        consensus = AgentConsensus()
        result = consensus.compute_consensus({
            "MedGemma": medgemma_result,
            "OpenBioLLM": openbio_result
        })
        
        assert isinstance(result, ConsensusResult)
        assert result.overall_status in ValidationStatus
        assert 0 <= result.agent_agreement <= 1
    
    def test_consensus_result_structure(self, mock_biomarker_summary, mock_system_results):
        """Test consensus result has correct structure."""
        medgemma = MedGemmaAgent()
        openbio = OpenBioLLMAgent()
        
        medgemma_result = medgemma.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        openbio_result = openbio.validate_consistency(mock_system_results)
        
        consensus = AgentConsensus()
        result = consensus.compute_consensus({
            "MedGemma": medgemma_result,
            "OpenBioLLM": openbio_result
        })
        
        result_dict = result.to_dict()
        
        assert "overall_status" in result_dict
        assert "overall_confidence" in result_dict
        assert "agent_agreement" in result_dict
        assert "combined_flags" in result_dict
        assert "recommendation" in result_dict
        assert "requires_human_review" in result_dict
    
    def test_empty_results_handling(self):
        """Test handling of empty agent results."""
        consensus = AgentConsensus()
        result = consensus.compute_consensus({})
        
        assert result.overall_status == ValidationStatus.UNCERTAIN
        assert result.agent_agreement == 0.0
    
    def test_recommendation_generation(self, mock_biomarker_summary, mock_system_results):
        """Test that consensus generates recommendations."""
        medgemma = MedGemmaAgent()
        medgemma_result = medgemma.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        
        consensus = AgentConsensus()
        result = consensus.compute_consensus({"MedGemma": medgemma_result})
        
        assert len(result.recommendation) > 0


class TestValidationFlag:
    """Tests for ValidationFlag."""
    
    def test_flag_creation(self):
        """Test validation flag creation."""
        flag = ValidationFlag(
            agent="TestAgent",
            severity=FlagSeverity.WARNING,
            category="test",
            message="Test warning message",
            system=PhysiologicalSystem.CNS
        )
        
        assert flag.agent == "TestAgent"
        assert flag.severity == FlagSeverity.WARNING
    
    def test_flag_serialization(self):
        """Test flag serialization."""
        flag = ValidationFlag(
            agent="TestAgent",
            severity=FlagSeverity.CRITICAL,
            category="plausibility",
            message="Critical issue detected"
        )
        
        flag_dict = flag.to_dict()
        
        assert flag_dict["severity"] == "critical"
        assert flag_dict["agent"] == "TestAgent"


class TestNonDecisionalConstraints:
    """Tests verifying all agents are non-decisional."""
    
    def test_validation_result_no_diagnosis_fields(self):
        """Verify ValidationResult has no diagnosis fields."""
        result = ValidationResult(
            agent_name="Test",
            model_used="test-model"
        )
        
        # Should have validation fields
        assert hasattr(result, 'status')
        assert hasattr(result, 'flags')
        assert hasattr(result, 'explanation')
        
        # Should NOT have clinical decision fields
        assert not hasattr(result, 'diagnosis')
        assert not hasattr(result, 'disease')
        assert not hasattr(result, 'treatment')
        assert not hasattr(result, 'prescription')
    
    def test_consensus_result_no_diagnosis_fields(self):
        """Verify ConsensusResult has no diagnosis fields."""
        result = ConsensusResult(
            overall_status=ValidationStatus.PLAUSIBLE,
            overall_confidence=0.8,
            agent_agreement=0.9,
            combined_flags=[],
            agent_results={},
            recommendation="Test recommendation"
        )
        
        # Should have consensus fields
        assert hasattr(result, 'overall_status')
        assert hasattr(result, 'recommendation')
        
        # Should NOT have clinical decision fields
        assert not hasattr(result, 'diagnosis')
        assert not hasattr(result, 'treatment_plan')


class TestIntegration:
    """Integration tests for agentic validation."""
    
    def test_full_validation_pipeline(self, mock_biomarker_summary, mock_system_results):
        """Test complete validation pipeline."""
        # 1. Create HF client
        client = HuggingFaceClient()
        
        # 2. Run MedGemma validation
        medgemma = MedGemmaAgent(client)
        plausibility_result = medgemma.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        
        # 3. Run OpenBioLLM consistency check
        openbio = OpenBioLLMAgent(client)
        consistency_result = openbio.validate_consistency(mock_system_results)
        
        # 4. Compute consensus
        consensus = AgentConsensus()
        final_result = consensus.compute_consensus({
            "MedGemma": plausibility_result,
            "OpenBioLLM": consistency_result
        })
        
        # Verify pipeline produces valid outputs
        assert plausibility_result.agent_name == "MedGemma"
        assert consistency_result.agent_name == "OpenBioLLM"
        assert final_result.overall_status in ValidationStatus
        assert len(final_result.recommendation) > 0
    
    def test_serialization(self, mock_biomarker_summary):
        """Test all results can be serialized."""
        agent = MedGemmaAgent()
        result = agent.validate_biomarkers(
            mock_biomarker_summary,
            PhysiologicalSystem.CARDIOVASCULAR
        )
        
        consensus = AgentConsensus()
        consensus_result = consensus.compute_consensus({"MedGemma": result})
        
        # Should serialize without errors
        result_dict = result.to_dict()
        consensus_dict = consensus_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert isinstance(consensus_dict, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit Tests for Validation Module (Phase 4)

Tests for signal quality, biomarker plausibility, cross-system consistency, and trust envelope.
"""
import pytest
import numpy as np
from typing import Dict, Any, List

from app.core.validation import (
    SignalQualityAssessor, ModalityQualityScore,
    BiomarkerPlausibilityValidator, PlausibilityResult,
    CrossSystemConsistencyChecker, ConsistencyResult,
    TrustEnvelope, TrustEnvelopeAggregator
)
from app.core.validation.signal_quality import Modality
from app.core.validation.biomarker_plausibility import ViolationType
from app.core.validation.cross_system_consistency import InconsistencyType
from app.core.validation.trust_envelope import SafetyFlag
from app.core.extraction.base import BiomarkerSet, Biomarker, PhysiologicalSystem


# Fixtures
@pytest.fixture
def sample_frames() -> List[np.ndarray]:
    """Generate sample camera frames."""
    frames = []
    for i in range(10):
        # Create synthetic frame with some variation
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def sample_poses() -> List[np.ndarray]:
    """Generate sample pose data."""
    poses = []
    for i in range(10):
        # 33 landmarks x 4 (x, y, z, visibility)
        pose = np.random.rand(33, 4).astype(np.float32)
        pose[:, 3] = 0.8 + 0.2 * np.random.rand(33)  # High visibility
        poses.append(pose)
    return poses


@pytest.fixture
def sample_ris_data() -> np.ndarray:
    """Generate sample RIS bioimpedance data."""
    t = np.linspace(0, 2, 2000)  # 2 seconds at 1000 Hz
    ris = np.zeros((2000, 16))
    
    for ch in range(16):
        # Base + cardiac + respiratory + noise
        base = 500 + ch * 10
        cardiac = 5 * np.sin(2 * np.pi * 1.2 * t)  # 72 bpm
        respiratory = 10 * np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
        noise = np.random.randn(2000) * 0.5
        ris[:, ch] = base + cardiac + respiratory + noise
    
    return ris.astype(np.float32)


@pytest.fixture
def sample_vitals() -> Dict[str, Any]:
    """Generate sample vital signs."""
    return {
        "heart_rate": 72,
        "respiratory_rate": 14,
        "spo2": 98,
        "temperature": 36.8,
        "systolic_bp": 120,
        "diastolic_bp": 80
    }


@pytest.fixture
def normal_cns_biomarkers() -> BiomarkerSet:
    """Create normal CNS biomarker set."""
    bms = BiomarkerSet(system=PhysiologicalSystem.CNS)
    bms.add(Biomarker("gait_variability", 0.05, "cv", 0.85, (0.02, 0.08)))
    bms.add(Biomarker("cns_stability_score", 85, "score", 0.80, (70, 100)))
    bms.add(Biomarker("tremor_resting", 0.02, "psd", 0.75, (0, 0.1)))
    return bms


@pytest.fixture
def abnormal_cardiovascular_biomarkers() -> BiomarkerSet:
    """Create abnormal cardiovascular biomarkers."""
    bms = BiomarkerSet(system=PhysiologicalSystem.CARDIOVASCULAR)
    bms.add(Biomarker("heart_rate", 250, "bpm", 0.90, (60, 100)))  # Impossible
    bms.add(Biomarker("systolic_bp", 120, "mmHg", 0.85, (90, 120)))
    bms.add(Biomarker("diastolic_bp", 130, "mmHg", 0.85, (60, 80)))  # Higher than systolic!
    return bms


class TestSignalQualityAssessor:
    """Tests for SignalQualityAssessor."""
    
    def test_assess_camera(self, sample_frames):
        """Test camera quality assessment."""
        assessor = SignalQualityAssessor()
        score = assessor.assess_camera(sample_frames)
        
        assert score.modality == Modality.CAMERA
        assert 0 <= score.overall_quality <= 1
        assert 0 <= score.continuity <= 1
        assert 0 <= score.noise_level <= 1
    
    def test_assess_camera_empty(self):
        """Test camera with no frames."""
        assessor = SignalQualityAssessor()
        score = assessor.assess_camera([])
        
        assert "No frames provided" in score.issues
        assert score.overall_quality == 0
    
    def test_assess_motion(self, sample_poses):
        """Test motion quality assessment."""
        assessor = SignalQualityAssessor()
        score = assessor.assess_motion(sample_poses)
        
        assert score.modality == Modality.MOTION
        assert 0 <= score.overall_quality <= 1
        assert score.continuity > 0.8  # All poses valid
    
    def test_assess_ris(self, sample_ris_data):
        """Test RIS quality assessment."""
        assessor = SignalQualityAssessor()
        score = assessor.assess_ris(sample_ris_data, sample_rate=1000)
        
        assert score.modality == Modality.RIS
        assert 0 <= score.overall_quality <= 1
        assert score.continuity > 0.9  # No NaN values
        assert score.snr > 0.3  # Should detect physiological content
    
    def test_assess_auxiliary(self, sample_vitals):
        """Test vital signs quality assessment."""
        assessor = SignalQualityAssessor()
        score = assessor.assess_auxiliary(sample_vitals)
        
        assert score.modality == Modality.AUXILIARY
        assert score.continuity == 1.0  # All vitals present
        assert score.noise_level == 1.0  # All in plausible range
    
    def test_assess_auxiliary_missing(self):
        """Test vitals with missing values."""
        assessor = SignalQualityAssessor()
        score = assessor.assess_auxiliary({"heart_rate": 72})
        
        assert score.continuity < 1.0  # Missing vitals
        assert len(score.issues) > 0
    
    def test_assess_all(self, sample_frames, sample_poses, sample_ris_data, sample_vitals):
        """Test assessing all modalities."""
        assessor = SignalQualityAssessor()
        results = assessor.assess_all(
            camera_frames=sample_frames,
            motion_poses=sample_poses,
            ris_data=sample_ris_data,
            vitals=sample_vitals
        )
        
        assert len(results) == 4
        assert Modality.CAMERA in results
        assert Modality.MOTION in results
        assert Modality.RIS in results
        assert Modality.AUXILIARY in results


class TestBiomarkerPlausibilityValidator:
    """Tests for BiomarkerPlausibilityValidator."""
    
    def test_validate_normal(self, normal_cns_biomarkers):
        """Test validating normal biomarkers."""
        validator = BiomarkerPlausibilityValidator()
        result = validator.validate(normal_cns_biomarkers)
        
        assert result.is_valid
        assert result.overall_plausibility > 0.8
        assert len([v for v in result.violations if v.severity >= 0.8]) == 0
    
    def test_validate_impossible_value(self, abnormal_cardiovascular_biomarkers):
        """Test detecting impossible values."""
        validator = BiomarkerPlausibilityValidator()
        result = validator.validate(abnormal_cardiovascular_biomarkers)
        
        assert not result.is_valid  # Should fail due to HR=250
        
        # Check for impossible value violation
        impossible = [v for v in result.violations 
                     if v.violation_type == ViolationType.IMPOSSIBLE_VALUE]
        assert len(impossible) > 0
    
    def test_validate_internal_contradiction(self, abnormal_cardiovascular_biomarkers):
        """Test detecting internal contradictions."""
        validator = BiomarkerPlausibilityValidator()
        result = validator.validate(abnormal_cardiovascular_biomarkers)
        
        # Should catch SBP <= DBP contradiction
        contradictions = [v for v in result.violations 
                         if v.violation_type == ViolationType.INTERNAL_CONTRADICTION]
        assert len(contradictions) > 0
    
    def test_validate_missing_required(self):
        """Test detecting missing required biomarkers."""
        validator = BiomarkerPlausibilityValidator()
        
        # Empty cardiovascular set
        bms = BiomarkerSet(system=PhysiologicalSystem.CARDIOVASCULAR)
        result = validator.validate(bms)
        
        missing = [v for v in result.violations 
                  if v.violation_type == ViolationType.MISSING_REQUIRED]
        assert len(missing) > 0  # heart_rate is required


class TestCrossSystemConsistencyChecker:
    """Tests for CrossSystemConsistencyChecker."""
    
    def test_check_consistent_systems(self):
        """Test with consistent systems."""
        checker = CrossSystemConsistencyChecker()
        
        # Create consistent biomarker sets
        cardio = BiomarkerSet(system=PhysiologicalSystem.CARDIOVASCULAR)
        cardio.add(Biomarker("heart_rate", 72, "bpm", 0.9, (60, 100)))
        cardio.add(Biomarker("hrv_rmssd", 50, "ms", 0.85, (20, 80)))
        
        cns = BiomarkerSet(system=PhysiologicalSystem.CNS)
        cns.add(Biomarker("cns_stability_score", 85, "score", 0.8, (70, 100)))
        cns.add(Biomarker("gait_variability", 0.05, "cv", 0.8, (0.02, 0.08)))
        
        result = checker.check_consistency({
            PhysiologicalSystem.CARDIOVASCULAR: cardio,
            PhysiologicalSystem.CNS: cns
        })
        
        assert result.overall_consistency > 0.8
        assert len(result.inconsistencies) == 0
    
    def test_check_respiratory_mismatch(self):
        """Test detecting respiratory rate mismatch."""
        checker = CrossSystemConsistencyChecker()
        
        nasal = BiomarkerSet(system=PhysiologicalSystem.NASAL)
        nasal.add(Biomarker("respiratory_rate", 12, "breaths/min", 0.9, (12, 20)))
        
        gi = BiomarkerSet(system=PhysiologicalSystem.GASTROINTESTINAL)
        gi.add(Biomarker("abdominal_respiratory_rate", 25, "breaths/min", 0.9, (12, 20)))
        
        result = checker.check_consistency({
            PhysiologicalSystem.NASAL: nasal,
            PhysiologicalSystem.GASTROINTESTINAL: gi
        })
        
        # Should detect the 13 breaths/min difference
        assert len(result.inconsistencies) > 0
        assert any("mismatch" in i.message.lower() for i in result.inconsistencies)


class TestTrustEnvelope:
    """Tests for TrustEnvelope."""
    
    def test_is_reliable(self):
        """Test reliability check."""
        envelope = TrustEnvelope(overall_reliability=0.8)
        assert envelope.is_reliable
        
        envelope = TrustEnvelope(overall_reliability=0.3)
        assert not envelope.is_reliable
    
    def test_requires_caveats(self):
        """Test caveat requirement check."""
        envelope = TrustEnvelope(overall_reliability=0.9)
        assert not envelope.requires_caveats
        
        envelope = TrustEnvelope(
            overall_reliability=0.7,
            safety_flags=[SafetyFlag.LOW_CONFIDENCE, SafetyFlag.DATA_QUALITY_ISSUE]
        )
        assert envelope.requires_caveats
    
    def test_adjusted_confidence(self):
        """Test confidence adjustment."""
        envelope = TrustEnvelope(confidence_penalty=0.3)
        
        original = 0.9
        adjusted = envelope.get_adjusted_confidence(original)
        
        assert adjusted < original
        assert 0.1 <= adjusted <= 0.99


class TestTrustEnvelopeAggregator:
    """Tests for TrustEnvelopeAggregator."""
    
    def test_aggregate_all(self, sample_frames, sample_ris_data, sample_vitals, normal_cns_biomarkers):
        """Test full aggregation."""
        # Get signal quality
        sq_assessor = SignalQualityAssessor()
        signal_quality = sq_assessor.assess_all(
            camera_frames=sample_frames,
            ris_data=sample_ris_data,
            vitals=sample_vitals
        )
        
        # Get plausibility
        pl_validator = BiomarkerPlausibilityValidator()
        plausibility = {
            normal_cns_biomarkers.system: pl_validator.validate(normal_cns_biomarkers)
        }
        
        # Aggregate
        aggregator = TrustEnvelopeAggregator()
        envelope = aggregator.aggregate(
            signal_quality=signal_quality,
            plausibility_results=plausibility
        )
        
        assert envelope.overall_reliability > 0
        assert envelope.data_quality_score > 0
        assert envelope.biomarker_plausibility > 0
        assert envelope.interpretation_guidance != ""
    
    def test_aggregate_with_issues(self, abnormal_cardiovascular_biomarkers):
        """Test aggregation with issues."""
        pl_validator = BiomarkerPlausibilityValidator()
        plausibility = {
            PhysiologicalSystem.CARDIOVASCULAR: pl_validator.validate(abnormal_cardiovascular_biomarkers)
        }
        
        aggregator = TrustEnvelopeAggregator()
        envelope = aggregator.aggregate(plausibility_results=plausibility)
        
        # Should have issues (warnings or critical)
        assert len(envelope.warnings) > 0 or len(envelope.critical_issues) > 0
        # Should have some safety flags set
        assert len(envelope.safety_flags) > 0
        # Confidence penalty should be applied
        assert envelope.confidence_penalty > 0.1
    
    def test_minimal_envelope(self):
        """Test creating minimal envelope."""
        aggregator = TrustEnvelopeAggregator()
        envelope = aggregator.create_minimal_envelope(reliability=0.6)
        
        assert envelope.overall_reliability == 0.6
        assert SafetyFlag.LOW_CONFIDENCE in envelope.safety_flags
        assert "simulated" in envelope.interpretation_guidance.lower()


class TestIntegration:
    """Integration tests for full validation pipeline."""
    
    def test_full_validation_pipeline(
        self, sample_frames, sample_poses, sample_ris_data, 
        sample_vitals, normal_cns_biomarkers
    ):
        """Test complete validation pipeline."""
        # 1. Signal Quality
        sq_assessor = SignalQualityAssessor()
        signal_quality = sq_assessor.assess_all(
            camera_frames=sample_frames,
            motion_poses=sample_poses,
            ris_data=sample_ris_data,
            vitals=sample_vitals
        )
        
        # 2. Biomarker Plausibility
        pl_validator = BiomarkerPlausibilityValidator()
        plausibility = {
            normal_cns_biomarkers.system: pl_validator.validate(normal_cns_biomarkers)
        }
        
        # 3. Cross-System Consistency (would need more systems)
        cs_checker = CrossSystemConsistencyChecker()
        consistency = cs_checker.check_consistency({
            normal_cns_biomarkers.system: normal_cns_biomarkers
        })
        
        # 4. Trust Envelope
        aggregator = TrustEnvelopeAggregator()
        envelope = aggregator.aggregate(
            signal_quality=signal_quality,
            plausibility_results=plausibility,
            consistency_result=consistency
        )
        
        # Validate envelope
        assert envelope.overall_reliability > 0.5
        assert envelope.is_reliable or len(envelope.critical_issues) > 0
        
        # Check serialization
        envelope_dict = envelope.to_dict()
        assert "overall_reliability" in envelope_dict
        assert "safety_flags" in envelope_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

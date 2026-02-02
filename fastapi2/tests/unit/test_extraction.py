"""
Unit Tests for Feature Extraction Module

Tests for all 9 physiological system extractors.
"""
import pytest
import numpy as np
from typing import Dict, Any

from app.core.extraction import (
    BaseExtractor,
    BiomarkerSet,
    CNSExtractor,
    CardiovascularExtractor,
    RenalExtractor,
    GastrointestinalExtractor,
    SkeletalExtractor,
    SkinExtractor,
    EyeExtractor,
    NasalExtractor,
    ReproductiveExtractor,
)
from app.core.extraction.base import Biomarker, PhysiologicalSystem


# Helper fixtures
@pytest.fixture
def sample_pose_sequence() -> list:
    """Generate sample pose sequence (30 frames, 33 landmarks, 4 values)."""
    np.random.seed(42)
    frames = []
    for i in range(60):  # 2 seconds at 30fps
        # Create pose with walking-like motion
        pose = np.random.rand(33, 4).astype(np.float32)
        # Add some temporal structure
        pose[:, 0] += 0.1 * np.sin(i / 10)  # X position
        pose[:, 1] += 0.05 * np.cos(i / 15)  # Y position
        pose[:, 3] = 0.8 + 0.1 * np.random.rand(33)  # Visibility
        frames.append(pose)
    return frames


@pytest.fixture
def sample_ris_data() -> np.ndarray:
    """Generate sample RIS data (1000 samples, 16 channels)."""
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    base = 500 * np.ones((1000, 16))
    
    # Add respiratory modulation
    resp = 20 * np.sin(2 * np.pi * 0.25 * t)[:, np.newaxis]
    # Add cardiac modulation
    cardiac = 5 * np.sin(2 * np.pi * 1.2 * t)[:, np.newaxis]
    
    return (base + resp + cardiac + np.random.randn(1000, 16) * 5).astype(np.float32)


@pytest.fixture
def sample_vital_signs() -> Dict[str, Any]:
    """Generate sample vital signs."""
    return {
        "heart_rate": 72,
        "hrv": 45,
        "systolic_bp": 118,
        "diastolic_bp": 76,
        "respiratory_rate": 15,
        "temperature": 36.6,
        "spo2": 98
    }


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Generate sample video frame."""
    np.random.seed(42)
    return np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)


class TestBiomarkerSet:
    """Tests for BiomarkerSet data structure."""
    
    def test_create_biomarker_set(self):
        """Test creating empty biomarker set."""
        bms = BiomarkerSet(system=PhysiologicalSystem.CNS)
        
        assert bms.system == PhysiologicalSystem.CNS
        assert len(bms.biomarkers) == 0
    
    def test_add_biomarker(self):
        """Test adding biomarkers."""
        bms = BiomarkerSet(system=PhysiologicalSystem.CARDIOVASCULAR)
        bm = Biomarker(
            name="heart_rate",
            value=72.0,
            unit="bpm",
            confidence=0.95,
            normal_range=(60, 100)
        )
        bms.add(bm)
        
        assert len(bms.biomarkers) == 1
        assert bms.get("heart_rate") is not None
        assert bms.get("heart_rate").value == 72.0
    
    def test_biomarker_abnormal_detection(self):
        """Test abnormal value detection."""
        bm_normal = Biomarker("test", 75, "unit", normal_range=(60, 100))
        bm_low = Biomarker("test", 50, "unit", normal_range=(60, 100))
        bm_high = Biomarker("test", 110, "unit", normal_range=(60, 100))
        
        assert bm_normal.is_abnormal() is False
        assert bm_low.is_abnormal() is True
        assert bm_high.is_abnormal() is True
    
    def test_to_feature_vector(self):
        """Test conversion to numpy array."""
        bms = BiomarkerSet(system=PhysiologicalSystem.CNS)
        bms.add(Biomarker("a", 1.0, "unit"))
        bms.add(Biomarker("b", 2.0, "unit"))
        bms.add(Biomarker("c", 3.0, "unit"))
        
        vec = bms.to_feature_vector()
        
        assert vec.shape == (3,)
        assert np.array_equal(vec, np.array([1.0, 2.0, 3.0], dtype=np.float32))


class TestCNSExtractor:
    """Tests for CNS extractor."""
    
    def test_extract_with_pose_data(self, sample_pose_sequence):
        """Test extraction with pose sequence."""
        extractor = CNSExtractor()
        
        result = extractor.extract({"pose_sequence": sample_pose_sequence})
        
        assert result.system == PhysiologicalSystem.CNS
        assert len(result.biomarkers) >= 4
        assert result.get("gait_variability") is not None
        assert result.get("posture_entropy") is not None
    
    def test_extract_without_data(self):
        """Test fallback to simulated values."""
        extractor = CNSExtractor()
        
        result = extractor.extract({})
        
        assert len(result.biomarkers) >= 3
        # Simulated values should have lower confidence
        for bm in result.biomarkers:
            assert bm.confidence <= 0.6


class TestCardiovascularExtractor:
    """Tests for cardiovascular extractor."""
    
    def test_extract_from_vitals(self, sample_vital_signs):
        """Test extraction from vital signs."""
        extractor = CardiovascularExtractor()
        
        result = extractor.extract({"vital_signs": sample_vital_signs})
        
        assert result.get("heart_rate") is not None
        assert result.get("heart_rate").value == 72
        assert result.get("hrv_rmssd") is not None
    
    def test_extract_from_ris(self, sample_ris_data):
        """Test extraction from RIS data."""
        extractor = CardiovascularExtractor()
        
        result = extractor.extract({"ris_data": sample_ris_data})
        
        assert result.get("heart_rate") is not None
        assert result.get("thoracic_impedance") is not None


class TestRenalExtractor:
    """Tests for renal extractor."""
    
    def test_extract_from_ris(self, sample_ris_data):
        """Test extraction from RIS data."""
        extractor = RenalExtractor()
        
        result = extractor.extract({"ris_data": sample_ris_data})
        
        assert result.get("fluid_asymmetry_index") is not None
        assert result.get("total_body_water_proxy") is not None
        assert result.get("extracellular_fluid_ratio") is not None


class TestGastrointestinalExtractor:
    """Tests for GI extractor."""
    
    def test_extract_from_ris(self, sample_ris_data):
        """Test extraction from RIS data."""
        extractor = GastrointestinalExtractor()
        
        result = extractor.extract({"ris_data": sample_ris_data})
        
        assert result.get("abdominal_rhythm_score") is not None
        assert result.get("visceral_motion_variance") is not None


class TestSkeletalExtractor:
    """Tests for skeletal extractor."""
    
    def test_extract_from_pose(self, sample_pose_sequence):
        """Test extraction from pose data."""
        extractor = SkeletalExtractor()
        
        result = extractor.extract({"pose_sequence": sample_pose_sequence})
        
        assert result.get("gait_symmetry_ratio") is not None
        assert result.get("stance_stability_score") is not None
        assert result.get("average_joint_rom") is not None


class TestSkinExtractor:
    """Tests for skin extractor."""
    
    def test_extract_from_frame(self, sample_frame):
        """Test extraction from video frame."""
        extractor = SkinExtractor()
        
        result = extractor.extract({"frames": [sample_frame]})
        
        assert result.get("texture_roughness") is not None
        assert result.get("skin_redness") is not None
        assert result.get("color_uniformity") is not None


class TestEyeExtractor:
    """Tests for eye extractor."""
    
    def test_extract_from_pose(self, sample_pose_sequence):
        """Test extraction from pose data."""
        extractor = EyeExtractor()
        
        result = extractor.extract({"pose_sequence": sample_pose_sequence})
        
        assert result.get("blink_rate") is not None
        assert result.get("gaze_stability_score") is not None
        assert result.get("fixation_duration") is not None


class TestNasalExtractor:
    """Tests for nasal extractor."""
    
    def test_extract_from_pose(self, sample_pose_sequence):
        """Test extraction from pose data."""
        extractor = NasalExtractor()
        
        result = extractor.extract({"pose_sequence": sample_pose_sequence})
        
        assert result.get("breathing_regularity") is not None
    
    def test_extract_from_ris(self, sample_ris_data):
        """Test extraction from RIS data."""
        extractor = NasalExtractor()
        
        result = extractor.extract({"ris_data": sample_ris_data})
        
        assert result.get("respiratory_rate") is not None
        assert result.get("breath_depth_index") is not None


class TestReproductiveExtractor:
    """Tests for reproductive proxy extractor."""
    
    def test_extract_from_vitals(self, sample_vital_signs):
        """Test extraction from vital signs."""
        extractor = ReproductiveExtractor()
        
        result = extractor.extract({"vital_signs": sample_vital_signs})
        
        assert result.get("autonomic_imbalance_index") is not None
        assert result.get("stress_response_proxy") is not None
        # Check disclaimer in metadata
        assert "disclaimer" in result.metadata


class TestIntegration:
    """Integration tests for extraction module."""
    
    def test_all_extractors_run(
        self,
        sample_pose_sequence,
        sample_ris_data,
        sample_vital_signs,
        sample_frame
    ):
        """Test that all extractors can run without errors."""
        data = {
            "pose_sequence": sample_pose_sequence,
            "ris_data": sample_ris_data,
            "vital_signs": sample_vital_signs,
            "frames": [sample_frame]
        }
        
        extractors = [
            CNSExtractor(),
            CardiovascularExtractor(),
            RenalExtractor(),
            GastrointestinalExtractor(),
            SkeletalExtractor(),
            SkinExtractor(),
            EyeExtractor(),
            NasalExtractor(),
            ReproductiveExtractor(),
        ]
        
        results = []
        for extractor in extractors:
            result = extractor.extract(data)
            results.append(result)
            
            # Verify each result
            assert result.system is not None
            assert len(result.biomarkers) > 0
            assert result.extraction_time_ms >= 0
        
        # Check all 9 systems covered
        systems = {r.system for r in results}
        assert len(systems) == 9
    
    def test_biomarker_serialization(self, sample_pose_sequence):
        """Test biomarker set serialization."""
        extractor = CNSExtractor()
        result = extractor.extract({"pose_sequence": sample_pose_sequence})
        
        # Serialize to dict
        result_dict = result.to_dict()
        
        assert "system" in result_dict
        assert "biomarkers" in result_dict
        assert isinstance(result_dict["biomarkers"], list)
        
        # Each biomarker should have required fields
        for bm_dict in result_dict["biomarkers"]:
            assert "name" in bm_dict
            assert "value" in bm_dict
            assert "unit" in bm_dict
            assert "confidence" in bm_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import pytest
import numpy as np
from unittest.mock import MagicMock
from app.core.extraction.skin import SkinExtractor
from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem

@pytest.fixture
def skin_extractor():
    return SkinExtractor()

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def create_mock_landmarks(yaw_offset=0.0, pitch_offset=0.0):
    """
    Create a list of 468 mock landmarks.
    Baseline (Frontal):
    Nose (1): (0.5, 0.5)
    Left Eye (33): (0.4, 0.4)
    Right Eye (263): (0.6, 0.4)
    Chin (152): (0.5, 0.7)
    Forehead (10): (0.5, 0.3)
    """
    landmarks = [MockLandmark(0.5, 0.5) for _ in range(468)]
    
    # Frontal coordinates
    nose_x, nose_y = 0.5, 0.5
    l_eye_x, l_eye_y = 0.4, 0.4
    r_eye_x, r_eye_y = 0.6, 0.4
    chin_x, chin_y = 0.5, 0.7
    forehead_x, forehead_y = 0.5, 0.3
    
    # Apply offsets for yaw (horizontal shift of nose relative to eyes)
    # If yaw > 0 (turned right), nose moves closer to right eye
    nose_x += yaw_offset
    # Apply offsets for pitch (vertical shift of nose relative to eye-line)
    nose_y += pitch_offset
    
    landmarks[1] = MockLandmark(nose_x, nose_y)
    landmarks[33] = MockLandmark(l_eye_x, l_eye_y)
    landmarks[263] = MockLandmark(r_eye_x, r_eye_y)
    landmarks[152] = MockLandmark(chin_x, chin_y)
    landmarks[10] = MockLandmark(forehead_x, forehead_y)
    
    return landmarks

def test_estimate_head_pose_frontal(skin_extractor):
    landmarks = create_mock_landmarks(yaw_offset=0.0, pitch_offset=0.0)
    yaw, pitch = skin_extractor._estimate_head_pose(landmarks)
    
    # Frontal should be approx 0,0
    assert abs(yaw) < 1.0
    assert abs(pitch) < 1.0

def test_estimate_head_pose_rotation(skin_extractor):
    # Significant yaw (nose closer to right eye)
    # dist_l = 0.55-0.4=0.15, dist_r = 0.6-0.55=0.05. Ratio = 3.0
    landmarks_yaw = create_mock_landmarks(yaw_offset=0.05, pitch_offset=0.0)
    yaw, pitch = skin_extractor._estimate_head_pose(landmarks_yaw)
    assert yaw > 15.0 # Should detect substantial rotation
    
    # Significant pitch (nose closer to chin)
    landmarks_pitch = create_mock_landmarks(yaw_offset=0.0, pitch_offset=0.05)
    yaw, pitch = skin_extractor._estimate_head_pose(landmarks_pitch)
    assert pitch > 10.0 # Should detect looking down

def test_asymmetry_gating_good_pose(skin_extractor):
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    thermal_data = {"thermal_asymmetry": 0.3}
    landmarks = create_mock_landmarks(yaw_offset=0.0, pitch_offset=0.0)
    
    skin_extractor._extract_from_thermal_v2(
        thermal_data, biomarker_set, pose_landmarks=landmarks
    )
    
    asymmetry_bm = next(bm for bm in biomarker_set.biomarkers if bm.name == "thermal_asymmetry")
    assert asymmetry_bm.confidence >= 0.9 # High confidence for frontal pose

def test_asymmetry_gating_bad_pose(skin_extractor):
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    thermal_data = {"thermal_asymmetry": 0.3}
    # Large yaw rotation
    landmarks = create_mock_landmarks(yaw_offset=0.1, pitch_offset=0.0)
    
    skin_extractor._extract_from_thermal_v2(
        thermal_data, biomarker_set, pose_landmarks=landmarks
    )
    
    asymmetry_bm = next(bm for bm in biomarker_set.biomarkers if bm.name == "thermal_asymmetry")
    assert asymmetry_bm.confidence < 0.4 # Confidence should be penalized
    assert "High head rotation" in asymmetry_bm.description


import pytest
import numpy as np
import cv2
from app.core.extraction.skin import SkinExtractor, SessionBaseline, BiomarkerSet

def test_skin_extract_integrated():
    extractor = SkinExtractor()
    
    # Mock _get_face_mask and internal analysis methods
    def mock_get_face_mask(frame):
        return np.ones(frame.shape[:2], dtype=np.uint8)*255, frame
    extractor._get_face_mask = mock_get_face_mask
    
    baseline = SessionBaseline(
        baseline_facial_temp=36.0,
        baseline_redness=130.0,
        baseline_yellowness=130.0,
        ambient_background_temp=22.0
    )
    
    data = {
        "thermal_data": {
            "fever_face_max": 35.2, # 35.2 + 0.8 = 36.0
            "fever_canthus_temp": 35.2,
            "background_temp": 22.0
        },
        "frames": [np.zeros((100, 100, 3), dtype=np.uint8)],
        "session_baseline": baseline
    }
    
    # Integrated test: Call extract WITHOUT explicit session_baseline
    biomarker_set = extractor.extract(data)
    
    # Check thermal deviation
    temp_dev = biomarker_set.get("skin_temperature_deviation")
    assert temp_dev is not None
    assert temp_dev.value == 0.0

def test_capture_session_baseline():
    extractor = SkinExtractor()
    
    # Mock thermal data (5 frames)
    # fever_face_max = 35.2 (will be calibrated with +0.8 to 36.0)
    # background_temp = 22.0
    thermal_frames = [
        {'fever_face_max': 35.2, 'background_temp': 22.0},
        {'fever_face_max': 35.3, 'background_temp': 22.1},
        {'fever_face_max': 35.2, 'background_temp': 22.0},
        {'fever_face_max': 35.1, 'background_temp': 21.9},
        {'fever_face_max': 35.2, 'background_temp': 22.0},
    ]
    
    # Mock RGB frames (black frames with a "face" in the middle)
    rgb_frames = []
    for _ in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a white square to simulate a face for the mask logic (though _get_face_mask uses MediaPipe)
        # For testing we might need to mock _get_face_mask or use a real image if we want to test the full path
        # But we can also test the SessionBaseline logic by mocking the results of _get_face_mask
        rgb_frames.append(frame)

    # Simplified test: Mock _get_face_mask to return a full mask
    def mock_get_face_mask(frame):
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        return mask, frame
    
    extractor._get_face_mask = mock_get_face_mask
    
    baseline = extractor.capture_session_baseline(thermal_frames, rgb_frames)
    
    assert isinstance(baseline, SessionBaseline)
    assert pytest.approx(baseline.baseline_facial_temp, 0.1) == 36.0
    assert pytest.approx(baseline.ambient_background_temp, 0.1) == 22.0
    # Black frame Lab 'a' and 'b' are 128 (neutral) in OpenCV
    assert pytest.approx(baseline.baseline_redness, 1.0) == 128.0 
    assert pytest.approx(baseline.baseline_yellowness, 1.0) == 128.0

def test_skin_extraction_with_baseline():
    extractor = SkinExtractor()
    biomarker_set = BiomarkerSet(system=extractor.system)
    
    # Case 1: Low ambient temperature (20.0°C) -> Normal deviation range should be (-1.5, 1.0)
    baseline = SessionBaseline(
        baseline_facial_temp=36.0,
        baseline_redness=130.0,
        baseline_yellowness=130.0,
        ambient_background_temp=20.0
    )
    
    thermal_data = {
        'fever_face_max': 35.2, # 35.2 + 0.8 = 36.0 (0 deviation)
        'fever_canthus_temp': 35.2,
        'fever_neck_temp': 34.0
    }
    
    extractor._extract_from_thermal_v2(thermal_data, biomarker_set, session_baseline=baseline)
    
    bm = next(b for b in biomarker_set.biomarkers if b.name == "skin_temperature_deviation")
    assert bm.value == 0.0
    assert bm.normal_range == (-1.5, 1.0)
    
    # Case 2: High ambient temperature (30.0°C) -> Normal deviation range should be (-1.0, 2.0)
    biomarker_set = BiomarkerSet(system=extractor.system)
    baseline.ambient_background_temp = 30.0
    thermal_data['fever_face_max'] = 36.2 # 36.2 + 0.8 = 37.0 (+1.0 deviation)
    
    extractor._extract_from_thermal_v2(thermal_data, biomarker_set, session_baseline=baseline)
    
    bm = next(b for b in biomarker_set.biomarkers if b.name == "skin_temperature_deviation")
    assert bm.value == 1.0
    assert bm.normal_range == (-1.0, 2.0)

def test_color_extraction_with_baseline():
    extractor = SkinExtractor()
    biomarker_set = BiomarkerSet(system=extractor.system)
    
    baseline = SessionBaseline(
        baseline_redness=140.0,
        baseline_yellowness=140.0
    )
    
    # Mock frame and mask
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Set pixels to have Lab a=150, b=150
    # Lab conversion is complex, let's just mock _analyze_skin_color_lab return
    # No, let's test the logic inside _analyze_skin_color_lab
    
    # We'll mock the cv2.cvtColor and mean calculations or just provide a frame that results in specific Lab values
    # Easier: Mock the part that computes the mean
    
    metrics = extractor._analyze_skin_color_lab(frame, mask, session_baseline=baseline)
    
    # Since frame is black (BGR 0,0,0), it converts to Lab (0, 128, 128)
    # Deviation from baseline (140, 140) should be (128-140) = -12
    assert metrics["redness"] == -12.0
    assert metrics["yellowness"] == -12.0

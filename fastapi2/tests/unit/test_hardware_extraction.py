

# import pytest (removed for manual run)
import numpy as np
from app.core.extraction.pulmonary import PulmonaryExtractor
from app.core.extraction.skin import SkinExtractor
from app.core.extraction.cardiovascular import CardiovascularExtractor
from app.core.extraction.base import PhysiologicalSystem

class TestHardwareExtraction:
    
    def test_pulmonary_radar_extraction(self):
        """Test pulmonary extraction from raw radar data."""
        extractor = PulmonaryExtractor()
        data = {
            "radar_data": {
                "radar": {
                    "respiration_rate": 16.5,
                    "breathing_depth": 0.8
                }
            }
        }
        
        result = extractor.extract(data)
        
        assert result.system == PhysiologicalSystem.PULMONARY
        rr = result.get("respiration_rate")
        depth = result.get("breathing_depth")
        
        assert rr is not None
        assert rr.value == 16.5
        assert rr.confidence >= 0.90
        
        assert depth is not None
        assert depth.value == 0.8

    def test_pulmonary_bridge_systems_extraction(self):
        """Test pulmonary extraction from bridge pre-processed list."""
        extractor = PulmonaryExtractor()
        data = {
            "systems": [
                {
                    "system": "pulmonary",
                    "biomarkers": [
                        {"name": "respiration_rate", "value": 18.0, "unit": "breaths/min"}
                    ]
                }
            ]
        }
        
        result = extractor.extract(data)
        rr = result.get("respiration_rate")
        assert rr is not None
        assert rr.value == 18.0

    def test_skin_thermal_extraction(self):
        """Test skin extraction from thermal sensor data."""
        extractor = SkinExtractor()
        data = {
            "esp32_data": {
                "thermal": {
                    "skin_temp_avg": 36.6,
                    "thermal_asymmetry": 0.1
                }
            }
        }
        
        result = extractor.extract(data)
        
        temp = result.get("skin_temperature")
        assert temp is not None
        assert temp.value == 36.6
        assert temp.unit == "celsius"
        assert temp.confidence >= 0.90

    def test_cardiovascular_radar_fusion(self):
        """Test CV extraction prioritizing radar HR over others."""
        extractor = CardiovascularExtractor()
        data = {
            "radar_data": {
                "radar": {
                    "heart_rate": 75.0
                }
            },
            # Provide dummy face frames to trigger rPPG logic
            "face_frames": [np.zeros((100, 100, 3), dtype=np.uint8)] * 30
        }
        
        result = extractor.extract(data)
        
        # Should have radar HR as primary "heart_rate"
        hr = result.get("heart_rate")
        assert hr is not None
        assert hr.value == 75.0
        assert hr.description == "Heart rate from 60GHz Radar"
        
        # rPPG HR should be stored separately if conflict
        # Note: rPPG logic might fail on dummy frames, but main HR must be 75
        
    def test_cardiovascular_bridge_systems(self):
        """Test CV extraction from bridge pre-processed radar HR."""
        extractor = CardiovascularExtractor()
        data = {
            "systems": [
                {
                    "system": "cardiovascular",
                    "biomarkers": [
                        {"name": "heart_rate_radar", "value": 72.0, "unit": "bpm"}
                    ]
                }
            ]
        }
        
        result = extractor.extract(data)
        hr = result.get("heart_rate")
        assert hr is not None
        assert hr.value == 72.0

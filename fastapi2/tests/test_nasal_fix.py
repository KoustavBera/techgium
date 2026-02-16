import sys
import os
import unittest
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock
sys.modules["mediapipe"] = MagicMock()
sys.modules["mediapipe.python"] = MagicMock()
sys.modules["mediapipe.python.solutions"] = MagicMock()

from app.core.extraction.nasal import NasalExtractor
from app.core.extraction.base import BiomarkerSet

class TestNasalExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = NasalExtractor()

    def test_radar_nested_structure(self):
        """Test extraction from nested radar data (HardwareManager format)."""
        data = {
            "radar_data": {
                "radar": {
                    "respiration_rate": 18.5,
                    "breath_intervals": [3.0, 3.2, 3.1, 2.9, 3.0]
                }
            }
        }
        
        result = self.extractor.extract(data)
        
        # Check Respiratory Rate
        rr_bm = next((bm for bm in result.biomarkers if bm.name == "respiratory_rate"), None)
        self.assertIsNotNone(rr_bm)
        self.assertEqual(rr_bm.value, 18.5)
        
        # Check Regularity Index
        reg_bm = next((bm for bm in result.biomarkers if bm.name == "respiratory_regularity_index"), None)
        self.assertIsNotNone(reg_bm)

    def test_thermal_aggregated_structure(self):
        """Test extraction from aggregated thermal data (HardwareManager format)."""
        data = {
            "thermal_data": {
                "nose_temp": 34.0,       # Mapped to nostril_mean
                "face_mean_temp": 33.0,  # Proxy for cheek_mean
                "thermal_asymmetry": 0.05 # Proxy for airflow symmetry
            }
        }
        
        result = self.extractor.extract(data)
        
        # Check Nasal Surface Temp Elevation (34.0 - 33.0 = 1.0)
        elev_bm = next((bm for bm in result.biomarkers if bm.name == "nasal_surface_temp_elevation"), None)
        self.assertIsNotNone(elev_bm)
        self.assertAlmostEqual(elev_bm.value, 1.0)
        
        # Check Airflow Symmetry (from thermal_asymmetry)
        sym_bm = next((bm for bm in result.biomarkers if bm.name == "airflow_thermal_symmetry_index"), None)
        self.assertIsNotNone(sym_bm)
        self.assertEqual(sym_bm.value, 0.05)

if __name__ == "__main__":
    unittest.main()


import sys
import os
from unittest.mock import MagicMock

# Mock cv2 and mediapipe before importing skin extractor
sys.modules["cv2"] = MagicMock()
sys.modules["mediapipe"] = MagicMock()
sys.modules["skimage"] = MagicMock()
sys.modules["skimage.feature"] = MagicMock()
sys.modules["skimage.exposure"] = MagicMock()

# Add project root to path
sys.path.append(os.getcwd())

from app.core.extraction.skin import SkinExtractor

def test_skin_extraction_fallback():
    extractor = SkinExtractor()
    
    # Case 1: Neck Temp Available (Standard)
    # Using the new flattened format expected by _extract_from_thermal_v2
    data_standard = {
        "thermal_data": {
            "fever_neck_temp": 36.5,
            "fever_canthus_temp": 37.0
        }
    }
    bs_standard = extractor.extract(data_standard)
    print("Standard Case (Neck Available):")
    found_std = False
    for bm in bs_standard.biomarkers:
        if bm.name == "skin_temperature" and bm.value == 36.5:
             print(f"  [OK] skin_temperature: {bm.value} (Expected: 36.5)")
             found_std = True
    if not found_std:
        print("  [FAIL] skin_temperature NOT FOUND or WRONG VALUE in standard case")


    # Case 2: Neck Temp Missing (Fallback needed)
    data_fallback = {
        "thermal_data": {
            "fever_neck_temp": None,
            "fever_canthus_temp": 37.2
        }
    }
    bs_fallback = extractor.extract(data_fallback)
    print("\nFallback Case (Neck Missing):")
    found_fallback = False
    for bm in bs_fallback.biomarkers:
        if bm.name == "skin_temperature":
            print(f"  [OK] skin_temperature: {bm.value} (Expected: 37.2)")
            found_fallback = True
    
    if not found_fallback:
        print("  [FAIL] skin_temperature NOT FOUND in fallback case!")

if __name__ == "__main__":
    test_skin_extraction_fallback()


import sys
import os
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.getcwd())

from app.core.extraction.skin import SkinExtractor

def test_skin_extractor():
    print("Initializing SkinExtractor...")
    try:
        extractor = SkinExtractor()
        print("SkinExtractor initialized successfully.")
    except Exception as e:
        print(f"FAILED to initialize SkinExtractor: {e}")
        return

    # Test 1: Empty Data
    print("\nTest 1: Empty Data Fallback")
    data = {}
    results = extractor.extract(data)
    print(f"Biomarkers count: {len(results.biomarkers)}")
    # Verify we got simulated/fallback values
    redness = next((b for b in results.biomarkers if b.name == "skin_redness"), None)
    if redness:
        print(f"Redness (Simulated): {redness.value} {redness.unit}")
    
    # Test 2: Black Image (No Face)
    print("\nTest 2: Black Image (No Face)")
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    data = {"frames": [black_frame]}
    results = extractor.extract(data)
    # Should fall back to simulation because no face detected
    texture = next((b for b in results.biomarkers if b.name == "texture_roughness"), None)
    if texture:
        print(f"Texture (Simulated): {texture.value}")

    # Test 3: Synthetic Face (White Circle on Black Background)
    # This might NOT trigger MediaPipe, but verifies the pipeline doesn't crash on weird images
    print("\nTest 3: Synthetic Image")
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(synthetic_frame, (320, 240), 100, (200, 200, 200), -1) # Light gray face-ish
    data = {"frames": [synthetic_frame]}
    results = extractor.extract(data)
    print("Extracted successfully (likely fallback if MP fails detection).")

    print("\nVerification Complete: Module loads and runs without crashing.")

if __name__ == "__main__":
    test_skin_extractor()

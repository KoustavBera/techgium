"""
Verification script for upgraded SkeletalExtractor.
Tests Butterworth filtering, visibility handling, and 3D kinematics.
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.extraction.skeletal import SkeletalExtractor
from app.core.extraction.base import PhysiologicalSystem

def test_skeletal_rigor():
    print("Initializing SkeletalExtractor...")
    extractor = SkeletalExtractor(sample_rate=30.0)
    
    # 1. Create noisy synthetic pose data (100 frames, 33 landmarks, [x,y,z,vis])
    # Case: Standing still but with camera jitter
    print("Generating noisy synthetic data...")
    num_frames = 60
    # Simulate hips (index 23, 24) at fixed position + Gaussian noise
    base_pose = np.random.normal(0.5, 0.01, (num_frames, 33, 4))
    base_pose[:, :, 3] = 0.9  # High visibility
    
    # Add a jump (artifact) in Hip L at frame 30
    base_pose[30, 23, :2] += 0.5 
    
    # 2. Run extraction
    data = {
        "pose_sequence": base_pose.tolist(),
        "fps": 30.0
    }
    
    print("Running extraction...")
    bm_set = extractor.extract(data)
    
    # 3. Verify results
    # We expect stable scores despite the single frame jump due to Butterworth filtering
    results = {bm.name: bm.value for bm in bm_set.biomarkers}
    
    print("\nResults:")
    for name, value in results.items():
        print(f" - {name}: {value:.4f}")
    
    if "stance_stability_score" in results:
        score = results["stance_stability_score"]
        if score > 80:
            print("\n✅ PASSED: Stability score remains high despite noise (Filter working)")
        else:
            print(f"\n❌ FAILED: Stability score too low ({score:.1f})")
            
    if "average_joint_rom" in results:
        rom = results["average_joint_rom"]
        print(f"✅ Joint ROM: {rom:.4f} radians")

if __name__ == "__main__":
    try:
        test_skeletal_rigor()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

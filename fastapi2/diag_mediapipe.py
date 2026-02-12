import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

def diag():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'pose_landmarker.task')
    
    print(f"Checking for model at: {model_path}")
    if not os.path.exists(model_path):
        print("❌ Model file NOT found!")
        return

    print(f"Model file size: {os.path.getsize(model_path)} bytes")

    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        print("Attempting to create PoseLandmarker...")
        landmarker = vision.PoseLandmarker.create_from_options(options)
        print("✅ PoseLandmarker created successfully!")
        
        # Test with dummy image
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
        print("Testing detection...")
        result = landmarker.detect(mp_image)
        print("✅ Detection test complete (No landmarks expected on black image)")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

if __name__ == "__main__":
    diag()

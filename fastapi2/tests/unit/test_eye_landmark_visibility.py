import unittest
import numpy as np
from unittest.mock import MagicMock
from app.core.hardware.drivers import CameraCapture

class TestEyeLandmarkExtraction(unittest.TestCase):
    def setUp(self):
        # Initialize without opening camera
        self.camera = CameraCapture(camera_index=-1)
        
        # Mock FaceMesh
        self.mock_mesh = MagicMock()
        self.camera.face_mesh = self.mock_mesh

    def test_landmark_visibility_preservation(self):
        """Verify that landmarks keep their actual visibility scores."""
        # Create mock landmark with specific visibility
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 0.85  # Specific value to check
        
        # Mock FaceMesh result
        mock_result = MagicMock()
        mock_face = MagicMock()
        mock_face.landmark = [mock_landmark] * 478 # Simplified
        mock_result.multi_face_landmarks = [mock_face]
        self.mock_mesh.process.return_value = mock_result
        
        # Test frame
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Run extraction
        roi, landmarks = self.camera.extract_face_features(dummy_frame)
        
        # Assertions
        self.assertIsNotNone(landmarks)
        # Check if the 4th column (visibility) matches our mock value
        self.assertEqual(landmarks[0, 3], 0.85, "Visibility score was not preserved!")
        
    def test_landmark_handled_when_visibility_missing(self):
        """Verify fallback to 1.0 if visibility attribute is missing for some reason."""
        class SimpleLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        mock_landmark = SimpleLandmark(0.5, 0.5, 0.0)
        
        mock_result = MagicMock()
        mock_face = MagicMock()
        mock_face.landmark = [mock_landmark] * 478
        mock_result.multi_face_landmarks = [mock_face]
        self.mock_mesh.process.return_value = mock_result
        
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi, landmarks = self.camera.extract_face_features(dummy_frame)
        
        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks[0, 3], 1.0, "Fallback visibility should be 1.0")

if __name__ == '__main__':
    unittest.main()

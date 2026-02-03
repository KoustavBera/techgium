"""
FaceMesh Ingestion

MediaPipe FaceMesh for precise facial landmark detection
with iris tracking support for enhanced eye biomarkers.
"""
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from app.utils import get_logger
from .sync import DataPacket, DataSynchronizer, ModalityType

logger = get_logger(__name__)


@dataclass
class FaceMeshData:
    """FaceMesh landmarks with metadata."""
    landmarks: np.ndarray  # (468+ x 4) [x, y, z, visibility]
    timestamp: float
    confidence: float
    has_iris: bool = False
    
    def get_eye_landmarks(self, left: bool = True) -> np.ndarray:
        """Extract eye region landmarks."""
        if left:
            indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        else:
            indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        return self.landmarks[indices]
    
    def get_iris_landmarks(self, left: bool = True) -> Optional[np.ndarray]:
        """Extract iris landmarks if available."""
        if not self.has_iris or self.landmarks.shape[0] < 478:
            return None
        if left:
            return self.landmarks[468:473]
        else:
            return self.landmarks[473:478]


class FaceMeshIngestion:
    """
    MediaPipe FaceMesh ingestion for facial landmark detection.
    
    Provides 468 facial landmarks + optional iris tracking (478 total).
    Optimized for eye biomarker extraction.
    """
    
    def __init__(
        self,
        synchronizer: Optional[DataSynchronizer] = None,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize FaceMesh ingestion.
        
        Args:
            synchronizer: Optional shared DataSynchronizer
            refine_landmarks: Enable iris tracking (468->478 landmarks)
            min_detection_confidence: Detection threshold
            min_tracking_confidence: Tracking threshold
        """
        self.synchronizer = synchronizer or DataSynchronizer()
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self._face_mesh = None
        self._frame_count = 0
        
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        else:
            logger.warning("MediaPipe not available, using simulation mode")
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe FaceMesh."""
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        logger.info(f"MediaPipe FaceMesh initialized (iris={'ON' if self.refine_landmarks else 'OFF'})")
    
    def process_frame(self, frame: np.ndarray) -> Optional[FaceMeshData]:
        """
        Extract facial landmarks from a video frame.
        
        Args:
            frame: BGR video frame
            
        Returns:
            FaceMeshData if face detected, None otherwise
        """
        timestamp = DataSynchronizer.current_timestamp_ms()
        
        if not MEDIAPIPE_AVAILABLE or self._face_mesh is None:
            return self._generate_synthetic_facemesh(timestamp)
        
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1] if len(frame.shape) == 3 else frame
        results = self._face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Extract first face
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Calculate confidence from visibility
        confidence = float(np.mean(landmarks_array[:, 3]))
        
        self._frame_count += 1
        return FaceMeshData(
            landmarks=landmarks_array,
            timestamp=timestamp,
            confidence=confidence,
            has_iris=self.refine_landmarks and landmarks_array.shape[0] >= 478
        )
    
    def _generate_synthetic_facemesh(self, timestamp: float) -> FaceMeshData:
        """Generate synthetic FaceMesh data for testing."""
        num_landmarks = 478 if self.refine_landmarks else 468
        
        # Generate plausible face landmarks
        landmarks = []
        for i in range(num_landmarks):
            # Distribute landmarks in face region (normalized 0-1)
            angle = (i / num_landmarks) * 2 * np.pi
            radius = 0.3 + 0.1 * np.sin(i * 0.1)
            
            x = 0.5 + radius * np.cos(angle) + np.random.normal(0, 0.01)
            y = 0.5 + radius * np.sin(angle) + np.random.normal(0, 0.01)
            z = np.random.normal(0, 0.02)
            visibility = np.clip(0.85 + np.random.normal(0, 0.1), 0, 1)
            
            landmarks.append([x, y, z, visibility])
        
        self._frame_count += 1
        return FaceMeshData(
            landmarks=np.array(landmarks, dtype=np.float32),
            timestamp=timestamp,
            confidence=0.85,
            has_iris=self.refine_landmarks
        )
    
    def create_facemesh_packet(self, facemesh_data: FaceMeshData) -> DataPacket:
        """
        Create a DataPacket from FaceMesh data.
        
        Args:
            facemesh_data: Extracted FaceMesh data
            
        Returns:
            DataPacket with facial landmarks
        """
        return self.synchronizer.create_packet(
            modality=ModalityType.MOTION,  # Reuse MOTION type
            data=facemesh_data.landmarks,
            metadata={
                "num_landmarks": facemesh_data.landmarks.shape[0],
                "confidence": facemesh_data.confidence,
                "has_iris": facemesh_data.has_iris,
                "type": "facemesh"
            },
            timestamp=facemesh_data.timestamp
        )
    
    def iter_facemesh_from_frames(
        self,
        frames: Generator[np.ndarray, None, None],
        create_packets: bool = True
    ) -> Generator[FaceMeshData, None, None]:
        """
        Process video frames and yield FaceMesh data.
        
        Args:
            frames: Generator of video frames
            create_packets: Whether to create DataPackets
            
        Yields:
            FaceMeshData for each frame with detected face
        """
        for frame in frames:
            facemesh_data = self.process_frame(frame)
            if facemesh_data is not None:
                if create_packets:
                    self.create_facemesh_packet(facemesh_data)
                yield facemesh_data
    
    def close(self) -> None:
        """Release resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            logger.info("MediaPipe FaceMesh released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

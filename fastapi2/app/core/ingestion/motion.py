"""
Motion Sensor Ingestion

Pose estimation using MediaPipe and motion sensor data parsing
for skeletal tracking and movement analysis.
"""
from typing import Optional, Dict, Any, List, Tuple, Generator
from dataclasses import dataclass
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from app.config import settings
from app.utils import get_logger
from .sync import DataPacket, DataSynchronizer, ModalityType

logger = get_logger(__name__)


@dataclass
class PoseLandmark:
    """Single pose landmark with 3D coordinates and visibility."""
    name: str
    x: float  # Normalized X (0-1)
    y: float  # Normalized Y (0-1)
    z: float  # Depth estimate
    visibility: float  # Confidence (0-1)


@dataclass
class PoseData:
    """Complete pose estimation result."""
    landmarks: List[PoseLandmark]
    timestamp: float
    confidence: float
    
    def to_array(self) -> np.ndarray:
        """Convert landmarks to numpy array (N x 4)."""
        return np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in self.landmarks
        ])
    
    def get_landmark(self, name: str) -> Optional[PoseLandmark]:
        """Get landmark by name."""
        for lm in self.landmarks:
            if lm.name.lower() == name.lower():
                return lm
        return None


# MediaPipe pose landmark names
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]


class MotionIngestion:
    """
    Motion and pose sensor data ingestion.
    
    Supports:
    - MediaPipe pose estimation from video frames
    - Raw motion sensor CSV parsing
    - Simulated pose data generation
    """
    
    def __init__(
        self,
        synchronizer: Optional[DataSynchronizer] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize motion ingestion.
        
        Args:
            synchronizer: Optional shared DataSynchronizer
            min_detection_confidence: MediaPipe detection threshold
            min_tracking_confidence: MediaPipe tracking threshold
        """
        self.synchronizer = synchronizer or DataSynchronizer()
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self._pose = None
        self._pose_count = 0
        
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        else:
            logger.warning("MediaPipe not available, using simulation mode")
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe pose estimation."""
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        logger.info("MediaPipe Pose initialized")
    
    def process_frame(self, frame: np.ndarray) -> Optional[PoseData]:
        """
        Extract pose from a video frame.
        
        Args:
            frame: BGR video frame
            
        Returns:
            PoseData if pose detected, None otherwise
        """
        timestamp = DataSynchronizer.current_timestamp_ms()
        
        if not MEDIAPIPE_AVAILABLE or self._pose is None:
            return self._generate_synthetic_pose(timestamp)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = frame[:, :, ::-1] if len(frame.shape) == 3 else frame
        results = self._pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks.append(PoseLandmark(
                name=POSE_LANDMARK_NAMES[idx],
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))
        
        # Calculate overall confidence
        confidence = np.mean([lm.visibility for lm in landmarks])
        
        self._pose_count += 1
        return PoseData(
            landmarks=landmarks,
            timestamp=timestamp,
            confidence=confidence
        )
    
    def _generate_synthetic_pose(self, timestamp: float) -> PoseData:
        """
        Generate synthetic pose data for testing.
        
        Simulates a walking human with natural movement.
        """
        t = self._pose_count / 30.0  # Simulated time in seconds
        
        # Base skeleton position with walking animation
        step_phase = np.sin(t * 2 * np.pi * 0.8)  # ~0.8 Hz walking frequency
        sway = np.sin(t * 2 * np.pi * 0.4) * 0.02  # Body sway
        
        landmarks = []
        for idx, name in enumerate(POSE_LANDMARK_NAMES):
            # Base position (normalized 0-1 coordinates)
            base_x = 0.5 + sway
            base_y = 0.3 + (idx / len(POSE_LANDMARK_NAMES)) * 0.6
            
            # Add natural variations based on landmark type
            if "hip" in name or "knee" in name or "ankle" in name:
                # Leg movement
                side = 1 if "left" in name else -1
                base_x += side * 0.1
                if "knee" in name or "ankle" in name:
                    base_y += step_phase * side * 0.03
            elif "shoulder" in name or "elbow" in name or "wrist" in name:
                # Arm swing
                side = 1 if "left" in name else -1
                base_x += side * 0.15
                base_y += 0.1 - step_phase * side * 0.02
            
            # Add noise
            noise_x = np.random.normal(0, 0.005)
            noise_y = np.random.normal(0, 0.005)
            noise_z = np.random.normal(0, 0.01)
            
            landmarks.append(PoseLandmark(
                name=name,
                x=float(np.clip(base_x + noise_x, 0, 1)),
                y=float(np.clip(base_y + noise_y, 0, 1)),
                z=float(noise_z),
                visibility=float(np.clip(0.8 + np.random.normal(0, 0.1), 0, 1))
            ))
        
        self._pose_count += 1
        return PoseData(
            landmarks=landmarks,
            timestamp=timestamp,
            confidence=0.85 + np.random.normal(0, 0.05)
        )
    
    def create_motion_packet(self, pose_data: PoseData) -> DataPacket:
        """
        Create a DataPacket from pose data.
        
        Args:
            pose_data: Extracted pose data
            
        Returns:
            DataPacket with motion data
        """
        return self.synchronizer.create_packet(
            modality=ModalityType.MOTION,
            data=pose_data.to_array(),
            metadata={
                "num_landmarks": len(pose_data.landmarks),
                "confidence": pose_data.confidence,
                "landmark_names": [lm.name for lm in pose_data.landmarks]
            },
            timestamp=pose_data.timestamp
        )
    
    def iter_poses_from_frames(
        self,
        frames: Generator[np.ndarray, None, None],
        create_packets: bool = True
    ) -> Generator[PoseData, None, None]:
        """
        Process video frames and yield pose data.
        
        Args:
            frames: Generator of video frames
            create_packets: Whether to create DataPackets
            
        Yields:
            PoseData for each frame with detected pose
        """
        for frame in frames:
            pose_data = self.process_frame(frame)
            if pose_data is not None:
                if create_packets:
                    self.create_motion_packet(pose_data)
                yield pose_data
    
    def calculate_gait_metrics(
        self,
        pose_sequence: List[PoseData]
    ) -> Dict[str, float]:
        """
        Calculate gait-related metrics from a pose sequence.
        
        Args:
            pose_sequence: List of PoseData over time
            
        Returns:
            Dictionary of gait metrics
        """
        if len(pose_sequence) < 2:
            return {}
        
        # Extract hip and ankle positions over time
        left_hip_y = [p.get_landmark("left_hip").y for p in pose_sequence]
        right_hip_y = [p.get_landmark("right_hip").y for p in pose_sequence]
        left_ankle_y = [p.get_landmark("left_ankle").y for p in pose_sequence]
        right_ankle_y = [p.get_landmark("right_ankle").y for p in pose_sequence]
        
        # Gait symmetry (comparing left/right movement patterns)
        left_stride = np.diff(left_ankle_y)
        right_stride = np.diff(right_ankle_y)
        
        symmetry = 1.0 - np.mean(np.abs(left_stride - right_stride))
        
        # Vertical hip oscillation (smoothness of gait)
        hip_center_y = [(l + r) / 2 for l, r in zip(left_hip_y, right_hip_y)]
        hip_variance = np.var(hip_center_y)
        
        # Stride variability
        stride_lengths = np.abs(left_stride) + np.abs(right_stride)
        stride_variability = np.std(stride_lengths) / (np.mean(stride_lengths) + 1e-6)
        
        return {
            "gait_symmetry": float(np.clip(symmetry, 0, 1)),
            "hip_variance": float(hip_variance),
            "stride_variability": float(stride_variability),
            "num_frames": len(pose_sequence)
        }
    
    def close(self) -> None:
        """Release resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
            logger.info("MediaPipe Pose released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

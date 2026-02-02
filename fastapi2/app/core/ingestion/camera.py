"""
Camera Video Ingestion

OpenCV-based video frame extraction with timestamp injection
for synchronized multimodal processing.
"""
from typing import Optional, Generator, Tuple, Dict, Any
from pathlib import Path
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from app.config import settings
from app.utils import get_logger, IngestionError
from .sync import DataPacket, DataSynchronizer, ModalityType

logger = get_logger(__name__)


class CameraIngestion:
    """
    Video ingestion handler using OpenCV.
    
    Supports:
    - Video file input
    - Webcam capture (device ID)
    - Simulated video streams
    """
    
    def __init__(
        self,
        source: Optional[str] = None,
        frame_rate: Optional[int] = None,
        synchronizer: Optional[DataSynchronizer] = None
    ):
        """
        Initialize camera ingestion.
        
        Args:
            source: Video file path, device ID, or None for simulation
            frame_rate: Target frame rate (FPS)
            synchronizer: Optional shared DataSynchronizer
        """
        self.source = source
        self.frame_rate = frame_rate or settings.video_frame_rate
        self.synchronizer = synchronizer or DataSynchronizer()
        
        self._capture: Optional[Any] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, using simulation mode")
    
    def open(self) -> bool:
        """
        Open video source for capture.
        
        Returns:
            True if successful, False otherwise
        """
        if not CV2_AVAILABLE:
            logger.info("Camera simulation mode active")
            return True
        
        try:
            if self.source is None:
                # Webcam
                self._capture = cv2.VideoCapture(0)
            elif isinstance(self.source, int):
                # Device ID
                self._capture = cv2.VideoCapture(self.source)
            elif Path(self.source).exists():
                # Video file
                self._capture = cv2.VideoCapture(self.source)
            else:
                raise IngestionError(
                    f"Video source not found: {self.source}",
                    modality="camera"
                )
            
            if not self._capture.isOpened():
                raise IngestionError(
                    f"Failed to open video source: {self.source}",
                    modality="camera"
                )
            
            # Get native properties
            native_fps = self._capture.get(cv2.CAP_PROP_FPS)
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(
                f"Camera opened: {width}x{height} @ {native_fps:.1f}fps, "
                f"target={self.frame_rate}fps"
            )
            return True
            
        except Exception as e:
            logger.error(f"Camera open failed: {e}")
            return False
    
    def close(self) -> None:
        """Release video capture resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Camera released")
    
    def read_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Read a single frame from video source.
        
        Returns:
            Tuple of (frame_array, timestamp_ms) or None if no frame
        """
        timestamp = DataSynchronizer.current_timestamp_ms()
        
        if not CV2_AVAILABLE or self._capture is None:
            # Simulation mode: generate synthetic frame
            frame = self._generate_synthetic_frame()
            return frame, timestamp
        
        ret, frame = self._capture.read()
        if not ret:
            return None
        
        self._frame_count += 1
        return frame, timestamp
    
    def _generate_synthetic_frame(
        self,
        width: int = 640,
        height: int = 480
    ) -> np.ndarray:
        """
        Generate a synthetic video frame for testing.
        
        Creates a frame with moving patterns to simulate human presence.
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            frame[y, :, :] = [
                int(30 + 20 * np.sin(self._frame_count / 30)),
                int(30 + 10 * np.sin(self._frame_count / 20)),
                40
            ]
        
        # Simulated body silhouette (moving ellipse)
        center_x = width // 2 + int(50 * np.sin(self._frame_count / 60))
        center_y = height // 2 + int(20 * np.cos(self._frame_count / 45))
        
        # Draw body shape
        y_indices, x_indices = np.ogrid[:height, :width]
        body_mask = (
            ((x_indices - center_x) / 60) ** 2 +
            ((y_indices - center_y) / 150) ** 2 <= 1
        )
        frame[body_mask] = [180, 140, 100]  # Skin-ish color
        
        # Head
        head_y = center_y - 170
        head_mask = (
            ((x_indices - center_x) / 40) ** 2 +
            ((y_indices - head_y) / 50) ** 2 <= 1
        )
        frame[head_mask] = [190, 150, 110]
        
        self._frame_count += 1
        return frame
    
    def iter_frames(
        self,
        max_frames: Optional[int] = None,
        create_packets: bool = True
    ) -> Generator[DataPacket, None, None]:
        """
        Iterate over video frames.
        
        Args:
            max_frames: Maximum frames to read (None for all)
            create_packets: Whether to create DataPackets
            
        Yields:
            DataPacket for each frame
        """
        frame_num = 0
        frame_interval_ms = 1000.0 / self.frame_rate
        
        while max_frames is None or frame_num < max_frames:
            result = self.read_frame()
            if result is None:
                break
            
            frame, timestamp = result
            
            if create_packets:
                packet = self.synchronizer.create_packet(
                    modality=ModalityType.CAMERA,
                    data=frame,
                    metadata={
                        "frame_number": frame_num,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                        "source": str(self.source) if self.source else "simulation"
                    },
                    timestamp=timestamp
                )
                yield packet
            
            frame_num += 1
        
        logger.info(f"Processed {frame_num} frames")
    
    def extract_frame_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract basic features from a frame.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            "mean_brightness": float(np.mean(frame)),
            "std_brightness": float(np.std(frame)),
        }
        
        # Color channel statistics
        if len(frame.shape) == 3:
            for i, channel in enumerate(["blue", "green", "red"]):
                features[f"{channel}_mean"] = float(np.mean(frame[:, :, i]))
                features[f"{channel}_std"] = float(np.std(frame[:, :, i]))
        
        # Detect presence (simple threshold)
        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
        features["presence_detected"] = float(np.std(gray)) > 30
        
        return features
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

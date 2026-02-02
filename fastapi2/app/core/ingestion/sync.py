"""
Time Synchronization Layer

Provides unified data packet structure and cross-modal time alignment
for synchronized multimodal data processing.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Generator
from enum import Enum
from collections import deque
import time
import uuid
import numpy as np

from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)


class ModalityType(str, Enum):
    """Supported data modality types."""
    CAMERA = "camera"
    MOTION = "motion"
    RIS = "ris"
    AUXILIARY = "auxiliary"
    HEARTBEAT = "heartbeat"
    THERMAL = "thermal"


@dataclass
class DataPacket:
    """
    Unified data packet structure for all modalities.
    
    Attributes:
        timestamp: Unix timestamp in milliseconds
        modality: Type of sensor/data source
        data: Raw data payload (numpy array)
        metadata: Modality-specific additional information
        session_id: Unique session identifier
        sequence_id: Sequential packet number within session
    """
    timestamp: float
    modality: ModalityType
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_id: int = 0
    
    def __post_init__(self):
        """Validate packet after initialization."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "modality": self.modality.value,
            "data_shape": list(self.data.shape),
            "data_dtype": str(self.data.dtype),
            "metadata": self.metadata,
            "session_id": self.session_id,
            "sequence_id": self.sequence_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_data: np.ndarray) -> "DataPacket":
        """Create packet from dictionary and raw data."""
        return cls(
            timestamp=data["timestamp"],
            modality=ModalityType(data["modality"]),
            data=raw_data,
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id", str(uuid.uuid4())),
            sequence_id=data.get("sequence_id", 0),
        )


class DataSynchronizer:
    """
    Cross-modal time synchronization manager.
    
    Maintains buffers for each modality and yields synchronized
    data packets within the configured tolerance window.
    """
    
    def __init__(
        self,
        tolerance_ms: float = None,
        buffer_size: int = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize synchronizer.
        
        Args:
            tolerance_ms: Maximum time difference for synchronized packets
            buffer_size: Maximum packets to buffer per modality
            session_id: Shared session identifier
        """
        self.tolerance_ms = tolerance_ms or settings.sync_tolerance_ms
        self.buffer_size = buffer_size or settings.buffer_size
        self.session_id = session_id or str(uuid.uuid4())
        
        # Per-modality buffers
        self._buffers: Dict[ModalityType, deque] = {
            modality: deque(maxlen=self.buffer_size)
            for modality in ModalityType
        }
        
        # Sequence counters per modality
        self._sequences: Dict[ModalityType, int] = {
            modality: 0 for modality in ModalityType
        }
        
        # Reference timestamp for alignment
        self._reference_time: Optional[float] = None
        
        logger.info(
            f"DataSynchronizer initialized: tolerance={self.tolerance_ms}ms, "
            f"buffer_size={self.buffer_size}, session={self.session_id}"
        )
    
    @staticmethod
    def current_timestamp_ms() -> float:
        """Get current timestamp in milliseconds."""
        return time.time() * 1000
    
    def add_packet(self, packet: DataPacket) -> None:
        """
        Add a data packet to the appropriate buffer.
        
        Args:
            packet: DataPacket to buffer
        """
        # Assign session and sequence
        packet.session_id = self.session_id
        packet.sequence_id = self._sequences[packet.modality]
        self._sequences[packet.modality] += 1
        
        # Set reference time from first packet
        if self._reference_time is None:
            self._reference_time = packet.timestamp
            logger.debug(f"Reference time set to {self._reference_time}")
        
        # Add to buffer
        self._buffers[packet.modality].append(packet)
        
        logger.debug(
            f"Buffered {packet.modality.value} packet "
            f"seq={packet.sequence_id} at t={packet.timestamp:.1f}ms"
        )
    
    def create_packet(
        self,
        modality: ModalityType,
        data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> DataPacket:
        """
        Create and buffer a new data packet.
        
        Args:
            modality: Data modality type
            data: Raw data array
            metadata: Additional metadata
            timestamp: Optional custom timestamp (defaults to current time)
            
        Returns:
            Created DataPacket
        """
        packet = DataPacket(
            timestamp=timestamp or self.current_timestamp_ms(),
            modality=modality,
            data=data,
            metadata=metadata or {},
        )
        self.add_packet(packet)
        return packet
    
    def get_synchronized_window(
        self,
        target_time: Optional[float] = None,
        required_modalities: Optional[List[ModalityType]] = None
    ) -> Optional[Dict[ModalityType, DataPacket]]:
        """
        Get synchronized packets across modalities for a time window.
        
        Args:
            target_time: Center time for synchronization window
            required_modalities: Modalities that must be present
            
        Returns:
            Dictionary of synchronized packets, or None if requirements not met
        """
        if target_time is None:
            # Use most recent reference
            target_time = self._reference_time or self.current_timestamp_ms()
        
        required = required_modalities or list(ModalityType)
        result: Dict[ModalityType, DataPacket] = {}
        
        for modality in required:
            buffer = self._buffers[modality]
            if not buffer:
                logger.debug(f"No packets in {modality.value} buffer")
                continue
            
            # Find closest packet within tolerance
            closest = min(
                buffer,
                key=lambda p: abs(p.timestamp - target_time)
            )
            
            if abs(closest.timestamp - target_time) <= self.tolerance_ms:
                result[modality] = closest
        
        # Check if all required modalities present
        if len(result) == len(required):
            logger.debug(f"Synchronized window at t={target_time:.1f}ms with {len(result)} modalities")
            return result
        
        return None
    
    def iter_synchronized(
        self,
        required_modalities: Optional[List[ModalityType]] = None,
        step_ms: float = 33.3  # ~30 FPS
    ) -> Generator[Dict[ModalityType, DataPacket], None, None]:
        """
        Iterate over synchronized data windows.
        
        Args:
            required_modalities: Modalities required for each window
            step_ms: Time step between windows
            
        Yields:
            Synchronized packet dictionaries
        """
        if self._reference_time is None:
            return
        
        current_time = self._reference_time
        
        # Find latest timestamp across all buffers
        max_time = max(
            (p.timestamp for buffer in self._buffers.values() for p in buffer),
            default=current_time
        )
        
        while current_time <= max_time:
            window = self.get_synchronized_window(
                target_time=current_time,
                required_modalities=required_modalities
            )
            if window:
                yield window
            current_time += step_ms
    
    def clear_before(self, timestamp: float) -> int:
        """
        Clear packets older than timestamp from all buffers.
        
        Args:
            timestamp: Cutoff timestamp in milliseconds
            
        Returns:
            Number of packets removed
        """
        removed = 0
        for modality in ModalityType:
            buffer = self._buffers[modality]
            initial_len = len(buffer)
            self._buffers[modality] = deque(
                (p for p in buffer if p.timestamp >= timestamp),
                maxlen=self.buffer_size
            )
            removed += initial_len - len(self._buffers[modality])
        
        logger.debug(f"Cleared {removed} packets before t={timestamp:.1f}ms")
        return removed
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about current buffer state."""
        stats = {
            "session_id": self.session_id,
            "reference_time": self._reference_time,
            "modalities": {}
        }
        
        for modality in ModalityType:
            buffer = self._buffers[modality]
            if buffer:
                timestamps = [p.timestamp for p in buffer]
                stats["modalities"][modality.value] = {
                    "count": len(buffer),
                    "min_time": min(timestamps),
                    "max_time": max(timestamps),
                    "sequence": self._sequences[modality]
                }
            else:
                stats["modalities"][modality.value] = {"count": 0}
        
        return stats

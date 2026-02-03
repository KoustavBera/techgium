"""
Unit Tests for Data Ingestion Module

Tests for camera, motion, RIS, auxiliary ingestion and synchronization.
"""
import pytest
import numpy as np
from pathlib import Path

from app.core.ingestion import (
    CameraIngestion,
    MotionIngestion,
    RISSimulator,
    AuxiliaryIngestion,
    DataSynchronizer,
    DataPacket,
    ModalityType
)
from app.core.ingestion.ris import RISConfiguration


class TestDataPacket:
    """Tests for DataPacket structure."""
    
    def test_create_packet(self):
        """Test basic packet creation."""
        data = np.array([1.0, 2.0, 3.0])
        packet = DataPacket(
            timestamp=1000.0,
            modality=ModalityType.CAMERA,
            data=data
        )
        
        assert packet.timestamp == 1000.0
        assert packet.modality == ModalityType.CAMERA
        assert np.array_equal(packet.data, data)
        assert isinstance(packet.session_id, str)
    
    def test_packet_to_dict(self):
        """Test packet serialization."""
        data = np.zeros((10, 5))
        packet = DataPacket(
            timestamp=2000.0,
            modality=ModalityType.RIS,
            data=data,
            metadata={"key": "value"}
        )
        
        result = packet.to_dict()
        
        assert result["timestamp"] == 2000.0
        assert result["modality"] == "ris"
        assert result["data_shape"] == [10, 5]
        assert result["metadata"]["key"] == "value"
    
    def test_packet_data_conversion(self):
        """Test automatic numpy conversion."""
        packet = DataPacket(
            timestamp=0,
            modality=ModalityType.AUXILIARY,
            data=[1, 2, 3, 4, 5]  # List instead of ndarray
        )
        
        assert isinstance(packet.data, np.ndarray)
        assert len(packet.data) == 5


class TestDataSynchronizer:
    """Tests for DataSynchronizer."""
    
    def test_create_synchronizer(self):
        """Test synchronizer initialization."""
        sync = DataSynchronizer(tolerance_ms=100.0, buffer_size=50)
        
        assert sync.tolerance_ms == 100.0
        assert sync.buffer_size == 50
        assert isinstance(sync.session_id, str)
    
    def test_create_packet(self):
        """Test packet creation through synchronizer."""
        sync = DataSynchronizer()
        data = np.array([1.0, 2.0])
        
        packet = sync.create_packet(
            modality=ModalityType.CAMERA,
            data=data,
            metadata={"test": True}
        )
        
        assert packet.session_id == sync.session_id
        assert packet.sequence_id == 0
        assert packet.metadata["test"] is True
    
    def test_sequence_increment(self):
        """Test sequence ID increments correctly."""
        sync = DataSynchronizer()
        
        p1 = sync.create_packet(ModalityType.CAMERA, np.zeros(1))
        p2 = sync.create_packet(ModalityType.CAMERA, np.zeros(1))
        p3 = sync.create_packet(ModalityType.RIS, np.zeros(1))
        
        assert p1.sequence_id == 0
        assert p2.sequence_id == 1
        assert p3.sequence_id == 0  # Different modality, starts at 0
    
    def test_buffer_stats(self):
        """Test buffer statistics."""
        sync = DataSynchronizer()
        
        for i in range(5):
            sync.create_packet(ModalityType.CAMERA, np.zeros(1))
        for i in range(3):
            sync.create_packet(ModalityType.RIS, np.zeros(1))
        
        stats = sync.get_buffer_stats()
        
        assert stats["modalities"]["camera"]["count"] == 5
        assert stats["modalities"]["ris"]["count"] == 3


class TestCameraIngestion:
    """Tests for CameraIngestion."""
    
    def test_create_camera(self):
        """Test camera initialization."""
        camera = CameraIngestion(frame_rate=30)
        
        assert camera.frame_rate == 30
        assert camera.synchronizer is not None
    
    def test_synthetic_frame_generation(self, sample_frame):
        """Test synthetic frame generation."""
        camera = CameraIngestion()
        
        frame, timestamp = camera.read_frame()
        
        assert frame is not None
        assert frame.shape[0] > 0  # Has height
        assert frame.shape[1] > 0  # Has width
        assert frame.shape[2] == 3  # RGB channels
        assert timestamp > 0
    
    def test_frame_iteration(self):
        """Test frame iterator."""
        camera = CameraIngestion()
        
        frames = list(camera.iter_frames(max_frames=5))
        
        assert len(frames) == 5
        assert all(isinstance(f, DataPacket) for f in frames)
        assert all(f.modality == ModalityType.CAMERA for f in frames)
    
    def test_extract_features(self, sample_frame):
        """Test frame feature extraction."""
        camera = CameraIngestion()
        
        features = camera.extract_frame_features(sample_frame)
        
        assert "mean_brightness" in features
        assert "std_brightness" in features
        assert "blue_mean" in features
        assert "presence_detected" in features
    
    def test_context_manager(self):
        """Test camera as context manager."""
        with CameraIngestion() as camera:
            frame, _ = camera.read_frame()
            assert frame is not None


class TestMotionIngestion:
    """Tests for MotionIngestion."""
    
    def test_create_motion(self):
        """Test motion ingestion initialization."""
        motion = MotionIngestion()
        
        assert motion.synchronizer is not None
        assert motion.min_detection_confidence == 0.5
    
    def test_synthetic_pose_generation(self, sample_frame):
        """Test synthetic pose generation."""
        motion = MotionIngestion()
        
        pose = motion.process_frame(sample_frame)
        
        assert pose is not None
        assert len(pose.landmarks) == 33  # MediaPipe pose landmarks
        assert pose.confidence > 0
    
    def test_pose_to_array(self, sample_frame):
        """Test pose data conversion to array."""
        motion = MotionIngestion()
        pose = motion.process_frame(sample_frame)
        
        arr = pose.to_array()
        
        assert arr.shape == (33, 4)
        assert arr.dtype == np.float64 or arr.dtype == np.float32
    
    def test_gait_metrics(self, sample_frame):
        """Test gait metrics calculation."""
        motion = MotionIngestion()
        
        # Generate multiple poses
        poses = [motion.process_frame(sample_frame) for _ in range(10)]
        
        metrics = motion.calculate_gait_metrics(poses)
        
        assert "gait_symmetry" in metrics
        assert "stride_variability" in metrics
        assert 0 <= metrics["gait_symmetry"] <= 1
    
    def test_motion_packet_creation(self, sample_frame):
        """Test motion packet creation."""
        motion = MotionIngestion()
        pose = motion.process_frame(sample_frame)
        
        packet = motion.create_motion_packet(pose)
        
        assert packet.modality == ModalityType.MOTION
        assert "num_landmarks" in packet.metadata
        assert packet.metadata["num_landmarks"] == 33


class TestRISSimulator:
    """Tests for RISSimulator."""
    
    def test_create_simulator(self):
        """Test RIS simulator initialization."""
        config = RISConfiguration(
            sample_rate=1000,
            num_channels=16
        )
        ris = RISSimulator(config=config)
        
        assert ris.config.sample_rate == 1000
        assert ris.config.num_channels == 16
    
    def test_generate_sample(self):
        """Test single sample generation."""
        ris = RISSimulator()
        
        sample = ris.generate_sample()
        
        assert sample.shape == (16,)
        assert sample.dtype == np.float32
        assert np.all(sample > 0)  # Impedance should be positive
    
    def test_generate_batch(self):
        """Test batch generation."""
        ris = RISSimulator()
        
        batch = ris.generate_batch(100)
        
        assert batch.shape == (100, 16)
    
    def test_physiological_params(self):
        """Test physiological parameter setting."""
        ris = RISSimulator()
        
        ris.set_physiological_params(
            heart_rate_bpm=80,
            resp_rate_bpm=12,
            body_fat_percent=25
        )
        
        assert ris._heart_rate == 80 / 60.0
        assert ris._resp_rate == 12 / 60.0
    
    def test_feature_extraction(self):
        """Test RIS feature extraction."""
        ris = RISSimulator()
        data = ris.generate_batch(1000)
        
        features = ris.extract_features(data)
        
        assert "mean_impedance" in features
        assert "std_impedance" in features
        assert "lr_asymmetry" in features
        assert "thorax_mean" in features
    
    def test_iter_samples(self):
        """Test sample iterator."""
        ris = RISSimulator()
        
        packets = list(ris.iter_samples(duration_seconds=0.5, batch_size=50))
        
        assert len(packets) > 0
        assert all(p.modality == ModalityType.RIS for p in packets)


class TestAuxiliaryIngestion:
    """Tests for AuxiliaryIngestion."""
    
    def test_create_auxiliary(self):
        """Test auxiliary ingestion initialization."""
        aux = AuxiliaryIngestion()
        
        assert aux.synchronizer is not None
        assert len(aux._data_cache) == 0
    
    def test_synthetic_heartbeat(self):
        """Test synthetic heartbeat generation."""
        aux = AuxiliaryIngestion()
        
        signal = aux.generate_synthetic_heartbeat(
            heart_rate_bpm=72,
            duration_seconds=5,
            sample_rate=250
        )
        
        assert len(signal) == 5 * 250
        assert signal.dtype == np.float32
    
    def test_synthetic_thermal(self):
        """Test synthetic thermal data generation."""
        aux = AuxiliaryIngestion()
        
        temps = aux.generate_synthetic_thermal(
            base_temperature=36.5,
            num_regions=10
        )
        
        assert len(temps) == 10
        assert np.all(temps > 30)  # Reasonable body temps
        assert np.all(temps < 42)
    
    def test_iter_synthetic(self):
        """Test synthetic auxiliary iterator."""
        aux = AuxiliaryIngestion()
        
        packets = list(aux.iter_synthetic_auxiliary(
            duration_seconds=1,
            sample_rate=10
        ))
        
        assert len(packets) == 10
        assert all(p.modality == ModalityType.AUXILIARY for p in packets)
    
    def test_vital_feature_extraction(self):
        """Test vital signs feature extraction."""
        aux = AuxiliaryIngestion()
        
        data = np.array([72, 16, 36.5, 98])
        features = aux.extract_vital_features(data)
        
        assert features["heart_rate"] == 72
        assert features["respiratory_rate"] == 16
        assert features["temperature"] == 36.5
        assert features["spo2"] == 98
    
    def test_load_csv_missing_file(self):
        """Test error handling for missing file."""
        aux = AuxiliaryIngestion()
        
        with pytest.raises(Exception):  # IngestionError
            aux.load_csv_dataset("nonexistent.csv")


class TestIntegration:
    """Integration tests for ingestion module."""
    
    def test_synchronized_multimodal_ingestion(self):
        """Test synchronized data from multiple sources."""
        sync = DataSynchronizer()
        
        camera = CameraIngestion(synchronizer=sync)
        motion = MotionIngestion(synchronizer=sync)
        ris = RISSimulator(synchronizer=sync)
        
        # Generate data from each source
        for _ in range(5):
            frame_packet = next(camera.iter_frames(max_frames=1))
            pose = motion.process_frame(frame_packet.data)
            if pose:
                motion.create_motion_packet(pose)
        
        for packet in ris.iter_samples(duration_seconds=0.1, batch_size=10):
            pass
        
        # Check all modalities present in synchronizer
        stats = sync.get_buffer_stats()
        
        assert stats["modalities"]["camera"]["count"] == 5
        assert stats["modalities"]["motion"]["count"] > 0
        assert stats["modalities"]["ris"]["count"] > 0
    
    def test_data_packet_consistency(self):
        """Test packet data consistency across modalities."""
        sync = DataSynchronizer()
        
        # Create packets from different sources
        camera = CameraIngestion(synchronizer=sync)
        ris = RISSimulator(synchronizer=sync)
        
        camera_packets = list(camera.iter_frames(max_frames=3))
        ris_packets = list(ris.iter_samples(duration_seconds=0.1))
        
        # All should have same session ID
        all_packets = camera_packets + ris_packets
        session_ids = set(p.session_id for p in all_packets)
        
        assert len(session_ids) == 1  # Single session


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

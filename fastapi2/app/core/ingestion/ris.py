"""
Radio Impedance Sensing (RIS) Stream Simulator

Generates realistic bioimpedance-like data streams for testing
before hardware arrives. Includes physiologically-inspired patterns.
"""
from typing import Optional, Dict, Any, Generator, List
from dataclasses import dataclass
import numpy as np
from scipy import signal

from app.config import settings
from app.utils import get_logger
from .sync import DataPacket, DataSynchronizer, ModalityType

logger = get_logger(__name__)


@dataclass
class RISConfiguration:
    """Configuration for RIS simulation."""
    sample_rate: int = 1000  # Hz
    num_channels: int = 16   # Number of sensor channels
    noise_level: float = 0.05
    baseline_impedance: float = 500.0  # Ohms
    impedance_variance: float = 50.0


class RISSimulator:
    """
    Simulated Radio Impedance Sensing data generator.
    
    Generates multi-channel bioimpedance data streams with:
    - Respiratory modulation
    - Cardiac signatures
    - Body composition proxies
    - Fluid distribution patterns
    """
    
    def __init__(
        self,
        config: Optional[RISConfiguration] = None,
        synchronizer: Optional[DataSynchronizer] = None
    ):
        """
        Initialize RIS simulator.
        
        Args:
            config: RIS simulation configuration
            synchronizer: Optional shared DataSynchronizer
        """
        self.config = config or RISConfiguration(
            sample_rate=settings.ris_sample_rate,
            num_channels=settings.ris_num_channels
        )
        self.synchronizer = synchronizer or DataSynchronizer()
        
        self._sample_count = 0
        self._respiration_phase = 0.0
        self._cardiac_phase = 0.0
        
        # Physiological parameters (can be varied for different subjects)
        self._resp_rate = 0.25  # Hz (15 breaths/min)
        self._heart_rate = 1.2  # Hz (72 bpm)
        
        logger.info(
            f"RIS Simulator initialized: {self.config.num_channels} channels "
            f"@ {self.config.sample_rate} Hz"
        )
    
    def set_physiological_params(
        self,
        heart_rate_bpm: float = 72,
        resp_rate_bpm: float = 15,
        body_fat_percent: float = 20
    ) -> None:
        """
        Set physiological parameters for simulation.
        
        Args:
            heart_rate_bpm: Heart rate in beats per minute
            resp_rate_bpm: Respiration rate in breaths per minute
            body_fat_percent: Body fat percentage (affects impedance)
        """
        self._heart_rate = heart_rate_bpm / 60.0
        self._resp_rate = resp_rate_bpm / 60.0
        
        # Adjust baseline impedance based on body composition
        # Higher fat content = higher impedance
        fat_factor = 1.0 + (body_fat_percent - 20) / 100
        self.config.baseline_impedance *= fat_factor
        
        logger.debug(
            f"Physiological params: HR={heart_rate_bpm}bpm, "
            f"RR={resp_rate_bpm}bpm, BF={body_fat_percent}%"
        )
    
    def generate_sample(self) -> np.ndarray:
        """
        Generate a single multi-channel RIS sample.
        
        Returns:
            NumPy array of shape (num_channels,) with impedance values
        """
        t = self._sample_count / self.config.sample_rate
        
        # Base impedance with channel variation
        channel_offsets = np.linspace(-0.1, 0.1, self.config.num_channels)
        base = self.config.baseline_impedance * (1 + channel_offsets)
        
        # Respiratory modulation (affects all channels)
        resp_signal = self.config.impedance_variance * np.sin(
            2 * np.pi * self._resp_rate * t
        )
        
        # Cardiac modulation (smaller amplitude, higher frequency)
        cardiac_signal = (self.config.impedance_variance * 0.1) * np.sin(
            2 * np.pi * self._heart_rate * t
        )
        
        # Regional variations (simulate body regions)
        # Channels 0-3: Upper thorax
        # Channels 4-7: Lower thorax
        # Channels 8-11: Abdomen
        # Channels 12-15: Lower body
        
        regional_factors = np.zeros(self.config.num_channels)
        regional_factors[0:4] = 1.0 + 0.1 * np.sin(2 * np.pi * self._resp_rate * t)
        regional_factors[4:8] = 1.0 + 0.15 * np.sin(2 * np.pi * self._resp_rate * t - 0.2)
        regional_factors[8:12] = 1.0 + 0.05 * np.sin(2 * np.pi * self._resp_rate * t * 0.5)
        regional_factors[12:16] = 1.0 + 0.02 * np.sin(2 * np.pi * self._heart_rate * t)
        
        # Combine signals
        sample = base * regional_factors + resp_signal + cardiac_signal
        
        # Add noise
        noise = np.random.normal(0, self.config.noise_level * self.config.baseline_impedance,
                                  self.config.num_channels)
        sample += noise
        
        self._sample_count += 1
        return sample.astype(np.float32)
    
    def generate_batch(self, num_samples: int) -> np.ndarray:
        """
        Generate a batch of RIS samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            NumPy array of shape (num_samples, num_channels)
        """
        return np.array([self.generate_sample() for _ in range(num_samples)])
    
    def iter_samples(
        self,
        duration_seconds: float,
        batch_size: int = 100,
        create_packets: bool = True
    ) -> Generator[DataPacket, None, None]:
        """
        Generate RIS data for specified duration.
        
        Args:
            duration_seconds: Total duration to generate
            batch_size: Samples per packet
            create_packets: Whether to create DataPackets
            
        Yields:
            DataPacket for each batch
        """
        total_samples = int(duration_seconds * self.config.sample_rate)
        num_batches = total_samples // batch_size
        
        for batch_idx in range(num_batches):
            batch_data = self.generate_batch(batch_size)
            timestamp = DataSynchronizer.current_timestamp_ms()
            
            if create_packets:
                packet = self.synchronizer.create_packet(
                    modality=ModalityType.RIS,
                    data=batch_data,
                    metadata={
                        "batch_index": batch_idx,
                        "sample_rate": self.config.sample_rate,
                        "num_channels": self.config.num_channels,
                        "samples_in_batch": batch_size,
                        "start_sample": batch_idx * batch_size,
                        "heart_rate_hz": self._heart_rate,
                        "resp_rate_hz": self._resp_rate
                    },
                    timestamp=timestamp
                )
                yield packet
        
        logger.info(f"Generated {num_batches} RIS batches ({total_samples} samples)")
    
    def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from RIS data.
        
        Args:
            data: RIS data array (samples x channels)
            
        Returns:
            Dictionary of extracted features
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        features = {}
        
        # Basic statistics per channel
        features["mean_impedance"] = float(np.mean(data))
        features["std_impedance"] = float(np.std(data))
        features["min_impedance"] = float(np.min(data))
        features["max_impedance"] = float(np.max(data))
        
        # Cross-channel asymmetry (left-right comparison)
        if data.shape[1] >= 2:
            left_channels = data[:, :data.shape[1]//2]
            right_channels = data[:, data.shape[1]//2:]
            features["lr_asymmetry"] = float(
                np.abs(np.mean(left_channels) - np.mean(right_channels)) /
                (np.mean(data) + 1e-6)
            )
        
        # Regional analysis
        if data.shape[1] >= 8:
            features["thorax_mean"] = float(np.mean(data[:, :8]))
            features["abdomen_mean"] = float(np.mean(data[:, 8:]))
            features["thorax_abdom_ratio"] = (
                features["thorax_mean"] / (features["abdomen_mean"] + 1e-6)
            )
        
        # Frequency analysis for respiration/cardiac detection
        if data.shape[0] >= self.config.sample_rate:
            # Compute FFT on first channel
            freqs = np.fft.fftfreq(data.shape[0], 1/self.config.sample_rate)
            fft_vals = np.abs(np.fft.fft(data[:, 0]))
            
            # Find respiration peak (0.1-0.5 Hz)
            resp_mask = (freqs > 0.1) & (freqs < 0.5)
            if np.any(resp_mask):
                resp_idx = np.argmax(fft_vals[resp_mask])
                features["detected_resp_rate_hz"] = float(freqs[resp_mask][resp_idx])
            
            # Find cardiac peak (0.8-2.5 Hz)
            cardiac_mask = (freqs > 0.8) & (freqs < 2.5)
            if np.any(cardiac_mask):
                cardiac_idx = np.argmax(fft_vals[cardiac_mask])
                features["detected_heart_rate_hz"] = float(freqs[cardiac_mask][cardiac_idx])
        
        return features
    
    def simulate_fluid_shift(
        self,
        shift_direction: str = "upper",  # "upper" or "lower"
        intensity: float = 0.1
    ) -> None:
        """
        Simulate body fluid redistribution.
        
        Args:
            shift_direction: Direction of fluid shift
            intensity: Magnitude of shift (0-1)
        """
        # This would affect regional impedance patterns
        if shift_direction == "upper":
            # Fluid moving to upper body (e.g., lying down)
            logger.debug(f"Simulating upward fluid shift, intensity={intensity}")
        else:
            # Fluid moving to lower body (e.g., orthostatic)
            logger.debug(f"Simulating downward fluid shift, intensity={intensity}")
    
    def reset(self) -> None:
        """Reset simulator state."""
        self._sample_count = 0
        self._respiration_phase = 0.0
        self._cardiac_phase = 0.0
        logger.debug("RIS simulator reset")

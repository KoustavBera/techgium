"""
Auxiliary Sensor Ingestion

Handles heartbeat, thermal, and other supplementary sensor data
from wearables, thermal cameras, or CSV datasets.
"""
from typing import Optional, Dict, Any, Generator, List
from pathlib import Path
import numpy as np
import pandas as pd

from app.config import settings
from app.utils import get_logger, IngestionError
from .sync import DataPacket, DataSynchronizer, ModalityType

logger = get_logger(__name__)


class AuxiliaryIngestion:
    """
    Auxiliary sensor data ingestion handler.
    
    Supports:
    - CSV dataset loading (heart rate, temperature, etc.)
    - Real-time sensor stream parsing
    - Simulated auxiliary signals
    """
    
    def __init__(
        self,
        synchronizer: Optional[DataSynchronizer] = None
    ):
        """
        Initialize auxiliary ingestion.
        
        Args:
            synchronizer: Optional shared DataSynchronizer
        """
        self.synchronizer = synchronizer or DataSynchronizer()
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._current_idx: Dict[str, int] = {}
    
    def load_csv_dataset(
        self,
        filepath: str,
        dataset_type: str = "generic"
    ) -> pd.DataFrame:
        """
        Load a CSV dataset into memory.
        
        Args:
            filepath: Path to CSV file
            dataset_type: Type of dataset for parsing hints
            
        Returns:
            Loaded DataFrame
        """
        path = Path(filepath)
        if not path.exists():
            raise IngestionError(
                f"Dataset not found: {filepath}",
                modality="auxiliary"
            )
        
        try:
            df = pd.read_csv(filepath)
            self._data_cache[str(path)] = df
            self._current_idx[str(path)] = 0
            
            logger.info(
                f"Loaded {dataset_type} dataset: {path.name}, "
                f"{len(df)} rows, {list(df.columns)}"
            )
            return df
            
        except Exception as e:
            raise IngestionError(
                f"Failed to load dataset: {e}",
                modality="auxiliary",
                details={"filepath": filepath}
            )
    
    def load_heart_rate_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load heart rate time-series dataset.
        
        Expected columns: T1, T2, T3, T4 (multiple measurements)
        """
        df = self.load_csv_dataset(filepath, "heart_rate")
        
        # Validate expected columns
        expected = ["T1", "T2", "T3", "T4"]
        if not all(col in df.columns for col in expected):
            logger.warning(
                f"Heart rate dataset missing expected columns. "
                f"Found: {list(df.columns)}"
            )
        
        return df
    
    def load_vital_signs_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load comprehensive vital signs dataset.
        
        Expected columns: Patient ID, Heart Rate, Respiratory Rate,
        Timestamp, Body Temperature, Oxygen Saturation, etc.
        """
        df = self.load_csv_dataset(filepath, "vital_signs")
        
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        return df
    
    def load_iot_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load IoT healthcare sensor dataset.
        
        Expected columns: Patient_ID, Timestamp, Sensor_Type,
        Temperature, Systolic_BP, Diastolic_BP, Heart_Rate, etc.
        """
        df = self.load_csv_dataset(filepath, "iot")
        return df
    
    def iter_dataset_rows(
        self,
        filepath: str,
        create_packets: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate over dataset rows as data points.
        
        Args:
            filepath: Path to loaded dataset
            create_packets: Whether to create DataPackets
            
        Yields:
            Dictionary of values for each row
        """
        path_key = str(Path(filepath))
        
        if path_key not in self._data_cache:
            self.load_csv_dataset(filepath)
        
        df = self._data_cache[path_key]
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            timestamp = DataSynchronizer.current_timestamp_ms()
            
            if create_packets:
                # Convert row to numpy array
                numeric_values = [
                    v for v in row_dict.values()
                    if isinstance(v, (int, float)) and not pd.isna(v)
                ]
                
                self.synchronizer.create_packet(
                    modality=ModalityType.AUXILIARY,
                    data=np.array(numeric_values, dtype=np.float32),
                    metadata={
                        "row_index": idx,
                        "source": Path(filepath).name,
                        **{k: v for k, v in row_dict.items() 
                           if isinstance(v, (str, int, float)) and not pd.isna(v)}
                    },
                    timestamp=timestamp
                )
            
            yield row_dict
    
    def generate_synthetic_heartbeat(
        self,
        heart_rate_bpm: float = 72,
        duration_seconds: float = 10,
        sample_rate: int = 250
    ) -> np.ndarray:
        """
        Generate synthetic heartbeat waveform (PPG-like).
        
        Args:
            heart_rate_bpm: Target heart rate
            duration_seconds: Duration of signal
            sample_rate: Samples per second
            
        Returns:
            NumPy array of heartbeat signal
        """
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples)
        
        heart_rate_hz = heart_rate_bpm / 60.0
        
        # Primary cardiac wave
        signal_data = np.sin(2 * np.pi * heart_rate_hz * t)
        
        # Add harmonics for realistic waveform
        signal_data += 0.3 * np.sin(4 * np.pi * heart_rate_hz * t - 0.5)
        signal_data += 0.1 * np.sin(6 * np.pi * heart_rate_hz * t)
        
        # Add heart rate variability
        hrv_modulation = 0.05 * np.sin(2 * np.pi * 0.25 * t)
        signal_data *= (1 + hrv_modulation)
        
        # Add noise
        signal_data += np.random.normal(0, 0.05, num_samples)
        
        return signal_data.astype(np.float32)
    
    def generate_synthetic_thermal(
        self,
        base_temperature: float = 36.5,
        variation: float = 0.5,
        num_regions: int = 10
    ) -> np.ndarray:
        """
        Generate synthetic thermal map data.
        
        Args:
            base_temperature: Base body temperature (Celsius)
            variation: Temperature variation range
            num_regions: Number of body regions
            
        Returns:
            NumPy array of regional temperatures
        """
        # Different regions have different baseline temperatures
        region_offsets = np.array([
            0.0,    # Core
            -0.5,   # Extremity 1
            -0.5,   # Extremity 2
            -0.3,   # Extremity 3
            -0.3,   # Extremity 4
            0.2,    # Head
            0.1,    # Neck
            0.0,    # Chest
            0.0,    # Back
            -0.2    # Abdomen
        ])[:num_regions]
        
        temperatures = base_temperature + region_offsets
        temperatures += np.random.normal(0, variation * 0.1, num_regions)
        
        return temperatures.astype(np.float32)
    
    def iter_synthetic_auxiliary(
        self,
        duration_seconds: float,
        sample_rate: int = 10,
        create_packets: bool = True
    ) -> Generator[DataPacket, None, None]:
        """
        Generate synthetic auxiliary sensor data.
        
        Args:
            duration_seconds: Total duration
            sample_rate: Samples per second
            create_packets: Whether to create DataPackets
            
        Yields:
            DataPacket for each sample
        """
        num_samples = int(duration_seconds * sample_rate)
        
        # Simulate varying heart rate
        base_hr = 72
        hr_variation = 5
        
        for i in range(num_samples):
            timestamp = DataSynchronizer.current_timestamp_ms()
            
            # Time-varying heart rate
            current_hr = base_hr + hr_variation * np.sin(i / num_samples * 2 * np.pi)
            
            # Generate data point
            data = np.array([
                current_hr,                           # Heart rate
                16 + np.random.normal(0, 1),          # Respiratory rate
                36.5 + np.random.normal(0, 0.2),      # Temperature
                97 + np.random.normal(0, 1),          # SpO2
            ], dtype=np.float32)
            
            if create_packets:
                packet = self.synchronizer.create_packet(
                    modality=ModalityType.AUXILIARY,
                    data=data,
                    metadata={
                        "sample_index": i,
                        "source": "synthetic",
                        "heart_rate_bpm": float(current_hr),
                        "respiratory_rate": float(data[1]),
                        "temperature_c": float(data[2]),
                        "spo2_percent": float(data[3])
                    },
                    timestamp=timestamp
                )
                yield packet
        
        logger.info(f"Generated {num_samples} synthetic auxiliary samples")
    
    def extract_vital_features(
        self,
        data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract features from vital signs data.
        
        Args:
            data: Vital signs array
            metadata: Optional metadata with labels
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        if metadata:
            # Use labeled data if available
            if "heart_rate_bpm" in metadata:
                features["heart_rate"] = metadata["heart_rate_bpm"]
            if "respiratory_rate" in metadata:
                features["respiratory_rate"] = metadata["respiratory_rate"]
            if "temperature_c" in metadata:
                features["temperature"] = metadata["temperature_c"]
            if "spo2_percent" in metadata:
                features["spo2"] = metadata["spo2_percent"]
        else:
            # Assume standard order
            if len(data) >= 1:
                features["heart_rate"] = float(data[0])
            if len(data) >= 2:
                features["respiratory_rate"] = float(data[1])
            if len(data) >= 3:
                features["temperature"] = float(data[2])
            if len(data) >= 4:
                features["spo2"] = float(data[3])
        
        return features
    
    def clear_cache(self) -> None:
        """Clear loaded dataset cache."""
        self._data_cache.clear()
        self._current_idx.clear()
        logger.debug("Auxiliary data cache cleared")

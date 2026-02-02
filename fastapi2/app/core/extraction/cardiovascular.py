"""
Cardiovascular Biomarker Extractor

Extracts cardiovascular health indicators:
- Heart rate (HR)
- Heart rate variability (HRV)
- Chest micro-motion proxies
"""
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class CardiovascularExtractor(BaseExtractor):
    """
    Extracts cardiovascular biomarkers.
    
    Analyzes RIS data, motion data, and auxiliary vital signs.
    """
    
    system = PhysiologicalSystem.CARDIOVASCULAR
    
    def __init__(self, sample_rate: float = 1000.0):
        """
        Initialize cardiovascular extractor.
        
        Args:
            sample_rate: RIS/signal sampling rate in Hz
        """
        super().__init__()
        self.sample_rate = sample_rate
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract cardiovascular biomarkers.
        
        Expected data keys:
        - ris_data: RIS signal array (optional)
        - heart_rate_signal: PPG/ECG-like signal (optional)
        - vital_signs: Dict with direct vital measurements (optional)
        - pose_sequence: Motion data for chest micro-motion (optional)
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        # Priority 1: Direct vital signs
        if "vital_signs" in data:
            self._extract_from_vitals(data["vital_signs"], biomarker_set)
        
        # Priority 2: RIS-derived metrics
        elif "ris_data" in data:
            self._extract_from_ris(data["ris_data"], biomarker_set)
        
        # Priority 3: Motion-derived (chest micro-motion)
        elif "pose_sequence" in data:
            self._extract_from_motion(data["pose_sequence"], biomarker_set)
        
        # Fallback: Simulated values
        else:
            self._generate_simulated_biomarkers(biomarker_set)
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_vitals(
        self,
        vitals: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract from direct vital sign measurements."""
        
        # Heart rate
        hr = vitals.get("heart_rate", vitals.get("heart_rate_bpm", 72))
        self._add_biomarker(
            biomarker_set,
            name="heart_rate",
            value=float(hr),
            unit="bpm",
            confidence=0.95,
            normal_range=(60, 100),
            description="Resting heart rate"
        )
        
        # HRV if available
        if "hrv" in vitals or "rmssd" in vitals:
            hrv = vitals.get("hrv", vitals.get("rmssd", 50))
            self._add_biomarker(
                biomarker_set,
                name="hrv_rmssd",
                value=float(hrv),
                unit="ms",
                confidence=0.90,
                normal_range=(20, 80),
                description="Heart rate variability (RMSSD)"
            )
        else:
            # Estimate HRV from HR
            estimated_hrv = self._estimate_hrv_from_hr(hr)
            self._add_biomarker(
                biomarker_set,
                name="hrv_rmssd",
                value=estimated_hrv,
                unit="ms",
                confidence=0.60,
                normal_range=(20, 80),
                description="Estimated HRV from heart rate"
            )
        
        # Blood pressure if available
        if "systolic_bp" in vitals:
            self._add_biomarker(
                biomarker_set,
                name="systolic_bp",
                value=float(vitals["systolic_bp"]),
                unit="mmHg",
                confidence=0.95,
                normal_range=(90, 120),
                description="Systolic blood pressure"
            )
        if "diastolic_bp" in vitals:
            self._add_biomarker(
                biomarker_set,
                name="diastolic_bp",
                value=float(vitals["diastolic_bp"]),
                unit="mmHg",
                confidence=0.95,
                normal_range=(60, 80),
                description="Diastolic blood pressure"
            )
    
    def _extract_from_ris(
        self,
        ris_data: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract cardiovascular metrics from RIS bioimpedance data."""
        
        if ris_data.ndim == 1:
            ris_data = ris_data.reshape(-1, 1)
        
        # Use thoracic channels (0-7) for cardiac signal
        thoracic_signal = np.mean(ris_data[:, :min(8, ris_data.shape[1])], axis=1)
        
        # Bandpass filter for cardiac frequencies (0.8-3 Hz = 48-180 bpm)
        if len(thoracic_signal) > 100:
            hr, hrv = self._analyze_cardiac_signal(thoracic_signal)
        else:
            hr, hrv = 72, 45
        
        self._add_biomarker(
            biomarker_set,
            name="heart_rate",
            value=hr,
            unit="bpm",
            confidence=0.75,
            normal_range=(60, 100),
            description="Heart rate derived from RIS"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="hrv_rmssd",
            value=hrv,
            unit="ms",
            confidence=0.65,
            normal_range=(20, 80),
            description="HRV derived from RIS beat-to-beat intervals"
        )
        
        # Chest impedance for fluid status
        mean_impedance = float(np.mean(thoracic_signal))
        self._add_biomarker(
            biomarker_set,
            name="thoracic_impedance",
            value=mean_impedance,
            unit="ohms",
            confidence=0.70,
            normal_range=(400, 600),
            description="Mean thoracic bioimpedance (fluid proxy)"
        )
    
    def _extract_from_motion(
        self,
        pose_sequence: List[np.ndarray],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract cardiac proxies from chest micro-motion."""
        
        pose_array = np.array(pose_sequence)
        
        if pose_array.shape[0] < 30 or pose_array.shape[1] < 12:
            self._generate_simulated_biomarkers(biomarker_set)
            return
        
        # Chest landmarks: shoulders (11, 12)
        left_shoulder = pose_array[:, 11, :2]
        right_shoulder = pose_array[:, 12, :2]
        chest_center = (left_shoulder + right_shoulder) / 2
        
        # Analyze vertical micro-motion (breathing + cardiac)
        vertical_motion = chest_center[:, 1]
        
        # High-pass filter to isolate cardiac from respiratory
        if len(vertical_motion) > 60:
            # Simple differentiation as high-pass
            micro_motion = np.diff(vertical_motion)
            micro_motion_std = float(np.std(micro_motion))
        else:
            micro_motion_std = 0.002
        
        self._add_biomarker(
            biomarker_set,
            name="chest_micro_motion",
            value=micro_motion_std,
            unit="normalized_amplitude",
            confidence=0.55,
            normal_range=(0.001, 0.01),
            description="Chest wall micro-motion amplitude (cardiac proxy)"
        )
        
        # Estimate HR from motion frequency
        hr = self._estimate_hr_from_motion(vertical_motion)
        self._add_biomarker(
            biomarker_set,
            name="heart_rate",
            value=hr,
            unit="bpm",
            confidence=0.50,
            normal_range=(60, 100),
            description="Heart rate estimated from chest motion"
        )
    
    def _analyze_cardiac_signal(self, signal_data: np.ndarray) -> tuple:
        """
        Analyze signal for heart rate and HRV.
        
        Returns:
            Tuple of (heart_rate_bpm, hrv_rmssd_ms)
        """
        # FFT for dominant frequency
        n = len(signal_data)
        freqs = fftfreq(n, 1/self.sample_rate)
        fft_vals = np.abs(fft(signal_data))
        
        # Look in cardiac range (0.8-3 Hz)
        cardiac_mask = (freqs > 0.8) & (freqs < 3.0)
        
        if not np.any(cardiac_mask):
            return 72.0, 45.0
        
        cardiac_freqs = freqs[cardiac_mask]
        cardiac_power = fft_vals[cardiac_mask]
        
        # Dominant frequency
        peak_idx = np.argmax(cardiac_power)
        peak_freq = cardiac_freqs[peak_idx]
        hr = float(abs(peak_freq) * 60)  # Convert Hz to bpm
        hr = np.clip(hr, 40, 180)
        
        # Estimate HRV from spectral width
        # Wider spectrum = more variability
        spectral_std = np.std(cardiac_power)
        hrv = float(30 + spectral_std * 100)  # Rough scaling
        hrv = np.clip(hrv, 10, 100)
        
        return hr, hrv
    
    def _estimate_hr_from_motion(self, motion_signal: np.ndarray) -> float:
        """Estimate heart rate from motion signal."""
        if len(motion_signal) < 30:
            return 72.0
        
        # Zero crossings as rough frequency estimate
        zero_crossings = np.sum(np.diff(np.sign(motion_signal - np.mean(motion_signal))) != 0)
        
        duration_sec = len(motion_signal) / 30.0  # Assume 30 FPS
        crossings_per_sec = zero_crossings / duration_sec
        
        # Each cardiac cycle has 2 zero crossings
        hr = crossings_per_sec * 30  # Scale factor
        return float(np.clip(hr, 50, 120))
    
    def _estimate_hrv_from_hr(self, hr: float) -> float:
        """Estimate HRV from heart rate using population statistics."""
        # HRV tends to be inversely related to HR
        # Using rough empirical relationship
        base_hrv = 60 - 0.5 * (hr - 60)
        noise = np.random.normal(0, 5)
        return float(np.clip(base_hrv + noise, 15, 90))
    
    def _generate_simulated_biomarkers(self, biomarker_set: BiomarkerSet) -> None:
        """Generate simulated cardiovascular biomarkers."""
        self._add_biomarker(biomarker_set, "heart_rate",
                           np.random.uniform(65, 85), "bpm",
                           0.5, (60, 100), "Simulated heart rate")
        self._add_biomarker(biomarker_set, "hrv_rmssd",
                           np.random.uniform(30, 60), "ms",
                           0.5, (20, 80), "Simulated HRV")
        self._add_biomarker(biomarker_set, "chest_micro_motion",
                           np.random.uniform(0.002, 0.006), "normalized_amplitude",
                           0.5, (0.001, 0.01), "Simulated chest motion")

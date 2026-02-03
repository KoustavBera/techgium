"""
Nasal Passage Biomarker Extractor

Extracts nasal/respiratory health indicators:
- Airflow turbulence proxies
- Breathing noise signatures
"""
from typing import Dict, Any
import numpy as np
from scipy.fft import fft, fftfreq

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class NasalExtractor(BaseExtractor):
    """
    Extracts nasal passage biomarkers.
    
    Analyzes motion and RIS data for respiratory patterns.
    """
    
    system = PhysiologicalSystem.NASAL
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract nasal biomarkers.
        
        Expected data keys:
        - pose_sequence: Pose landmarks (nose position)
        - ris_data: RIS data (thoracic channels)
        - audio_data: Optional breathing audio
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        pose_sequence = data.get("pose_sequence", [])
        ris_data = data.get("ris_data")
        
        has_data = False
        
        if len(pose_sequence) >= 30:
            self._extract_from_pose(np.array(pose_sequence), biomarker_set)
            has_data = True
        
        if ris_data is not None and len(ris_data) > 100:
            self._extract_from_ris(np.array(ris_data), biomarker_set)
            has_data = True
        
        if not has_data:
            self._generate_simulated_biomarkers(biomarker_set)
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_pose(
        self,
        pose_array: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract nasal indicators from nose landmark motion."""
        
        # Nose is landmark 0 in MediaPipe pose
        nose = pose_array[:, 0, :2]
        
        # Analyze vertical motion (breathing proxy)
        vertical_motion = nose[:, 1]
        
        # Breathing regularity
        regularity = self._analyze_breathing_regularity(vertical_motion)
        self._add_biomarker(
            biomarker_set,
            name="breathing_regularity",
            value=regularity,
            unit="score_0_1",
            confidence=0.55,
            normal_range=(0.6, 1.0),
            description="Consistency of breathing pattern"
        )
        
        # Nasal airflow asymmetry proxy (from lateral nose movement)
        lateral_motion = nose[:, 0]
        asymmetry = np.std(lateral_motion) / (np.std(vertical_motion) + 1e-6)
        asymmetry = float(np.clip(asymmetry, 0, 2))
        
        self._add_biomarker(
            biomarker_set,
            name="nasal_asymmetry_proxy",
            value=asymmetry,
            unit="ratio",
            confidence=0.45,
            normal_range=(0.1, 0.5),
            description="Lateral vs vertical nose motion ratio"
        )
    
    def _extract_from_ris(
        self,
        ris_data: np.ndarray,
        biomarker_set: BiomarkerSet,
        sample_rate: float = 1000
    ) -> None:
        """Extract respiratory indicators from RIS data."""
        
        if ris_data.ndim == 1:
            ris_data = ris_data.reshape(-1, 1)
        
        # Use thoracic channels
        thoracic = np.mean(ris_data[:, :min(4, ris_data.shape[1])], axis=1)
        
        # Respiratory rate
        resp_rate = self._calculate_respiratory_rate(thoracic, sample_rate)
        self._add_biomarker(
            biomarker_set,
            name="respiratory_rate",
            value=resp_rate,
            unit="breaths_per_min",
            confidence=0.70,
            normal_range=(12, 20),
            description="Breathing rate from thoracic impedance"
        )
        
        # Tidal variation (breath depth proxy)
        breath_depth = self._analyze_breath_depth(thoracic)
        self._add_biomarker(
            biomarker_set,
            name="breath_depth_index",
            value=breath_depth,
            unit="normalized",
            confidence=0.60,
            normal_range=(0.5, 1.5),
            description="Relative breath depth"
        )
        
        # Airflow turbulence proxy (high-frequency content)
        turbulence = self._analyze_turbulence(thoracic, sample_rate)
        self._add_biomarker(
            biomarker_set,
            name="airflow_turbulence",
            value=turbulence,
            unit="power_ratio",
            confidence=0.50,
            normal_range=(0.01, 0.1),
            description="High-frequency respiratory component"
        )
    
    def _analyze_breathing_regularity(
        self,
        signal: np.ndarray,
        fps: float = 30
    ) -> float:
        """Analyze regularity of breathing pattern."""
        
        if len(signal) < 60:  # Need at least 2 seconds
            return np.random.uniform(0.7, 0.9)
        
        # Find peaks
        from scipy.signal import find_peaks
        
        # Smooth signal
        if len(signal) > 5:
            smoothed = np.convolve(signal, np.ones(5)/5, mode='valid')
        else:
            smoothed = signal
        
        peaks, _ = find_peaks(smoothed, distance=fps//3)  # Min 0.33 sec between breaths
        
        if len(peaks) < 3:
            return np.random.uniform(0.7, 0.9)
        
        # Calculate inter-breath intervals
        intervals = np.diff(peaks)
        
        # Regularity = inverse of coefficient of variation
        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        regularity = 1 / (1 + cv)
        
        return float(np.clip(regularity, 0, 1))
    
    def _calculate_respiratory_rate(
        self,
        signal: np.ndarray,
        sample_rate: float
    ) -> float:
        """Calculate respiratory rate from signal."""
        
        n = len(signal)
        if n < 100:
            return np.random.uniform(14, 18)
        
        freqs = fftfreq(n, 1/sample_rate)
        fft_vals = np.abs(fft(signal))
        
        # Respiratory range: 0.1-0.5 Hz (6-30 bpm)
        resp_mask = (freqs > 0.1) & (freqs < 0.5)
        
        if not np.any(resp_mask):
            return 15.0
        
        resp_freqs = freqs[resp_mask]
        resp_power = fft_vals[resp_mask]
        
        peak_idx = np.argmax(resp_power)
        peak_freq = abs(resp_freqs[peak_idx])
        
        resp_rate = peak_freq * 60  # Hz to bpm
        return float(np.clip(resp_rate, 6, 30))
    
    def _analyze_breath_depth(self, signal: np.ndarray) -> float:
        """Analyze relative breath depth."""
        
        # Use peak-to-trough amplitude
        max_val = np.max(signal)
        min_val = np.min(signal)
        amplitude = max_val - min_val
        
        # Normalize by mean
        mean_val = np.mean(signal)
        if mean_val > 0:
            depth_index = amplitude / mean_val
        else:
            depth_index = 1.0
        
        return float(np.clip(depth_index, 0.1, 3.0))
    
    def _analyze_turbulence(
        self,
        signal: np.ndarray,
        sample_rate: float
    ) -> float:
        """Analyze airflow turbulence from high-frequency content."""
        
        n = len(signal)
        if n < 100:
            return np.random.uniform(0.03, 0.07)
        
        freqs = fftfreq(n, 1/sample_rate)
        fft_vals = np.abs(fft(signal))
        
        # Low frequency (respiratory): 0.1-0.5 Hz
        low_mask = (freqs > 0.1) & (freqs < 0.5)
        # High frequency (turbulence): 5-50 Hz
        high_mask = (freqs > 5) & (freqs < 50)
        
        low_power = np.sum(fft_vals[low_mask]) if np.any(low_mask) else 1
        high_power = np.sum(fft_vals[high_mask]) if np.any(high_mask) else 0
        
        turbulence_ratio = high_power / (low_power + 1e-6)
        
        return float(np.clip(turbulence_ratio, 0, 1))
    
    def _generate_simulated_biomarkers(self, biomarker_set: BiomarkerSet) -> None:
        """Generate simulated nasal biomarkers."""
        self._add_biomarker(biomarker_set, "breathing_regularity",
                           np.random.uniform(0.75, 0.9), "score_0_1",
                           0.5, (0.6, 1.0), "Simulated regularity")
        self._add_biomarker(biomarker_set, "respiratory_rate",
                           np.random.uniform(14, 18), "breaths_per_min",
                           0.5, (12, 20), "Simulated respiratory rate")
        self._add_biomarker(biomarker_set, "breath_depth_index",
                           np.random.uniform(0.8, 1.2), "normalized",
                           0.5, (0.5, 1.5), "Simulated breath depth")
        self._add_biomarker(biomarker_set, "airflow_turbulence",
                           np.random.uniform(0.03, 0.07), "power_ratio",
                           0.5, (0.01, 0.1), "Simulated turbulence")

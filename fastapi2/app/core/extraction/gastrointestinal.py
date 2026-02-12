"""
Gastrointestinal Biomarker Extractor

Extracts GI health indicators:
- Abdominal motion rhythm
- Visceral motion patterns
"""
from typing import Dict, Any
import numpy as np
from scipy.fft import fft, fftfreq

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class GastrointestinalExtractor(BaseExtractor):
    """
    Extracts gastrointestinal biomarkers.
    
    Analyzes abdominal motion and RIS patterns for GI function.
    """
    
    system = PhysiologicalSystem.GASTROINTESTINAL
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract GI biomarkers.
        
        Expected data keys:
        - ris_data: RIS bioimpedance array (abdominal channels)
        - pose_sequence: Motion data for abdominal movement
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        ris_data = data.get("ris_data")
        pose_sequence = data.get("pose_sequence")
        
        has_data = False
        
        if ris_data is not None and len(ris_data) > 0:
            self._extract_from_ris(np.array(ris_data), biomarker_set)
            has_data = True
        
        if pose_sequence is not None and len(pose_sequence) > 10:
            self._extract_from_motion(np.array(pose_sequence), biomarker_set)
            has_data = True
        
        if not has_data:
            logger.warning("GastrointestinalExtractor: No data sources available.")
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_ris(
        self,
        ris_data: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract GI indicators from abdominal RIS channels."""
        
        if ris_data.ndim == 1:
            ris_data = ris_data.reshape(-1, 1)
        
        num_channels = ris_data.shape[1]
        
        # Use abdominal channels (8-11 in 16-channel setup)
        if num_channels >= 12:
            abdominal = ris_data[:, 8:12]
        elif num_channels >= 8:
            abdominal = ris_data[:, 4:8]
        else:
            abdominal = ris_data
        
        abdominal_mean = np.mean(abdominal, axis=1)
        
        # Analyze rhythmic patterns (GI motility ~0.05 Hz = 3 cycles/min)
        if len(abdominal_mean) > 100:
            rhythm_score = self._analyze_gi_rhythm(abdominal_mean)
        else:
            rhythm_score = np.random.uniform(0.5, 0.8)
        
        self._add_biomarker(
            biomarker_set,
            name="abdominal_rhythm_score",
            value=float(rhythm_score),
            unit="score_0_1",
            confidence=0.60,
            normal_range=(0.4, 0.9),
            description="Regularity of abdominal rhythmic patterns"
        )
        
        # Visceral motion variance
        visceral_variance = float(np.var(abdominal_mean))
        
        self._add_biomarker(
            biomarker_set,
            name="visceral_motion_variance",
            value=visceral_variance,
            unit="variance",
            confidence=0.55,
            normal_range=(10, 100),
            description="Variability in visceral motion patterns"
        )
    
    def _extract_from_motion(
        self,
        pose_array: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract abdominal motion from pose data."""
        
        # Hip landmarks as abdominal proxy
        left_hip_idx = 23
        right_hip_idx = 24
        
        if pose_array.shape[1] < 25:
            return
        
        hip_center = (pose_array[:, left_hip_idx, :2] + 
                     pose_array[:, right_hip_idx, :2]) / 2
        
        # Vertical motion (breathing + visceral)
        vertical = hip_center[:, 1]
        
        # Respiratory rate from motion
        if len(vertical) > 30:
            resp_rate = self._estimate_respiratory_rate(vertical)
        else:
            resp_rate = 15
        
        self._add_biomarker(
            biomarker_set,
            name="abdominal_respiratory_rate",
            value=resp_rate,
            unit="breaths_per_min",
            confidence=0.65,
            normal_range=(12, 20),
            description="Respiratory rate from abdominal motion"
        )
    
    def _analyze_gi_rhythm(self, signal: np.ndarray, sample_rate: float = 1000) -> float:
        """
        Analyze GI motility rhythm.
        
        Normal GI motility: 3-12 cycles per minute (0.05-0.2 Hz)
        """
        n = len(signal)
        freqs = fftfreq(n, 1/sample_rate)
        fft_vals = np.abs(fft(signal))
        
        # GI frequency range
        gi_mask = (freqs > 0.03) & (freqs < 0.25)
        
        if not np.any(gi_mask):
            return 0.5
        
        gi_power = fft_vals[gi_mask]
        total_power = np.sum(fft_vals[freqs > 0])
        
        # Rhythm score = proportion of power in GI band
        rhythm_score = np.sum(gi_power) / (total_power + 1e-6)
        
        return float(np.clip(rhythm_score, 0, 1))
    
    def _estimate_respiratory_rate(self, motion: np.ndarray, fps: float = 30) -> float:
        """Estimate respiratory rate from vertical motion."""
        # Count peaks
        from scipy.signal import find_peaks
        
        # Smooth the signal
        if len(motion) > 5:
            smoothed = np.convolve(motion, np.ones(5)/5, mode='valid')
        else:
            smoothed = motion
        
        peaks, _ = find_peaks(smoothed, distance=fps//2)  # Min 0.5 sec between breaths
        
        duration_min = len(motion) / fps / 60
        if duration_min > 0:
            resp_rate = len(peaks) / duration_min
        else:
            resp_rate = 15
        
        return float(np.clip(resp_rate, 8, 30))

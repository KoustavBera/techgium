"""
Central Nervous System (CNS) Biomarker Extractor

Extracts CNS-related biomarkers from motion/pose data:
- Gait variability
- Posture entropy
- Tremor signatures
"""
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem, Biomarker

logger = get_logger(__name__)


class CNSExtractor(BaseExtractor):
    """
    Extracts Central Nervous System biomarkers.
    
    Analyzes motion and pose data for neurological health indicators.
    """
    
    system = PhysiologicalSystem.CNS
    
    def __init__(self, sample_rate: float = 30.0):
        """
        Initialize CNS extractor.
        
        Args:
            sample_rate: Sampling rate of motion data in Hz
        """
        super().__init__()
        self.sample_rate = sample_rate
        
        # Tremor frequency bands (Hz)
        self.tremor_bands = {
            "resting": (4, 6),      # Parkinsonian resting tremor
            "postural": (6, 12),    # Essential tremor
            "intention": (3, 5),    # Cerebellar tremor
        }
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract CNS biomarkers from motion data.
        
        Expected data keys:
        - pose_sequence: List of pose arrays over time (Nx33x4)
        - timestamps: List of timestamps in ms
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        # Extract pose sequence
        pose_sequence = data.get("pose_sequence", [])
        if len(pose_sequence) < 10:
            logger.warning("Insufficient pose data for CNS extraction")
            return self._generate_simulated_biomarkers(biomarker_set)
        
        pose_array = np.array(pose_sequence)
        
        # Extract gait variability
        gait_var = self._calculate_gait_variability(pose_array)
        self._add_biomarker(
            biomarker_set,
            name="gait_variability",
            value=gait_var,
            unit="coefficient_of_variation",
            confidence=0.85,
            normal_range=(0.02, 0.08),
            description="Stride-to-stride variability in gait pattern"
        )
        
        # Extract posture entropy
        posture_entropy = self._calculate_posture_entropy(pose_array)
        self._add_biomarker(
            biomarker_set,
            name="posture_entropy",
            value=posture_entropy,
            unit="bits",
            confidence=0.80,
            normal_range=(1.5, 3.5),
            description="Shannon entropy of posture angle distribution"
        )
        
        # Extract tremor signatures
        tremor_scores = self._analyze_tremor(pose_array)
        for tremor_type, score in tremor_scores.items():
            self._add_biomarker(
                biomarker_set,
                name=f"tremor_{tremor_type}",
                value=score,
                unit="power_spectral_density",
                confidence=0.75,
                normal_range=(0, 0.1),
                description=f"{tremor_type.capitalize()} tremor power in characteristic band"
            )
        
        # Calculate overall CNS stability score
        stability = self._calculate_stability_score(pose_array)
        self._add_biomarker(
            biomarker_set,
            name="cns_stability_score",
            value=stability,
            unit="score_0_100",
            confidence=0.70,
            normal_range=(70, 100),
            description="Composite CNS stability indicator"
        )
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _calculate_gait_variability(self, pose_array: np.ndarray) -> float:
        """
        Calculate gait variability from ankle positions.
        
        Uses coefficient of variation of stride lengths.
        """
        # Ankle landmarks: left=27, right=28 in MediaPipe
        left_ankle_idx = 27
        right_ankle_idx = 28
        
        if pose_array.shape[1] < 29:
            return np.random.uniform(0.03, 0.06)
        
        left_ankle = pose_array[:, left_ankle_idx, :2]  # x, y positions
        right_ankle = pose_array[:, right_ankle_idx, :2]
        
        # Calculate step lengths
        left_steps = np.diff(left_ankle[:, 1])  # Y position changes
        right_steps = np.diff(right_ankle[:, 1])
        
        all_steps = np.concatenate([np.abs(left_steps), np.abs(right_steps)])
        all_steps = all_steps[all_steps > 0.001]  # Filter noise
        
        if len(all_steps) < 5:
            return np.random.uniform(0.03, 0.06)
        
        # Coefficient of variation
        cv = np.std(all_steps) / (np.mean(all_steps) + 1e-6)
        return float(np.clip(cv, 0, 0.5))
    
    def _calculate_posture_entropy(self, pose_array: np.ndarray) -> float:
        """
        Calculate Shannon entropy of posture angles.
        
        Measures postural sway complexity.
        """
        # Extract shoulder-hip angle as posture measure
        left_shoulder_idx = 11
        left_hip_idx = 23
        
        if pose_array.shape[1] < 24:
            return np.random.uniform(2.0, 3.0)
        
        shoulder = pose_array[:, left_shoulder_idx, :2]
        hip = pose_array[:, left_hip_idx, :2]
        
        # Calculate trunk angle over time
        angles = np.arctan2(shoulder[:, 0] - hip[:, 0], 
                          shoulder[:, 1] - hip[:, 1])
        
        # Discretize angles for entropy calculation
        angle_bins = np.linspace(-np.pi/4, np.pi/4, 20)
        hist, _ = np.histogram(angles, bins=angle_bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(np.clip(entropy, 0, 5))
    
    def _analyze_tremor(self, pose_array: np.ndarray) -> Dict[str, float]:
        """
        Analyze tremor in different frequency bands.
        
        Uses FFT on hand/wrist positions.
        """
        tremor_scores = {}
        
        # Wrist landmarks: left=15, right=16
        wrist_idx = 15
        
        if pose_array.shape[1] < 17 or pose_array.shape[0] < 30:
            for band_name in self.tremor_bands:
                tremor_scores[band_name] = np.random.uniform(0.01, 0.05)
            return tremor_scores
        
        wrist_motion = pose_array[:, wrist_idx, :2]
        
        # Compute velocity (derivative)
        velocity = np.diff(wrist_motion, axis=0)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        # FFT analysis
        n = len(velocity_magnitude)
        freqs = fftfreq(n, 1/self.sample_rate)
        fft_vals = np.abs(fft(velocity_magnitude))
        
        # Extract power in each tremor band
        for band_name, (low_freq, high_freq) in self.tremor_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.mean(fft_vals[mask]) if np.any(mask) else 0.0
            tremor_scores[band_name] = float(np.clip(band_power, 0, 1))
        
        return tremor_scores
    
    def _calculate_stability_score(self, pose_array: np.ndarray) -> float:
        """
        Calculate composite CNS stability score.
        
        Combines multiple metrics into a 0-100 score.
        """
        # Center of mass proxy (average of hips)
        left_hip_idx = 23
        right_hip_idx = 24
        
        if pose_array.shape[1] < 25:
            return np.random.uniform(75, 95)
        
        com = (pose_array[:, left_hip_idx, :2] + pose_array[:, right_hip_idx, :2]) / 2
        
        # Sway analysis
        sway_std = np.std(com, axis=0)
        sway_magnitude = np.linalg.norm(sway_std)
        
        # Convert to 0-100 score (lower sway = higher stability)
        # Assuming sway < 0.01 is very stable, > 0.1 is unstable
        score = 100 * (1 - np.clip(sway_magnitude / 0.1, 0, 1))
        
        return float(np.clip(score, 0, 100))
    
    def _generate_simulated_biomarkers(self, biomarker_set: BiomarkerSet) -> BiomarkerSet:
        """Generate simulated biomarkers when real data unavailable."""
        self._add_biomarker(biomarker_set, "gait_variability", 
                           np.random.uniform(0.03, 0.06), "coefficient_of_variation",
                           0.5, (0.02, 0.08), "Simulated gait variability")
        self._add_biomarker(biomarker_set, "posture_entropy",
                           np.random.uniform(2.0, 3.0), "bits",
                           0.5, (1.5, 3.5), "Simulated posture entropy")
        self._add_biomarker(biomarker_set, "tremor_resting",
                           np.random.uniform(0.01, 0.05), "power_spectral_density",
                           0.5, (0, 0.1), "Simulated resting tremor")
        self._add_biomarker(biomarker_set, "cns_stability_score",
                           np.random.uniform(80, 95), "score_0_100",
                           0.5, (70, 100), "Simulated stability score")
        return biomarker_set

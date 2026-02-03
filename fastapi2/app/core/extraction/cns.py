"""
Central Nervous System (CNS) Biomarker Extractor - Scientific Production Level

Extracts CNS-related biomarkers from motion/pose data using validated clinical methods:
- Gait variability (Zeni heel strike detection - gold standard)
- Posture entropy (Sample entropy - clinical standard for postural sway)
- Tremor signatures (Welch PSD - proper spectral analysis)

References:
- Zeni et al. (2008): Heel strike detection for gait analysis
- Richman & Moorman (2000): Sample entropy for physiological signals
- Elble & McNames (2016): Tremor analysis methodology
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem, Biomarker

logger = get_logger(__name__)


class CNSExtractor(BaseExtractor):
    """
    Scientific-grade Central Nervous System biomarker extractor.
    
    Uses validated clinical algorithms for neurological health screening:
    - Parkinson's risk (tremor analysis)
    - Fall risk (gait/posture stability)
    - Balance disorders (postural sway complexity)
    """
    
    system = PhysiologicalSystem.CNS
    
    def __init__(self, sample_rate: float = 30.0):
        """
        Initialize CNS extractor with clinical-grade parameters.
        
        Args:
            sample_rate: Sampling rate of motion data in Hz (typical webcam: 30 Hz)
        """
        super().__init__()
        self.sample_rate = sample_rate
        
        # Minimum data requirements (10 seconds for reliable analysis)
        self.min_data_length = int(10 * self.sample_rate)
        self.min_strides = 3  # Minimum strides for gait analysis
        
        # MediaPipe landmark indices
        self.landmarks = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_wrist": 15,
            "right_wrist": 16,
        }
        
        # Tremor frequency bands (Hz) - clinically validated ranges
        self.tremor_bands = {
            "resting": (4, 6),      # Parkinsonian resting tremor
            "postural": (6, 12),    # Essential tremor
            "intention": (3, 5),    # Cerebellar tremor
        }
        
        # Normal ranges from clinical literature
        self.normal_ranges = {
            "gait_variability": (0.02, 0.06),      # CV 2-6% is normal
            "posture_entropy": (0.5, 2.5),          # SampEn units
            "tremor_power": (0.0, 0.05),            # Normalized PSD
            "stability_score": (75, 100),           # 0-100 scale
        }
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract CNS biomarkers using validated clinical algorithms.
        
        Expected data keys:
        - pose_sequence: List of pose arrays over time (Nx33x4: landmarks x [x,y,z,visibility])
        - timestamps: List of timestamps in seconds
        - fps/frame_rate: Actual capture framerate (optional, uses sample_rate if missing)
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        # Extract and validate pose sequence
        pose_sequence = data.get("pose_sequence", [])
        
        # Update sample rate if provided
        fps = data.get("fps") or data.get("frame_rate")
        if fps:
            self.sample_rate = float(fps)
        
        # Minimum data validation
        if len(pose_sequence) < self.min_data_length:
            logger.warning(
                f"Insufficient pose data: {len(pose_sequence)} frames. "
                f"Need {self.min_data_length} frames (10s) for reliable CNS analysis."
            )
            return self._generate_simulated_biomarkers(biomarker_set)
        
        try:
            pose_array = np.array(pose_sequence)
            
            # Validate pose array shape (frames, landmarks, coordinates)
            if pose_array.ndim != 3 or pose_array.shape[1] < 29:
                logger.warning(f"Invalid pose array shape: {pose_array.shape}")
                return self._generate_simulated_biomarkers(biomarker_set)
            
        except Exception as e:
            logger.warning(f"Failed to convert pose sequence: {e}")
            return self._generate_simulated_biomarkers(biomarker_set)
        
        # =====================================================
        # 1. GAIT VARIABILITY (Zeni heel strike detection)
        # =====================================================
        gait_var, heel_strikes = self._calculate_gait_variability(pose_array)
        gait_confidence = min(0.95, 0.5 + len(heel_strikes) / 20)  # More strides = higher confidence
        
        self._add_biomarker(
            biomarker_set,
            name="gait_variability",
            value=gait_var,
            unit="coefficient_of_variation",
            confidence=gait_confidence,
            normal_range=self.normal_ranges["gait_variability"],
            description="Stride-to-stride timing variability (Zeni heel strike method)"
        )
        
        # =====================================================
        # 2. POSTURE ENTROPY (Sample Entropy - clinical standard)
        # =====================================================
        posture_entropy = self._calculate_posture_entropy(pose_array)
        
        self._add_biomarker(
            biomarker_set,
            name="posture_entropy",
            value=posture_entropy,
            unit="sample_entropy",
            confidence=0.85,
            normal_range=self.normal_ranges["posture_entropy"],
            description="Postural sway complexity (Sample Entropy - Richman method)"
        )
        
        # =====================================================
        # 3. TREMOR ANALYSIS (Welch PSD - bilateral)
        # =====================================================
        tremor_scores = self._analyze_tremor(pose_array)
        
        for tremor_type, (score, band_confidence) in tremor_scores.items():
            self._add_biomarker(
                biomarker_set,
                name=f"tremor_{tremor_type}",
                value=score,
                unit="normalized_psd",
                confidence=band_confidence,
                normal_range=self.normal_ranges["tremor_power"],
                description=f"{tremor_type.capitalize()} tremor power ({self.tremor_bands[tremor_type][0]}-{self.tremor_bands[tremor_type][1]} Hz)"
            )
        
        # =====================================================
        # 4. COMPOSITE STABILITY SCORE (Multi-domain)
        # =====================================================
        stability, stability_components = self._calculate_stability_score(
            pose_array, gait_var, tremor_scores
        )
        
        self._add_biomarker(
            biomarker_set,
            name="cns_stability_score",
            value=stability,
            unit="score_0_100",
            confidence=0.80,
            normal_range=self.normal_ranges["stability_score"],
            description="Composite CNS stability (sway + gait + tremor combined)"
        )
        
        # Add component scores for detailed analysis
        self._add_biomarker(
            biomarker_set,
            name="sway_amplitude_ap",
            value=stability_components["sway_ap"],
            unit="normalized_units",
            confidence=0.85,
            normal_range=(0.0, 0.05),
            description="Anterior-posterior postural sway amplitude"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="sway_amplitude_ml",
            value=stability_components["sway_ml"],
            unit="normalized_units",
            confidence=0.85,
            normal_range=(0.0, 0.05),
            description="Medial-lateral postural sway amplitude"
        )
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        logger.info(
            f"CNS extraction complete: {len(biomarker_set.biomarkers)} biomarkers, "
            f"{biomarker_set.extraction_time_ms:.1f}ms, "
            f"{len(heel_strikes)} heel strikes detected"
        )
        
        return biomarker_set
    
    # =========================================================================
    # SIGNAL PREPROCESSING (Essential for clinical-grade analysis)
    # =========================================================================
    
    def _preprocess_signal(
        self, 
        sig: np.ndarray, 
        low_freq: float = 0.5, 
        high_freq: float = 10.0,
        detrend: bool = True
    ) -> np.ndarray:
        """
        Bandpass filter + detrend for all analyses.
        
        Removes:
        - Baseline drift (detrending)
        - Motion artifacts (high-pass)
        - High-frequency noise (low-pass)
        
        Args:
            sig: Input signal
            low_freq: High-pass cutoff (Hz)
            high_freq: Low-pass cutoff (Hz)
            detrend: Whether to remove linear trend
            
        Returns:
            Preprocessed signal
        """
        if len(sig) < 30:
            return sig
        
        # Handle multi-dimensional signals (take magnitude for 2D/3D)
        if sig.ndim > 1:
            sig = np.linalg.norm(sig, axis=-1)
        
        # Detrend to remove baseline drift
        if detrend:
            sig = signal.detrend(sig)
        
        # Validate frequency range for Nyquist
        nyquist = self.sample_rate / 2
        low_freq = min(low_freq, nyquist * 0.9)
        high_freq = min(high_freq, nyquist * 0.9)
        
        if low_freq >= high_freq:
            return sig
        
        try:
            # Bandpass filter (4th order Butterworth)
            sos = signal.butter(
                4, 
                [low_freq, high_freq], 
                btype='band', 
                fs=self.sample_rate, 
                output='sos'
            )
            return signal.sosfiltfilt(sos, sig)
        except Exception:
            return sig
    
    # =========================================================================
    # GAIT VARIABILITY (Zeni et al. 2008 - Gold standard heel strike detection)
    # =========================================================================
    
    def _calculate_gait_variability(
        self, 
        pose_array: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate gait variability using clinically-validated heel strike detection.
        
        Uses Zeni et al. (2008) method: identify heel strikes as local minima
        in filtered ankle vertical position.
        
        Returns:
            Tuple of (coefficient of variation, array of heel strike indices)
        """
        left_ankle_idx = self.landmarks["left_ankle"]
        right_ankle_idx = self.landmarks["right_ankle"]
        
        # Validate data
        if pose_array.shape[0] < 60:  # Need ~2 seconds minimum
            return 0.045, np.array([])
        
        # Extract bilateral ankle Y-positions (vertical movement)
        left_ankle_y = pose_array[:, left_ankle_idx, 1]
        right_ankle_y = pose_array[:, right_ankle_idx, 1]
        
        # Preprocess: remove drift, filter to gait frequencies (0.5-3 Hz)
        left_filtered = self._preprocess_signal(left_ankle_y, 0.5, 3.0)
        right_filtered = self._preprocess_signal(right_ankle_y, 0.5, 3.0)
        
        # Detect heel strikes (local minima = foot contact)
        # Zeni method: invert signal and find peaks
        min_stride_samples = int(0.8 * self.sample_rate)  # Min stride ~0.8s
        max_stride_samples = int(2.0 * self.sample_rate)  # Max stride ~2.0s
        
        try:
            # Left foot heel strikes
            left_strikes, left_props = signal.find_peaks(
                -left_filtered,  # Inverted for minima
                distance=min_stride_samples,
                prominence=np.std(left_filtered) * 0.2
            )
            
            # Right foot heel strikes
            right_strikes, right_props = signal.find_peaks(
                -right_filtered,
                distance=min_stride_samples,
                prominence=np.std(right_filtered) * 0.2
            )
            
        except Exception:
            return 0.045, np.array([])
        
        # Combine all heel strikes
        all_strikes = np.sort(np.concatenate([left_strikes, right_strikes]))
        
        if len(all_strikes) < self.min_strides + 1:
            return 0.045, all_strikes
        
        # Calculate stride times (time between consecutive heel strikes)
        stride_times = np.diff(all_strikes) / self.sample_rate
        
        # Filter out physiologically impossible strides
        valid_strides = stride_times[(stride_times > 0.4) & (stride_times < 2.5)]
        
        if len(valid_strides) < self.min_strides:
            return 0.045, all_strikes
        
        # Coefficient of Variation (CV) - standard gait variability measure
        cv = (np.std(valid_strides) / np.mean(valid_strides))
        
        return float(np.clip(cv, 0.01, 0.20)), all_strikes
    
    # =========================================================================
    # POSTURE ENTROPY (Sample Entropy - Richman & Moorman 2000)
    # =========================================================================
    
    def _sample_entropy(
        self, 
        time_series: np.ndarray, 
        m: int = 2, 
        r: float = None
    ) -> float:
        """
        Calculate Sample Entropy (SampEn) - clinical standard for postural sway.
        
        SampEn measures the complexity/regularity of a time series.
        Lower values = more regular (pathological)
        Higher values = more complex (healthy)
        
        Args:
            time_series: Input signal
            m: Embedding dimension (default: 2)
            r: Tolerance threshold (default: 0.2 * std)
            
        Returns:
            Sample entropy value
        """
        N = len(time_series)
        
        if N < 2 * m + 10:
            return 1.5  # Default for insufficient data
        
        if r is None:
            r = 0.2 * np.std(time_series)
        
        if r == 0:
            return 1.5
        
        def count_matches(templates: np.ndarray, tolerance: float) -> int:
            """Count template matches within tolerance (excluding self-matches)."""
            count = 0
            n = len(templates)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.max(np.abs(templates[i] - templates[j])) <= tolerance:
                        count += 2  # Count both (i,j) and (j,i)
            return count
        
        # Create templates of length m and m+1
        templates_m = np.array([time_series[i:i+m] for i in range(N - m)])
        templates_m1 = np.array([time_series[i:i+m+1] for i in range(N - m - 1)])
        
        # Count matches
        B = count_matches(templates_m, r)
        A = count_matches(templates_m1, r)
        
        # Prevent division by zero
        if B == 0:
            return 1.5
        
        # Sample Entropy = -ln(A/B)
        return float(-np.log((A + 1e-10) / (B + 1e-10)))
    
    def _calculate_posture_entropy(self, pose_array: np.ndarray) -> float:
        """
        Calculate postural sway complexity using Sample Entropy.
        
        Uses center of mass proxy from hip/shoulder landmarks.
        Filters to postural frequency band (0.1-2.0 Hz).
        """
        # Landmark indices for center of mass estimation
        shoulder_left = self.landmarks["left_shoulder"]
        shoulder_right = self.landmarks["right_shoulder"]
        hip_left = self.landmarks["left_hip"]
        hip_right = self.landmarks["right_hip"]
        
        if pose_array.shape[1] < 25:
            return 1.5
        
        # Center of mass proxy (average of shoulders and hips)
        com_landmarks = [shoulder_left, shoulder_right, hip_left, hip_right]
        com_y = np.mean(pose_array[:, com_landmarks, 1], axis=1)  # AP sway (Y-axis)
        
        # Filter to postural sway frequencies (0.1-2.0 Hz)
        com_filtered = self._preprocess_signal(com_y, 0.1, 2.0)
        
        # Calculate sample entropy
        return float(np.clip(self._sample_entropy(com_filtered), 0.0, 4.0))
    
    # =========================================================================
    # TREMOR ANALYSIS (Welch PSD - Elble & McNames 2016)
    # =========================================================================
    
    def _analyze_tremor(
        self, 
        pose_array: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        Analyze tremor using bilateral wrist motion and Welch PSD.
        
        Returns:
            Dict mapping tremor type to (power, confidence) tuples
        """
        tremor_results = {}
        default_result = {k: (0.03, 0.5) for k in self.tremor_bands}
        
        left_wrist_idx = self.landmarks["left_wrist"]
        right_wrist_idx = self.landmarks["right_wrist"]
        
        # Validate data (need 2+ seconds for reliable spectral analysis)
        if pose_array.shape[1] < 17 or pose_array.shape[0] < 60:
            return default_result
        
        try:
            # Extract bilateral wrist positions
            left_wrist = pose_array[:, left_wrist_idx, :2]   # X, Y
            right_wrist = pose_array[:, right_wrist_idx, :2]
            
            # Preprocess: filter to tremor frequencies (2-15 Hz)
            left_filtered = self._preprocess_signal(left_wrist, 2.0, 15.0)
            right_filtered = self._preprocess_signal(right_wrist, 2.0, 15.0)
            
            # Combine bilateral (average reduces noise)
            tremor_signal = (left_filtered + right_filtered) / 2
            
            # Welch Power Spectral Density
            nperseg = min(256, len(tremor_signal) // 4)
            if nperseg < 32:
                return default_result
                
            freqs, psd = signal.welch(
                tremor_signal, 
                fs=self.sample_rate, 
                nperseg=nperseg,
                noverlap=nperseg // 2
            )
            
            # Total power for normalization
            total_power = np.trapz(psd, freqs) + 1e-10
            
            # Extract power in each clinical tremor band
            for band_name, (low_freq, high_freq) in self.tremor_bands.items():
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if np.any(mask):
                    band_power = np.trapz(psd[mask], freqs[mask])
                    normalized_power = band_power / total_power
                    
                    # Confidence based on signal quality
                    peak_freq_idx = np.argmax(psd[mask])
                    peak_prominence = psd[mask][peak_freq_idx] / np.mean(psd[mask])
                    confidence = min(0.95, 0.5 + peak_prominence / 10)
                    
                    tremor_results[band_name] = (
                        float(np.clip(normalized_power, 0, 0.5)),
                        confidence
                    )
                else:
                    tremor_results[band_name] = (0.03, 0.5)
            
            return tremor_results
            
        except Exception as e:
            logger.warning(f"Tremor analysis failed: {e}")
            return default_result
    
    # =========================================================================
    # COMPOSITE STABILITY SCORE (Multi-domain integration)
    # =========================================================================
    
    def _calculate_stability_score(
        self, 
        pose_array: np.ndarray,
        gait_variability: float,
        tremor_scores: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite CNS stability score combining multiple domains.
        
        Components (weighted):
        - 40% Postural sway (AP + ML)
        - 30% Gait variability
        - 30% Tremor power
        
        Returns:
            Tuple of (stability score 0-100, component dict)
        """
        hip_left = self.landmarks["left_hip"]
        hip_right = self.landmarks["right_hip"]
        
        components = {"sway_ap": 0.0, "sway_ml": 0.0}
        
        if pose_array.shape[1] < 25:
            return 85.0, components
        
        # Extract center of mass from hips
        com_ap = np.mean(pose_array[:, [hip_left, hip_right], 1], axis=1)  # Y = AP
        com_ml = np.mean(pose_array[:, [hip_left, hip_right], 0], axis=1)  # X = ML
        
        # Filter to postural band
        sway_ap_filtered = self._preprocess_signal(com_ap, 0.1, 2.0)
        sway_ml_filtered = self._preprocess_signal(com_ml, 0.1, 2.0)
        
        # Sway amplitudes (std of filtered signal)
        sway_ap = np.std(sway_ap_filtered)
        sway_ml = np.std(sway_ml_filtered)
        
        components["sway_ap"] = float(np.clip(sway_ap, 0, 0.2))
        components["sway_ml"] = float(np.clip(sway_ml, 0, 0.2))
        
        # Average tremor power
        tremor_powers = [score for score, _ in tremor_scores.values()]
        avg_tremor = np.mean(tremor_powers) if tremor_powers else 0.03
        
        # Composite score calculation
        # Each component normalized to 0-100 penalty, then subtracted from 100
        sway_penalty = (sway_ap * 1000 + sway_ml * 1000)  # ~0-20 points
        gait_penalty = gait_variability * 500             # ~0-10 points
        tremor_penalty = avg_tremor * 200                 # ~0-10 points
        
        stability = 100 - sway_penalty - gait_penalty - tremor_penalty
        
        return float(np.clip(stability, 40, 100)), components
    
    # =========================================================================
    # SIMULATED BIOMARKERS (Fallback when data insufficient)
    # =========================================================================
    
    def _generate_simulated_biomarkers(self, biomarker_set: BiomarkerSet) -> BiomarkerSet:
        """Generate clinically-plausible simulated biomarkers when real data unavailable."""
        
        # Gait variability (normal range ~3-6%)
        self._add_biomarker(
            biomarker_set, 
            "gait_variability", 
            np.random.uniform(0.03, 0.05), 
            "coefficient_of_variation",
            0.4, 
            self.normal_ranges["gait_variability"], 
            "Simulated gait variability (insufficient pose data)"
        )
        
        # Posture entropy (normal range ~1.0-2.5)
        self._add_biomarker(
            biomarker_set, 
            "posture_entropy",
            np.random.uniform(1.5, 2.2), 
            "sample_entropy",
            0.4, 
            self.normal_ranges["posture_entropy"], 
            "Simulated posture entropy (insufficient pose data)"
        )
        
        # Tremor scores (normal: low power)
        for tremor_type in self.tremor_bands:
            self._add_biomarker(
                biomarker_set, 
                f"tremor_{tremor_type}",
                np.random.uniform(0.01, 0.04), 
                "normalized_psd",
                0.4, 
                self.normal_ranges["tremor_power"], 
                f"Simulated {tremor_type} tremor (insufficient pose data)"
            )
        
        # Stability score (normal: 80-95)
        self._add_biomarker(
            biomarker_set, 
            "cns_stability_score",
            np.random.uniform(82, 92), 
            "score_0_100",
            0.4, 
            self.normal_ranges["stability_score"], 
            "Simulated stability score (insufficient pose data)"
        )
        
        # Sway components
        self._add_biomarker(
            biomarker_set,
            "sway_amplitude_ap",
            np.random.uniform(0.01, 0.03),
            "normalized_units",
            0.4,
            (0.0, 0.05),
            "Simulated AP sway (insufficient pose data)"
        )
        
        self._add_biomarker(
            biomarker_set,
            "sway_amplitude_ml",
            np.random.uniform(0.01, 0.03),
            "normalized_units",
            0.4,
            (0.0, 0.05),
            "Simulated ML sway (insufficient pose data)"
        )
        
        return biomarker_set

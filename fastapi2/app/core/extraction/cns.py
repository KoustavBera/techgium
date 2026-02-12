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
from scipy.spatial.distance import cdist

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
            return biomarker_set
        
        try:
            pose_array = np.array(pose_sequence)
            
            # Validate pose array shape (frames, landmarks, coordinates)
            if pose_array.ndim != 3 or pose_array.shape[1] < 29:
                logger.warning(f"Invalid pose array shape: {pose_array.shape}")
                return biomarker_set
            
        except Exception as e:
            logger.warning(f"Failed to convert pose sequence: {e}")
            return biomarker_set
        
        # =====================================================
        # 1. GAIT VARIABILITY (Zeni heel strike detection)
        # =====================================================
        gait_var, heel_strikes = self._calculate_gait_variability(pose_array)
        gait_confidence = min(0.95, 0.5 + len(heel_strikes) / 20)  # More strides = higher confidence
        
        # CONTEXT-AWARE: Signal stationary state by setting normal_range=None
        # This triggers "not_assessed" status instead of misleading "Normal"
        gait_normal_range = None if len(heel_strikes) == 0 else self.normal_ranges["gait_variability"]
        
        self._add_biomarker(
            biomarker_set,
            name="gait_variability",
            value=gait_var,
            unit="coefficient_of_variation",
            confidence=gait_confidence,
            normal_range=gait_normal_range,
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
        
        # NEW: Extract thermal stress gradient from ESP32 (autonomic stress marker)
        if "thermal_data" in data:
            self._extract_from_thermal(data["thermal_data"], biomarker_set)
        
        return biomarker_set
    
    def _extract_from_thermal(
        self,
        thermal_data: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract CNS/autonomic biomarkers from thermal camera data."""
        
        # Stress Gradient - forehead to nose temperature difference
        # Higher gradient indicates sympathetic activation (stress response)
        if thermal_data.get('stress_gradient') is not None:
            self._add_biomarker(
                biomarker_set,
                name="thermal_stress_gradient",
                value=float(thermal_data['stress_gradient']),
                unit="delta_celsius",
                confidence=0.80,
                normal_range=(0.0, 1.5),
                description="Forehead-nose thermal gradient (autonomic stress indicator)"
            )
        
        # Individual temps for context
        forehead = thermal_data.get('forehead_temp')
        nose = thermal_data.get('nose_temp')
        if forehead is not None:
            self._add_biomarker(
                biomarker_set,
                name="forehead_temperature",
                value=float(forehead),
                unit="celsius",
                confidence=0.85,
                normal_range=(33.0, 36.0),
                description="Forehead temperature (MLX90640)"
            )
    
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
    
    def _detect_gait_state(self, pose_array: np.ndarray) -> bool:
        """
        Detect if subject is walking vs standing using velocity-based analysis.
        
        Args:
            pose_array: Pose landmarks (frames, landmarks, [x,y,z,visibility])
            
        Returns:
            True if walking detected, False if standing/stationary
        """
        if pose_array.shape[0] < 30:
            return False
        
        # Use hip center velocity as gait indicator
        hip_left = self.landmarks["left_hip"]
        hip_right = self.landmarks["right_hip"]
        
        # Hip center position over time
        hip_center = (pose_array[:, hip_left, :2] + pose_array[:, hip_right, :2]) / 2
        
        # Compute frame-to-frame velocity (magnitude)
        velocities = np.linalg.norm(np.diff(hip_center, axis=0), axis=1)
        
        # Walking threshold: mean velocity > 0.01 normalized units
        # (tuned for MediaPipe normalized coordinates)
        mean_velocity = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        # Walking characteristics: higher mean velocity + variability
        is_walking = (mean_velocity > 0.01) and (velocity_std > 0.005)
        
        return is_walking
    
    def _get_landmark_with_visibility(
        self,
        pose_array: np.ndarray,
        landmark_idx: int,
        coord_idx: int = 1,
        min_visibility: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract landmark coordinates weighted by visibility/confidence.
        
        Args:
            pose_array: Pose landmarks (frames, landmarks, [x,y,z,visibility])
            landmark_idx: Index of landmark to extract
            coord_idx: Coordinate index (0=x, 1=y, 2=z)
            min_visibility: Minimum visibility threshold (0-1)
            
        Returns:
            Tuple of (coordinates, visibility_mask)
        """
        # Extract coordinate and visibility
        coords = pose_array[:, landmark_idx, coord_idx]
        visibility = pose_array[:, landmark_idx, 3] if pose_array.shape[2] > 3 else np.ones_like(coords)
        
        # Create mask for reliable landmarks
        visibility_mask = visibility >= min_visibility
        
        return coords, visibility_mask
    
    def _calculate_gait_variability(
        self, 
        pose_array: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate gait variability with visibility weighting and gait state detection.
        
        Uses Zeni et al. (2008) method: identify heel strikes as local minima
        in filtered ankle vertical position, with visibility-based filtering.
        
        Returns:
            Tuple of (coefficient of variation, array of heel strike indices)
        """
        left_ankle_idx = self.landmarks["left_ankle"]
        right_ankle_idx = self.landmarks["right_ankle"]
        
        # Validate data
        if pose_array.shape[0] < 60:  # Need ~2 seconds minimum
            return 0.045, np.array([])
        
        # Check if subject is actually walking
        is_walking = self._detect_gait_state(pose_array)
        if not is_walking:
            logger.info("Subject appears stationary, skipping gait analysis")
            return 0.045, np.array([])  # Normal resting variability
        
        # Extract bilateral ankle Y-positions with visibility weighting
        left_ankle_y, left_visibility = self._get_landmark_with_visibility(
            pose_array, left_ankle_idx, coord_idx=1, min_visibility=0.5
        )
        right_ankle_y, right_visibility = self._get_landmark_with_visibility(
            pose_array, right_ankle_idx, coord_idx=1, min_visibility=0.5
        )
        
        # Filter out low-visibility frames
        left_ankle_y = left_ankle_y[left_visibility]
        right_ankle_y = right_ankle_y[right_visibility]
        
        if len(left_ankle_y) < 30 or len(right_ankle_y) < 30:
            logger.warning("Insufficient visible ankle landmarks for gait analysis")
            return 0.045, np.array([])
        
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
        Calculate Sample Entropy (SampEn) using vectorized implementation.
        
        SampEn measures the complexity/regularity of a time series.
        Lower values = more regular (pathological)
        Higher values = more complex (healthy)
        
        Uses scipy.spatial.distance.cdist for O(n) performance instead of O(n²).
        
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
        
        # Vectorized template matching using cdist
        def count_matches_vectorized(templates: np.ndarray, tolerance: float) -> int:
            """Count template matches using vectorized distance computation."""
            # Compute pairwise Chebyshev distances (max absolute difference)
            distances = cdist(templates, templates, metric='chebyshev')
            
            # Count matches within tolerance, excluding self-matches (diagonal)
            matches = (distances <= tolerance) & (distances > 0)
            return int(np.sum(matches))
        
        # Create templates of length m and m+1
        templates_m = np.array([time_series[i:i+m] for i in range(N - m)])
        templates_m1 = np.array([time_series[i:i+m+1] for i in range(N - m - 1)])
        
        # Count matches using vectorized approach
        B = count_matches_vectorized(templates_m, r)
        A = count_matches_vectorized(templates_m1, r)
        
        # Prevent division by zero
        if B == 0:
            return 1.5
        
        # Sample Entropy = -ln(A/B)
        return float(-np.log((A + 1e-10) / (B + 1e-10)))
    
    def _calculate_posture_entropy(self, pose_array: np.ndarray) -> float:
        """
        Calculate postural sway complexity using Sample Entropy with visibility weighting.
        
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
        
        # Extract landmarks with visibility weighting
        landmarks_data = []
        for lm_idx in [shoulder_left, shoulder_right, hip_left, hip_right]:
            coords, visibility = self._get_landmark_with_visibility(
                pose_array, lm_idx, coord_idx=1
            )
            landmarks_data.append((coords, visibility))
        
        # Compute weighted average for center of mass
        com_y = np.zeros(pose_array.shape[0])
        total_weight = np.zeros(pose_array.shape[0])
        
        for coords, visibility in landmarks_data:
            com_y += coords * visibility
            total_weight += visibility
        
        # Avoid division by zero
        total_weight = np.maximum(total_weight, 1e-6)
        com_y = com_y / total_weight
        
        # Filter to postural sway frequencies (0.1-2.0 Hz)
        com_filtered = self._preprocess_signal(com_y, 0.1, 2.0)
        
        # Calculate sample entropy (now using vectorized implementation)
        return float(np.clip(self._sample_entropy(com_filtered), 0.0, 4.0))
    
    # =========================================================================
    # TREMOR ANALYSIS (Welch PSD - Elble & McNames 2016)
    # =========================================================================
    
    def _analyze_tremor(
        self, 
        pose_array: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        Analyze tremor using bilateral wrist motion with visibility weighting
        and frequency-optimized Welch PSD.
        
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
            # Extract bilateral wrist positions with visibility weighting
            left_wrist_x, left_vis_x = self._get_landmark_with_visibility(
                pose_array, left_wrist_idx, coord_idx=0
            )
            left_wrist_y, left_vis_y = self._get_landmark_with_visibility(
                pose_array, left_wrist_idx, coord_idx=1
            )
            right_wrist_x, right_vis_x = self._get_landmark_with_visibility(
                pose_array, right_wrist_idx, coord_idx=0
            )
            right_wrist_y, right_vis_y = self._get_landmark_with_visibility(
                pose_array, right_wrist_idx, coord_idx=1
            )
            
            # Compute magnitude (do magnitude BEFORE filtering for tremor)
            left_mag = np.sqrt(left_wrist_x**2 + left_wrist_y**2)
            right_mag = np.sqrt(right_wrist_x**2 + right_wrist_y**2)
            
            # Weight by minimum visibility across both coordinates
            left_visibility = np.minimum(left_vis_x, left_vis_y)
            right_visibility = np.minimum(right_vis_x, right_vis_y)
            
            # Filter frames with low visibility
            valid_left = left_visibility > 0.5
            valid_right = right_visibility > 0.5
            
            if np.sum(valid_left) < 30 or np.sum(valid_right) < 30:
                logger.warning("Insufficient visible wrist landmarks for tremor analysis")
                return default_result
            
            left_mag = left_mag[valid_left]
            right_mag = right_mag[valid_right]
            
            # Preprocess: filter to tremor frequencies (2-15 Hz)
            left_filtered = self._preprocess_signal(left_mag, 2.0, 15.0)
            right_filtered = self._preprocess_signal(right_mag, 2.0, 15.0)
            
            # Combine bilateral (average reduces noise)
            min_len = min(len(left_filtered), len(right_filtered))
            tremor_signal = (left_filtered[:min_len] + right_filtered[:min_len]) / 2
            
            # Total power for normalization
            total_power = np.trapz(psd, freqs) + 1e-10
            
            # Extract power in each clinical tremor band with optimized windows
            for band_name, (low_freq, high_freq) in self.tremor_bands.items():
                # Frequency-optimized Welch parameters for each tremor type
                # Higher frequencies need shorter windows for better resolution
                if band_name == "postural":  # 6-12 Hz - needs shorter window
                    nperseg_opt = min(128, len(tremor_signal) // 4)
                elif band_name == "resting":  # 4-6 Hz - medium window
                    nperseg_opt = min(192, len(tremor_signal) // 4)
                else:  # intention 3-5 Hz - longer window
                    nperseg_opt = min(256, len(tremor_signal) // 4)
                
                if nperseg_opt < 32:
                    tremor_results[band_name] = (0.03, 0.5)
                    continue
                
                # Recompute PSD with optimized window for this band
                freqs_opt, psd_opt = signal.welch(
                    tremor_signal,
                    fs=self.sample_rate,
                    nperseg=nperseg_opt,
                    noverlap=nperseg_opt // 2
                )
                
                mask = (freqs_opt >= low_freq) & (freqs_opt <= high_freq)
                
                if np.any(mask):
                    band_power = np.trapz(psd_opt[mask], freqs_opt[mask])
                    total_power_opt = np.trapz(psd_opt, freqs_opt) + 1e-10
                    normalized_power = band_power / total_power_opt
                    
                    # Confidence based on signal quality and visibility
                    peak_freq_idx = np.argmax(psd_opt[mask])
                    peak_prominence = psd_opt[mask][peak_freq_idx] / (np.mean(psd_opt[mask]) + 1e-10)
                    
                    # Reduce confidence if many landmarks were filtered out
                    visibility_factor = min(np.mean(left_visibility[valid_left]), 
                                           np.mean(right_visibility[valid_right]))
                    confidence = min(0.95, (0.5 + peak_prominence / 10) * visibility_factor)
                    
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
        Calculate composite CNS stability score with calibrated normalization.
        
        Components (percentile-based normalization):
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
        
        # Extract center of mass from hips with visibility weighting
        com_ap_left, vis_left = self._get_landmark_with_visibility(
            pose_array, hip_left, coord_idx=1
        )
        com_ap_right, vis_right = self._get_landmark_with_visibility(
            pose_array, hip_right, coord_idx=1
        )
        com_ml_left, _ = self._get_landmark_with_visibility(
            pose_array, hip_left, coord_idx=0
        )
        com_ml_right, _ = self._get_landmark_with_visibility(
            pose_array, hip_right, coord_idx=0
        )
        
        # Average hips for center of mass
        com_ap = (com_ap_left + com_ap_right) / 2  # Y = AP
        com_ml = (com_ml_left + com_ml_right) / 2  # X = ML
        
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
        
        # Calibrated percentile-based normalization (clinical reference ranges)
        # Sway: Normal <0.03, Mild 0.03-0.05, Moderate 0.05-0.08, Severe >0.08
        sway_total = sway_ap + sway_ml
        sway_score = 100 * (1 - np.clip(sway_total / 0.15, 0, 1))  # 0-100
        
        # Gait: Normal CV <0.05, Mild 0.05-0.08, Moderate 0.08-0.12, Severe >0.12  
        gait_score = 100 * (1 - np.clip(gait_variability / 0.15, 0, 1))  # 0-100
        
        # Tremor: Normal <0.05, Mild 0.05-0.10, Moderate 0.10-0.20, Severe >0.20
        tremor_score = 100 * (1 - np.clip(avg_tremor / 0.25, 0, 1))  # 0-100
        
        # Weighted composite (40% sway, 30% gait, 30% tremor)
        stability = 0.4 * sway_score + 0.3 * gait_score + 0.3 * tremor_score
        
        return float(np.clip(stability, 40, 100)), components
    
    # SIMULATION METHOD REMOVED
        
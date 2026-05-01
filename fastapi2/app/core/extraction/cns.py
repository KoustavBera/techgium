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
        
        # Adaptive noise floor — set during baseline calibration in manager.py
        # Default: 0.003 (empirically validated for 720p @ 30fps MediaPipe output)
        self.motion_noise_floor: float = 0.003
        
        # Minimum data requirements (relaxed from 10s to ~7s for reliability)
        self.min_data_length = 200 # Previously 300 (10s @ 30fps)
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

        # ── Motion quality gate (NEW) ──────────────────────────────────────────
        motion_quality = float(data.get("motion_quality", 1.0))
        if motion_quality < 0.40:
            logger.warning(
                f"Motion quality score {motion_quality:.2f} too low for CNS analysis. "
                "Returning empty biomarker set to avoid false positives."
            )
            # Add a single "not_assessed" biomarker so the report is informative
            self._add_biomarker(
                biomarker_set,
                name="cns_data_quality",
                value=motion_quality,
                unit="quality_score",
                confidence=1.0,
                normal_range=None,  # triggers "Not Assessed" status
                description="CNS analysis skipped: pose tracking quality insufficient"
            )
            return biomarker_set
        
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

    def calibrate_noise_floor(self, still_pose_sequence: List[np.ndarray]) -> float:
        """
        Estimate per-device noise floor from 2 seconds of "subject still" data.
        
        Call this during the INITIALIZING phase in manager.py before any scan.
        The result is stored in self.motion_noise_floor and used in _analyze_tremor().
        
        Args:
            still_pose_sequence: ~60 frames of pose data while subject is still
        
        Returns:
            Estimated noise floor (std of wrist velocity during stillness)
        """
        if len(still_pose_sequence) < 30:
            logger.warning("Too few frames for noise calibration — using default")
            return self.motion_noise_floor

        try:
            pose_array = np.array(still_pose_sequence)
            left_wrist_idx  = self.landmarks["left_wrist"]
            right_wrist_idx = self.landmarks["right_wrist"]

            lx = pose_array[:, left_wrist_idx,  0]
            ly = pose_array[:, left_wrist_idx,  1]
            rx = pose_array[:, right_wrist_idx, 0]
            ry = pose_array[:, right_wrist_idx, 1]

            left_vel  = np.sqrt(np.diff(lx)**2 + np.diff(ly)**2)
            right_vel = np.sqrt(np.diff(rx)**2 + np.diff(ry)**2)

            # Use 95th percentile of velocity as conservative noise floor
            # (not mean, to avoid outlier spikes from tracking glitches)
            noise_estimate = float(np.percentile(
                np.concatenate([left_vel, right_vel]), 95
            ))

            # Sanity clamp: never go below 0.001 or above 0.010
            noise_estimate = float(np.clip(noise_estimate, 0.001, 0.010))
            self.motion_noise_floor = noise_estimate

            logger.info(
                f"Noise floor calibrated: {noise_estimate:.5f} "
                f"(was default: 0.003)"
            )
            return noise_estimate

        except Exception as e:
            logger.warning(f"Noise calibration failed: {e} — using default")
            return self.motion_noise_floor
    
    def _extract_from_thermal(
        self,
        thermal_data: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract CNS/autonomic biomarkers from thermal data with artifact rejection."""

        stress_gradient = thermal_data.get('stress_gradient')
        forehead = thermal_data.get('forehead_temp')
        nose = thermal_data.get('nose_temp')

        # ── Artifact detection ─────────────────────────────────────────────────
        thermal_confidence = 0.80  # base confidence

        if stress_gradient is not None:
            # Gradients > 3°C almost always indicate ROI drift or hair occlusion
            if abs(stress_gradient) > 3.0:
                logger.warning(
                    f"Thermal gradient {stress_gradient:.2f}°C > 3°C — "
                    "likely ROI artifact. Reducing confidence to 0.3."
                )
                thermal_confidence = 0.30

            # Implausible if forehead is colder than nose (anatomy)
            if forehead is not None and nose is not None:
                if forehead < nose - 0.5:  # forehead should be warmer or equal
                    logger.warning(
                        "Thermal: forehead cooler than nose — possible ROI swap or hair artifact"
                    )
                    thermal_confidence = min(thermal_confidence, 0.35)

            # Temporal consistency check
            gradient_history = thermal_data.get('gradient_history', [])
            if len(gradient_history) >= 3:
                gradient_std = float(np.std(gradient_history))
                if gradient_std > 1.5:
                    logger.warning(
                        f"Thermal gradient unstable over time (std={gradient_std:.2f}°C)"
                    )
                    thermal_confidence *= 0.6

            self._add_biomarker(
                biomarker_set,
                name="thermal_stress_gradient",
                value=float(np.clip(stress_gradient, 0.0, 5.0)),
                unit="delta_celsius",
                confidence=thermal_confidence,
                normal_range=(0.0, 1.5),
                description="Forehead-nose thermal gradient (autonomic stress indicator)"
            )

        if forehead is not None:
            # Plausibility: skin temp must be 30–38°C
            forehead_conf = 0.85 if 30.0 <= forehead <= 38.0 else 0.30
            self._add_biomarker(
                biomarker_set,
                name="forehead_temperature",
                value=float(forehead),
                unit="celsius",
                confidence=forehead_conf,
                normal_range=(33.0, 36.5),
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
        # POSTURE DETECTION: Check if subject is seated
        # A seated subject has no postural sway, leading to artificially low entropy (0.3-0.4)
        # We check the aspect ratio of the bounding box of the torso.
        
        # Approximate torso height/width
        if pose_array.shape[1] > 24:
            shoulders_y = (pose_array[:, 11, 1] + pose_array[:, 12, 1]) / 2
            hips_y = (pose_array[:, 23, 1] + pose_array[:, 24, 1]) / 2
            torso_height = np.mean(abs(shoulders_y - hips_y))
            
            shoulders_x = abs(pose_array[:, 11, 0] - pose_array[:, 12, 0])
            hips_x = abs(pose_array[:, 23, 0] - pose_array[:, 24, 0])
            torso_width = np.mean((shoulders_x + hips_x) / 2)
            
            # Aspect ratio check (Standing usually > 1.5, Seated/Cropped < 1.2)
            # This is a heuristic.
            if torso_width > 0:
                aspect_ratio = torso_height / torso_width
                if aspect_ratio < 1.2:
                    logger.warning(f"CNS: Low torso aspect ratio ({aspect_ratio:.2f}). Subject likely seated. Skipping sway analysis.")
                    return 1.5 # Return "Normal/high" complexity to avoid "Abnormal/Low" flag
        
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
        
        # ── Sway amplitude gate ────────────────────────────────────────────────────
        # If the filtered sway signal has negligible amplitude, the person is
        # effectively stationary. Measuring entropy of ~0 noise gives artificially
        # LOW SampEn (0.0–0.2) which is indistinguishable from Parkinson's rigidity.
        # Return a healthy mid-range value instead of a pathological one.
        sway_std = np.std(com_filtered)
        SWAY_NOISE_FLOOR = 0.002  # ~2.5px at 1280px — below this is camera jitter
        if sway_std < SWAY_NOISE_FLOOR:
            logger.info(
                f"Sway std {sway_std:.5f} below noise floor — "
                "subject is stationary, returning healthy entropy baseline"
            )
            return 1.5  # Healthy mid-range (normal: 0.5–2.5)

        # Also enforce minimum tolerance to prevent microscopic r values
        r_min = max(0.2 * sway_std, SWAY_NOISE_FLOOR * 2)
        return float(np.clip(self._sample_entropy(com_filtered, r=r_min), 0.0, 4.0))
    
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
        # Healthy default: near-zero tremor, high confidence
        default_result = {k: (0.0, 0.9) for k in self.tremor_bands}

        left_wrist_idx  = self.landmarks["left_wrist"]
        right_wrist_idx = self.landmarks["right_wrist"]

        if pose_array.shape[1] < 17 or pose_array.shape[0] < 60:
            return default_result

        try:
            # ── Step 1: Extract positions ──────────────────────────────────────
            left_x,  left_vis_x  = self._get_landmark_with_visibility(pose_array, left_wrist_idx,  0)
            left_y,  left_vis_y  = self._get_landmark_with_visibility(pose_array, left_wrist_idx,  1)
            right_x, right_vis_x = self._get_landmark_with_visibility(pose_array, right_wrist_idx, 0)
            right_y, right_vis_y = self._get_landmark_with_visibility(pose_array, right_wrist_idx, 1)

            left_visibility  = np.minimum(left_vis_x,  left_vis_y)
            right_visibility = np.minimum(right_vis_x, right_vis_y)

            valid_left  = left_visibility  > 0.5
            valid_right = right_visibility > 0.5

            if np.sum(valid_left) < 30 or np.sum(valid_right) < 30:
                logger.warning("Insufficient visible wrist landmarks for tremor analysis")
                return default_result

            # ── Step 2: VELOCITY (frame-to-frame diff) — THE CRITICAL FIX ──────
            # Tremor is OSCILLATORY MOVEMENT, not absolute position.
            # np.diff gives displacement per frame → actual motion signal.
            lx = left_x[valid_left];   ly = left_y[valid_left]
            rx = right_x[valid_right]; ry = right_y[valid_right]

            left_mag  = np.sqrt(np.diff(lx)**2 + np.diff(ly)**2)
            right_mag = np.sqrt(np.diff(rx)**2 + np.diff(ry)**2)

            # ── Step 3: Absolute motion amplitude gate ───────────────────────
            # If person is still, std of velocity ≈ camera quantization noise (<0.003).
            # Genuine tremor produces std > 0.003 in normalized MediaPipe coords.
            # At 1280px width, 0.003 ≈ ~4 pixels of movement — below this is jitter.
            motion_amplitude = (np.std(left_mag) + np.std(right_mag)) / 2
            # Use adaptive noise floor (set by calibrate_noise_floor(), default 0.003)
            if motion_amplitude < self.motion_noise_floor:
                logger.info(
                    f"Motion amplitude {motion_amplitude:.5f} < noise_floor "
                    f"{self.motion_noise_floor:.5f} — returning healthy baseline"
                )
                return default_result

            # ── Step 4: Preprocess and combine ──────────────────────────────
            left_filtered  = self._preprocess_signal(left_mag,  2.0, 15.0)
            right_filtered = self._preprocess_signal(right_mag, 2.0, 15.0)
            min_len = min(len(left_filtered), len(right_filtered))
            tremor_signal = (left_filtered[:min_len] + right_filtered[:min_len]) / 2

            # ── Step 5: PNR-based band scoring ──────────────────────────────
            # PNR = peak_power_in_band / mean_noise_floor
            # Genuine tremor: sharp peak → PNR ≥ 3.0
            # White noise:   flat PSD  → PNR ≈ 1.0–1.5
            for band_name, (low_freq, high_freq) in self.tremor_bands.items():
                if band_name == "postural":
                    nperseg_opt = min(128, len(tremor_signal) // 4)
                elif band_name == "resting":
                    nperseg_opt = min(192, len(tremor_signal) // 4)
                else:
                    nperseg_opt = min(256, len(tremor_signal) // 4)

                if nperseg_opt < 32:
                    tremor_results[band_name] = (0.0, 0.7)
                    continue

                freqs, psd = signal.welch(
                    tremor_signal,
                    fs=self.sample_rate,
                    nperseg=nperseg_opt,
                    noverlap=nperseg_opt // 2
                )

                mask = (freqs >= low_freq) & (freqs <= high_freq)
                if not np.any(mask):
                    tremor_results[band_name] = (0.0, 0.7)
                    continue

                # Noise floor = mean PSD across entire spectrum
                noise_floor = np.mean(psd) + 1e-10
                peak_in_band = np.max(psd[mask])
                pnr = peak_in_band / noise_floor

                # Only report tremor if there is a genuine spectral peak
                if pnr < 3.0:
                    # Flat spectrum = noise. Healthy.
                    tremor_score = 0.0
                    confidence = 0.85
                else:
                    # Map PNR 3→10 to score 0.01→0.05 (stays within normal_range for mild peaks)
                    tremor_score = float(np.clip((pnr - 3.0) / 140.0, 0.0, 0.5))
                    # Confidence scales with PNR (sharper peak = more confident)
                    confidence = float(np.clip(0.5 + (pnr - 3.0) / 20.0, 0.55, 0.95))

                tremor_results[band_name] = (tremor_score, confidence)

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
        
        # ── Velocity-based sway (matches tremor fix) ───────────────────────────────
        # Sway = rate of CoM displacement, not absolute position.
        # Absolute position drifts with subject's distance from camera.
        com_ap_raw = (com_ap_left + com_ap_right) / 2
        com_ml_raw = (com_ml_left + com_ml_right) / 2

        # Velocity = frame-to-frame change in CoM position
        com_ap_velocity = np.diff(com_ap_raw)
        com_ml_velocity = np.diff(com_ml_raw)

        # Filter to postural sway frequency band
        sway_ap_filtered = self._preprocess_signal(com_ap_velocity, 0.1, 2.0)
        sway_ml_filtered = self._preprocess_signal(com_ml_velocity, 0.1, 2.0)

        # Sway = std of velocity (how much CoM is oscillating)
        sway_ap = np.std(sway_ap_filtered)
        sway_ml = np.std(sway_ml_filtered)

        # Scale: velocity std in normalized units. Normal standing sway ≈ 0.0005–0.002.
        # Rescale thresholds: normal < 0.003, abnormal > 0.008
        components["sway_ap"] = float(np.clip(sway_ap, 0, 0.05))
        components["sway_ml"] = float(np.clip(sway_ml, 0, 0.05))
        
        # Average tremor power
        tremor_powers = [score for score, _ in tremor_scores.values()]
        avg_tremor = np.mean(tremor_powers) if tremor_powers else 0.03
        
        # Calibrated percentile-based normalization (clinical reference ranges)
        # Old threshold 0.15 was for position-based sway. New velocity-based threshold:
        sway_total = sway_ap + sway_ml
        sway_score = 100 * (1 - np.clip(sway_total / 0.012, 0, 1))  # 0.012 = severe velocity sway
        
        # Gait: Normal CV <0.05, Mild 0.05-0.08, Moderate 0.08-0.12, Severe >0.12  
        gait_score = 100 * (1 - np.clip(gait_variability / 0.15, 0, 1))  # 0-100
        
        # Tremor: Normal <0.05, Mild 0.05-0.10, Moderate 0.10-0.20, Severe >0.20
        tremor_score = 100 * (1 - np.clip(avg_tremor / 0.25, 0, 1))  # 0-100
        
        # Weighted composite (40% sway, 30% gait, 30% tremor)
        stability = 0.4 * sway_score + 0.3 * gait_score + 0.3 * tremor_score
        
        return float(np.clip(stability, 40, 100)), components
    
    # SIMULATION METHOD REMOVED
        
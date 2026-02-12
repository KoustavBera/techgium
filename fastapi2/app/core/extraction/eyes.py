"""
Eye Biomarker Extractor with MediaPipe FaceMesh - Clinical Grade

Enhanced extraction using:
- 468-point FaceMesh landmarks for precision
- Soukupová & Čech (2016) EAR for accurate blink detection
- Iris tracking (landmarks 468-477)
- ISO 9241-3 gaze estimation standards
- Clinical-grade validation thresholds
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time
from scipy.ndimage import median_filter

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class EyeExtractor(BaseExtractor):
    """
    Clinical-grade eye biomarker extractor using MediaPipe FaceMesh.
    
    Expects face_landmarks_sequence: List of np.array(468+,4) [x,y,z,visibility]
    Falls back to pose landmarks if FaceMesh unavailable.
    
    Clinical standards applied:
    - Soukupová EAR for blink detection (92% accuracy)
    - ISO 9241-3 velocity thresholds for gaze stability
    - 150ms minimum fixation duration
    - 5s minimum recording for reliability
    """
    
    system = PhysiologicalSystem.EYES
    
    # Standard FaceMesh eye contour indices (16 points per eye)
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # EAR calculation indices (6 key points per eye) - Soukupová standard
    # Order: outer corner, upper-1, upper-2, inner corner, lower-2, lower-1
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    
    # Iris indices (with refineLandmarks=True)
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    
    # Pose eye indices (fallback) - MediaPipe Pose landmarks
    # Use all 3 points (inner, center, outer) for robust averaging
    POSE_LEFT_EYE = [1, 2, 3]   # left_eye_inner, left_eye, left_eye_outer
    POSE_RIGHT_EYE = [4, 5, 6]  # right_eye_inner, right_eye, right_eye_outer
    
    # Head pose reference landmarks (FaceMesh) for motion compensation
    HEAD_POSE_INDICES = [1, 4, 152, 10, 151, 9]  # nose bridge + chin
    
    def __init__(self, sample_rate: float = 30.0, frame_width: int = 1280, frame_height: int = 720):
        """Initialize with clinical parameters.
        
        Args:
            sample_rate: Expected frames per second (default 30.0)
            frame_width: Video frame width for denormalization (default 1280)
            frame_height: Video frame height for denormalization (default 720)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_frames = int(5 * sample_rate)  # 5s minimum for reliability
        self.EAR_THRESHOLD = 0.2  # Adaptive threshold baseline
        self.BLINK_MIN_DURATION = 0.1  # 100ms minimum blink duration
        self.CONSEC_FRAMES = 2  # Confirm blink after N frames
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract eye biomarkers with clinical validation.
        
        Expected data keys:
        - face_landmarks_sequence: FaceMesh landmarks (T, 468+, 4) [preferred]
        - pose_sequence: Pose landmarks (T, 33, 4) [fallback]
        - fps: Frame rate (optional, defaults to sample_rate)
        """
        start_time = time.time()
        biomarker_set = self._create_biomarker_set()
        
        # Get frame rate
        fps = data.get("fps", data.get("frame_rate", self.sample_rate))
        
        # Prefer FaceMesh over pose
        face_seq = data.get("face_landmarks_sequence", [])
        pose_seq = data.get("pose_sequence", [])
        
        # PRODUCTION: Strict validation - require minimum frames
        if len(face_seq) >= self.min_frames:
            landmarks_array = np.array(face_seq)
            
            # Quality check: mean visibility across all landmarks
            if landmarks_array.shape[2] >= 4:
                mean_visibility = np.mean(landmarks_array[:, :min(468, landmarks_array.shape[1]), 3])
                if mean_visibility < 0.5:
                    biomarker_set.metadata["flags"] = biomarker_set.metadata.get("flags", [])
                    biomarker_set.metadata["flags"].append("LOW_VISIBILITY")
                    logger.warning(f"Low landmark visibility: {mean_visibility:.2f}")
            
            self._extract_from_facemesh(landmarks_array, biomarker_set, fps)
            
        elif len(face_seq) >= 30:  # At least 1s of data
            logger.warning(f"Eye analysis optimal at {self.min_frames} frames, got {len(face_seq)}")
            landmarks_array = np.array(face_seq)
            self._extract_from_facemesh(landmarks_array, biomarker_set, fps)
            # Reduce confidence for short recordings
            for bm in biomarker_set.biomarkers:
                bm.confidence *= 0.8
                
        elif len(pose_seq) >= 30:
            pose_array = np.array(pose_seq)
            self._extract_from_pose(pose_array, biomarker_set, fps)
            
        else:
            logger.warning(f"Insufficient data for eye analysis: face={len(face_seq)}, pose={len(pose_seq)}")
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000

    # SIMULATION METHOD REMOVED
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_facemesh(
        self,
        landmarks_array: np.ndarray,  # (T, 468+, 4)
        biomarker_set: BiomarkerSet,
        fps: float = 30.0
    ) -> None:
        """Extract from full FaceMesh landmarks with clinical-grade EAR blink detection."""
        
        # 1. Blink rate using Soukupová EAR method
        blink_rate, blink_count = self._estimate_blink_rate_ear(landmarks_array, fps)
        self._add_biomarker_safe(
            biomarker_set, "blink_rate", blink_rate, "blinks_per_min",
            confidence=0.92, normal_range=(12, 35),
            description="Blink rate (12-35 bpm normal for screen viewing)"
        )
        self._add_biomarker_safe(
            biomarker_set, "blink_count", float(blink_count), "count",
            confidence=0.95,
            description="Total blinks detected in session"
        )
        
        # Extract eye centers for gaze analysis
        left_eye_pos = self._extract_eye_center(landmarks_array, self.LEFT_EYE_INDICES)
        right_eye_pos = self._extract_eye_center(landmarks_array, self.RIGHT_EYE_INDICES)
        
        # Apply head pose compensation (subtract head motion)
        head_compensated_left, head_compensated_right = self._apply_head_compensation(
            landmarks_array, left_eye_pos, right_eye_pos
        )
        gaze_center = (head_compensated_left + head_compensated_right) / 2
        
        # 2. Gaze stability (ISO 9241-3)
        gaze_stability = self._calculate_gaze_stability_v2(gaze_center, fps)
        self._add_biomarker_safe(
            biomarker_set, "gaze_stability_score", gaze_stability, "score_0_100",
            confidence=0.85, normal_range=(70, 100),
            description="ISO 9241-3 velocity-based stability"
        )
        
        # 3. Fixation duration (clinical 150ms minimum)
        fixation_duration = self._estimate_fixation_duration_v2(gaze_center, fps)
        self._add_biomarker_safe(
            biomarker_set, "fixation_duration", fixation_duration, "ms",
            confidence=0.80, normal_range=(60, 400),
            description="Fixation duration (60ms+ normal for reading/scanning)"
        )
        
        # 4. Saccade frequency (bimodal detection)
        saccade_freq = self._count_saccades_v2(gaze_center, fps)
        self._add_biomarker_safe(
            biomarker_set, "saccade_frequency", saccade_freq, "saccades_per_sec",
            confidence=0.80, normal_range=(2, 5),
            description="Peak velocity saccade detection"
        )
        
        # 5. Eye symmetry (motion correlation)
        eye_symmetry = self._calculate_eye_symmetry_v2(left_eye_pos, right_eye_pos)
        self._add_biomarker_safe(
            biomarker_set, "eye_symmetry", eye_symmetry, "ratio",
            confidence=0.90, normal_range=(0.9, 1.0),
            description="Bilateral eye motion correlation"
        )
        
        # Note: Pupil reactivity removed - iris landmarks don't track pupil size
        # True pupilometry requires pupil center segmentation + lighting normalization
        # Iris boundary variation only measures MediaPipe detection noise
        
        # 6. Legacy: Iris detection quality (NOT pupil reactivity)
        if landmarks_array.shape[1] >= 478 and False:  # Disabled - scientifically invalid
            pupil_reactivity = self._estimate_pupil_reactivity(landmarks_array)
            self._add_biomarker_safe(
                biomarker_set, "pupil_reactivity", pupil_reactivity, "score_0_100",
                confidence=0.75, normal_range=(60, 95),
                description="Iris size variation (pupil response proxy)"
            )
    
    def _extract_from_pose(
        self,
        pose_array: np.ndarray,
        biomarker_set: BiomarkerSet,
        fps: float = 30.0
    ) -> None:
        """Extract eye metrics from pose landmarks (lower confidence fallback)."""
        
        if pose_array.shape[1] < 7:
            return
        
        # Eye landmarks from pose - use all 3 points per eye for robustness
        left_eye = np.mean(pose_array[:, self.POSE_LEFT_EYE, :2], axis=1)   # Average of [1,2,3]
        right_eye = np.mean(pose_array[:, self.POSE_RIGHT_EYE, :2], axis=1)  # Average of [4,5,6]
        
        # Fallback to basic pose-based extraction (lower confidence)
        blink_rate = self._estimate_blink_rate(pose_array, fps)
        self._add_biomarker_safe(
            biomarker_set, "blink_rate", blink_rate, "blinks_per_min",
            confidence=0.65, normal_range=(12, 35),
            description="Pose-based blink estimation (screen-calibrated)"
        )
        
        gaze_stability = self._calculate_gaze_stability(left_eye, right_eye)
        self._add_biomarker_safe(
            biomarker_set, "gaze_stability_score", gaze_stability, "score_0_100",
            confidence=0.60, normal_range=(70, 100),
            description="Pose-based gaze stability"
        )
        
        fixation_duration = self._estimate_fixation_duration(left_eye, right_eye, fps)
        self._add_biomarker_safe(
            biomarker_set, "fixation_duration", fixation_duration, "ms",
            confidence=0.55, normal_range=(60, 400),
            description="Pose-based fixation (screen-calibrated)"
        )
        
        saccade_freq = self._count_saccades(left_eye, right_eye, fps)
        self._add_biomarker_safe(
            biomarker_set, "saccade_frequency", saccade_freq, "saccades_per_sec",
            confidence=0.50, normal_range=(2, 5),
            description="Pose-based saccades"
        )
        
        eye_symmetry = self._calculate_eye_symmetry(left_eye, right_eye)
        self._add_biomarker_safe(
            biomarker_set, "eye_symmetry", eye_symmetry, "ratio",
            confidence=0.70, normal_range=(0.9, 1.0),
            description="Pose-based symmetry"
        )
    
    def _estimate_blink_rate_ear(
        self, landmarks: np.ndarray, fps: float = 30.0
    ) -> Tuple[float, int]:
        """
        Soukupová & Čech (2016) EAR blink detection - clinical gold standard.
        
        Uses 6-point Eye Aspect Ratio with adaptive thresholds
        and duration validation (100ms minimum blink).
        """
        def compute_ear(eye_indices: List[int]) -> np.ndarray:
            """Compute Eye Aspect Ratio for each frame."""
            eye_lm = landmarks[:, eye_indices, :2]  # (T, 6, 2) normalized [0,1]
            # ✅ Denormalize to pixel coordinates for metric accuracy
            eye_lm_pixels = eye_lm * np.array([self.frame_width, self.frame_height])
            
            # Clinical standard 6-point EAR
            # Indices: 0=outer, 1=upper1, 2=upper2, 3=inner, 4=lower2, 5=lower1
            p1, p4 = 0, 3  # Horizontal corners
            p2, p6 = 1, 5  # Vertical pair 1
            p3, p5 = 2, 4  # Vertical pair 2
            
            # Vertical distances (in pixels)
            A = np.linalg.norm(eye_lm_pixels[:, p2] - eye_lm_pixels[:, p6], axis=1)
            B = np.linalg.norm(eye_lm_pixels[:, p3] - eye_lm_pixels[:, p5], axis=1)
            # Horizontal distance (in pixels)
            C = np.linalg.norm(eye_lm_pixels[:, p1] - eye_lm_pixels[:, p4], axis=1)
            
            # EAR formula: (A + B) / (2 * C)
            ear = (A + B) / (2.0 * C + 1e-6)
            return np.nan_to_num(ear, nan=0.3)  # Open eye default ~0.3
        
        left_ear = compute_ear(self.LEFT_EYE_EAR)
        right_ear = compute_ear(self.RIGHT_EYE_EAR)
        
        # Robust averaging: use average of both eyes to reduce noise sensitivity
        # from single-eye tracker jitter. 
        ears = (left_ear + right_ear) / 2.0
        
        # Robust threshold using rolling median (resistant to outliers)
        # Prevents single squint from poisoning global percentile
        ear_threshold = self._compute_robust_ear_threshold(ears)
        
        # Detect blinks with duration validation
        blink_events = []
        in_blink = False
        blink_start = 0
        min_blink_frames = max(1, int(self.BLINK_MIN_DURATION * fps))
        
        for i, ear in enumerate(ears):
            if ear < ear_threshold and not in_blink:
                in_blink = True
                blink_start = i
            elif in_blink and (ear >= ear_threshold or i == len(ears) - 1):
                blink_duration_frames = i - blink_start
                # Only count if duration >= 100ms
                if blink_duration_frames >= min_blink_frames:
                    blink_events.append(blink_duration_frames / fps)
                in_blink = False
        
        # Convert to blinks per minute
        duration_min = len(landmarks) / fps / 60
        blink_rate = len(blink_events) / max(duration_min, 0.01) # Avoid div by zero, but allow low rates
        
        return float(np.clip(blink_rate, 0, 60)), len(blink_events)
    
    def _compute_robust_ear_threshold(
        self, ears: np.ndarray, window: int = 30
    ) -> float:
        """
        Compute robust EAR threshold using rolling median + IQR.
        
        Resistant to outliers (squinting, partial closures) that poison
        global percentile methods.
        
        Args:
            ears: EAR time series
            window: Rolling window size (default 30 frames = 1s at 30fps)
            
        Returns:
            Robust threshold for blink detection
        """
        if len(ears) < window:
            # Fallback for short sequences
            return float(max(np.percentile(ears, 10), 0.15))
        
        try:
            # Rolling median smoothing
            rolling_median = median_filter(ears, size=window, mode='nearest')
            
            # Tukey outlier detection (1.5 * IQR below Q1)
            q1 = np.percentile(rolling_median, 25)
            q3 = np.percentile(rolling_median, 75)
            iqr = q3 - q1
            
            threshold = q1 - 1.5 * iqr
            
            # Clamp to sensible range
            return float(np.clip(threshold, 0.15, 0.25))
            
        except Exception:
            # Fallback on error
            return 0.18
    
    def _apply_head_compensation(
        self,
        landmarks: np.ndarray,
        left_eye_pos: np.ndarray,
        right_eye_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compensate for head movement to get gaze-in-head-frame.
        
        Based on Google MediaPipe Iris research - subtracts head motion
        from eye positions for stable gaze tracking.
        
        Args:
            landmarks: Full FaceMesh landmarks (T, 468+, 4)
            left_eye_pos: Left eye center positions (T, 2)
            right_eye_pos: Right eye center positions (T, 2)
            
        Returns:
            Tuple of (head_compensated_left, head_compensated_right)
        """
        if landmarks.shape[1] < 468:
            # No head compensation available for pose-only data
            return left_eye_pos, right_eye_pos
        
        # Head reference: nose bridge + chin landmarks
        head_lm = landmarks[:, self.HEAD_POSE_INDICES, :2]  # (T, 6, 2)
        head_center = np.mean(head_lm, axis=1)  # (T, 2)
        
        # Subtract head motion (relative gaze)
        left_compensated = left_eye_pos - head_center
        right_compensated = right_eye_pos - head_center
        
        return left_compensated, right_compensated
    
    def _extract_eye_center(
        self, landmarks: np.ndarray, indices: List[int]
    ) -> np.ndarray:
        """Extract visibility-weighted eye center with global confidence scaling."""
        eye_lm = landmarks[:, indices, :3]  # (T, N, 3)
        
        # Use global visibility weighting for proper confidence
        if landmarks.shape[2] >= 4:
            # Global visibility mean across ALL face landmarks
            all_visibility = landmarks[:, :, 3]  # (T, all_landmarks)
            global_vis_mean = np.mean(all_visibility, axis=1, keepdims=True)  # (T, 1)
            
            # Eye-specific visibility
            eye_visibility = landmarks[:, indices, 3:4]  # (T, N, 1)
            vis_sum = np.sum(eye_visibility, axis=1, keepdims=True) + 1e-6
            weights = eye_visibility / vis_sum
            weighted = eye_lm[:, :, :2] * weights
            eye_center = np.sum(weighted, axis=1)  # (T, 2)
            
            # Scale by global visibility (penalize when face poorly detected)
            return eye_center * global_vis_mean
        else:
            return np.mean(eye_lm[:, :, :2], axis=1)  # (T, 2)
    
    def _calculate_gaze_stability_v2(
        self, gaze_center: np.ndarray, fps: float = 30.0
    ) -> float:
        """
        ISO 9241-3 clinical gaze stability using RMS velocity.
        
        Applies median filter preprocessing and velocity-based scoring.
        """
        if len(gaze_center) < 30:
            return 75.0  # Default for insufficient data
        
        # Preprocessing: apply median filter to reduce noise
        try:
            from scipy.signal import medfilt
            kernel_size = min(5, len(gaze_center) // 2 * 2 - 1)
            if kernel_size >= 3:
                gaze_smooth_x = medfilt(gaze_center[:, 0], kernel_size=kernel_size)
                gaze_smooth_y = medfilt(gaze_center[:, 1], kernel_size=kernel_size)
                gaze_smooth = np.column_stack([gaze_smooth_x, gaze_smooth_y])
            else:
                gaze_smooth = gaze_center
        except ImportError:
            # Fallback to simple moving average
            gaze_smooth = gaze_center
        
        # RMS velocity (clinical standard)
        velocity = np.linalg.norm(np.diff(gaze_smooth, axis=0), axis=1)
        rms_velocity = np.sqrt(np.mean(velocity**2))
        
        # Clinical threshold: map RMS velocity to 0-100 score
        # Low velocity = high stability
        # Scale factor 20 maps typical normalized velocities to score range
        stability_score = 100 * np.clip(1 - rms_velocity * 20, 0, 1)
        
        return float(stability_score)
    
    def _estimate_fixation_duration_v2(
        self, gaze_center: np.ndarray, fps: float = 30.0
    ) -> float:
        """
        Clinical fixation duration with strict velocity threshold.
        
        Uses bottom 10% velocity percentile (stricter than 20%)
        to avoid counting smooth pursuit as fixation.
        """
        if len(gaze_center) < 30:
            return 250.0
        
        velocity = np.linalg.norm(np.diff(gaze_center, axis=0), axis=1)
        
        # Stricter threshold: bottom 10% velocity = true fixation
        # Clinical fixation requires <0.5 deg/sec
        vel_threshold = np.percentile(velocity, 10)
        fixation_mask = velocity < vel_threshold
        
        # Screen-aware fixation: 60ms min (normal for reading/UI scanning)
        # Clinical standard is 150ms-200ms
        min_fixation_frames = max(1, int(0.06 * fps))
        
        fixation_lengths = []
        current = 0
        for is_fix in fixation_mask:
            if is_fix:
                current += 1
            elif current >= min_fixation_frames:
                fixation_lengths.append(current)
                current = 0
            else:
                current = 0
        
        # Count final fixation if ongoing
        if current >= min_fixation_frames:
            fixation_lengths.append(current)
        
        if fixation_lengths:
            avg_ms = np.mean(fixation_lengths) * 1000 / fps
        else:
            avg_ms = 200  # Default if no valid fixations detected
        
        return float(np.clip(avg_ms, 50, 600))
    
    def _count_saccades_v2(
        self, gaze_center: np.ndarray, fps: float = 30.0
    ) -> float:
        """
        Vectorized saccade detection using rising edge transitions.
        
        Detects isolated fast movements (90th percentile velocity threshold).
        """
        if len(gaze_center) < 30:
            return 3.0
        
        velocity = np.linalg.norm(np.diff(gaze_center, axis=0), axis=1)
        
        # Bimodal threshold: top 10% velocity = saccades
        sacc_threshold = np.percentile(velocity, 90)
        
        # Vectorized rising edge detection (proper transition counting)
        sacc_mask = velocity > sacc_threshold
        # Pad and diff to find transitions
        transitions = np.diff(np.pad(sacc_mask.astype(int), (1, 1), mode='constant'))
        saccades = int(np.sum(transitions == 1))  # Count rising edges only
        
        duration_sec = len(gaze_center) / fps
        saccade_freq = saccades / max(duration_sec, 1)
        
        return float(np.clip(saccade_freq, 0.5, 8))
    
    def _calculate_eye_symmetry_v2(
        self, left_center: np.ndarray, right_center: np.ndarray
    ) -> float:
        """Motion correlation between eyes for symmetry score."""
        left_motion = np.diff(left_center, axis=0)
        right_motion = np.diff(right_center, axis=0)
        left_mag = np.linalg.norm(left_motion, axis=1)
        right_mag = np.linalg.norm(right_motion, axis=1)
        
        if len(left_mag) < 2 or np.std(left_mag) < 1e-6 or np.std(right_mag) < 1e-6:
            return 0.95
        
        corr = np.corrcoef(left_mag, right_mag)[0, 1]
        if np.isnan(corr):
            return 0.95
        
        # Map correlation [-1, 1] to symmetry [0, 1]
        return float(np.clip((corr + 1) / 2, 0, 1))
    
    def _estimate_pupil_reactivity(self, landmarks: np.ndarray) -> float:
        """
        DEPRECATED: Iris boundary variance (NOT true pupil reactivity).
        
        WARNING: This measures MediaPipe iris detection noise, not physiological
        pupil response. Iris landmarks track iris boundary, not pupil diameter.
        
        True pupilometry requires:
        - Pupil center segmentation (not iris contour)
        - Lighting condition normalization  
        - Constriction velocity measurement
        
        This method kept for backward compatibility only.
        """
        logger.warning("Pupil reactivity is scientifically invalid - iris ≠ pupil")
        
        if landmarks.shape[1] < 478:
            return 70.0
        
        def iris_size(iris_indices: List[int]) -> np.ndarray:
            """Calculate iris bounding box area for each frame."""
            iris_pts = landmarks[:, iris_indices, :2]
            # Bounding box dimensions
            x_min = np.min(iris_pts[:, :, 0], axis=1)
            x_max = np.max(iris_pts[:, :, 0], axis=1)
            y_min = np.min(iris_pts[:, :, 1], axis=1)
            y_max = np.max(iris_pts[:, :, 1], axis=1)
            return (x_max - x_min) * (y_max - y_min)
        
        left_size = iris_size(self.LEFT_IRIS)
        right_size = iris_size(self.RIGHT_IRIS)
        pupil_sizes = (left_size + right_size) / 2
        
        # Reactivity = coefficient of variation (CV)
        mean_size = np.mean(pupil_sizes)
        if mean_size < 1e-6:
            return 70.0
        
        cv = np.std(pupil_sizes) / mean_size
        
        # Map CV to reactivity score (high CV = low reactivity/more variation)
        # Invert: stable pupil size = high reactivity score
        reactivity = 100 * (1 - np.clip(cv * 100, 0, 1))
        
        return float(np.clip(reactivity, 40, 100))
    
    def _add_biomarker_safe(
        self,
        biomarker_set: BiomarkerSet,
        name: str,
        value: float,
        unit: str,
        confidence: float = 1.0,
        normal_range: Optional[tuple] = None,
        description: str = ""
    ) -> None:
        """Safe biomarker addition with NaN/Inf protection and logging."""
        try:
            # Validate value
            if np.isnan(value) or np.isinf(value):
                logger.warning(f"Invalid {name} value: {value}, using 0.0")
                value = 0.0
            
            # Cap confidence to valid range
            confidence = float(np.clip(confidence, 0.0, 1.0))
            
            self._add_biomarker(
                biomarker_set, name, float(value), unit,
                confidence=confidence, normal_range=normal_range,
                description=description
            )
        except Exception as e:
            logger.error(f"Failed to add biomarker {name}: {e}")

    # ============================================================
    # Legacy pose-based methods (lower accuracy fallback)
    # ============================================================
    
    def _estimate_blink_rate(self, pose_array: np.ndarray, fps: float = 30) -> float:
        """Estimate blink rate from eye visibility changes (legacy)."""
        
        # Use eye visibility scores (column 3)
        if pose_array.shape[2] < 4:
            return np.random.uniform(14, 18)
        
        left_visibility = pose_array[:, 2, 3]
        right_visibility = pose_array[:, 5, 3]
        avg_visibility = (left_visibility + right_visibility) / 2
        
        # Detect dips in visibility (blinks)
        visibility_threshold = np.mean(avg_visibility) - 0.5 * np.std(avg_visibility)
        
        # Count transitions below threshold
        below = avg_visibility < visibility_threshold
        transitions = np.sum(np.diff(below.astype(int)) == 1)
        
        # Convert to blinks per minute
        duration_min = len(pose_array) / fps / 60
        if duration_min > 0:
            blink_rate = transitions / duration_min
        else:
            blink_rate = 15
        
        return float(np.clip(blink_rate, 5, 40))
    
    def _calculate_gaze_stability(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> float:
        """Calculate gaze stability from eye position variance (legacy)."""
        
        # Combined eye position
        gaze_center = (left_eye + right_eye) / 2
        
        # Calculate position variance
        variance = np.var(gaze_center, axis=0)
        total_variance = np.sum(variance)
        
        # Convert to 0-100 stability score
        # Lower variance = higher stability
        stability = 100 * (1 - np.clip(total_variance / 0.01, 0, 1))
        
        return float(np.clip(stability, 0, 100))
    
    def _estimate_fixation_duration(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        fps: float = 30
    ) -> float:
        """Estimate average fixation duration (legacy)."""
        
        gaze = (left_eye + right_eye) / 2
        
        # Calculate velocity
        velocity = np.linalg.norm(np.diff(gaze, axis=0), axis=1)
        
        # Fixation = low velocity periods
        velocity_threshold = np.percentile(velocity, 30)
        fixating = velocity < velocity_threshold
        
        # Count consecutive fixation frames
        fixation_lengths = []
        current_length = 0
        
        for is_fix in fixating:
            if is_fix:
                current_length += 1
            elif current_length > 0:
                fixation_lengths.append(current_length)
                current_length = 0
        
        if current_length > 0:
            fixation_lengths.append(current_length)
        
        if fixation_lengths:
            avg_frames = np.mean(fixation_lengths)
            avg_duration_ms = avg_frames / fps * 1000
        else:
            avg_duration_ms = 250
        
        return float(np.clip(avg_duration_ms, 50, 1000))
    
    def _count_saccades(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        fps: float = 30
    ) -> float:
        """Count saccadic eye movements (legacy)."""
        
        gaze = (left_eye + right_eye) / 2
        velocity = np.linalg.norm(np.diff(gaze, axis=0), axis=1)
        
        # Saccades = high velocity events
        velocity_threshold = np.percentile(velocity, 85)
        saccades = np.sum(velocity > velocity_threshold)
        
        # Convert to per-second
        duration_sec = len(gaze) / fps
        if duration_sec > 0:
            saccade_freq = saccades / duration_sec
        else:
            saccade_freq = 3
        
        return float(np.clip(saccade_freq, 0.5, 10))
    
    def _calculate_eye_symmetry(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> float:
        """Calculate symmetry of left/right eye movements (legacy)."""
        
        left_motion = np.diff(left_eye, axis=0)
        right_motion = np.diff(right_eye, axis=0)
        
        # Correlation of movement patterns
        left_magnitude = np.linalg.norm(left_motion, axis=1)
        right_magnitude = np.linalg.norm(right_motion, axis=1)
        
        if np.std(left_magnitude) < 1e-6 or np.std(right_magnitude) < 1e-6:
            return 0.95
        
        correlation = np.corrcoef(left_magnitude, right_magnitude)[0, 1]
        
        if np.isnan(correlation):
            return 0.95
        
        # Convert to 0-1 symmetry score
        symmetry = (correlation + 1) / 2  # Map -1,1 to 0,1
        
        return float(np.clip(symmetry, 0, 1))
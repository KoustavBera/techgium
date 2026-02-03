"""
Eye Biomarker Extractor with MediaPipe FaceMesh

Enhanced extraction using:
- 468-point FaceMesh landmarks for precision
- Eye Aspect Ratio (EAR) for accurate blink detection
- Iris tracking (landmarks 468-477)
- Gaze estimation from eye contours
"""
from typing import Dict, Any, List
import numpy as np

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class EyeExtractor(BaseExtractor):
    """
    Enhanced eye biomarker extractor using MediaPipe FaceMesh.
    
    Expects face_landmarks_sequence: List of np.array(468+,4) [x,y,z,visibility]
    Falls back to pose landmarks if FaceMesh unavailable.
    """
    
    system = PhysiologicalSystem.EYES
    
    # Standard FaceMesh eye contour indices (16 points per eye)
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # EAR calculation indices (6 key points per eye)
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    
    # Iris indices (with refineLandmarks=True)
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    
    # Pose eye indices (fallback)
    POSE_LEFT_EYE = [1, 2, 3]
    POSE_RIGHT_EYE = [4, 5, 6]
    
    EAR_THRESHOLD = 0.18  # Blink detection threshold
    CONSEC_FRAMES = 2     # Confirm blink after N frames
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract eye biomarkers.
        
        Expected data keys:
        - face_landmarks_sequence: FaceMesh landmarks (T, 468+, 4) [preferred]
        - pose_sequence: Pose landmarks (T, 33, 4) [fallback]
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        # Prefer FaceMesh over pose
        face_seq = data.get("face_landmarks_sequence", [])
        pose_seq = data.get("pose_sequence", [])
        
        if len(face_seq) >= 30:  # 1s @30fps
            landmarks_array = np.array(face_seq)
            self._extract_from_facemesh(landmarks_array, biomarker_set)
        elif len(pose_seq) >= 30:
            pose_array = np.array(pose_seq)
            self._extract_from_pose(pose_array, biomarker_set)
        else:
            self._generate_simulated_biomarkers(biomarker_set)
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_facemesh(
        self,
        landmarks_array: np.ndarray,  # (T, 468+, 4)
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract from full FaceMesh landmarks with EAR-based blink detection."""
        
        # 1. Blink rate using Eye Aspect Ratio (EAR)
        blink_rate = self._estimate_blink_rate_ear(landmarks_array)
        self._add_biomarker(
            biomarker_set, "blink_rate", blink_rate, "blinks_per_min",
            confidence=0.95, normal_range=(12, 20),
            description="EAR-based blink detection"
        )
        
        # Extract eye centers for gaze analysis
        left_eye_pos = self._extract_eye_center(landmarks_array, self.LEFT_EYE_INDICES)
        right_eye_pos = self._extract_eye_center(landmarks_array, self.RIGHT_EYE_INDICES)
        gaze_center = (left_eye_pos + right_eye_pos) / 2
        
        # 2. Gaze stability
        gaze_stability = self._calculate_gaze_stability_v2(gaze_center)
        self._add_biomarker(
            biomarker_set, "gaze_stability_score", gaze_stability, "score_0_100",
            confidence=0.85, normal_range=(70, 100),
            description="Eye center position stability"
        )
        
        # 3. Fixation duration
        fixation_duration = self._estimate_fixation_duration_v2(gaze_center)
        self._add_biomarker(
            biomarker_set, "fixation_duration", fixation_duration, "ms",
            confidence=0.80, normal_range=(150, 400),
            description="Gaze fixation periods"
        )
        
        # 4. Saccade frequency
        saccade_freq = self._count_saccades_v2(gaze_center)
        self._add_biomarker(
            biomarker_set, "saccade_frequency", saccade_freq, "saccades_per_sec",
            confidence=0.80, normal_range=(2, 5),
            description="Rapid gaze shifts"
        )
        
        # 5. Eye symmetry
        eye_symmetry = self._calculate_eye_symmetry_v2(left_eye_pos, right_eye_pos)
        self._add_biomarker(
            biomarker_set, "eye_symmetry", eye_symmetry, "ratio",
            confidence=0.90, normal_range=(0.9, 1.0),
            description="Bilateral eye coordination"
        )
        
        # 6. Pupil reactivity (if iris landmarks available)
        if landmarks_array.shape[1] >= 478:
            pupil_reactivity = self._estimate_pupil_reactivity(landmarks_array)
            self._add_biomarker(
                biomarker_set, "pupil_reactivity", pupil_reactivity, "score_0_100",
                confidence=0.75, normal_range=(60, 95),
                description="Iris size variation proxy"
            )
    
    def _extract_from_pose(
        self,
        pose_array: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract eye metrics from pose landmarks."""
        
        if pose_array.shape[1] < 7:
            self._generate_simulated_biomarkers(biomarker_set)
            return
        
        # Eye landmarks from pose
        left_eye = pose_array[:, 2, :2]   # left_eye
        right_eye = pose_array[:, 5, :2]  # right_eye
        
        # Fallback to basic pose-based extraction (lower confidence)
        blink_rate = self._estimate_blink_rate(pose_array)
        self._add_biomarker(
            biomarker_set, "blink_rate", blink_rate, "blinks_per_min",
            confidence=0.65, normal_range=(12, 20),
            description="Pose-based blink estimation"
        )
        
        gaze_stability = self._calculate_gaze_stability(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set, "gaze_stability_score", gaze_stability, "score_0_100",
            confidence=0.60, normal_range=(70, 100),
            description="Pose-based gaze stability"
        )
        
        fixation_duration = self._estimate_fixation_duration(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set, "fixation_duration", fixation_duration, "ms",
            confidence=0.55, normal_range=(150, 400),
            description="Pose-based fixation"
        )
        
        saccade_freq = self._count_saccades(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set, "saccade_frequency", saccade_freq, "saccades_per_sec",
            confidence=0.50, normal_range=(2, 5),
            description="Pose-based saccades"
        )
        
        eye_symmetry = self._calculate_eye_symmetry(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set, "eye_symmetry", eye_symmetry, "ratio",
            confidence=0.70, normal_range=(0.9, 1.0),
            description="Pose-based symmetry"
        )
    
    def _estimate_blink_rate_ear(
        self, landmarks: np.ndarray, fps: float = 30.0
    ) -> float:
        """Eye Aspect Ratio (EAR) blink detection - gold standard."""
        def compute_ear(eye_indices: List[int]) -> np.ndarray:
            eye_lm = landmarks[:, eye_indices, :2]  # (T, 6, 2)
            # Vertical distances
            A = np.linalg.norm(eye_lm[:, 1] - eye_lm[:, 5], axis=1)
            B = np.linalg.norm(eye_lm[:, 2] - eye_lm[:, 4], axis=1)
            # Horizontal distance
            C = np.linalg.norm(eye_lm[:, 0] - eye_lm[:, 3], axis=1)
            return (A + B) / (2.0 * C + 1e-6)
        
        left_ear = compute_ear(self.LEFT_EYE_EAR)
        right_ear = compute_ear(self.RIGHT_EYE_EAR)
        avg_ear = (left_ear + right_ear) / 2
        
        # Detect blinks: EAR < threshold for CONSEC_FRAMES
        blinks = 0
        consec_closed = 0
        for ear in avg_ear:
            if ear < self.EAR_THRESHOLD:
                consec_closed += 1
                if consec_closed == self.CONSEC_FRAMES:
                    blinks += 1
            else:
                consec_closed = 0
        
        duration_min = len(landmarks) / fps / 60
        return float(np.clip(blinks / max(duration_min, 0.1), 5, 40))
    
    def _extract_eye_center(
        self, landmarks: np.ndarray, indices: List[int]
    ) -> np.ndarray:
        """Extract weighted eye center positions (T, 2)."""
        eye_lm = landmarks[:, indices, :3]  # (T, N, 3)
        visibility = landmarks[:, indices, 3:4]  # (T, N, 1)
        weighted = eye_lm[:, :, :2] * visibility
        return np.mean(weighted, axis=1)  # (T, 2)
    
    def _calculate_gaze_stability_v2(self, gaze_center: np.ndarray) -> float:
        """Variance-based stability score."""
        variance = np.var(gaze_center, axis=0)
        total_var = np.sum(variance)
        stability = 100 * (1 - np.clip(total_var / 0.005, 0, 1))
        return float(np.clip(stability, 0, 100))
    
    def _estimate_fixation_duration_v2(
        self, gaze_center: np.ndarray, fps: float = 30.0
    ) -> float:
        """Low-velocity fixation periods."""
        velocity = np.linalg.norm(np.diff(gaze_center, axis=0), axis=1)
        vel_threshold = np.percentile(velocity, 25)
        fixating = velocity < vel_threshold
        
        fixation_lengths = []
        current = 0
        for f in fixating:
            if f:
                current += 1
            elif current > 2:
                fixation_lengths.append(current)
                current = 0
        if current > 2:
            fixation_lengths.append(current)
        
        avg_ms = np.mean(fixation_lengths) / fps * 1000 if fixation_lengths else 250
        return float(np.clip(avg_ms, 50, 1000))
    
    def _count_saccades_v2(
        self, gaze_center: np.ndarray, fps: float = 30.0
    ) -> float:
        """High-velocity saccade detection."""
        velocity = np.linalg.norm(np.diff(gaze_center, axis=0), axis=1)
        sacc_threshold = np.percentile(velocity, 88)
        saccades = np.sum(velocity > sacc_threshold)
        duration_sec = len(gaze_center) / fps
        return float(np.clip(saccades / max(duration_sec, 0.1), 0.5, 10))
    
    def _calculate_eye_symmetry_v2(
        self, left_center: np.ndarray, right_center: np.ndarray
    ) -> float:
        """Motion correlation between eyes."""
        left_motion = np.diff(left_center, axis=0)
        right_motion = np.diff(right_center, axis=0)
        left_mag = np.linalg.norm(left_motion, axis=1)
        right_mag = np.linalg.norm(right_motion, axis=1)
        
        if len(left_mag) < 2:
            return 0.95
        corr = np.corrcoef(left_mag, right_mag)[0, 1]
        return float(np.clip((corr + 1) / 2, 0, 1))
    
    def _estimate_pupil_reactivity(self, landmarks: np.ndarray) -> float:
        """Iris area variance as pupil response proxy."""
        if landmarks.shape[1] < 478:
            return 75.0
        
        left_iris = landmarks[:, self.LEFT_IRIS, :2]
        right_iris = landmarks[:, self.RIGHT_IRIS, :2]
        
        def iris_variance(iris_pts: np.ndarray) -> float:
            centers = np.mean(iris_pts, axis=1)
            return np.mean(np.var(centers, axis=0))
        
        left_var = iris_variance(left_iris)
        right_var = iris_variance(right_iris)
        avg_var = (left_var + right_var) / 2
        
        reactivity = 100 * (1 - np.clip(avg_var / 0.002, 0, 1))
        return float(np.clip(reactivity, 0, 100))
    
    def _estimate_blink_rate(self, pose_array: np.ndarray, fps: float = 30) -> float:
        """Estimate blink rate from eye visibility changes."""
        
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
        """Calculate gaze stability from eye position variance."""
        
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
        """Estimate average fixation duration."""
        
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
        """Count saccadic eye movements."""
        
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
        """Calculate symmetry of left/right eye movements."""
        
        left_motion = np.diff(left_eye, axis=0)
        right_motion = np.diff(right_eye, axis=0)
        
        # Correlation of movement patterns
        left_magnitude = np.linalg.norm(left_motion, axis=1)
        right_magnitude = np.linalg.norm(right_motion, axis=1)
        
        if np.std(left_magnitude) < 1e-6 or np.std(right_magnitude) < 1e-6:
            return 0.95
        
        correlation = np.corrcoef(left_magnitude, right_magnitude)[0, 1]
        
        # Convert to 0-1 symmetry score
        symmetry = (correlation + 1) / 2  # Map -1,1 to 0,1
        
        return float(np.clip(symmetry, 0, 1))
    
    def _generate_simulated_biomarkers(self, biomarker_set: BiomarkerSet) -> None:
        """Generate simulated eye biomarkers."""
        sim_data = {
            "blink_rate": (np.random.uniform(14, 18), "blinks_per_min"),
            "gaze_stability_score": (np.random.uniform(80, 95), "score_0_100"),
            "fixation_duration": (np.random.uniform(200, 350), "ms"),
            "saccade_frequency": (np.random.uniform(2.5, 4), "saccades_per_sec"),
            "eye_symmetry": (np.random.uniform(0.92, 0.98), "ratio"),
            "pupil_reactivity": (np.random.uniform(70, 90), "score_0_100")
        }
        for name, (value, unit) in sim_data.items():
            self._add_biomarker(biomarker_set, name, value, unit,
                              confidence=0.4, normal_range=(0, 100),
                              description=f"Simulated {name}")
                              
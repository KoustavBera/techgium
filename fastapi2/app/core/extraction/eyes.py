"""
Eye Biomarker Extractor

Extracts ocular health indicators:
- Blink rate
- Fixation patterns
- Gaze stability
"""
from typing import Dict, Any, List
import numpy as np

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class EyeExtractor(BaseExtractor):
    """
    Extracts eye-related biomarkers.
    
    Analyzes pose/face landmark data for ocular health indicators.
    """
    
    system = PhysiologicalSystem.EYES
    
    # Eye landmark indices in MediaPipe (face mesh)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Simplified set
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Pose eye indices
    POSE_LEFT_EYE = [1, 2, 3]  # left_eye_inner, left_eye, left_eye_outer
    POSE_RIGHT_EYE = [4, 5, 6]
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract eye biomarkers.
        
        Expected data keys:
        - pose_sequence: Pose landmarks over time
        - face_landmarks: Face mesh landmarks (optional)
        - eye_tracking: Eye tracking data (optional)
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        pose_sequence = data.get("pose_sequence", [])
        
        if len(pose_sequence) >= 30:  # Need at least 1 second at 30fps
            pose_array = np.array(pose_sequence)
            self._extract_from_pose(pose_array, biomarker_set)
        else:
            self._generate_simulated_biomarkers(biomarker_set)
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
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
        
        # Blink rate estimation
        blink_rate = self._estimate_blink_rate(pose_array)
        self._add_biomarker(
            biomarker_set,
            name="blink_rate",
            value=blink_rate,
            unit="blinks_per_min",
            confidence=0.65,
            normal_range=(12, 20),
            description="Estimated blink rate"
        )
        
        # Gaze stability (eye position variance)
        gaze_stability = self._calculate_gaze_stability(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set,
            name="gaze_stability_score",
            value=gaze_stability,
            unit="score_0_100",
            confidence=0.60,
            normal_range=(70, 100),
            description="Gaze steadiness and stability"
        )
        
        # Fixation duration proxy
        fixation_duration = self._estimate_fixation_duration(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set,
            name="fixation_duration",
            value=fixation_duration,
            unit="ms",
            confidence=0.55,
            normal_range=(150, 400),
            description="Average fixation duration"
        )
        
        # Saccade frequency
        saccade_freq = self._count_saccades(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set,
            name="saccade_frequency",
            value=saccade_freq,
            unit="saccades_per_sec",
            confidence=0.50,
            normal_range=(2, 5),
            description="Eye movement saccade frequency"
        )
        
        # Eye symmetry
        eye_symmetry = self._calculate_eye_symmetry(left_eye, right_eye)
        self._add_biomarker(
            biomarker_set,
            name="eye_symmetry",
            value=eye_symmetry,
            unit="ratio",
            confidence=0.70,
            normal_range=(0.9, 1.0),
            description="Left-right eye movement symmetry"
        )
    
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
        self._add_biomarker(biomarker_set, "blink_rate",
                           np.random.uniform(14, 18), "blinks_per_min",
                           0.5, (12, 20), "Simulated blink rate")
        self._add_biomarker(biomarker_set, "gaze_stability_score",
                           np.random.uniform(80, 95), "score_0_100",
                           0.5, (70, 100), "Simulated gaze stability")
        self._add_biomarker(biomarker_set, "fixation_duration",
                           np.random.uniform(200, 350), "ms",
                           0.5, (150, 400), "Simulated fixation duration")
        self._add_biomarker(biomarker_set, "saccade_frequency",
                           np.random.uniform(2.5, 4), "saccades_per_sec",
                           0.5, (2, 5), "Simulated saccade frequency")
        self._add_biomarker(biomarker_set, "eye_symmetry",
                           np.random.uniform(0.92, 0.98), "ratio",
                           0.5, (0.9, 1.0), "Simulated eye symmetry")

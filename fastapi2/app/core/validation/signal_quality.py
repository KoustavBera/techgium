"""
Signal Quality Assessment Module

Computes quality metrics for each sensing modality using physics-based analysis.
NO ML/AI - purely signal processing based.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats

from app.core.ingestion.sync import DataPacket
from app.utils import get_logger

logger = get_logger(__name__)


class Modality(str, Enum):
    """Sensing modalities."""
    CAMERA = "camera"
    MOTION = "motion"
    RIS = "ris"
    AUXILIARY = "auxiliary"


@dataclass
class ModalityQualityScore:
    """Quality assessment for a single modality."""
    modality: Modality
    continuity: float = 0.0        # 0-1: temporal continuity of signal
    noise_level: float = 0.0       # 0-1: inverse of noise (1 = clean)
    snr: float = 0.0               # 0-1: signal-to-noise ratio normalized
    dropout_rate: float = 0.0      # 0-1: inverse of dropout percentage
    artifact_level: float = 0.0    # 0-1: inverse of artifact contamination
    overall_quality: float = 0.0   # 0-1: weighted aggregate
    
    issues: List[str] = field(default_factory=list)
    
    def compute_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted overall quality score."""
        if weights is None:
            weights = {
                "continuity": 0.25,
                "noise_level": 0.20,
                "snr": 0.25,
                "dropout_rate": 0.15,
                "artifact_level": 0.15
            }
        
        self.overall_quality = (
            self.continuity * weights["continuity"] +
            self.noise_level * weights["noise_level"] +
            self.snr * weights["snr"] +
            self.dropout_rate * weights["dropout_rate"] +
            self.artifact_level * weights["artifact_level"]
        )
        return self.overall_quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "modality": self.modality.value,
            "continuity": round(self.continuity, 3),
            "noise_level": round(self.noise_level, 3),
            "snr": round(self.snr, 3),
            "dropout_rate": round(self.dropout_rate, 3),
            "artifact_level": round(self.artifact_level, 3),
            "overall_quality": round(self.overall_quality, 3),
            "issues": self.issues
        }


class SignalQualityAssessor:
    """
    Assesses signal quality for all sensing modalities.
    
    Uses physics-based signal processing - NO ML/AI.
    """
    
    def __init__(self):
        """Initialize assessor."""
        self._assessment_count = 0
        logger.info("SignalQualityAssessor initialized (NO-ML)")
    
    def assess_camera(self, frames: List[np.ndarray], timestamps: Optional[List[float]] = None) -> ModalityQualityScore:
        """
        Assess camera signal quality.
        
        Checks:
        - Frame continuity (consistent timing)
        - Image noise (variance analysis)
        - Motion artifacts (blur detection)
        - Dropouts (missing frames)
        
        Args:
            frames: List of BGR image arrays
            timestamps: Optional list of timestamps in ms
            
        Returns:
            ModalityQualityScore for camera
        """
        score = ModalityQualityScore(modality=Modality.CAMERA)
        issues = []
        
        if not frames:
            score.issues = ["No frames provided"]
            return score
        
        n_frames = len(frames)
        
        # Continuity: Check temporal consistency
        if timestamps and len(timestamps) >= 2:
            intervals = np.diff(timestamps)
            expected_interval = np.median(intervals)
            if expected_interval > 0:
                variation = np.std(intervals) / expected_interval
                score.continuity = float(np.clip(1.0 - variation, 0, 1))
                if variation > 0.3:
                    issues.append(f"Frame timing unstable (CV={variation:.2f})")
            else:
                score.continuity = 0.5
        else:
            score.continuity = 0.8  # Assume good if no timestamps
        
        # Noise level: Estimate from high-frequency content
        noise_estimates = []
        for frame in frames[:min(10, n_frames)]:  # Sample first 10 frames
            if frame is not None and frame.size > 0:
                gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
                # Laplacian variance as noise proxy
                laplacian_var = np.var(gray)
                noise_estimates.append(laplacian_var)
        
        if noise_estimates:
            avg_noise = np.mean(noise_estimates)
            # Normalize: high variance = good detail, but very high = noise
            # Optimal range: 100-2000
            if avg_noise < 10:
                score.noise_level = 0.3  # Too flat/dark
                issues.append("Low image variance - possibly underexposed")
            elif avg_noise > 5000:
                score.noise_level = 0.4  # Too noisy
                issues.append("High image noise detected")
            else:
                score.noise_level = float(np.clip(avg_noise / 2000, 0.5, 1.0))
        else:
            score.noise_level = 0.5
        
        # SNR: Signal-to-noise ratio from frame differences
        if n_frames >= 2:
            diffs = []
            for i in range(min(5, n_frames - 1)):
                if frames[i] is not None and frames[i+1] is not None:
                    diff = np.abs(frames[i].astype(float) - frames[i+1].astype(float))
                    diffs.append(np.mean(diff))
            
            if diffs:
                mean_diff = np.mean(diffs)
                # Low diff = stable signal, high diff = motion/noise
                if mean_diff < 1:
                    score.snr = 0.95
                elif mean_diff > 50:
                    score.snr = 0.4
                    issues.append("High inter-frame variation")
                else:
                    score.snr = float(np.clip(1.0 - mean_diff / 100, 0.4, 0.95))
        else:
            score.snr = 0.7
        
        # Dropout rate: Check for None or empty frames
        valid_frames = sum(1 for f in frames if f is not None and f.size > 0)
        score.dropout_rate = float(valid_frames / n_frames) if n_frames > 0 else 0.0
        if score.dropout_rate < 0.9:
            issues.append(f"Frame dropouts detected ({(1-score.dropout_rate)*100:.1f}%)")
        
        # Artifact level: Blur detection via edge analysis
        blur_scores = []
        for frame in frames[:min(10, n_frames)]:
            if frame is not None and frame.size > 0:
                gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
                # Sobel edge detection
                dx = np.diff(gray, axis=1)
                dy = np.diff(gray, axis=0)
                edge_strength = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
                blur_scores.append(edge_strength)
        
        if blur_scores:
            avg_edge = np.mean(blur_scores)
            # Good edges: 10-50, blurry: <5
            score.artifact_level = float(np.clip(avg_edge / 30, 0.3, 1.0))
            if avg_edge < 5:
                issues.append("Possible motion blur detected")
        else:
            score.artifact_level = 0.7
        
        score.issues = issues
        score.compute_overall()
        self._assessment_count += 1
        
        return score
    
    def assess_motion(self, poses: List[np.ndarray], timestamps: Optional[List[float]] = None) -> ModalityQualityScore:
        """
        Assess motion/pose signal quality.
        
        Checks:
        - Pose detection continuity
        - Landmark confidence
        - Physiological plausibility of motion
        - Sudden jumps (tracking failures)
        
        Args:
            poses: List of pose arrays (Nx33x4)
            timestamps: Optional list of timestamps
            
        Returns:
            ModalityQualityScore for motion
        """
        score = ModalityQualityScore(modality=Modality.MOTION)
        issues = []
        
        if not poses:
            score.issues = ["No pose data provided"]
            return score
        
        n_poses = len(poses)
        valid_poses = [p for p in poses if p is not None and len(p) > 0]
        
        # Continuity: Ratio of valid detections
        score.continuity = float(len(valid_poses) / n_poses) if n_poses > 0 else 0.0
        if score.continuity < 0.8:
            issues.append(f"Pose detection gaps ({(1-score.continuity)*100:.1f}% missing)")
        
        # Noise level: Based on landmark visibility/confidence
        if valid_poses:
            confidences = []
            for pose in valid_poses:
                if len(pose.shape) == 2 and pose.shape[1] >= 4:
                    # Visibility is typically in column 3
                    vis = pose[:, 3] if pose.shape[1] > 3 else np.ones(pose.shape[0])
                    confidences.append(np.mean(vis))
            
            if confidences:
                avg_conf = np.mean(confidences)
                score.noise_level = float(np.clip(avg_conf, 0, 1))
                if avg_conf < 0.5:
                    issues.append("Low landmark detection confidence")
        else:
            score.noise_level = 0.0
        
        # SNR: Motion smoothness (jerky motion = low quality)
        if len(valid_poses) >= 3:
            velocities = []
            for i in range(len(valid_poses) - 1):
                p1, p2 = valid_poses[i], valid_poses[i+1]
                if p1.shape == p2.shape:
                    vel = np.mean(np.abs(p2[:, :3] - p1[:, :3]))
                    velocities.append(vel)
            
            if velocities:
                # Smooth motion has low velocity variance
                vel_cv = np.std(velocities) / (np.mean(velocities) + 1e-6)
                score.snr = float(np.clip(1.0 - vel_cv / 2, 0.3, 1.0))
                if vel_cv > 1.5:
                    issues.append("Jerky motion detected - possible tracking errors")
        else:
            score.snr = 0.5
        
        # Dropout rate: Same as continuity for motion
        score.dropout_rate = score.continuity
        
        # Artifact level: Check for impossible landmark positions
        artifact_count = 0
        for pose in valid_poses:
            if len(pose.shape) == 2:
                # Check for values outside [0, 1] for normalized coords
                out_of_range = np.sum((pose[:, :2] < -0.5) | (pose[:, :2] > 1.5))
                if out_of_range > pose.shape[0] * 0.1:  # >10% landmarks bad
                    artifact_count += 1
        
        score.artifact_level = float(1.0 - artifact_count / len(valid_poses)) if valid_poses else 0.0
        if artifact_count > 0:
            issues.append(f"Out-of-bounds landmarks in {artifact_count} frames")
        
        score.issues = issues
        score.compute_overall()
        self._assessment_count += 1
        
        return score
    
    def assess_ris(self, ris_data: np.ndarray, sample_rate: int = 1000) -> ModalityQualityScore:
        """
        Assess RIS bioimpedance signal quality.
        
        Checks:
        - Signal continuity (no gaps)
        - Noise floor estimation
        - Physiological frequency content (cardiac, respiratory)
        - Saturation/clipping artifacts
        
        Args:
            ris_data: RIS signal array (samples x channels)
            sample_rate: Sampling rate in Hz
            
        Returns:
            ModalityQualityScore for RIS
        """
        score = ModalityQualityScore(modality=Modality.RIS)
        issues = []
        
        if ris_data is None or ris_data.size == 0:
            score.issues = ["No RIS data provided"]
            return score
        
        # Ensure 2D
        if len(ris_data.shape) == 1:
            ris_data = ris_data.reshape(-1, 1)
        
        n_samples, n_channels = ris_data.shape
        
        # Continuity: Check for NaN/Inf values
        valid_samples = np.sum(~np.isnan(ris_data) & ~np.isinf(ris_data))
        total_samples = ris_data.size
        score.continuity = float(valid_samples / total_samples) if total_samples > 0 else 0.0
        if score.continuity < 0.99:
            issues.append(f"Invalid RIS samples: {(1-score.continuity)*100:.2f}%")
        
        # Replace invalid for further analysis
        ris_clean = np.nan_to_num(ris_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Noise level: High-frequency noise estimation
        noise_scores = []
        for ch in range(n_channels):
            channel = ris_clean[:, ch]
            if len(channel) > 10:
                # High-pass filter to extract noise
                diff2 = np.diff(channel, n=2)
                noise_power = np.var(diff2)
                signal_power = np.var(channel) + 1e-10
                # Lower noise ratio = better
                noise_ratio = noise_power / signal_power
                noise_scores.append(1.0 / (1.0 + noise_ratio))
        
        score.noise_level = float(np.mean(noise_scores)) if noise_scores else 0.5
        
        # SNR: Check for physiological frequency content
        if n_samples >= sample_rate:  # At least 1 second
            snr_scores = []
            for ch in range(min(4, n_channels)):  # Check first 4 channels
                channel = ris_clean[:, ch]
                
                # FFT analysis
                freqs = np.fft.rfftfreq(len(channel), 1/sample_rate)
                fft_mag = np.abs(np.fft.rfft(channel))
                
                # Cardiac band: 0.8-3 Hz (48-180 bpm)
                cardiac_mask = (freqs >= 0.8) & (freqs <= 3.0)
                cardiac_power = np.sum(fft_mag[cardiac_mask]**2) if np.any(cardiac_mask) else 0
                
                # Respiratory band: 0.1-0.5 Hz (6-30 breaths/min)
                resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
                resp_power = np.sum(fft_mag[resp_mask]**2) if np.any(resp_mask) else 0
                
                # Total power
                total_power = np.sum(fft_mag**2) + 1e-10
                
                # Good signal has strong cardiac + respiratory content
                physio_ratio = (cardiac_power + resp_power) / total_power
                snr_scores.append(physio_ratio)
            
            score.snr = float(np.clip(np.mean(snr_scores) * 5, 0.3, 1.0))  # Scale up
            if score.snr < 0.5:
                issues.append("Weak physiological signal content in RIS")
        else:
            score.snr = 0.5
        
        # Dropout rate: Check for constant values (stuck sensor)
        dropout_channels = 0
        for ch in range(n_channels):
            channel = ris_clean[:, ch]
            if np.std(channel) < 1e-6:  # No variation
                dropout_channels += 1
        
        score.dropout_rate = float(1.0 - dropout_channels / n_channels) if n_channels > 0 else 0.0
        if dropout_channels > 0:
            issues.append(f"{dropout_channels} RIS channels showing no variation")
        
        # Artifact level: Check for clipping/saturation
        clip_count = 0
        for ch in range(n_channels):
            channel = ris_clean[:, ch]
            ch_max, ch_min = np.max(channel), np.min(channel)
            # Check if values cluster at extremes
            at_max = np.sum(channel >= ch_max * 0.99) / len(channel)
            at_min = np.sum(channel <= ch_min * 1.01 + 1e-6) / len(channel)
            if at_max > 0.05 or at_min > 0.05:
                clip_count += 1
        
        score.artifact_level = float(1.0 - clip_count / n_channels) if n_channels > 0 else 0.0
        if clip_count > 0:
            issues.append(f"{clip_count} RIS channels showing clipping")
        
        score.issues = issues
        score.compute_overall()
        self._assessment_count += 1
        
        return score
    
    def assess_auxiliary(self, vitals: Dict[str, Any]) -> ModalityQualityScore:
        """
        Assess auxiliary vital signs quality.
        
        Checks:
        - Data completeness
        - Value plausibility
        - Sensor reading consistency
        
        Args:
            vitals: Dictionary of vital signs
            
        Returns:
            ModalityQualityScore for auxiliary
        """
        score = ModalityQualityScore(modality=Modality.AUXILIARY)
        issues = []
        
        if not vitals:
            score.issues = ["No vital signs provided"]
            return score
        
        expected_keys = ["heart_rate", "respiratory_rate", "spo2", "temperature", 
                        "systolic_bp", "diastolic_bp"]
        
        # Continuity: Check completeness
        present = sum(1 for k in expected_keys if k in vitals and vitals[k] is not None)
        score.continuity = float(present / len(expected_keys))
        if score.continuity < 0.8:
            missing = [k for k in expected_keys if k not in vitals or vitals[k] is None]
            issues.append(f"Missing vitals: {', '.join(missing)}")
        
        # Noise level: Based on plausibility
        valid_values = 0
        total_values = 0
        
        plausibility_ranges = {
            "heart_rate": (30, 200),
            "respiratory_rate": (4, 40),
            "spo2": (70, 100),
            "temperature": (35, 42),
            "systolic_bp": (70, 200),
            "diastolic_bp": (40, 130)
        }
        
        for key, (low, high) in plausibility_ranges.items():
            if key in vitals and vitals[key] is not None:
                total_values += 1
                val = vitals[key]
                if low <= val <= high:
                    valid_values += 1
                else:
                    issues.append(f"{key}={val} outside plausible range [{low}, {high}]")
        
        score.noise_level = float(valid_values / total_values) if total_values > 0 else 0.0
        
        # SNR: Check internal consistency
        consistency_score = 1.0
        if "systolic_bp" in vitals and "diastolic_bp" in vitals:
            sbp, dbp = vitals.get("systolic_bp", 0), vitals.get("diastolic_bp", 0)
            if sbp and dbp and sbp <= dbp:
                consistency_score -= 0.3
                issues.append("Systolic BP <= Diastolic BP (impossible)")
        
        score.snr = float(consistency_score)
        
        # Dropout rate: Same as continuity
        score.dropout_rate = score.continuity
        
        # Artifact level: Check for obviously wrong values
        artifact_issues = 0
        if vitals.get("heart_rate", 0) > 250:
            artifact_issues += 1
        if vitals.get("spo2", 100) < 50:
            artifact_issues += 1
        
        score.artifact_level = float(1.0 - artifact_issues / 6)
        
        score.issues = issues
        score.compute_overall()
        self._assessment_count += 1
        
        return score
    
    def assess_all(
        self,
        camera_frames: Optional[List[np.ndarray]] = None,
        motion_poses: Optional[List[np.ndarray]] = None,
        ris_data: Optional[np.ndarray] = None,
        vitals: Optional[Dict[str, Any]] = None,
        timestamps: Optional[Dict[str, List[float]]] = None
    ) -> Dict[Modality, ModalityQualityScore]:
        """
        Assess quality for all available modalities.
        
        Returns:
            Dict mapping modality to quality score
        """
        results = {}
        ts = timestamps or {}
        
        if camera_frames is not None:
            results[Modality.CAMERA] = self.assess_camera(
                camera_frames, ts.get("camera")
            )
        
        if motion_poses is not None:
            results[Modality.MOTION] = self.assess_motion(
                motion_poses, ts.get("motion")
            )
        
        if ris_data is not None:
            results[Modality.RIS] = self.assess_ris(ris_data)
        
        if vitals is not None:
            results[Modality.AUXILIARY] = self.assess_auxiliary(vitals)
        
        logger.debug(f"Assessed {len(results)} modalities")
        return results

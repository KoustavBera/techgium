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

# from app.core.ingestion.sync import DataPacket
from app.utils import get_logger

logger = get_logger(__name__)


class Modality(str, Enum):
    """Sensing modalities."""
    CAMERA = "camera"
    MOTION = "motion"
    RADAR = "radar"      # Seeed MR60BHA2 mmWave
    THERMAL = "thermal"  # MLX90640 thermal camera
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
            # Normalize: high variance = good detail, low variance = blurry
            # Optimal range: 50-3000
            if avg_noise < 50:
                score.noise_level = 0.3  # Too blurry/flat
                issues.append("Low image variance - possibly underexposed or blurry")
            elif avg_noise > 3000:
                score.noise_level = 0.6  # Excessive noise but still usable
                issues.append("High image variance detected")
            else:
                score.noise_level = float(np.clip((avg_noise - 50) / 2950, 0.3, 1.0))
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
                # Low diff = frozen frames (bad), moderate = natural motion (good)
                if mean_diff < 2:
                    score.snr = 0.4  # Frozen frames
                    issues.append("Very low inter-frame variation - possible frozen frames")
                elif mean_diff > 80:
                    score.snr = 0.5  # Too much motion
                    issues.append("High inter-frame variation - excessive motion")
                else:
                    # Optimal around 20-40 diff
                    score.snr = float(np.clip(1.0 - abs(mean_diff - 30) / 50, 0.5, 1.0))
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
        
        # Dropout rate: count completely missing frames (None)
        # Note: continuity handles invalid poses; dropout only handles None frames
        missing_frames = sum(1 for p in poses if p is None)
        score.dropout_rate = float(1.0 - missing_frames / n_poses) if n_poses > 0 else 0.0
        
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
    
    def assess_radar(self, radar_data: Dict[str, Any]) -> ModalityQualityScore:
        """
        Assess Seeed MR60BHA2 mmWave radar data quality.
        
        Checks:
        - Presence detection status
        - Heart rate plausibility (40-200 bpm)
        - Respiration rate plausibility (5-40 rpm)
        - Signal consistency
        
        Args:
            radar_data: Dict with keys: heart_rate, respiration_rate, 
                       presence_detected, breathing_depth (optional)
            
        Returns:
            ModalityQualityScore for radar
        """
        score = ModalityQualityScore(modality=Modality.RADAR)
        issues = []
        
        if not radar_data:
            score.issues = ["No radar data provided"]
            return score
        
        # Continuity: Check presence detection
        presence = radar_data.get("presence_detected", False)
        if not presence:
            score.continuity = 0.3
            issues.append("No presence detected by radar")
        else:
            score.continuity = 1.0
        
        # Noise level: Check if values are within plausible ranges
        hr = radar_data.get("heart_rate")
        rr = radar_data.get("respiration_rate")
        
        valid_count = 0
        total_checks = 0
        
        if hr is not None:
            total_checks += 1
            if 40 <= hr <= 200:
                valid_count += 1
            else:
                issues.append(f"Heart rate {hr:.0f} bpm outside plausible range [40, 200]")
        
        if rr is not None:
            total_checks += 1
            if 5 <= rr <= 40:
                valid_count += 1
            else:
                issues.append(f"Respiration rate {rr:.1f} rpm outside plausible range [5, 40]")
        
        score.noise_level = float(valid_count / total_checks) if total_checks > 0 else 0.5
        
        # SNR: Based on breathing depth (signal strength proxy)
        breathing_depth = radar_data.get("breathing_depth", 0.5)
        if breathing_depth is not None:
            score.snr = float(np.clip(breathing_depth, 0.0, 1.0))
            if breathing_depth < 0.3:
                issues.append("Weak breathing signal detected")
        else:
            score.snr = 0.7
        
        # Dropout rate: Based on presence
        score.dropout_rate = 1.0 if presence else 0.3
        
        # Artifact level: Check for stuck/invalid readings
        artifact_issues = 0
        if hr is not None and (hr == 0 or np.isnan(hr)):
            artifact_issues += 1
        if rr is not None and (rr == 0 or np.isnan(rr)):
            artifact_issues += 1
        
        score.artifact_level = float(1.0 - artifact_issues / 2) if total_checks > 0 else 0.5
        
        score.issues = issues
        score.compute_overall()
        self._assessment_count += 1
        
        return score
    
    def assess_thermal(self, thermal_data: Dict[str, Any]) -> ModalityQualityScore:
        """
        Assess MLX90640 thermal camera data quality.
        
        Checks:
        - Temperature plausibility (skin: 30-42°C)
        - Thermal map validity (no dead pixels)
        - Thermal asymmetry reasonableness (< 2.0°C)
        
        Args:
            thermal_data: Dict with keys: skin_temp_avg, skin_temp_max,
                         thermal_asymmetry, thermal_map (optional 32x24 array)
            
        Returns:
            ModalityQualityScore for thermal
        """
        score = ModalityQualityScore(modality=Modality.THERMAL)
        issues = []
        
        if not thermal_data:
            score.issues = ["No thermal data provided"]
            return score
        
        # Continuity: Check if we have basic temperature readings
        temp_avg = thermal_data.get("skin_temp_avg")
        temp_max = thermal_data.get("skin_temp_max")
        
        if temp_avg is not None and temp_max is not None:
            score.continuity = 1.0
        elif temp_avg is not None or temp_max is not None:
            score.continuity = 0.7
        else:
            score.continuity = 0.3
            issues.append("Missing temperature readings")
        
        # Noise level: Check temperature plausibility
        valid_temps = 0
        total_temps = 0
        
        if temp_avg is not None:
            total_temps += 1
            if 30.0 <= temp_avg <= 42.0:
                valid_temps += 1
            else:
                issues.append(f"Average skin temp {temp_avg:.1f}°C outside range [30, 42]")
        
        if temp_max is not None:
            total_temps += 1
            if 30.0 <= temp_max <= 45.0:
                valid_temps += 1
            else:
                issues.append(f"Max skin temp {temp_max:.1f}°C outside range [30, 45]")
        
        score.noise_level = float(valid_temps / total_temps) if total_temps > 0 else 0.5
        
        # SNR: Based on thermal asymmetry (lower is better signal)
        asymmetry = thermal_data.get("thermal_asymmetry", 0.5)
        if asymmetry is not None:
            # Smooth decay: 1.0 at 0°C, ~0.67 at 1°C, ~0.33 at 2°C, floor at 0.3
            score.snr = float(np.clip(1.0 - asymmetry / 3.0, 0.3, 1.0))
            
            if asymmetry > 2.0:
                issues.append(f"High thermal asymmetry ({asymmetry:.1f}°C) may indicate measurement error")
        else:
            score.snr = 0.7
        
        # Dropout rate: Check thermal map for dead pixels (if available)
        thermal_map = thermal_data.get("thermal_map")
        if thermal_map is not None:
            try:
                thermal_arr = np.array(thermal_map)
                # Dead pixels are typically 0 or very low values
                dead_pixels = np.sum(thermal_arr < 20.0)
                total_pixels = thermal_arr.size
                score.dropout_rate = float(1.0 - dead_pixels / total_pixels) if total_pixels > 0 else 0.5
                if dead_pixels > total_pixels * 0.05:
                    issues.append(f"{dead_pixels} dead pixels detected in thermal map")
            except Exception:
                score.dropout_rate = 0.5
        else:
            score.dropout_rate = 0.8  # Assume OK if no map provided
        
        # Artifact level: Check for sensor errors (all same value, extreme values)
        artifact_issues = 0
        if temp_avg is not None and temp_max is not None:
            if temp_avg > temp_max:
                artifact_issues += 1
                issues.append("Average temp > Max temp (sensor error)")
            if abs(temp_max - temp_avg) > 10:
                artifact_issues += 1
                issues.append("Large temp difference may indicate artifact")
        
        score.artifact_level = float(1.0 - artifact_issues / 2)
        
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
        
        max_artifact_checks = 2  # HR > 250, SpO2 < 50
        score.artifact_level = float(1.0 - artifact_issues / max_artifact_checks)
        
        score.issues = issues
        score.compute_overall()
        self._assessment_count += 1
        
        return score
    
    def assess_all(
        self,
        camera_frames: Optional[List[np.ndarray]] = None,
        motion_poses: Optional[List[np.ndarray]] = None,
        radar_data: Optional[Dict[str, Any]] = None,
        thermal_data: Optional[Dict[str, Any]] = None,
        vitals: Optional[Dict[str, Any]] = None,
        timestamps: Optional[Dict[str, List[float]]] = None
    ) -> Dict[Modality, ModalityQualityScore]:
        """
        Assess quality for all available modalities.
        
        Args:
            camera_frames: List of video frames
            motion_poses: List of pose landmarks
            radar_data: Dict from Seeed MR60BHA2 radar
            thermal_data: Dict from MLX90640 thermal camera
            vitals: Dict of auxiliary vital signs
            timestamps: Optional timestamps per modality
        
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
        
        if radar_data is not None:
            results[Modality.RADAR] = self.assess_radar(radar_data)
        
        if thermal_data is not None:
            results[Modality.THERMAL] = self.assess_thermal(thermal_data)
        
        if vitals is not None:
            results[Modality.AUXILIARY] = self.assess_auxiliary(vitals)
        
        logger.debug(f"Assessed {len(results)} modalities")
        return results


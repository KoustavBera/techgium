"""
Skin Biomarker Extractor

Extracts skin health indicators from camera data:
- Surface texture roughness (GLCM)
- Color maps / pigmentation analysis (CIELab)
- Lesion morphology detection (Interface only)
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)

# Physiological fallback values for sensor failures
FALLBACK_VALUES = {
    "skin_temperature": 36.5,
    "skin_temperature_max": 37.0,
    "thermal_asymmetry": 0.2,
    "texture_roughness": 15.0,
    "skin_redness": 0.45,
    "skin_yellowness": 0.35,
    "color_uniformity": 0.85,
    "lesion_count": 0.0
}


@dataclass
class SessionBaseline:
    """Session-specific environmental baseline (not stored after scan)."""
    baseline_facial_temp: float = 36.0
    baseline_redness: float = 0.0  # CIELab *a
    baseline_yellowness: float = 0.0  # CIELab *b
    ambient_background_temp: float = 25.0  # Room temp from thermal camera
    ambient_light_level: float = 120.0  # Average RGB intensity

    def to_dict(self) -> Dict[str, float]:
        return {
            "baseline_facial_temp": self.baseline_facial_temp,
            "baseline_redness": self.baseline_redness,
            "baseline_yellowness": self.baseline_yellowness,
            "ambient_background_temp": self.ambient_background_temp,
            "ambient_light_level": self.ambient_light_level
        }


class SkinExtractor(BaseExtractor):
    """
    Extracts skin biomarkers from visual data.
    
    Analyzes camera frames for dermatological indicators using Computer Vision:
    - Face Detection: MediaPipe Face Mesh
    - Color Analysis: CIELab Color Space
    - Texture Analysis: GLCM (Gray Level Co-occurrence Matrix)
    """
    
    system = PhysiologicalSystem.SKIN
    
    def __init__(self):
        super().__init__()
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def _safe_normal_range(self, bm_range: Any) -> Optional[tuple]:
        """Convert normal_range to safe tuple format."""
        try:
            if isinstance(bm_range, (list, tuple)) and len(bm_range) == 2:
                return tuple(float(x) for x in bm_range)
        except (TypeError, ValueError):
            pass
        return None
    
    def _get_fallback_value(self, name: str) -> float:
        """Get physiologically reasonable fallback value for a biomarker."""
        return FALLBACK_VALUES.get(name, 0.0)
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract skin biomarkers.
        
        Expected data keys:
        - frames: List of video frames (HxWx3 arrays, BGR format)
        - esp32_data: Dict containing thermal metrics from MLX90640
        - systems: List of pre-processed systems from bridge
        - session_baseline: Optional baseline context for normalization
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        session_baseline = data.get("session_baseline")
        
        # Priority 1: Hardware Thermal Data (ESP32/MLX90640)
        has_thermal = False
        
        # NEW FORMAT: Flattened thermal_data from bridge.py
        if "thermal_data" in data:
            # Try to get landmarks from frames for pose gating
            pose_landmarks = None
            frames = data.get("frames", data.get("face_frames", []))
            if frames:
                _, _, pose_landmarks = self._get_face_mask(frames[0])
                
            self._extract_from_thermal_v2(
                data["thermal_data"], 
                biomarker_set, 
                session_baseline,
                pose_landmarks=pose_landmarks
            )
            has_thermal = True
        # OLD FORMAT: Nested esp32_data
        elif "esp32_data" in data:
            self._extract_from_thermal(data["esp32_data"], biomarker_set)
            has_thermal = True
        elif "systems" in data:
            # Check for pre-processed thermal data
            for sys in data["systems"]:
                if sys.get("system") == "skin":
                    for bm in sys.get("biomarkers", []):
                        self._add_biomarker_safe(
                            biomarker_set,
                            name=bm["name"],
                            value=bm["value"],
                            unit=bm["unit"],
                            confidence=0.95,
                            normal_range=self._safe_normal_range(bm.get("normal_range")),
                            description="From Thermal Camera (MLX90640)"
                        )
                        has_thermal = True
        
        # Priority 2: Visual Analysis (Webcam)
        frames = data.get("frames", data.get("face_frames", []))
        if len(frames) > 0:
            frame = np.array(frames[0]) if not isinstance(frames[0], np.ndarray) else frames[0]
            self._extract_from_frame(frame, biomarker_set, session_baseline)
        elif not has_thermal:
            logger.warning("SkinExtractor: No data sources available.")
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_frame(
        self,
        frame: np.ndarray,
        biomarker_set: BiomarkerSet,
        session_baseline: Optional[SessionBaseline] = None
    ) -> None:
        """Extract skin metrics from a video frame."""
        
        if frame.ndim < 2:
            return
        
        # 1. Face Segmentation (MediaPipe)
        face_mask, face_roi_crop, landmarks = self._get_face_mask(frame)
        
        if face_mask is None:
            logger.warning("Skin: No face detected.")
            return

        # 2. Texture Analysis (GLCM on Green Channel of ROI)
        texture_roughness = self._analyze_texture_glcm(face_roi_crop, face_mask)
        self._add_biomarker(
            biomarker_set,
            name="texture_roughness",
            value=texture_roughness,
            unit="glcm_contrast",
            confidence=0.75,
            normal_range=(0.0, 5.0),  # Calibrated for webcam GLCM (32 levels, bilateral filtered)
            description="Skin surface texture (GLCM Contrast, multi-angle)"
        )
        
        # 3. Color Analysis (CIELab on Masked Face)
        color_metrics = self._analyze_skin_color_lab(frame, face_mask, session_baseline)
        
        self._add_biomarker(
            biomarker_set,
            name="skin_redness",
            value=color_metrics["redness"],
            unit="lab_deviation",
            confidence=0.85,
            normal_range=(-10.0, 10.0) if session_baseline else (0.0, 25.0),
            description="Skin redness (Lab a* deviation from session baseline)" if session_baseline else "Skin redness (Lab a* deviation from neutral)"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="skin_yellowness",
            value=color_metrics["yellowness"],
            unit="lab_deviation",
            confidence=0.80,
            normal_range=(-10.0, 10.0) if session_baseline else (0.0, 25.0),
            description="Skin yellowness (Lab b* deviation from session baseline)" if session_baseline else "Skin yellowness (Lab b* deviation from neutral)"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="color_uniformity",
            value=color_metrics["uniformity"],
            unit="entropy_inv",
            confidence=0.70,
            normal_range=(0.25, 1.0),  # Calibrated: Real skin has natural variation (0.25-0.7 typical)
            description="Skin tone uniformity (Inverse Entropy)"
        )
        
        # 4. Lesion Detection (Placeholder for Future ML Model)
        self._add_biomarker(
            biomarker_set,
            name="lesion_count",
            value=0.0,
            unit="count",
            confidence=0.0, # Explicitly 0 to indicate Disabled
            normal_range=(0, 5),
            description="Skin lesions (Disabled: Requires ML Model)"
        )
        
        # 5. Head Pose Logging (Informational)
        if landmarks:
            yaw, pitch = self._estimate_head_pose(landmarks)
            logger.info(f"Skin: Visual head pose estimation - Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°")
    
    def _get_face_mask(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[Any]]]:
        """
        Generate a binary mask for the face skin using MediaPipe Face Mesh.
        Returns: (full_frame_mask, face_crop_roi, landmarks)
        """
        h, w = frame.shape[:2]
        
        # Resize large frames for MediaPipe efficiency
        max_dim = 640
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            frame_resized = frame
        
        h_r, w_r = frame_resized.shape[:2]
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None, None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Expanded skin region indices (100+ points for better coverage)
        # Face oval + forehead + cheeks (excluding eyes, mouth, eyebrows)
        skin_indices = [
            # Face oval
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
            # Forehead
            151, 108, 69, 299, 337, 151, 9, 107, 66, 105, 104, 63,
            # Cheeks expanded
            206, 207, 187, 123, 116, 117, 118, 119, 120, 121, 128, 245,
            426, 427, 411, 352, 345, 346, 347, 348, 349, 350, 357, 465,
            # Chin
            194, 32, 140, 171, 175, 396, 369, 262, 418
        ]
        
        points = []
        for idx in skin_indices:
            if idx < len(landmarks):
                pt = landmarks[idx]
                # Scale back to original frame coordinates
                points.append((int(pt.x * w), int(pt.y * h)))
            
        if len(points) < 10:
            return None, None, None
            
        # Create mask using convex hull for better coverage
        hull = cv2.convexHull(np.array(points))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Dilate mask to include skin edges, then erode to remove boundary noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Get Bounding Box for Crop
        x, y, w_box, h_box = cv2.boundingRect(hull)
        # Ensure crop is within bounds with padding
        pad = 10
        x, y = max(0, x - pad), max(0, y - pad)
        w_box, h_box = min(w - x, w_box + 2*pad), min(h - y, h_box + 2*pad)
        
        crop = frame[y:y+h_box, x:x+w_box]
        mask_crop = mask[y:y+h_box, x:x+w_box]
        
        # Apply mask to crop (black out non-face)
        processed_crop = cv2.bitwise_and(crop, crop, mask=mask_crop)
        
        return mask, processed_crop, landmarks

    def _analyze_texture_glcm(self, face_crop: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Analyze texture using GLCM (Gray Level Co-occurrence Matrix).
        Uses multi-distance and multi-angle for robustness.
        Metric: Mean Contrast across all distance/angle combinations.
        """
        if face_crop.size == 0:
            return self._get_fallback_value("texture_roughness")
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(face_crop, 9, 75, 75)
        
        # Convert to grayscale and resize for consistency
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256)).astype(np.uint8)
        
        # Quantize to 32 levels for faster GLCM computation
        gray_quantized = (gray // 8).astype(np.uint8)
        
        # Calculate GLCM with multiple distances and angles
        # Distances: 1, 2, 4 pixels; Angles: 0°, 45°, 90°, 135°
        distances = [1, 2, 4]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, 
                           levels=32, symmetric=True, normed=True)
        
        # Calculate mean Contrast across all distance/angle combinations
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        
        # Literature suggests skin contrast typically ranges 10-100
        # Return raw contrast value (no arbitrary scaling)
        return float(contrast)

    def _estimate_head_pose(self, landmarks: List[Any]) -> Tuple[float, float]:
        """
        Estimate head orientation (Yaw, Pitch) from landmarks.
        Yaw: Rotation left/right (0 = center)
        Pitch: Rotation up/down (0 = center)
        """
        if not landmarks or len(landmarks) < 468:
            return 0.0, 0.0

        # Point Indices:
        # 1: Nose Tip
        # 33, 133: Left Eye Outer/Inner
        # 362, 263: Right Eye Inner/Outer
        # 152: Chin
        # 10: Forehead/Top of Face
        
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        chin = landmarks[152]
        forehead = landmarks[10]

        # 1. Yaw Estimation (Horizontal Rotation)
        # Ratio of distances from nose tip to eye outer corners
        dist_left = abs(nose_tip.x - left_eye.x)
        dist_right = abs(nose_tip.x - right_eye.x)
        
        if dist_right == 0: dist_right = 0.01
        yaw_ratio = dist_left / dist_right
        
        # Logarithmic mapping: ratio=1.0 -> 0°, ratio=2.0 -> ~20°
        yaw = np.degrees(np.log(yaw_ratio)) * 20.0

        # 2. Pitch Estimation (Vertical Rotation)
        # Ratio of (Nose-Forehead) / (Nose-Chin)
        dist_up = abs(nose_tip.y - forehead.y)
        dist_down = abs(nose_tip.y - chin.y)
        
        if dist_down == 0: dist_down = 0.01
        pitch_ratio = dist_up / dist_down
        
        # Baseline vertical ratio for frontal face is approx 1.0
        pitch = np.degrees(np.log(pitch_ratio)) * 15.0
        
        return float(yaw), float(pitch)

    def _analyze_skin_color_lab(self, frame: np.ndarray, mask: np.ndarray, session_baseline: Optional[SessionBaseline] = None) -> Dict[str, float]:
        """
        Analyze skin color in CIELab space using the face mask.
        - L: Lightness (Ignored for color metrics to reduce lighting bias)
        - a*: Green-Red component (Redness)
        - b*: Blue-Yellow component (Yellowness)
        
        If session_baseline is provided, metrics are returned as deviations from baseline.
        Otherwise, they are deviations from neutral gray (128).
        """
        if frame.size == 0:
            return {
                "redness": self._get_fallback_value("skin_redness"),
                "yellowness": self._get_fallback_value("skin_yellowness"),
                "uniformity": self._get_fallback_value("color_uniformity")
            }
            
        # Convert to Lab
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        
        # Extract channels
        l_channel, a_channel, b_channel = cv2.split(lab_frame)
        
        # Apply mask to get only skin pixels
        skin_pixels_a = a_channel[mask == 255]
        skin_pixels_b = b_channel[mask == 255]
        
        if len(skin_pixels_a) == 0:
            return {
                "redness": self._get_fallback_value("skin_redness"),
                "yellowness": self._get_fallback_value("skin_yellowness"),
                "uniformity": self._get_fallback_value("color_uniformity")
            }

        # Calculate metrics with deviation scaling
        # If baseline exists, compute deviation from baseline
        # Otherwise, 128 is neutral, so we subtract 128 and scale by 2
        raw_a_mean = np.mean(skin_pixels_a)
        raw_b_mean = np.mean(skin_pixels_b)
        
        if session_baseline:
            redness = float(raw_a_mean - session_baseline.baseline_redness)
            yellowness = float(raw_b_mean - session_baseline.baseline_yellowness)
        else:
            redness = float((raw_a_mean - 128.0) * 2.0)
            yellowness = float((raw_b_mean - 128.0) * 2.0)
        
        # Uniformity: Entropy of the a* channel (pigmentation variation)
        # Lower entropy = Higher uniformity
        try:
            # Ensure integer type for bincount
            skin_a_int = skin_pixels_a.astype(np.int32)
            counts = np.bincount(skin_a_int, minlength=256)
            probs = counts[counts > 0] / len(skin_pixels_a)
            
            # Handle edge case: if all pixels are same value, entropy = 0
            if len(probs) <= 1:
                entropy = 0.0
            else:
                # Avoid log(0) by filtering already done above
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            # Invert scale: Max realistic entropy ~6 bits for skin.
            # Map 0 (uniform) -> 1.0, 6 (varied) -> 0.0
            uniformity = max(0.0, min(1.0, 1.0 - (entropy / 6.0)))
        except Exception:
            uniformity = 0.8  # Fallback
             
        return {
            "redness": float(redness),
            "yellowness": float(yellowness),
            "uniformity": float(uniformity)
        }
    
    def _detect_lesions(self, frame: np.ndarray) -> int:
        """Disabled Lesion Detection."""
        return 0

    # SIMULATION METHOD REMOVED

    def _extract_from_thermal(self, thermal_data: Dict[str, Any], biomarker_set: BiomarkerSet) -> None:
        """Extract skin metrics from thermal sensor data (OLD FORMAT)."""
        data = thermal_data.get("thermal", {})
        
        if "skin_temp_avg" in data:
            self._add_biomarker_safe(
                biomarker_set,
                name="skin_temperature",
                value=float(data["skin_temp_avg"]),
                unit="celsius",
                confidence=0.90,
                normal_range=(35.5, 37.5),
                description="Average facial skin temperature (MLX90640)"
            )
            
        if "skin_temp_max" in data:
            self._add_biomarker_safe(
                biomarker_set,
                name="skin_temperature_max",
                value=float(data["skin_temp_max"]),
                unit="celsius",
                confidence=0.90,
                normal_range=(36.0, 38.0),
                description="Max facial skin temperature (Inner canthus proxy)"
            )
            
        if "thermal_asymmetry" in data:
            self._add_biomarker_safe(
                biomarker_set,
                name="thermal_asymmetry",
                value=float(data["thermal_asymmetry"]),
                unit="delta_celsius",
                confidence=0.85,
                normal_range=(0.0, 0.5),
                description="Thermal asymmetry (Left vs Right)"
            )

    def capture_session_baseline(
        self, 
        thermal_frames: List[Dict[str, Any]], 
        rgb_frames: List[np.ndarray]
    ) -> SessionBaseline:
        """
        Capture environmental baseline from initial data.
        
        Args:
            thermal_frames: List of thermal data dicts
            rgb_frames: List of RGB frames
        """
        logger.info(f"Skin: Capturing session baseline from {len(thermal_frames)} thermal and {len(rgb_frames)} RGB frames")
        
        # 1. Thermal Baseline
        facial_temps = []
        background_temps = []
        CALIBRATION_OFFSET = 0.8
        
        for data in thermal_frames:
            face_max = data.get('fever_face_max')
            if face_max is not None:
                facial_temps.append(face_max + CALIBRATION_OFFSET)
            
            bg_temp = data.get('background_temp')
            if bg_temp is not None:
                background_temps.append(bg_temp)
                
        baseline_temp = np.median(facial_temps) if facial_temps else 36.0
        ambient_temp = np.median(background_temps) if background_temps else 25.0
        
        # 2. RGB Baseline
        redness_values = []
        yellowness_values = []
        light_levels = []
        
        for frame in rgb_frames:
            if frame is None or frame.size == 0:
                continue
            
            mask, _, _ = self._get_face_mask(frame)
            if mask is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                a_chan = lab[:, :, 1]
                b_chan = lab[:, :, 2]
                redness_values.append(np.mean(a_chan[mask == 255]))
                yellowness_values.append(np.mean(b_chan[mask == 255]))
                light_levels.append(np.mean(frame))
        
        baseline_redness = np.median(redness_values) if redness_values else 128.0
        baseline_yellowness = np.median(yellowness_values) if yellowness_values else 128.0
        baseline_light = np.median(light_levels) if light_levels else 120.0
        
        baseline = SessionBaseline(
            baseline_facial_temp=float(baseline_temp),
            baseline_redness=float(baseline_redness),
            baseline_yellowness=float(baseline_yellowness),
            ambient_background_temp=float(ambient_temp),
            ambient_light_level=float(baseline_light)
        )
        
        logger.info(f"Skin: Session baseline established: {baseline.to_dict()}")
        return baseline

    def _extract_from_thermal_v2(
        self, 
        thermal_data: Dict[str, Any], 
        biomarker_set: BiomarkerSet,
        session_baseline: Optional[SessionBaseline] = None,
        pose_landmarks: Optional[List[Any]] = None
    ) -> None:
        """Extract skin metrics from flattened thermal data (NEW FORMAT v2)."""
        # THERMAL CALIBRATION: Hardware consistently reads ~0.8°C lower than actual
        # Applying offset to bring readings into clinical range
        CALIBRATION_OFFSET = 0.8
        
        # Skin Temperature from canthus (core body proxy - medically validated)
        neck_temp = thermal_data.get('fever_neck_temp')
        canthus_temp = thermal_data.get('fever_canthus_temp')
        face_max = thermal_data.get('fever_face_max')  # NEW
        
        # Apply calibration
        if neck_temp is not None: neck_temp += CALIBRATION_OFFSET
        if canthus_temp is not None: canthus_temp += CALIBRATION_OFFSET
        if face_max is not None: face_max += CALIBRATION_OFFSET
        
        # FIXED: Prioritize face_max > canthus > neck
        # face_max captures the absolute hottest point (likely inner canthus) even if ROIs are slightly off
        if face_max is not None:
            final_skin_temp = face_max
        elif canthus_temp is not None:
             final_skin_temp = canthus_temp
        else:
             final_skin_temp = neck_temp
        
        # Validation: Warn if neck temp is suspiciously low compared to canthus
        if neck_temp is not None and canthus_temp is not None:
            temp_diff = abs(canthus_temp - neck_temp)
            if temp_diff > 5.0:  # More than 5°C difference is abnormal
                logger.warning(
                    f"Skin: Large temperature difference detected - "
                    f"Canthus: {canthus_temp:.1f}°C, Neck: {neck_temp:.1f}°C (Δ={temp_diff:.1f}°C). "
                    f"Using canthus (more reliable for core temp). "
                    f"Check thermal camera positioning or neck ROI calibration."
                )
        
        if final_skin_temp is not None:
            if session_baseline:
                thermal_deviation = final_skin_temp - session_baseline.baseline_facial_temp
                
                # Adjust normal range based on ambient temperature
                if session_baseline.ambient_background_temp < 22.0:
                    adjusted_range = (-1.5, 1.0)
                elif session_baseline.ambient_background_temp > 28.0:
                    adjusted_range = (-1.0, 2.0)
                else:
                    adjusted_range = (-1.0, 1.0)
                
                self._add_biomarker_safe(
                    biomarker_set,
                    name="skin_temperature_deviation",
                    value=float(thermal_deviation),
                    unit="delta_celsius",
                    confidence=0.92 if canthus_temp is not None else 0.70,
                    normal_range=adjusted_range,
                    description=f"Skin temp deviation from session baseline ({session_baseline.baseline_facial_temp:.1f}°C)"
                )
            else:
                 self._add_biomarker_safe(
                    biomarker_set,
                    name="skin_temperature",
                    value=float(final_skin_temp),
                    unit="celsius",
                    confidence=0.92 if canthus_temp is not None else 0.70,
                    normal_range=(35.5, 37.5),
                    description="Body temperature (Inner canthus - medical standard)"
                )
        
        # Max Temperature from inner canthus or face_max (most accurate core temp proxy)
        # Use face_max if available (firmware 2.0), otherwise fallback to canthus
        max_temp_source = face_max if face_max is not None else thermal_data.get('fever_canthus_temp')
        
        if max_temp_source is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="skin_temperature_max",
                value=float(max_temp_source),
                unit="celsius",
                confidence=0.95 if face_max is not None else 0.90,
                normal_range=(36.0, 38.0),
                description="Peak facial temperature (Fever indicator)"
            )
        
        # Inflammation Index from hot pixel percentage
        if thermal_data.get('inflammation_pct') is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="inflammation_index",
                value=float(thermal_data['inflammation_pct']),
                unit="percent",
                confidence=0.75,
                normal_range=(0.0, 5.0),
                description="Localized inflammation (hot pixel %, MLX90640)"
            )
        
        # Face mean temperature for context
        if thermal_data.get('face_mean_temp') is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="face_mean_temperature",
                value=float(thermal_data['face_mean_temp']),
                unit="celsius",
                confidence=0.85,
                normal_range=(34.0, 37.0),
                description="Average face temperature (MLX90640)"
            )
        
        # Thermal Stability from canthus temperature range (temporal consistency)
        # Lower range = more stable reading = higher measurement quality
        if thermal_data.get('thermal_stability') is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="thermal_stability",
                value=float(thermal_data['thermal_stability']),
                unit="delta_celsius",
                confidence=0.80,
                normal_range=(0.0, 0.8),
                description="Thermal measurement stability (canthus range, MLX90640)"
            )
            
        # NEW Phase 1.3: Thermal Asymmetry with Pose Gating
        if thermal_data.get('thermal_asymmetry') is not None:
            asymmetry_val = float(thermal_data['thermal_asymmetry'])
            confidence = 0.85
            description = "Thermal asymmetry (Left vs Right)"
            
            # GATING: If pose is provided, check frontal alignment
            if pose_landmarks:
                yaw, pitch = self._estimate_head_pose(pose_landmarks)
                
                # Loose thresholds for consumer hardware/movement
                if abs(yaw) > 12.0 or abs(pitch) > 12.0:
                    logger.warning(
                        f"Skin: High head rotation detected (Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°). "
                        f"Reducing asymmetry confidence."
                    )
                    confidence *= 0.3 # Heavy penalty for non-frontal pose
                    description += f" (Warning: High head rotation {yaw:.1f}°/{pitch:.1f}°)"
                else:
                    confidence = 0.95 # Bonus for confirmed frontal pose
            
            self._add_biomarker_safe(
                biomarker_set,
                name="thermal_asymmetry",
                value=asymmetry_val,
                unit="delta_celsius",
                confidence=confidence,
                normal_range=(0.0, 0.5),
                description=description
            )

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
        """Safe biomarker addition with fallback for invalid values."""
        try:
            # Handle NaN/Inf with physiological fallback instead of silent drop
            if np.isnan(value) or np.isinf(value):
                fallback = self._get_fallback_value(name)
                logger.warning(
                    f"Skin: Invalid {name} value: {value}. "
                    f"Using fallback: {fallback} (confidence reduced by 50%)"
                )
                value = fallback
                confidence *= 0.5  # Reduce confidence for fallback data

            self._add_biomarker(
                biomarker_set, name, float(value), unit,
                confidence=confidence, normal_range=normal_range,
                description=description
            )
        except Exception as e:
            logger.error(f"Skin: Failed to add biomarker {name}: {e}")
            # Last resort: try fallback value
            try:
                self._add_biomarker(
                    biomarker_set, name, self._get_fallback_value(name), unit,
                    confidence=0.1, normal_range=normal_range,
                    description=f"{description} (fallback due to error)"
                )
            except:
                pass  # Give up gracefully
            
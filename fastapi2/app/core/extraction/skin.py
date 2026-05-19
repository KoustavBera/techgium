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

# Reference physiological values — for documentation only.
# These are NOT used as substitutes for real measurements.
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
    baseline_canthus_temp: float = 35.5 # NEW: inner eye specific baseline
    baseline_redness: float = 0.0  # CIELab *a
    baseline_yellowness: float = 0.0  # CIELab *b
    ambient_background_temp: float = 25.0  # Room temp from thermal camera
    ambient_light_level: float = 120.0  # Average RGB intensity
    dht11_ambient_temp: Optional[float] = None   # DHT11 co-located sensor (v2 firmware)
    dht11_humidity: Optional[float] = None        # DHT11 relative humidity (v2 firmware)

    def to_dict(self) -> Dict[str, float]:
        d = {
            "baseline_facial_temp": self.baseline_facial_temp,
            "baseline_canthus_temp": self.baseline_canthus_temp,
            "baseline_redness": self.baseline_redness,
            "baseline_yellowness": self.baseline_yellowness,
            "ambient_background_temp": self.ambient_background_temp,
            "ambient_light_level": self.ambient_light_level,
        }
        if self.dht11_ambient_temp is not None:
            d["dht11_ambient_temp"] = self.dht11_ambient_temp
        if self.dht11_humidity is not None:
            d["dht11_humidity"] = self.dht11_humidity
        return d


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
            min_detection_confidence=0.3  # Lowered from 0.5 for better detection on tight crops
        )
    
    def _safe_normal_range(self, bm_range: Any) -> Optional[tuple]:
        """Convert normal_range to safe tuple format."""
        try:
            if isinstance(bm_range, (list, tuple)) and len(bm_range) == 2:
                return tuple(float(x) for x in bm_range)
        except (TypeError, ValueError):
            pass
        return None
    
    def _get_fallback_value(self, name: str) -> None:
        """Fallback values are no longer used as substitutes for real measurements."""
        return None
    
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
        
        # Prefer raw (uncropped) frames for FaceMesh — tight ROI crops confuse FaceMesh
        # because it can't see the full face context. raw_face_frames are full camera frames.
        raw_frames = data.get("raw_face_frames", [])
        roi_frames = data.get("frames", data.get("face_frames", []))
        # Use raw frames if available, otherwise fall back to ROI crops
        best_frames_for_facemesh = raw_frames if raw_frames else roi_frames
        
        # NEW FORMAT: Flattened thermal_data from bridge.py
        if "thermal_data" in data:
            # Try to get landmarks from the best available frame for pose gating
            pose_landmarks = None
            if best_frames_for_facemesh:
                frame_for_pose = best_frames_for_facemesh[0]
                if not isinstance(frame_for_pose, np.ndarray):
                    frame_for_pose = np.array(frame_for_pose)
                _, _, pose_landmarks = self._get_face_mask(frame_for_pose)
                if pose_landmarks:
                    logger.info("Skin: Got pose landmarks from raw frame for thermal gating.")
                
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
        # Use raw uncropped frames for FaceMesh — much better detection than tight crops
        if best_frames_for_facemesh:
            frame = best_frames_for_facemesh[0]
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            logger.info(f"Skin: Running visual analysis on {'raw' if raw_frames else 'ROI'} frame (shape: {frame.shape})")
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
        if texture_roughness is not None:
            self._add_biomarker(
                biomarker_set,
                name="texture_roughness",
                value=texture_roughness,
                unit="glcm_contrast",
                confidence=0.75,
                normal_range=(0.0, 5.0),  # Calibrated for webcam GLCM (32 levels, bilateral filtered)
                description="Skin surface texture (GLCM Contrast, multi-angle)"
            )
        else:
            logger.warning("Skin: Texture analysis skipped (empty crop)")
        
        # 3. Color Analysis (CIELab on Masked Face)
        color_metrics = self._analyze_skin_color_lab(frame, face_mask, session_baseline)

        if color_metrics is not None:
            self._add_biomarker(
                biomarker_set,
                name="skin_redness",
                value=color_metrics["redness"],
                unit="normalized_score",
                confidence=0.85,
                normal_range=(0.0, 0.5),
                description="Skin redness (normalized 0-1, higher = more red)"
            )

            self._add_biomarker(
                biomarker_set,
                name="skin_yellowness",
                value=color_metrics["yellowness"],
                unit="normalized_score",
                confidence=0.80,
                normal_range=(0.0, 0.5),
                description="Skin yellowness (normalized 0-1, higher = more yellow)"
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
        else:
            logger.warning("Skin: Color analysis skipped (empty frame or no skin pixels)")
        
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
            logger.warning("Skin: _analyze_texture_glcm received empty crop; skipping texture analysis")
            return None
        
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
        # Stable Linear Delta Ratio: (L-R)/(L+R)
        dist_left = abs(nose_tip.x - left_eye.x)
        dist_right = abs(nose_tip.x - right_eye.x)
        
        denominator = dist_left + dist_right
        if denominator == 0: denominator = 0.01
        
        # Linear shift: -1.0 (Full Left) to 1.0 (Full Right)
        yaw_shift = (dist_left - dist_right) / denominator
        # Map to degrees (~ ±90°)
        yaw = yaw_shift * 90.0
        
        # 2. Pitch Estimation (Vertical Rotation)
        dist_up = abs(nose_tip.y - forehead.y)
        dist_down = abs(nose_tip.y - chin.y)
        
        denominator_v = dist_up + dist_down
        if denominator_v == 0: denominator_v = 0.01
        
        pitch_shift = (dist_up - dist_down) / denominator_v
        pitch = pitch_shift * 90.0
        
        return float(np.clip(yaw, -90, 90)), float(np.clip(pitch, -90, 90))

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
            logger.warning("Skin: _analyze_skin_color_lab received empty frame; skipping color analysis")
            return None
            
        # Convert to Lab
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        
        # Auto White Balance (Grey World Assumption) to correct lighting
        avg_scene_color = np.mean(frame, axis=(0, 1))
        # Protect against div by zero
        avg_scene_color = np.maximum(avg_scene_color, 1.0) 
        
        # Calculate gain to make average scene gray (128, 128, 128)
        gain = 128.0 / avg_scene_color
        
        # Apply gain (simple Von Kries model)
        # Note: This is computationally expensive on full frame, so we apply correction 
        # to the EXTRACTED mean values instead of the whole image array
        
        # Extract channels
        l_channel, a_channel, b_channel = cv2.split(lab_frame)
        
        # Apply mask to get only skin pixels
        skin_pixels_a = a_channel[mask == 255]
        skin_pixels_b = b_channel[mask == 255]
        
        if len(skin_pixels_a) == 0:
            logger.warning("Skin: No skin pixels found in mask; skipping color analysis")
            return None

        # Calculate raw means
        raw_a_mean = np.mean(skin_pixels_a)
        raw_b_mean = np.mean(skin_pixels_b)
        
        # Apply pseudo-AWB correction to the MEANS based on scene lighting bias
        # If scene is too yellow (low b in Lab? No, Lab b is Blue-Yellow), we adjust.
        # Actually simplest AWB is on RGB, then convert. 
        # Since we already have Lab, let's normalize deviations using a context-aware approach.
        
        # Improved Approach: Differential analysis against background
        # If we assume background wall is neutral-ish, we can subtract its color bias.
        # For now, we will use a "Soft White Balance" that dampens extreme values driven by lighting
        
        # Dampening factor for lighting artifacts (heuristic)
        lighting_offset_a = (avg_scene_color[2] - 128) * 0.1 # Red channel bias influence
        lighting_offset_b = (avg_scene_color[0] - 128) * 0.1 # Blue channel bias influence
        
        if session_baseline:
            raw_redness = float(raw_a_mean - session_baseline.baseline_redness)
            raw_yellowness = float(raw_b_mean - session_baseline.baseline_yellowness)
        else:
            # Deviation from neutral (128 in OpenCV uint8 Lab)
            raw_redness = float(raw_a_mean - 128.0 - lighting_offset_a)
            raw_yellowness = float(raw_b_mean - 128.0 - lighting_offset_b)
        
        # Normalize to 0-1 range:
        # CIELab a*/b* deviation range is roughly -50 to +50 for skin.
        # We map: 0 deviation -> 0.0, +50 deviation -> 1.0
        # Negative deviations (greenish/bluish) are clamped to 0.
        MAX_LAB_DEVIATION = 50.0
        redness = float(np.clip(raw_redness / MAX_LAB_DEVIATION, 0.0, 1.0))
        yellowness = float(np.clip(raw_yellowness / MAX_LAB_DEVIATION, 0.0, 1.0))
        
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
            "redness": redness,
            "yellowness": yellowness,
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
        facial_temps_raw = []
        canthus_temps_raw = []
        background_temps = []
        dht11_temps = []
        dht11_humidities = []

        for data in thermal_frames:
            face_max = data.get('fever_face_max')
            if face_max is not None:
                facial_temps_raw.append(float(face_max))

            canthus = data.get('fever_canthus_temp')
            if canthus is not None:
                canthus_temps_raw.append(float(canthus))

            bg_temp = data.get('background_temp')
            if bg_temp is not None:
                background_temps.append(float(bg_temp))

            # v2 firmware: DHT11 readings passed through thermal_frames
            d11 = data.get('dht11_ambient_temp')
            if d11 is not None and 15.0 <= d11 <= 45.0:
                dht11_temps.append(float(d11))
            d11h = data.get('dht11_humidity')
            if d11h is not None and 0.0 <= d11h <= 100.0:
                dht11_humidities.append(float(d11h))

        # Dynamic AC-room correction: each 1°C below 25°C adds ~0.15°C more offset.
        # Prefer DHT11 (v2 firmware) over background-pixel proxy (v1 firmware).
        if dht11_temps:
            ambient_temp = float(np.median(dht11_temps))
            cal_source = "DHT11"
        elif background_temps:
            ambient_temp = float(np.median(background_temps))
            cal_source = "thermal_background_proxy"
        else:
            ambient_temp = 25.0
            cal_source = "default_25C"
        CALIBRATION_OFFSET = 0.8 + max(0.0, (25.0 - ambient_temp) * 0.15)
        logger.info(
            f"Skin baseline: dynamic offset={CALIBRATION_OFFSET:.2f}°C "
            f"(room={ambient_temp:.1f}°C, source={cal_source})"
        )

        facial_temps = [t + CALIBRATION_OFFSET for t in facial_temps_raw]
        canthus_temps = [t + CALIBRATION_OFFSET for t in canthus_temps_raw]

        baseline_temp = float(np.median(facial_temps)) if facial_temps else 36.0
        baseline_canthus = float(np.median(canthus_temps)) if canthus_temps else baseline_temp - 0.5
        
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
            baseline_canthus_temp=float(baseline_canthus),
            baseline_redness=float(baseline_redness),
            baseline_yellowness=float(baseline_yellowness),
            ambient_background_temp=float(ambient_temp),
            ambient_light_level=float(baseline_light),
            dht11_ambient_temp=float(np.median(dht11_temps)) if dht11_temps else None,
            dht11_humidity=float(np.median(dht11_humidities)) if dht11_humidities else None,
        )
        
        logger.info(f"Skin: Session baseline established: {baseline.to_dict()}")
        return baseline

    def _apply_thermal_homography(
        self,
        rgb_canthus_pixels: Dict,
        thermal_grid: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        Thermal Homography-guided canthus temperature extraction.

        Maps the RGB FaceMesh inner-canthus coordinates onto the MLX90640 32x24
        thermal grid using a scale-only homography matrix (no calibration board needed).

        Math:
          H = diag(thermal_w / rgb_w,  thermal_h / rgb_h,  1.0)  (3x3 affine scale)
          [tx, ty, 1]^T = H @ [rx, ry, 1]^T  →  clamp to [0..31, 0..23]
          Extract max temperature in a 3x3 neighbourhood around (tx, ty).

        Falls back gracefully (returns None) if:
          - thermal_grid is None (hardware not available)
          - projected point falls outside grid after clamping

        The scale-only approximation is valid when the cameras are mounted
        co-axially and the user is centred in frame. For offset cameras a
        full calibration board (findHomography) can compute a more precise H.

        Args:
            rgb_canthus_pixels: dict from manager._compute_canthus_pixels()
              {'right': (rx, ry), 'left': (lx, ly),
               'rgb_frame_shape': (h, w), 'thermal_frame_shape': (24, 32)}
            thermal_grid: 2D np.ndarray of shape (24, 32) with raw temperatures.
                          If None, the function returns None immediately.

        Returns:
            float: Maximum temperature (°C) in a 3x3 neighbourhood around the
                   projected inner-canthus coordinates, or None on failure.
        """
        if thermal_grid is None or rgb_canthus_pixels is None:
            return None

        X_OFFSET_PX = -1 # Account for -1 pixel shift in thermal projection

        try:
            rgb_h, rgb_w = rgb_canthus_pixels.get('rgb_frame_shape', (720, 1280))
            th_h, th_w   = rgb_canthus_pixels.get('thermal_frame_shape', (24, 32))

            # Build scale-only homography (no distortion, just resolution scaling)
            sx = th_w / max(rgb_w, 1)
            sy = th_h / max(rgb_h, 1)

            # Sample both canthi and take higher temperature (inner canthus is hottest)
            temps = []
            for key in ['right', 'left']:
                px, py = rgb_canthus_pixels.get(key, (None, None))
                if px is None:
                    continue

                # Project RGB pixel -> thermal pixel
                tx = int(round(px * sx)) + X_OFFSET_PX # Apply X offset
                ty = int(round(py * sy))

                # Clamp to grid bounds
                tx = max(0, min(th_w - 1, tx))
                ty = max(0, min(th_h - 1, ty))

                # Extract 3x3 neighbourhood (handles boundaries safely with np slicing)
                r0 = max(0, ty - 1);  r1 = min(th_h, ty + 2)
                c0 = max(0, tx - 1);  c1 = min(th_w, tx + 2)
                patch = thermal_grid[r0:r1, c0:c1]

                if patch.size > 0:
                    temps.append(float(np.max(patch)))

                logger.info(
                    f"Thermal homography [{key}]: RGB({px:.0f},{py:.0f}) "
                    f"→ Thermal({tx},{ty}) → patch_max={temps[-1]:.1f}°C"
                    if temps else
                    f"Thermal homography [{key}]: RGB({px:.0f},{py:.0f}) "
                    f"→ Thermal({tx},{ty}) — empty patch"
                )

            if not temps:
                return None

            guided_temp = float(max(temps))   # Take hottest canthus as core proxy
            logger.info(f"Thermal homography ✅  guided_canthus_temp = {guided_temp:.2f}°C")
            return guided_temp

        except Exception as e:
            logger.warning(f"Thermal homography failed: {e} — falling back to face_max")
            return None

    def _extract_from_thermal_v2(
        self, 
        thermal_data: Dict[str, Any], 
        biomarker_set: BiomarkerSet,
        session_baseline: Optional[SessionBaseline] = None,
        pose_landmarks: Optional[List[Any]] = None
    ) -> None:
        """Extract skin metrics from flattened thermal data (NEW FORMAT v2).
        
        Calibration accuracy hierarchy:
          Room temp source: DHT11 (v2 firmware) > thermal_bg_proxy (v1) > default 25°C
          Canthus temp source: canthus_top5 > rolling_canthus_mean > canthus_mean > neck_mean
          Confidence modifiers: glasses_detected penalty, esp32_confidence scaling
        """
        # ══ 1. CALIBRATION OFFSET ════════════════════════════════════════════
        # DHT11 (v2 firmware): co-located sensor, no emissivity/angle error.
        # room_temp_calibration (v1 firmware): background-pixel proxy, more noise.
        # Formula: 0.8°C base (MLX90640 characteristic under-read) +
        #          0.15°C per 1°C the room is below 25°C (AC convective cooling).
        dht11_temp = thermal_data.get('dht11_ambient_temp')
        room_temp_proxy = thermal_data.get('room_temp_calibration', 25.0)

        if dht11_temp is not None and 15.0 <= dht11_temp <= 45.0:
            ambient_room_temp = dht11_temp
            calibration_source = "DHT11"
        else:
            ambient_room_temp = room_temp_proxy
            calibration_source = "thermal_background_proxy"

        CALIBRATION_OFFSET = 0.8 + max(0.0, (25.0 - ambient_room_temp) * 0.15)
        logger.info(
            f"Thermal v2: offset={CALIBRATION_OFFSET:.2f}°C "
            f"(room={ambient_room_temp:.1f}°C, source={calibration_source})"
        )

        # ══ 2. GLASSES CONFIDENCE MODIFIER ══════════════════════════════════
        # When glasses are detected, the canthus ROI may be partially occluded
        # by the frame edge — particularly the inner canthus (tear-duct hotspot).
        glasses_detected = thermal_data.get('glasses_detected', False)
        esp32_confidence = float(thermal_data.get('esp32_confidence', 1.0))
        esp32_confidence = max(0.3, min(1.0, esp32_confidence))  # clamp to [0.3, 1.0]

        # Base confidence tiers for temperature readings
        base_thermal_conf = 0.92
        if glasses_detected:
            base_thermal_conf = min(base_thermal_conf, 0.78)
            logger.info("Skin: Glasses detected — canthus confidence reduced (occlusion risk)")

        # Scale by ESP32's own quality estimate
        thermal_conf = round(base_thermal_conf * esp32_confidence, 3)

        # ══ 3. MOTION GATING ════════════════════════════════════════════════
        # Rapid head movement causes "inflammation" pixel smearing artifacts.
        if pose_landmarks:
            yaw, pitch = self._estimate_head_pose(pose_landmarks)
            is_stable_pose = abs(yaw) < 10.0 and abs(pitch) < 10.0

            if not is_stable_pose:
                logger.warning(
                    f"Thermal: Unstable pose (Yaw={yaw:.1f}, Pitch={pitch:.1f}). "
                    f"Skipping inflammation/stability analysis to prevent artifacts."
                )
                # Still extract basic temp, but drop noisy derived metrics
                for key in ('inflammation_pct', 'thermal_stability', 'thermal_asymmetry'):
                    thermal_data.pop(key, None)

        # ══ 4. CANTHUS TEMPERATURE SELECTION ════════════════════════════════
        # Accuracy hierarchy (highest to lowest):
        #   canthus_top5       — 5 hottest pixels mean; best single-frame estimate
        #   rolling_canthus_mean — firmware multi-frame rolling avg (already smoothed)
        #   canthus_mean       — mean of all ROI pixels (includes cold border pixels)
        #   neck_mean          — fallback; anatomically different site
        neck_temp    = thermal_data.get('fever_neck_temp')
        canthus_mean = thermal_data.get('fever_canthus_temp')
        face_max     = thermal_data.get('fever_face_max')
        canthus_top5 = thermal_data.get('canthus_top5')
        rolling_mean = thermal_data.get('rolling_canthus_mean')

        # Apply calibration offset to each raw reading
        if neck_temp    is not None: neck_temp    += CALIBRATION_OFFSET
        if canthus_mean is not None: canthus_mean += CALIBRATION_OFFSET
        if face_max     is not None: face_max     += CALIBRATION_OFFSET
        if canthus_top5 is not None: canthus_top5 += CALIBRATION_OFFSET
        if rolling_mean is not None: rolling_mean += CALIBRATION_OFFSET

        # Prefer canthus_top5 if the top5–mean gap is physiologically plausible (0–2°C).
        # A gap > 2°C signals a hot-spot artifact (e.g. reflection), so fall back.
        best_canthus = None
        canthus_source = "none"
        if (canthus_top5 is not None and canthus_mean is not None
                and 0.0 <= (canthus_top5 - canthus_mean) <= 2.0):
            best_canthus = canthus_top5
            canthus_source = "canthus_top5"
        elif rolling_mean is not None:
            best_canthus = rolling_mean
            canthus_source = "rolling_canthus_mean"
        elif canthus_mean is not None:
            best_canthus = canthus_mean
            canthus_source = "canthus_mean"

        logger.info(
            f"Thermal v2 canthus selection: {canthus_source}={best_canthus:.2f}°C "
            f"(top5={canthus_top5}, mean={canthus_mean}, rolling={rolling_mean})"
            if best_canthus is not None else
            f"Thermal v2: no valid canthus reading"
        )

        # Homography-guided canthus logging (raw grid unavailable in firmware v2)
        rgb_canthus_pixels = thermal_data.get('rgb_canthus_pixels')
        if rgb_canthus_pixels is not None:
            logger.info(
                f"Thermal homography guidance received: "
                f"right={rgb_canthus_pixels.get('right')}  "
                f"left={rgb_canthus_pixels.get('left')}  "
                f"rgb_shape={rgb_canthus_pixels.get('rgb_frame_shape')}  "
                f"thermal_shape={rgb_canthus_pixels.get('thermal_frame_shape')}"
            )

        # Final temperature: best_canthus > face_max > neck
        if best_canthus is not None:
            final_skin_temp = best_canthus
        elif face_max is not None:
            final_skin_temp = face_max
        else:
            final_skin_temp = neck_temp

        # Sanity check: warn on large canthus–neck divergence
        if neck_temp is not None and best_canthus is not None:
            temp_diff = abs(best_canthus - neck_temp)
            if temp_diff > 5.0:
                logger.warning(
                    f"Skin: Large canthus–neck divergence — "
                    f"canthus ({canthus_source})={best_canthus:.1f}°C, "
                    f"neck={neck_temp:.1f}°C (Δ={temp_diff:.1f}°C). "
                    f"Check camera positioning or neck ROI calibration."
                )

        # ══ 5. CORE TEMPERATURE BIOMARKER ═══════════════════════════════════
        glasses_note = " (⚠️ glasses detected — canthus occlusion risk)" if glasses_detected else ""

        if final_skin_temp is not None:
            if session_baseline:
                # Metric-aware deviation: compare peak-to-peak or canthus-to-canthus
                if face_max is not None and best_canthus is None:
                    thermal_deviation = face_max - session_baseline.baseline_facial_temp
                    desc_suffix = f"Peak baseline ({session_baseline.baseline_facial_temp:.1f}°C)"
                else:
                    thermal_deviation = final_skin_temp - session_baseline.baseline_canthus_temp
                    desc_suffix = f"Canthus baseline ({session_baseline.baseline_canthus_temp:.1f}°C)"

                # Use DHT11 ambient_temp from session_baseline when available for range adjust
                ref_ambient = (
                    session_baseline.dht11_ambient_temp
                    if session_baseline.dht11_ambient_temp is not None
                    else session_baseline.ambient_background_temp
                )
                if ref_ambient < 22.0:
                    adjusted_range = (-1.5, 1.0)
                elif ref_ambient > 28.0:
                    adjusted_range = (-1.0, 2.0)
                else:
                    adjusted_range = (-1.0, 1.0)

                self._add_biomarker_safe(
                    biomarker_set,
                    name="skin_temperature_deviation",
                    value=float(thermal_deviation),
                    unit="delta_celsius",
                    confidence=thermal_conf,
                    normal_range=adjusted_range,
                    description=f"Skin temp deviation from {desc_suffix} [{canthus_source}]{glasses_note}"
                )
            else:
                self._add_biomarker_safe(
                    biomarker_set,
                    name="skin_temperature",
                    value=float(final_skin_temp),
                    unit="celsius",
                    confidence=thermal_conf,
                    normal_range=(35.5, 37.5),
                    description=f"Body temperature ({canthus_source} - medical standard){glasses_note}"
                )

        # Peak temperature (fever screening)
        peak_source = best_canthus if best_canthus is not None else face_max
        if peak_source is None:
            # Last resort: un-calibrated canthus from raw field
            _raw_canthus = thermal_data.get('fever_canthus_temp')
            peak_source = _raw_canthus
        if peak_source is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="skin_temperature_max",
                value=float(peak_source),
                unit="celsius",
                confidence=min(0.95, thermal_conf + 0.03),
                normal_range=(36.0, 38.0),
                description=f"Peak facial temperature — fever indicator [{canthus_source}]{glasses_note}"
            )

        # ══ 6. INFLAMMATION INDEX ════════════════════════════════════════════
        if thermal_data.get('inflammation_pct') is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="inflammation_index",
                value=float(thermal_data['inflammation_pct']),
                unit="percent",
                confidence=0.75 * esp32_confidence,
                normal_range=(0.0, 5.0),
                description="Localized inflammation (hot pixel %, MLX90640)"
            )

        # ══ 7. FACE MEAN TEMPERATURE ═════════════════════════════════════════
        if thermal_data.get('face_mean_temp') is not None:
            face_mean_calibrated = float(thermal_data['face_mean_temp']) + CALIBRATION_OFFSET
            self._add_biomarker_safe(
                biomarker_set,
                name="face_mean_temperature",
                value=face_mean_calibrated,
                unit="celsius",
                confidence=0.85 * esp32_confidence,
                normal_range=(33.5, 37.0),
                description="Average face temperature (MLX90640, calibrated)"
            )

        # ══ 8. THERMAL STABILITY ═════════════════════════════════════════════
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

        # ══ 9. THERMAL ASYMMETRY WITH POSE GATING ════════════════════════════
        if thermal_data.get('thermal_asymmetry') is not None:
            asymmetry_val = float(thermal_data['thermal_asymmetry'])
            asym_conf = 0.85
            description = "Thermal asymmetry (Left vs Right)"

            if pose_landmarks:
                yaw, pitch = self._estimate_head_pose(pose_landmarks)
                if abs(yaw) > 12.0 or abs(pitch) > 12.0:
                    logger.warning(
                        f"Skin: High head rotation (Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°) "
                        f"— reducing asymmetry confidence."
                    )
                    asym_conf *= 0.3
                    description += f" (⚠️ head rotation {yaw:.1f}°/{pitch:.1f}°)"
                else:
                    asym_conf = 0.95  # frontal pose confirmed

            self._add_biomarker_safe(
                biomarker_set,
                name="thermal_asymmetry",
                value=asymmetry_val,
                unit="delta_celsius",
                confidence=asym_conf * esp32_confidence,
                normal_range=(0.0, 0.5),
                description=description
            )

        # ══ 10. NEW v2 FIRMWARE BIOMARKERS ══════════════════════════════════
        # Forehead temperature (v2: dedicated forehead_mean ROI)
        forehead_raw = thermal_data.get('forehead_temp')
        if forehead_raw is not None:
            forehead_cal = float(forehead_raw) + CALIBRATION_OFFSET
            self._add_biomarker_safe(
                biomarker_set,
                name="forehead_temperature",
                value=forehead_cal,
                unit="celsius",
                confidence=0.82 * esp32_confidence,
                normal_range=(33.5, 36.5),
                description=f"Forehead temperature (MLX90640 forehead ROI, calibrated, {calibration_source})"
            )

        # Nasal temperature (v2: dedicated nose_mean ROI)
        nose_raw = thermal_data.get('nose_temp')
        if nose_raw is not None:
            nose_cal = float(nose_raw) + CALIBRATION_OFFSET
            self._add_biomarker_safe(
                biomarker_set,
                name="nasal_temperature",
                value=nose_cal,
                unit="celsius",
                confidence=0.78 * esp32_confidence,
                normal_range=(29.0, 34.0),
                description=f"Nasal temperature (MLX90640 nose ROI, calibrated, {calibration_source})"
            )

        # DHT11 ambient environment (v2 only — informational, not a clinical biomarker)
        if dht11_temp is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="ambient_temperature",
                value=float(dht11_temp),
                unit="celsius",
                confidence=0.98,
                normal_range=(18.0, 32.0),
                description="Room ambient temperature (DHT11 co-located sensor)"
            )
        dht11_humidity = thermal_data.get('dht11_humidity')
        if dht11_humidity is not None:
            self._add_biomarker_safe(
                biomarker_set,
                name="ambient_humidity",
                value=float(dht11_humidity),
                unit="percent_rh",
                confidence=0.95,
                normal_range=(30.0, 70.0),
                description="Room relative humidity (DHT11 co-located sensor)"
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
        """Safe biomarker addition. Drops the biomarker if the value is invalid (NaN/Inf)."""
        try:
            if np.isnan(value) or np.isinf(value):
                logger.warning(
                    f"Skin: Invalid {name} value ({value}) — dropping biomarker. "
                    f"A fake fallback value would misrepresent the measurement."
                )
                return  # Do NOT substitute a hardcoded value

            self._add_biomarker(
                biomarker_set, name, float(value), unit,
                confidence=confidence, normal_range=normal_range,
                description=description
            )
        except Exception as e:
            logger.error(f"Skin: Failed to add biomarker {name}: {e} — skipping")
            # Do NOT inject a fallback value; a fake number is worse than no number.
            
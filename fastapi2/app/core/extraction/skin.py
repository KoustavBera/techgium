"""
Skin Biomarker Extractor

Extracts skin health indicators from camera data:
- Surface texture roughness
- Lesion morphology detection
- Color maps / pigmentation analysis
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

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


class SkinExtractor(BaseExtractor):
    """
    Extracts skin biomarkers from visual data.
    
    Analyzes camera frames for dermatological indicators.
    """
    
    system = PhysiologicalSystem.SKIN
    
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
        - frames: List of video frames (HxWx3 arrays)
        - esp32_data: Dict containing thermal metrics from MLX90640
        - systems: List of pre-processed systems from bridge
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        # Priority 1: Hardware Thermal Data (ESP32/MLX90640)
        has_thermal = False
        if "esp32_data" in data:
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
        frames = data.get("frames", [])
        if len(frames) > 0:
            frame = np.array(frames[0]) if not isinstance(frames[0], np.ndarray) else frames[0]
            self._extract_from_frame(frame, biomarker_set)
        elif not has_thermal:
            self._generate_simulated_biomarkers(biomarker_set)
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_frame(
        self,
        frame: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract skin metrics from a video frame."""
        
        if frame.ndim < 2:
            self._generate_simulated_biomarkers(biomarker_set)
            return
        
        # Extract face ROI for more accurate skin analysis
        face_roi = self._extract_face_roi(frame)
        
        # Texture analysis using local variance
        texture_roughness = self._analyze_texture(face_roi)
        self._add_biomarker(
            biomarker_set,
            name="texture_roughness",
            value=texture_roughness,
            unit="variance_score",
            confidence=0.35,  # Lowered: experimental without proper validation
            normal_range=(5, 25),
            description="Skin surface texture roughness (experimental)"
        )
        
        # Color analysis on face ROI only
        color_metrics = self._analyze_skin_color(face_roi)
        
        self._add_biomarker(
            biomarker_set,
            name="skin_redness",
            value=color_metrics["redness"],
            unit="normalized_intensity",
            confidence=0.30,  # Lowered: no lighting normalization
            normal_range=(0.3, 0.6),
            description="Skin redness/erythema level (experimental)"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="skin_yellowness",
            value=color_metrics["yellowness"],
            unit="normalized_intensity",
            confidence=0.30,  # Lowered: no clinical validation
            normal_range=(0.2, 0.5),
            description="Skin yellowness/jaundice proxy (experimental)"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="color_uniformity",
            value=color_metrics["uniformity"],
            unit="score_0_1",
            confidence=0.30,  # Lowered: basic algorithm
            normal_range=(0.7, 1.0),
            description="Skin color uniformity (experimental)"
        )
        
        # DISABLED: Lesion detection (unvalidated algorithm - too many false positives)
        # Will be re-enabled with proper morphological analysis + clinical validation
        self._add_biomarker(
            biomarker_set,
            name="lesion_count",
            value=0.0,  # Conservative: disabled until proper CV model available
            unit="count",
            confidence=0.10,  # Very low: feature disabled
            normal_range=(0, 5),
            description="Skin abnormalities (feature disabled - pending validation)"
        )
    
    def _extract_face_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract approximate face region to reduce background pollution.
        
        Uses simple center-crop heuristic. For production, use MediaPipe Face Mesh.
        """
        if frame.size == 0 or frame.ndim < 2:
            return frame
        
        h, w = frame.shape[:2]
        
        # Conservative face region: center 50% of frame
        # Assumes subject is centered (walk-through chamber setup)
        y_start = h // 4
        y_end = 3 * h // 4
        x_start = w // 4
        x_end = 3 * w // 4
        
        return frame[y_start:y_end, x_start:x_end]
    
    def _analyze_texture(self, frame: np.ndarray) -> float:
        """Analyze texture using local variance."""
        if frame.size == 0 or frame.ndim < 2:
            return self._get_fallback_value("texture_roughness")
        
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame
        
        # Calculate local variance using sliding window
        window_size = 5
        h, w = gray.shape
        
        if h < window_size * 2 or w < window_size * 2:
            return np.random.uniform(10, 20)
        
        # Subsample for efficiency
        step = max(1, min(h, w) // 50)
        variances = []
        
        for y in range(window_size, h - window_size, step):
            for x in range(window_size, w - window_size, step):
                patch = gray[y-window_size:y+window_size, x-window_size:x+window_size]
                variances.append(np.var(patch))
        
        if variances:
            return float(np.mean(variances))
        return self._get_fallback_value("texture_roughness")
    
    def _analyze_skin_color(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze skin color characteristics (on face ROI)."""
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] < 3:
            return {
                "redness": self._get_fallback_value("skin_redness"),
                "yellowness": self._get_fallback_value("skin_yellowness"),
                "uniformity": self._get_fallback_value("color_uniformity")
            }
        
        # Normalize to 0-1 range
        frame_norm = frame.astype(np.float32) / 255.0
        
        # Extract color channels (assuming BGR)
        blue = frame_norm[:, :, 0]
        green = frame_norm[:, :, 1]
        red = frame_norm[:, :, 2]
        
        # Redness: ratio of red to other channels
        redness = np.mean(red) / (np.mean(green) + np.mean(blue) + 1e-6)
        redness = float(np.clip(redness / 2, 0, 1))
        
        # Yellowness: red + green relative to blue
        yellowness = (np.mean(red) + np.mean(green)) / (2 * np.mean(blue) + 1e-6)
        yellowness = float(np.clip(yellowness / 3, 0, 1))
        
        # Uniformity: inverse of color variance
        color_std = np.mean([np.std(red), np.std(green), np.std(blue)])
        uniformity = float(1 - np.clip(color_std * 2, 0, 0.5))
        
        return {
            "redness": redness,
            "yellowness": yellowness,
            "uniformity": uniformity
        }
    
    def _detect_lesions(self, frame: np.ndarray) -> int:
        """DISABLED: Lesion detection pending proper validation.
        
        Previous algorithm had critical flaws:
        - 2σ outliers detect image noise, not medical lesions
        - No morphological analysis (single pixels ≠ lesions)
        - Background pollution (clothes, hair counted as lesions)
        - Zero clinical validation
        
        TODO for production:
        - Implement connected component analysis
        - Add morphological operations (erosion/dilation)
        - Use GLCM texture features
        - Validate against dermatology datasets
        - Consider pre-trained skin lesion detection model
        """
        # Conservative approach: return 0 until proper algorithm validated
        return 0
    
    def _generate_simulated_biomarkers(self, biomarker_set: BiomarkerSet) -> None:
        """Generate simulated skin biomarkers."""
        self._add_biomarker(biomarker_set, "texture_roughness",
                           np.random.uniform(10, 20), "variance_score",
                           0.5, (5, 25), "Simulated texture")
        self._add_biomarker(biomarker_set, "skin_redness",
                           np.random.uniform(0.4, 0.5), "normalized_intensity",
                           0.5, (0.3, 0.6), "Simulated redness")
        self._add_biomarker(biomarker_set, "skin_yellowness",
                           np.random.uniform(0.3, 0.4), "normalized_intensity",
                           0.5, (0.2, 0.5), "Simulated yellowness")
        self._add_biomarker(biomarker_set, "color_uniformity",
                           np.random.uniform(0.8, 0.95), "score_0_1",
                           0.5, (0.7, 1.0), "Simulated uniformity")
        self._add_biomarker(biomarker_set, "lesion_count",
                           float(np.random.randint(0, 3)), "count",
                           0.5, (0, 5), "Simulated lesion count")

    def _extract_from_thermal(self, thermal_data: Dict[str, Any], biomarker_set: BiomarkerSet) -> None:
        """Extract skin metrics from thermal sensor data."""
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
            
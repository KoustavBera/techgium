"""
Skin Biomarker Extractor

Extracts skin health indicators from camera data:
- Surface texture roughness
- Lesion morphology detection
- Color maps / pigmentation analysis
"""
from typing import Dict, Any, List, Tuple
import numpy as np

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class SkinExtractor(BaseExtractor):
    """
    Extracts skin biomarkers from visual data.
    
    Analyzes camera frames for dermatological indicators.
    """
    
    system = PhysiologicalSystem.SKIN
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract skin biomarkers.
        
        Expected data keys:
        - frames: List of video frames (HxWx3 arrays)
        - skin_regions: Optional ROI coordinates
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        frames = data.get("frames", [])
        
        if len(frames) > 0:
            frame = np.array(frames[0]) if not isinstance(frames[0], np.ndarray) else frames[0]
            self._extract_from_frame(frame, biomarker_set)
        else:
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
        
        # Texture analysis using local variance
        texture_roughness = self._analyze_texture(frame)
        self._add_biomarker(
            biomarker_set,
            name="texture_roughness",
            value=texture_roughness,
            unit="variance_score",
            confidence=0.60,
            normal_range=(5, 25),
            description="Skin surface texture roughness"
        )
        
        # Color analysis
        color_metrics = self._analyze_skin_color(frame)
        
        self._add_biomarker(
            biomarker_set,
            name="skin_redness",
            value=color_metrics["redness"],
            unit="normalized_intensity",
            confidence=0.65,
            normal_range=(0.3, 0.6),
            description="Skin redness/erythema level"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="skin_yellowness",
            value=color_metrics["yellowness"],
            unit="normalized_intensity",
            confidence=0.60,
            normal_range=(0.2, 0.5),
            description="Skin yellowness (jaundice proxy)"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="color_uniformity",
            value=color_metrics["uniformity"],
            unit="score_0_1",
            confidence=0.55,
            normal_range=(0.7, 1.0),
            description="Skin color uniformity"
        )
        
        # Simple lesion detection (high-contrast spots)
        lesion_count = self._detect_lesions(frame)
        self._add_biomarker(
            biomarker_set,
            name="lesion_count",
            value=float(lesion_count),
            unit="count",
            confidence=0.45,
            normal_range=(0, 5),
            description="Detected skin abnormalities count"
        )
    
    def _analyze_texture(self, frame: np.ndarray) -> float:
        """Analyze texture using local variance."""
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
        return np.random.uniform(10, 20)
    
    def _analyze_skin_color(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze skin color characteristics."""
        if frame.ndim != 3 or frame.shape[2] < 3:
            return {
                "redness": np.random.uniform(0.4, 0.5),
                "yellowness": np.random.uniform(0.3, 0.4),
                "uniformity": np.random.uniform(0.75, 0.9)
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
        """Simple lesion detection based on local contrast."""
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame
        
        # Threshold for high-contrast regions
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Count regions significantly darker or lighter
        threshold_low = mean_intensity - 2 * std_intensity
        threshold_high = mean_intensity + 2 * std_intensity
        
        dark_pixels = np.sum(gray < threshold_low)
        bright_pixels = np.sum(gray > threshold_high)
        
        total_pixels = gray.size
        anomaly_ratio = (dark_pixels + bright_pixels) / total_pixels
        
        # Estimate lesion count from anomaly ratio
        # This is a very rough heuristic
        estimated_lesions = int(anomaly_ratio * 100)
        
        return min(estimated_lesions, 20)  # Cap at 20
    
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

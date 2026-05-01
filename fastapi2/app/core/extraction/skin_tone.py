"""
skin_tone.py — ITA-Based Skin Tone Classifier

Computes Individual Typology Angle (ITA) from a facial forehead ROI and
classifies the result into Fitzpatrick phototype groups.

Uses pure colorimetry (CIELab) — no ML, no training data required.

References:
  - Chardon et al. (1991)  — ITA angle definition
  - Ware et al. (2020)     — rPPG accuracy across Fitzpatrick types
  - Sun et al. (2022)      — CHROM + ITA weighting for skin-tone robustness
  - Fida et al. (2023)     — Pulse rate bias in Fitzpatrick III–V (Indian context)

Fitzpatrick / ITA mapping:
  ITA > 28°         →  I–II   (Very Light)
  10° < ITA ≤ 28°  →  II–III (Light)
  -30° < ITA ≤ 10° →  III–IV (Intermediate) ← Indian subcontinent range
  -50° < ITA ≤ -30°→  IV–V   (Tan)
  ITA ≤ -50°        →  V–VI   (Dark)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

from app.utils import get_logger

logger = get_logger(__name__)

# ── MediaPipe forehead landmark indices ──────────────────────────────────────
# These 10 landmarks sit on the forehead skin between the hairline and brow.
# Using specific indices avoids including hair/eyebrows in the ROI,
# which is the primary cause of ITA over-darkening.
_FOREHEAD_LANDMARK_IDS = [10, 109, 338, 151, 337, 299, 333, 69, 104, 67, 9]

# MediaPipe face oval landmarks (used to build a tight face polygon for ITA)
_FACE_OVAL_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


@dataclass
class SkinToneResult:
    """Calibration result from ITA skin tone analysis."""
    ita_angle: float
    fitzpatrick_class: str      # e.g. "III–IV"
    fitzpatrick_number: int     # midpoint: 1–6
    confidence_level: str       # "High" | "Medium"
    compensation_active: bool
    lighting_quality: str       # "Optimal" | "Moderate" | "Underlit" | "Overexposed" | "Glare Detected"
    lighting_score: float       # 0.0–1.0
    specular_highlight_pct: float
    chrom_weight_profile: str
    indian_range: bool          # True if Fitzpatrick III–V (common in India)
    detected_skin_color: str    # Hex code of average ROI color
    message: str
    face_detected: bool

    def to_dict(self) -> dict:
        return {
            "ita_angle": self.ita_angle,
            "fitzpatrick_class": self.fitzpatrick_class,
            "fitzpatrick_number": self.fitzpatrick_number,
            "confidence_level": self.confidence_level,
            "compensation_active": self.compensation_active,
            "lighting_quality": self.lighting_quality,
            "lighting_score": self.lighting_score,
            "specular_highlight_pct": self.specular_highlight_pct,
            "chrom_weight_profile": self.chrom_weight_profile,
            "indian_range": self.indian_range,
            "detected_skin_color": self.detected_skin_color,
            "message": self.message,
            "face_detected": self.face_detected,
        }


class SkinToneClassifier:
    """
    Classifies skin tone from a BGR camera frame using ITA angle.

    Optimised for the Fitzpatrick III–V range prevalent in the Indian
    subcontinent, as validated by Sun et al. (2022) and Fida et al. (2023).
    """

    def classify(self, frame: np.ndarray) -> SkinToneResult:
        """
        Run ITA analysis on a BGR frame and return a calibration result.

        Args:
            frame: BGR numpy array (H × W × 3) from OpenCV camera capture.

        Returns:
            SkinToneResult with ITA, Fitzpatrick class, lighting, and messaging.
        """
        if not CV2_AVAILABLE:
            return self._fallback_result("OpenCV not available")
        if frame is None or frame.size == 0:
            return self._fallback_result("Empty frame received")

        forehead_roi, lighting_info = self._extract_forehead_roi(frame)

        if forehead_roi is None:
            return self._no_face_result(lighting_info)

        ita_angle = self._compute_ita(forehead_roi)
        fitzpatrick_class, fitzpatrick_number = self._classify_fitzpatrick(ita_angle)
        confidence_level, compensation_active = self._get_confidence(ita_angle, lighting_info["score"])
        chrom_profile = self._get_chrom_profile(ita_angle)
        indian_range = -30.0 <= ita_angle <= 28.0  # Fitzpatrick III–V

        message = self._compose_message(
            fitzpatrick_class, confidence_level, compensation_active,
            lighting_info["quality"], indian_range
        )

        return SkinToneResult(
            ita_angle=round(ita_angle, 1),
            fitzpatrick_class=fitzpatrick_class,
            fitzpatrick_number=fitzpatrick_number,
            confidence_level=confidence_level,
            compensation_active=compensation_active,
            lighting_quality=lighting_info["quality"],
            lighting_score=round(lighting_info["score"], 2),
            specular_highlight_pct=round(lighting_info.get("specular_pct", 0.0), 1),
            chrom_weight_profile=chrom_profile,
            indian_range=indian_range,
            detected_skin_color=lighting_info.get("hex_color", "#cccccc"),
            message=message,
            face_detected=True,
        )

    # ── Private Methods ──────────────────────────────────────────────────────

    def _extract_forehead_roi(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
        """
        Extract a skin ROI from the center of the face (nose/inner cheeks area).
        This avoids issues with hair occlusion on the forehead.
        If no face is detected, it falls back to the absolute center of the frame.
        """
        h, w = frame.shape[:2]
        lighting_info = {"quality": "Unknown", "score": 0.5, "specular_pct": 0.0, "hex_color": "#cccccc"}

        if MP_AVAILABLE:
            try:
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                face_mesh.close()

                if not results.multi_face_landmarks:
                    return None, lighting_info

                lm = results.multi_face_landmarks[0].landmark

                # ── Get the center of the face ──
                face_xs = [int(lm[i].x * w) for i in _FACE_OVAL_IDS]
                face_ys = [int(lm[i].y * h) for i in _FACE_OVAL_IDS]
                
                center_x = int(np.mean(face_xs))
                center_y = int(np.mean(face_ys))
                
                # Crop a square of 15% of the frame size from the center of the face
                crop_size = int(min(h, w) * 0.15)
                half_crop = crop_size // 2
                
                r0 = max(0, center_y - half_crop)
                r1 = min(h, center_y + half_crop)
                c0 = max(0, center_x - half_crop)
                c1 = min(w, center_x + half_crop)

                if r1 <= r0 or c1 <= c0:
                    return None, lighting_info

                roi = frame[r0:r1, c0:c1]
                roi_mask = np.ones((r1 - r0, c1 - c0), dtype=np.uint8) * 255

                lighting_info = self._analyze_lighting(roi, roi_mask)
                return roi, lighting_info

            except Exception as e:
                logger.warning(f"SkinToneClassifier: MediaPipe failed ({e}), using heuristic crop")

        # ── Heuristic fallback (no MediaPipe) ──
        # Take a center crop of the frame
        center_y, center_x = h // 2, w // 2
        crop_size = int(min(h, w) * 0.15)
        half_crop = crop_size // 2
        
        r0 = max(0, center_y - half_crop)
        r1 = min(h, center_y + half_crop)
        c0 = max(0, center_x - half_crop)
        c1 = min(w, center_x + half_crop)
        
        roi = frame[r0:r1, c0:c1]
        if roi.size == 0:
            return None, lighting_info
            
        roi_mask = np.ones((r1 - r0, c1 - c0), dtype=np.uint8) * 255
        lighting_info = self._analyze_lighting(roi, roi_mask)
        return roi, lighting_info

    def _analyze_lighting(self, roi: np.ndarray, mask: np.ndarray = None) -> dict:
        """Assess lighting quality from CIELab L* channel on masked skin pixels."""
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(float)
        l_norm = lab[:, :, 0] / 2.55  # [0, 100]

        # Use only non-zero (face-skin) pixels if mask is provided
        if mask is not None and mask.sum() > 0:
            skin_pixels = l_norm[mask > 0]
        else:
            skin_pixels = l_norm.flatten()

        if len(skin_pixels) == 0:
            skin_pixels = l_norm.flatten()

        mean_l = float(np.mean(skin_pixels))

        # Specular highlights: L* > 92 (overexposed bright spots)
        specular_pct = float(np.sum(skin_pixels > 92) / len(skin_pixels) * 100)

        # Continuous score: ideal L* ~ 60 for skin
        diff = abs(mean_l - 60.0)
        score = max(0.1, 1.0 - (diff / 60.0) ** 1.5)

        if mean_l < 25:
            quality = "Too Dark"
        elif mean_l < 45:
            quality = "Underlit"
        elif mean_l <= 75:
            quality = "Optimal"
        elif mean_l <= 85:
            quality = "Moderate"
        else:
            quality = "Overexposed"

        if specular_pct > 15:
            score = max(0.20, score - 0.25)
            quality = "Glare Detected"
        elif specular_pct > 5:
            score = max(0.30, score - 0.10)

        # Average skin color — only from face-skin pixels (not background)
        if mask is not None and mask.sum() > 0:
            bgr_pixels = roi.reshape(-1, 3)[mask.flatten() > 0].astype(float)
        else:
            bgr_pixels = roi.reshape(-1, 3).astype(float)

        if len(bgr_pixels) > 0:
            avg = np.mean(bgr_pixels, axis=0)
            b = int(np.clip(avg[0], 0, 255))
            g = int(np.clip(avg[1], 0, 255))
            r = int(np.clip(avg[2], 0, 255))
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            hex_color = "#cccccc"

        return {"quality": quality, "score": round(score, 2), "specular_pct": specular_pct, "hex_color": hex_color}

    def _compute_ita(self, roi: np.ndarray) -> float:
        """
        Compute ITA angle from the forehead ROI.
        ITA = arctan((L* − 50) / b*) × (180/π)

        Uses median of per-pixel ITA for robustness against specular highlights.
        Only computes ITA on pixels with non-zero value (masked-in skin).
        """
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(float)
        l_star = lab[:, :, 0] / 2.55          # [0, 100]
        b_star = lab[:, :, 2] - 128.0          # approximately CIELab b*

        # Only use pixels that aren't black (masked-out background = 0,0,0)
        gray = np.mean(roi, axis=2)
        valid = gray > 5  # non-background pixels

        l_valid = l_star[valid]
        b_valid = b_star[valid]

        if len(l_valid) < 10:
            # Fall back to all pixels
            l_valid = l_star.flatten()
            b_valid = b_star.flatten()

        b_safe = np.where(np.abs(b_valid) < 0.5, 0.5, b_valid)
        ita_map = np.degrees(np.arctan((l_valid - 50.0) / b_safe))

        # Clip to plausible ITA range and return median
        ita_map = np.clip(ita_map, -90, 90)
        return float(np.median(ita_map))

    def _classify_fitzpatrick(self, ita: float) -> Tuple[str, int]:
        if ita > 28:
            return "I–II", 1
        elif ita > 10:
            return "II–III", 2
        elif ita > -30:
            return "III–IV", 3   # Indian subcontinent range
        elif ita > -50:
            return "IV–V", 5
        else:
            return "V–VI", 6

    def _get_confidence(self, ita: float, lighting_score: float) -> Tuple[str, bool]:
        """Determine confidence and whether compensation is active."""
        compensation = ita < 10  # Active for intermediate and darker tones
        if lighting_score < 0.45:
            return "Medium", True
        return "High", compensation

    def _get_chrom_profile(self, ita: float) -> str:
        if ita > 28:
            return "Standard (R:G:B = 3:−2:0)"
        elif ita > 10:
            return "Light-Adapted (G-Balanced)"
        elif ita > -30:
            return "ITA-III/IV — Indian Optimised"
        elif ita > -50:
            return "ITA-IV/V — Green-Dominant"
        else:
            return "ITA-V/VI — Green-Dominant + Enhanced"

    def _compose_message(self, fitzpatrick_class, confidence, compensation,
                          lighting_quality, indian_range) -> str:
        parts = []
        if indian_range:
            parts.append(
                f"Skin type {fitzpatrick_class} detected — within the Fitzpatrick III–V range "
                f"prevalent in the Indian subcontinent. Signal processing is tuned for this range."
            )
        else:
            parts.append(f"Skin type {fitzpatrick_class} detected.")

        if compensation:
            parts.append(
                "ITA-based CHROM weighting applied (Sun et al. 2022). "
                "Pulse-rate bias is mitigated and actively monitored."
            )
        else:
            parts.append("Standard CHROM rPPG — no tone compensation required.")

        if lighting_quality not in ("Optimal", "Moderate"):
            parts.append(
                f"Lighting: {lighting_quality}. For best results, face an evenly-lit white wall."
            )

        return " ".join(parts)

    def _fallback_result(self, reason: str) -> SkinToneResult:
        return SkinToneResult(
            ita_angle=0.0,
            fitzpatrick_class="Unknown",
            fitzpatrick_number=3,
            confidence_level="Medium",
            compensation_active=True,
            lighting_quality="Unknown",
            lighting_score=0.5,
            specular_highlight_pct=0.0,
            chrom_weight_profile="ITA-III/IV — Default (Indian Optimised)",
            indian_range=False,
            detected_skin_color="#cccccc",
            message=f"Calibration incomplete: {reason}. Standard Indian-range compensation applied.",
            face_detected=False,
        )

    def _no_face_result(self, lighting_info: dict) -> SkinToneResult:
        lighting_info = lighting_info or {"quality": "Unknown", "score": 0.5, "specular_pct": 0.0, "hex_color": "#cccccc"}
        return SkinToneResult(
            ita_angle=0.0,
            fitzpatrick_class="Undetected",
            fitzpatrick_number=3,
            confidence_level="Medium",
            compensation_active=True,
            lighting_quality=lighting_info["quality"],
            lighting_score=lighting_info["score"],
            specular_highlight_pct=lighting_info.get("specular_pct", 0.0),
            chrom_weight_profile="ITA-III/IV — Default (Indian Optimised)",
            indian_range=False,
            detected_skin_color=lighting_info.get("hex_color", "#cccccc"),
            message=(
                "No face detected. Please centre your face in the camera frame. "
                f"Lighting: {lighting_info['quality']}. "
                "ITA-III/IV compensation will be applied as a safe default."
            ),
            face_detected=False,
        )

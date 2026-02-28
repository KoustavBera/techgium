"""
Visual Disease Classifier — Roboflow API (Phase 1 POC)

Runs 3 models IN PARALLEL using ThreadPoolExecutor to minimise latency:
  - Skin Lesions   : my-ham10000/2   → full face crop
  - Eye Disease    : eye-disease-jio2h/5   → tight eye crop
  - Conjunctivitis : eye-detection-ci8qu/2 → tight eye crop (reused)

Crops are computed ONCE from MediaPipe FaceMesh landmarks where possible,
falling back to the full frame gracefully.
"""

import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple

try:
    from inference_sdk import InferenceHTTPClient
    INFERENCE_AVAILABLE = True
except ImportError:
    InferenceHTTPClient = None
    INFERENCE_AVAILABLE = False

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh eye landmark indices (left eye outer bounding region)
# We use left eye because it's typically better lit on a webcam.
# ---------------------------------------------------------------------------
_LEFT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

# ---------------------------------------------------------------------------
# Human-readable display names for raw model class labels.
# HAM10000 uses abbreviations; eye models use mixed-case labels.
# Any class NOT in this dict is displayed as-is (title-cased).
# ---------------------------------------------------------------------------
CLASS_DISPLAY_NAMES: Dict[str, str] = {
    # HAM10000 skin lesion classes
    "nv":     "Melanocytic Nevi (Moles)",
    "mel":    "Melanoma",
    "bkl":    "Benign Keratosis",
    "bcc":    "Basal Cell Carcinoma",
    "akiec":  "Actinic Keratoses",
    "vasc":   "Vascular Lesion",
    "df":     "Dermatofibroma",
    # Eye disease model classes
    "cataract":       "Cataract",
    "jaundice":       "Jaundice (Eye)",
    "normal_eye":     "Normal Eye",
    "pterygium":      "Pterygium",
    # Conjunctivitis model classes
    "conjunctivitis": "Conjunctivitis",
    "not_affected":   "No Conjunctivitis",
    # Measles (legacy)
    "measles":        "Measles",
    "normal":         "Normal Skin",
}


class VisualDiseaseClassifier(BaseExtractor):
    """
    API-based visual disease extractor.
    Sends auto-cropped face/eye images to 3 Roboflow models in parallel.
    """

    system = PhysiologicalSystem.VISUAL_DISEASE

    # --- Model definitions -------------------------------------------------
    MODELS = {
        "skin_lesion": {
            "model_id": "my-ham10000/2",
            "crop": "face",            # use the pre-cropped face ROI
            "min_confidence": 0.3,     # HAM10000 — show any notable detection
        },
        "eye_disease": {
            "model_id": "eye-disease-jio2h/5",
            "crop": "eye",             # tight eye crop via landmarks
            "min_confidence": 0.4,
        },
        "conjunctivitis": {
            "model_id": "eye-detection-ci8qu/2",
            "crop": "eye",             # reuses the same eye crop
            "min_confidence": 0.7,     # Raised — this model produces many false positives
        },
    }

    def __init__(self):
        super().__init__()
        self.roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

        if INFERENCE_AVAILABLE and self.roboflow_api_key:
            # One shared client — thread-safe for reads
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.roboflow_api_key,
            )
            logger.info("VisualClassifier: Roboflow client initialised.")
        else:
            self.client = None
            if not INFERENCE_AVAILABLE:
                logger.warning("VisualClassifier: inference_sdk not installed — pip install inference-sdk")
            elif not self.roboflow_api_key:
                logger.warning("VisualClassifier: ROBOFLOW_API_KEY missing from .env")

    # -----------------------------------------------------------------------
    # PUBLIC: extract
    # -----------------------------------------------------------------------
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        import time
        t0 = time.time()
        biomarker_set = self._create_biomarker_set()

        if not self.client:
            return biomarker_set

        # -- Resolve images -------------------------------------------------
        raw_frames = data.get("raw_face_frames", [])
        face_frames = data.get("face_frames", [])
        face_landmarks_sequence = data.get("face_landmarks_sequence", [])

        if not raw_frames and not face_frames:
            logger.warning("VisualClassifier: No frames available, skipping.")
            return biomarker_set

        # Face crop: prefer pre-cropped ROI, else fall back to raw frame
        face_img = face_frames[0] if face_frames else raw_frames[0]

        # Eye crop: derived once from MediaPipe landmarks + raw frame
        eye_img = self._crop_eye_region(
            raw_frames[0] if raw_frames else face_img,
            face_landmarks_sequence,
        )

        # Map crop type → resolved image
        crop_images = {
            "face": face_img,
            "eye": eye_img,
        }

        # -- Run all models in parallel -------------------------------------
        results = self._run_models_parallel(crop_images)

        # -- Store as biomarkers -------------------------------------------
        for prefix, predictions in results.items():
            model_cfg = self.MODELS[prefix]
            min_conf = model_cfg.get("min_confidence", 0.4)

            for pred in predictions:
                confidence = float(pred.get("confidence", 0.0))
                raw_class = str(pred.get("class", "unknown")).replace(" ", "_").lower()

                # Skip predictions below per-model threshold
                if confidence < min_conf:
                    logger.debug(f"VisualClassifier [{prefix}]: {raw_class} = {confidence:.2f} < {min_conf} threshold, skipping.")
                    continue

                # Resolve friendly display name
                display_name = CLASS_DISPLAY_NAMES.get(raw_class, raw_class.replace("_", " ").title())
                bm_name = f"{prefix}_{raw_class}"

                self._add_biomarker(
                    biomarker_set,
                    name=bm_name,
                    value=confidence,
                    unit="probability",
                    confidence=confidence,
                    normal_range=(0.4, 1.0),   # Reversed: >40% → normal
                    description=(
                        f"Visual AI ({model_cfg['model_id']}): {display_name}"
                    ),
                )
                level = "normal" if confidence > 0.4 else "⚠ HIGH"
                logger.info(f"VisualClassifier [{prefix}]: {display_name} = {confidence:.2f}  [{level}]")

        biomarker_set.extraction_time_ms = (time.time() - t0) * 1000
        self._extraction_count += 1
        logger.info(f"VisualClassifier: done in {biomarker_set.extraction_time_ms:.0f} ms")
        return biomarker_set

    # -----------------------------------------------------------------------
    # PRIVATE: parallel inference
    # -----------------------------------------------------------------------
    def _run_models_parallel(
        self, crop_images: Dict[str, np.ndarray]
    ) -> Dict[str, list]:
        """
        Dispatch all 3 Roboflow API calls simultaneously.
        Returns {prefix: [prediction_dicts]}.
        """
        results = {}

        def _call(prefix: str, model_config: dict) -> Tuple[str, list]:
            crop_type = model_config["crop"]
            img = crop_images.get(crop_type)
            if img is None or img.size == 0:
                logger.warning(f"VisualClassifier [{prefix}]: no valid crop, skipping.")
                return prefix, []
            try:
                response = self.client.infer(img, model_id=model_config["model_id"])
                preds = response.get("predictions", []) if isinstance(response, dict) else []
                logger.debug(f"VisualClassifier [{prefix}]: {len(preds)} prediction(s) received.")
                return prefix, preds
            except Exception as e:
                logger.error(f"VisualClassifier [{prefix}] API error: {e}")
                return prefix, []

        # 3 threads, one per model — collapses ~3s sequential → ~1s
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_call, prefix, cfg): prefix
                for prefix, cfg in self.MODELS.items()
            }
            for future in as_completed(futures):
                prefix, preds = future.result()
                results[prefix] = preds

        return results

    # -----------------------------------------------------------------------
    # PRIVATE: eye crop
    # -----------------------------------------------------------------------
    def _crop_eye_region(
        self,
        frame: np.ndarray,
        face_landmarks_sequence: list,
        padding: float = 0.35,
    ) -> np.ndarray:
        """
        Extract a tight bounding-box crop around the left eye using
        MediaPipe FaceMesh landmarks.  Falls back to the upper-centre
        quarter of the frame if landmarks are unavailable.

        Args:
            frame:   Raw BGR frame (H × W × 3)
            face_landmarks_sequence: list of [468×3] landmark arrays
            padding: fractional padding around the bounding box

        Returns:
            Cropped BGR eye image (may be fallback if landmarks absent)
        """
        h, w = frame.shape[:2]

        if face_landmarks_sequence:
            lm = face_landmarks_sequence[0]  # use first landmark set
            if isinstance(lm, np.ndarray) and lm.shape[0] > max(_LEFT_EYE_INDICES):
                pts = lm[_LEFT_EYE_INDICES, :2]  # (N, 2) normalised x,y
                x_min = float(pts[:, 0].min()) * w
                x_max = float(pts[:, 0].max()) * w
                y_min = float(pts[:, 1].min()) * h
                y_max = float(pts[:, 1].max()) * h

                bw = x_max - x_min
                bh = y_max - y_min
                x1 = max(0, int(x_min - bw * padding))
                x2 = min(w, int(x_max + bw * padding))
                y1 = max(0, int(y_min - bh * padding))
                y2 = min(h, int(y_max + bh * padding))

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    logger.debug(f"VisualClassifier: Eye crop from landmarks {x1},{y1}→{x2},{y2}")
                    return crop

        # Fallback: upper-centre quarter (rough eye region without landmarks)
        logger.info("VisualClassifier: Landmark crop failed — using heuristic eye region.")
        y1, y2 = int(h * 0.15), int(h * 0.55)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        fallback = frame[y1:y2, x1:x2]
        return fallback if fallback.size > 0 else frame

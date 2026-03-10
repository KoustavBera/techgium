"""
Visual Disease Classifier — Roboflow API (Phase 1 POC)

Runs 4 models IN PARALLEL using ThreadPoolExecutor to minimise latency:
  - Skin Lesions   : my-ham10000/2        → full face crop
  - Eye Disease    : eye-disease-jio2h/5  → tight eye crop
  - Conjunctivitis : eye-detection-ci8qu/2 → tight eye crop (reused)
  - Measles        : measles-f0wxa/2      → full face crop

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
# MediaPipe FaceMesh eye landmark indices.
# Both eyes are analysed; the prediction with the higher confidence wins.
# ---------------------------------------------------------------------------
_LEFT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]
_RIGHT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
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
            "min_confidence": 0.85,    # Raised higher — this model has many false positives
        },
        "measles": {
            "model_id": "measles-f0wxa/2",
            "crop": "face",            # full face crop — looks for rash patterns
            "min_confidence": 0.45,
        },
    }

    # Classes that represent a HEALTHY / BENIGN finding.
    # High confidence in these = good. High confidence in anything else = concern.
    HEALTHY_CLASSES = {"normal_eye", "not_affected", "normal", "nv"}

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
        raw_frame = raw_frames[0] if raw_frames else face_img

        # Eye crop: try BOTH eyes from landmarks (on the raw frame so coords are correct).
        # face_img is passed as the safe fallback if landmarks are absent — it is
        # already a tight face ROI so the heuristic box will land on the eyes.
        left_eye_img  = self._crop_eye_region(
            raw_frame, face_landmarks_sequence,
            eye_indices=_LEFT_EYE_INDICES,
            fallback_frame=face_img,
        )
        right_eye_img = self._crop_eye_region(
            raw_frame, face_landmarks_sequence,
            eye_indices=_RIGHT_EYE_INDICES,
            fallback_frame=None,   # don't double-fallback on right eye
        )

        # Pick the sharper / larger crop to send to the eye models.
        # If the right-eye crop is valid and bigger, prefer it; otherwise left.
        if right_eye_img is not None and right_eye_img.size > 0 and \
                right_eye_img.size >= left_eye_img.size:
            eye_img = right_eye_img
            logger.debug("VisualClassifier: using RIGHT eye crop (larger crop selected)")
        else:
            eye_img = left_eye_img
            logger.debug("VisualClassifier: using LEFT eye crop")

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

                # Healthy classes: high confidence = normal.
                # Disease classes: high confidence = abnormal (above normal).
                is_healthy = raw_class in self.HEALTHY_CLASSES
                normal_range = (0.4, 1.0) if is_healthy else (0.0, 0.4)
                level = "normal" if (is_healthy and confidence >= 0.4) or (not is_healthy and confidence <= 0.4) else "⚠ HIGH"

                self._add_biomarker(
                    biomarker_set,
                    name=bm_name,
                    value=confidence,
                    unit="probability",
                    confidence=confidence,
                    normal_range=normal_range,
                    description=(
                        f"Visual AI ({model_cfg['model_id']}): {display_name}"
                    ),
                )
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
                # Option C: preprocess to match clinical training distribution
                img_processed = self._preprocess_for_model(img, crop_type)
                # Roboflow models expect RGB; OpenCV frames are BGR.
                img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
                response = self.client.infer(img_rgb, model_id=model_config["model_id"])
                preds = response.get("predictions", []) if isinstance(response, dict) else []
                logger.debug(f"VisualClassifier [{prefix}]: {len(preds)} prediction(s) received.")
                return prefix, preds
            except Exception as e:
                logger.error(f"VisualClassifier [{prefix}] API error: {e}")
                return prefix, []

        # 4 threads, one per model — collapses ~4s sequential → ~1s
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_call, prefix, cfg): prefix
                for prefix, cfg in self.MODELS.items()
            }
            for future in as_completed(futures):
                prefix, preds = future.result()
                results[prefix] = preds

        return results

    # -----------------------------------------------------------------------
    # PRIVATE: clinical-grade image preprocessing (Option C)
    # -----------------------------------------------------------------------
    def _preprocess_for_model(
        self, img: np.ndarray, crop_type: str
    ) -> np.ndarray:
        """
        Preprocess a webcam crop to better match the clinical training distribution
        of the Roboflow models. Two techniques:

        1. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
           Normalises local lighting variations to mimic the controlled-lighting
           conditions of dermoscopy / slit-lamp photographs used in training.
           Applied per-channel in LAB colour space to avoid colour shift.

        2. Resize to 512×512
           Most clinical image models (HAM10000, eye disease datasets) were trained
           on 224-512px images. A tight crop of a webcam face delivers far fewer
           pixels than that. Upscaling with INTER_CUBIC interpolation improves
           feature clarity without artefacts.

        3. Unsharp mask (eye crops only)
           Eye crops are particularly small (~200×80px). Sharpening helps the
           model resolve fine details (blood vessels, scleral redness).
        """
        TARGET_SIZE = (512, 512)

        try:
            # Step 1: CLAHE in LAB colour space (preserves hue, fixes luminance)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l_channel)
            lab_eq = cv2.merge([l_eq, a_channel, b_channel])
            img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

            # Step 2: Resize to target size using cubic interpolation
            img_resized = cv2.resize(img_eq, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

            # Step 3: Unsharp mask for eye crops (very small input patches)
            if crop_type == "eye":
                gaussian = cv2.GaussianBlur(img_resized, (0, 0), sigmaX=2.0)
                img_resized = cv2.addWeighted(img_resized, 1.5, gaussian, -0.5, 0)

            logger.debug(
                f"VisualClassifier: preprocessed {crop_type} crop "
                f"{img.shape[:2]} → {TARGET_SIZE} (CLAHE + resize"
                f"{' + sharpen' if crop_type == 'eye' else ''})"
            )
            return img_resized

        except Exception as e:
            # Never block inference due to preprocessing failure
            logger.warning(f"VisualClassifier: preprocessing failed ({e}), using raw crop.")
            return img

    # -----------------------------------------------------------------------
    # PRIVATE: eye crop
    # -----------------------------------------------------------------------
    def _crop_eye_region(
        self,
        frame: np.ndarray,
        face_landmarks_sequence: list,
        eye_indices: list = None,
        padding: float = 0.35,
        fallback_frame: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract a tight bounding-box crop around one eye using MediaPipe
        FaceMesh landmarks.

        Args:
            frame:                  Raw BGR frame (landmarks computed against this).
            face_landmarks_sequence: List of [468×3] landmark arrays.
            eye_indices:            Which landmark indices to use (left or right eye).
                                    Defaults to _LEFT_EYE_INDICES.
            padding:                Fractional padding around the bounding box.
            fallback_frame:         If given, the heuristic fallback box is applied to
                                    THIS frame instead of `frame`.  Pass `face_img` here
                                    so the fallback stays inside the face ROI and the
                                    eye heuristic actually lands on the eyes.

        Returns:
            Cropped BGR eye image, or None if impossible to crop.
        """
        if eye_indices is None:
            eye_indices = _LEFT_EYE_INDICES

        h, w = frame.shape[:2]

        if face_landmarks_sequence:
            lm = face_landmarks_sequence[0]  # use first landmark set
            if isinstance(lm, np.ndarray) and lm.shape[0] > max(eye_indices):
                pts = lm[eye_indices, :2]  # (N, 2) normalised x,y
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

        # Fix 3: Fallback uses `fallback_frame` (face ROI) if provided.
        # The face ROI is already zoomed-in, so the heuristic box lands on the eyes.
        # If no fallback_frame is given, return None so the caller can skip this eye.
        if fallback_frame is None:
            return None

        logger.info("VisualClassifier: Landmark crop failed — using heuristic eye region on face ROI.")
        fh, fw = fallback_frame.shape[:2]
        y1, y2 = int(fh * 0.15), int(fh * 0.55)
        x1, x2 = int(fw * 0.20), int(fw * 0.80)
        fallback = fallback_frame[y1:y2, x1:x2]
        return fallback if fallback.size > 0 else fallback_frame

"""
Skeletal Structure Biomarker Extractor

Extracts musculoskeletal health indicators from MediaPipe Pose (BlazePose 33-point):
  1. Bilateral joint symmetry   — works seated or standing
  2. Stance stability & sway    — postural sway amplitude (CoM)
  3. Sway entropy               — Sample Entropy of CoM (balance quality)
  4. Posture alignment          — ear-shoulder-hip forward head / trunk tilt
  5. Joint range of motion      — elbow and knee angle range
  6. Gait step symmetry         — ankle-based stride, only when walking

Core biomarkers (1-5) are ALWAYS computed regardless of subject mobility.
Gait biomarkers (6) are computed opportunistically when walking is detected.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import signal

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem, Biomarker

logger = get_logger(__name__)


class SkeletalExtractor(BaseExtractor):
    """
    Extracts skeletal structure biomarkers from MediaPipe Pose data.

    All primary biomarkers work on a stationary kiosk subject.
    Gait analysis runs opportunistically when walking is detected.
    """

    system = PhysiologicalSystem.SKELETAL

    # BlazePose 33-point landmark indices
    LM = {
        "nose": 0,
        "left_eye": 2, "right_eye": 5,
        "left_ear": 7, "right_ear": 8,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13,    "right_elbow": 14,
        "left_wrist": 15,    "right_wrist": 16,
        "left_hip": 23,      "right_hip": 24,
        "left_knee": 25,     "right_knee": 26,
        "left_ankle": 27,    "right_ankle": 28,
    }

    # All symmetric joint pairs (label, left_idx, right_idx)
    SYMMETRIC_JOINTS = [
        ("shoulder", 11, 12),
        ("elbow",    13, 14),
        ("wrist",    15, 16),
        ("hip",      23, 24),
        ("knee",     25, 26),
        ("ankle",    27, 28),
    ]

    def __init__(self, sample_rate: float = 30.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_visibility = 0.5
        self._walk_threshold_pps = 40.0  # px/s above which = walking

    # ──────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────

    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract skeletal biomarkers.

        Expected data keys:
          pose_sequence : List[np.ndarray]  shape (N, 33, 4) — x,y,z,visibility
          fps / frame_rate : float          optional, overrides 30 Hz default
        """
        import time
        t0 = time.time()
        bs = self._create_biomarker_set()

        pose_sequence = data.get("pose_sequence", [])
        fps = data.get("fps") or data.get("frame_rate")
        if fps:
            self.sample_rate = float(fps)

        if len(pose_sequence) < 10:
            logger.warning(f"Skeletal: only {len(pose_sequence)} pose frames — skipping.")
            bs.extraction_time_ms = (time.time() - t0) * 1000
            self._extraction_count += 1
            return bs

        pose_arr = np.array(pose_sequence, dtype=np.float32)  # (F, 33, 4)

        is_walking = self._detect_walking(pose_arr)
        mode = "walking" if is_walking else "stationary"
        logger.info(f"Skeletal: {len(pose_arr)} frames, mode={mode}")

        # Always-run biomarkers (work for any subject)
        self._extract_bilateral_symmetry(pose_arr, bs)
        self._extract_stance_stability(pose_arr, bs)   # also produces sway_entropy
        self._extract_joint_kinematics(pose_arr, bs)
        self._extract_posture_alignment(pose_arr, bs)

        # Walking-only biomarkers
        if is_walking:
            self._extract_gait_step_symmetry(pose_arr, bs)

        bs.extraction_time_ms = (time.time() - t0) * 1000
        self._extraction_count += 1
        return bs

    # ──────────────────────────────────────────────────────────────────
    # Walking detection
    # ──────────────────────────────────────────────────────────────────

    def _detect_walking(self, pose_arr: np.ndarray) -> bool:
        """Return True if hips/shoulders show walking-level velocity."""
        if pose_arr.shape[1] <= 24:
            return False
        key = pose_arr[:, [11, 12, 23, 24], :2]            # (F, 4, 2)
        disp = np.linalg.norm(np.diff(key, axis=0), axis=2)  # (F-1, 4)
        avg_pps = float(np.mean(disp)) * self.sample_rate
        return avg_pps >= self._walk_threshold_pps

    # ──────────────────────────────────────────────────────────────────
    # 1. Bilateral symmetry — ALWAYS runs (works seated/standing)
    # ──────────────────────────────────────────────────────────────────

    def _extract_bilateral_symmetry(
        self, pose_arr: np.ndarray, bs: BiomarkerSet
    ) -> None:
        """
        Compute gait_symmetry_ratio from bilateral joint ROM variance.

        Even a stationary subject has micro-oscillations from breathing and
        postural sway — these are sufficient to measure left-right ROM asymmetry.
        Confidence scales with how many joint pairs are visible.
        """
        scores: List[float] = []
        used: List[str] = []

        for label, idx_l, idx_r in self.SYMMETRIC_JOINTS:
            if pose_arr.shape[1] <= max(idx_l, idx_r):
                continue

            l3d, l_vis = self._lm3d(pose_arr, idx_l)
            r3d, r_vis = self._lm3d(pose_arr, idx_r)
            mask = (l_vis > self.min_visibility) & (r_vis > self.min_visibility)
            if mask.sum() < 10:
                continue

            lf = self._bandpass(l3d[mask])
            rf = self._bandpass(r3d[mask])

            l_range = np.linalg.norm(lf.max(0) - lf.min(0))
            r_range = np.linalg.norm(rf.max(0) - rf.min(0))
            avg = (l_range + r_range) / 2 + 1e-6
            scores.append(float(np.clip(1.0 - abs(l_range - r_range) / avg, 0.0, 1.0)))
            used.append(label)

        if not scores:
            logger.warning("Skeletal: no valid joint pairs for bilateral symmetry.")
            return

        n = len(scores)
        confidence = float(np.clip(0.70 + (n / len(self.SYMMETRIC_JOINTS)) * 0.22, 0.70, 0.92))

        self._add_biomarker(
            bs,
            name="gait_symmetry_ratio",
            value=float(np.mean(scores)),
            unit="ratio",
            confidence=confidence,
            normal_range=(0.85, 1.0),
            description=f"Bilateral joint ROM symmetry ({', '.join(used)} joints)",
        )
        bs.metadata["bilateral_joints_used"] = used

    # ──────────────────────────────────────────────────────────────────
    # 2. Stance stability + sway entropy — ALWAYS runs
    # ──────────────────────────────────────────────────────────────────

    def _extract_stance_stability(
        self, pose_arr: np.ndarray, bs: BiomarkerSet
    ) -> None:
        """
        Compute stance_stability_score, sway_velocity, and sway_entropy
        from the hip center-of-mass trajectory.

        Normalization: all sway values are divided by shoulder width so
        the score is independent of subject distance from the camera.
        """
        if pose_arr.shape[1] <= 24:
            return

        hl, vl = self._lm3d(pose_arr, 23)
        hr, vr = self._lm3d(pose_arr, 24)
        mask = (vl > self.min_visibility) & (vr > self.min_visibility)
        if mask.sum() < 10:
            return

        com = (hl[mask] + hr[mask]) / 2            # (N, 3)
        filt = self._bandpass(com, low=0.1, high=2.0) # physiological sway band

        sway_px = float(np.sqrt(np.std(filt[:, 0]) ** 2 + np.std(filt[:, 1]) ** 2))
        sw_px   = self._shoulder_width(pose_arr)
        norm_sway = sway_px / max(sw_px, 1.0)

        # Stability score: 0 = severe (≥2% of shoulder width), 100 = perfect
        stability = float(np.clip(100.0 * (1.0 - norm_sway / 0.02), 0.0, 100.0))
        self._add_biomarker(
            bs,
            name="stance_stability_score",
            value=stability,
            unit="score_0_100",
            confidence=0.90,
            normal_range=(75, 100),
            description="Postural stability (CoM sway / shoulder width, lower = more stable)",
        )

        # Sway velocity
        vel = float(np.mean(np.linalg.norm(np.diff(filt, axis=0), axis=1)))
        self._add_biomarker(
            bs,
            name="sway_velocity",
            value=vel,
            unit="normalized_units_per_frame",
            confidence=0.85,
            normal_range=(0.001, 0.01),
            description="Average CoM velocity (filtered, normalized)",
        )

        # Sway entropy — Sample Entropy of anterior-posterior component
        # High entropy = healthy adaptive balance; low = rigid/stiff (Parkinson risk)
        if len(filt) >= 30:
            se = float(np.clip(self._sample_entropy(filt[:, 1], m=2, r=0.2), 0.0, 3.0))
            self._add_biomarker(
                bs,
                name="sway_entropy",
                value=se,
                unit="sample_entropy",
                confidence=0.82,
                normal_range=(0.3, 2.0),
                description="Sample Entropy of AP sway (healthy=0.3–2.0; low=rigid/stiff)",
            )

    # ──────────────────────────────────────────────────────────────────
    # 3. Posture alignment — ALWAYS runs
    # ──────────────────────────────────────────────────────────────────

    def _extract_posture_alignment(
        self, pose_arr: np.ndarray, bs: BiomarkerSet
    ) -> None:
        """
        posture_score (0–100) from ear-shoulder-hip alignment.

        Three sub-scores, each worth ~33 points:
          A. Forward head: horizontal ear offset from shoulder
          B. Shoulder tilt: angle of shoulder line from horizontal
          C. Trunk tilt: angle of shoulder→hip vector from vertical

        Falls back to B+C only when ear landmarks have low visibility.
        Uses median pose frame for robust single-frame estimate.
        """
        if pose_arr.shape[1] <= 24:
            return

        pm = np.median(pose_arr, axis=0)  # (33, 4) median

        def xy(i: int) -> np.ndarray:
            return pm[i, :2]

        def vis(i: int) -> float:
            return float(pm[i, 3])

        l_sh, r_sh   = xy(11), xy(12)
        l_hip, r_hip = xy(23), xy(24)

        if vis(11) < 0.4 or vis(12) < 0.4 or vis(23) < 0.4 or vis(24) < 0.4:
            logger.debug("Skeletal: posture skipped — low shoulder/hip visibility.")
            return

        sh_mid  = (l_sh + r_sh) / 2
        hip_mid = (l_hip + r_hip) / 2
        sw      = float(np.linalg.norm(r_sh - l_sh)) + 1e-6

        # B. Shoulder tilt
        sh_vec  = r_sh - l_sh
        sh_tilt = abs(float(np.degrees(np.arctan2(sh_vec[1], sh_vec[0]))))
        sh_pen  = min(sh_tilt / 15.0, 1.0)

        # C. Trunk tilt from vertical
        trunk_vec  = sh_mid - hip_mid
        trunk_tilt = abs(float(np.degrees(np.arctan2(trunk_vec[0], abs(trunk_vec[1]) + 1e-6))))
        trunk_pen  = min(trunk_tilt / 20.0, 1.0)

        ears_ok = vis(7) >= 0.4 and vis(8) >= 0.4

        if ears_ok:
            ear_mid = (xy(7) + xy(8)) / 2
            fwd_norm = abs(ear_mid[0] - sh_mid[0]) / sw
            fwd_pen  = min(fwd_norm / 0.30, 1.0)
            score    = float(np.clip(100.0 - (fwd_pen + sh_pen + trunk_pen) * 33.3, 0.0, 100.0))
            conf     = 0.88
            desc     = "Posture score: forward head + shoulder tilt + trunk tilt"
        else:
            # No ears: shoulder + trunk only (2 components × 50 points)
            score = float(np.clip(100.0 - (sh_pen + trunk_pen) * 50.0, 0.0, 100.0))
            conf  = 0.75
            desc  = "Posture score: shoulder tilt + trunk tilt (ear landmarks occluded)"

        self._add_biomarker(
            bs,
            name="posture_score",
            value=score,
            unit="score_0_100",
            confidence=conf,
            normal_range=(70, 100),
            description=desc,
        )

    # ──────────────────────────────────────────────────────────────────
    # 4. Joint kinematics (ROM) — ALWAYS runs
    # ──────────────────────────────────────────────────────────────────

    def _extract_joint_kinematics(
        self, pose_arr: np.ndarray, bs: BiomarkerSet
    ) -> None:
        """Joint range-of-motion: elbow (shoulder→elbow→wrist) and knee (hip→knee→ankle)."""
        roms: Dict[str, float] = {}

        if pose_arr.shape[1] > 16:
            for side, (si, ei, wi) in [("left", (11, 13, 15)), ("right", (12, 14, 16))]:
                ang = self._joint_angles(pose_arr[:, si, :2], pose_arr[:, ei, :2], pose_arr[:, wi, :2])
                roms[f"elbow_{side}"] = float(ang.max() - ang.min())

        if pose_arr.shape[1] > 28:
            for side, (hi, ki, ai) in [("left", (23, 25, 27)), ("right", (24, 26, 28))]:
                ang = self._joint_angles(pose_arr[:, hi, :2], pose_arr[:, ki, :2], pose_arr[:, ai, :2])
                roms[f"knee_{side}"] = float(ang.max() - ang.min())

        if not roms:
            return

        avg_rom = float(np.mean(list(roms.values())))
        self._add_biomarker(
            bs,
            name="average_joint_rom",
            value=avg_rom,
            unit="radians",
            confidence=0.88,
            normal_range=(0.3, 0.8),
            description="Average elbow + knee range of motion (radians)",
        )
        bs.metadata["joint_roms"] = {k: round(v, 4) for k, v in roms.items()}

    # ──────────────────────────────────────────────────────────────────
    # 5. Gait step symmetry — WALKING only
    # ──────────────────────────────────────────────────────────────────

    def _extract_gait_step_symmetry(
        self, pose_arr: np.ndarray, bs: BiomarkerSet
    ) -> None:
        """Step-length symmetry from ankle Y-position oscillation (requires walking)."""
        if pose_arr.shape[1] <= 28:
            return

        l_y, l_vis = pose_arr[:, 27, 1], pose_arr[:, 27, 3]
        r_y, r_vis = pose_arr[:, 28, 1], pose_arr[:, 28, 3]

        valid_l = l_vis > self.min_visibility
        valid_r = r_vis > self.min_visibility

        if valid_l.sum() < 10 or valid_r.sum() < 10:
            return

        l_clean = self._bandpass(l_y[valid_l])
        r_clean = self._bandpass(r_y[valid_r])

        l_steps = np.abs(np.diff(l_clean))
        r_steps = np.abs(np.diff(r_clean))
        l_steps = l_steps[l_steps > 0.005]
        r_steps = r_steps[r_steps > 0.005]

        if len(l_steps) < 5 or len(r_steps) < 5:
            return

        sym = 1.0 - abs(np.mean(l_steps) - np.mean(r_steps)) / (
            0.5 * (np.mean(l_steps) + np.mean(r_steps)) + 1e-6
        )
        self._add_biomarker(
            bs,
            name="step_length_symmetry",
            value=float(np.clip(sym, 0.0, 1.0)),
            unit="ratio",
            confidence=0.82,
            normal_range=(0.85, 1.0),
            description="Ankle motion step-length symmetry (walking mode)",
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _lm3d(self, pose_arr: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (xyz, visibility) arrays for landmark idx."""
        return pose_arr[:, idx, :3], pose_arr[:, idx, 3]

    def _shoulder_width(self, pose_arr: np.ndarray) -> float:
        """Median shoulder width in pixels (scale proxy)."""
        if pose_arr.shape[1] <= 12:
            return 100.0
        dists = np.linalg.norm(pose_arr[:, 11, :2] - pose_arr[:, 12, :2], axis=1)
        valid = dists[dists > 0]
        return float(np.median(valid)) if len(valid) > 0 else 100.0

    def _joint_angles(
        self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> np.ndarray:
        """Per-frame angle at p2 in the p1→p2→p3 chain (radians)."""
        v1 = p1 - p2
        v2 = p3 - p2
        dot  = np.sum(v1 * v2, axis=1)
        mag  = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6
        return np.arccos(np.clip(dot / mag, -1.0, 1.0))

    def _bandpass(
        self,
        sig: np.ndarray,
        low: float = 0.5,
        high: float = 5.0,
    ) -> np.ndarray:
        """
        Butterworth bandpass filter. Handles 1-D and N-D arrays.
        Returns original signal unchanged if it is too short or filter fails.
        """
        if len(sig) < 15:
            return sig
        nyq = self.sample_rate / 2.0
        lo  = low  / nyq
        hi  = min(high / nyq, 0.99)
        try:
            sos = signal.butter(4, [lo, hi], btype="band", output="sos")
            axis = 0
            return signal.sosfiltfilt(sos, sig, axis=axis)
        except Exception:
            return sig

    # Keep old name as alias so existing call-sites in tests don't break
    _preprocess_signal = _bandpass

    def _sample_entropy(
        self, x: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        """
        Vectorised Sample Entropy (SampEn).

        Parameters
        ----------
        x : 1-D signal
        m : template length
        r : tolerance (fraction of std)

        O(N²) but fast with numpy broadcasting for N≤500.
        SampEn ∈ (0, ~3): higher = more complex / healthier balance.
        """
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        if N < 2 * (m + 1):
            return 0.0
        r_abs = r * (np.std(x) + 1e-10)

        def _phi(m_: int) -> int:
            # Build template matrix and count matches
            templates = np.array([x[i: i + m_] for i in range(N - m_)])  # (N-m, m)
            count = 0
            for i in range(N - m_):
                diffs = np.max(np.abs(templates - templates[i]), axis=1)
                # exclude self-match (i==i)
                count += int(np.sum(diffs <= r_abs)) - 1
            return count

        A = _phi(m + 1)
        B = _phi(m)
        if B == 0:
            return 0.0
        return float(-np.log((A + 1e-10) / (B + 1e-10)))
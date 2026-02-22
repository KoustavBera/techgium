"""
Cardiovascular Biomarker Extractor

Extracts cardiovascular health indicators:
- Heart rate (HR)
- Heart rate variability (HRV)
- Chest micro-motion proxies
"""
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import signal

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class CardiovascularExtractor(BaseExtractor):
    """
    Extracts cardiovascular biomarkers.
    
    Analyzes RIS data, motion data, and auxiliary vital signs.
    """
    
    system = PhysiologicalSystem.CARDIOVASCULAR
    
    def __init__(self, sample_rate: float = 1000.0):
        """
        Initialize cardiovascular extractor.
        
        Args:
            sample_rate: RIS/signal sampling rate in Hz
        """
        super().__init__()
        self.sample_rate = sample_rate
    
    def preprocess_signal(
        self, 
        signal_data: np.ndarray, 
        fs: float,
        lowcut: float = 0.8,
        highcut: float = 3.0
    ) -> np.ndarray:
        """
        Preprocess physiological signal for cardiac analysis.
        
        Applies detrending, bandpass filtering, and normalization.
        
        Args:
            signal_data: Raw signal array
            fs: Sampling frequency in Hz
            lowcut: Low cutoff frequency (default 0.8 Hz = 48 bpm)
            highcut: High cutoff frequency (default 3.0 Hz = 180 bpm)
            
        Returns:
            Preprocessed signal array
        """
        # Detrend to remove DC offset and linear trends
        detrended = np.asarray(signal.detrend(signal_data))
        
        # Bandpass filter for cardiac frequencies
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Use second-order sections for numerical stability
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered = np.asarray(signal.sosfilt(sos, detrended))
        
        # Normalize to unit variance
        std = float(np.std(filtered))
        if std > 1e-6:  # Avoid division by zero
            normalized: np.ndarray = filtered / std
        else:
            normalized = filtered
        
        return normalized
    
    def validate_signal(
        self,
        signal_data: np.ndarray,
        fs: float,
        min_duration_sec: float = 10.0
    ) -> tuple[bool, float]:
        """
        Validate signal for cardiac analysis.
        
        Args:
            signal_data: Signal array to validate
            fs: Sampling frequency in Hz
            min_duration_sec: Minimum required duration in seconds
            
        Returns:
            Tuple of (is_valid, confidence_penalty)
        """
        duration_sec = len(signal_data) / fs
        
        if duration_sec < min_duration_sec:
            # Penalize confidence based on how short the signal is
            confidence_penalty = duration_sec / min_duration_sec
            return False, confidence_penalty
        
        # Check for flat/dead signal
        if np.std(signal_data) < 1e-6:
            return False, 0.1
        
        return True, 1.0
    
    def compute_signal_quality(
        self,
        signal_data: np.ndarray,
        fs: float
    ) -> float:
        """
        Compute signal quality score based on SNR.
        
        Args:
            signal_data: Preprocessed signal
            fs: Sampling frequency
            
        Returns:
            Quality score between 0 and 1
        """
        # FFT to get frequency content
        n = len(signal_data)
        freqs = np.fft.fftfreq(n, d=1 / fs)
        fft_vals = np.abs(np.fft.fft(signal_data))
        
        # Cardiac band (0.8-3 Hz)
        cardiac_mask = (freqs > 0.8) & (freqs < 3.0)
        # Noise band (>3 Hz only, excluding respiratory 0.1-0.5 Hz which contains valid signal)
        noise_mask = (freqs > 3.0) & (freqs < fs/2)
        
        if not np.any(cardiac_mask) or not np.any(noise_mask):
            return 0.5  # Default quality
        
        signal_power = np.mean(fft_vals[cardiac_mask] ** 2)
        noise_power = np.mean(fft_vals[noise_mask] ** 2)
        
        # SNR in dB, then map to 0-1 range
        if noise_power > 1e-6:
            snr_db = 10 * np.log10(signal_power / noise_power)
            # Map SNR: 0dB->0.3, 10dB->0.7, 20dB+->1.0
            quality = np.clip(0.3 + snr_db / 40, 0.0, 1.0)
        else:
            quality = 1.0
        
        return float(quality)
    
    def compute_proper_hrv(
        self, 
        signal_data: np.ndarray, 
        fs: float,
        is_preprocessed: bool = True
    ) -> float:
        """
        Compute proper HRV using peak detection and RMSSD.
        
        RMSSD = Root Mean Square of Successive Differences of RR intervals.
        This is the correct way to compute HRV, not spectral methods.
        
        Args:
            signal_data: Cardiac signal (should be preprocessed with bandpass filter)
            fs: Sampling frequency in Hz
            is_preprocessed: If True, assumes signal is already filtered (default: True)
            
        Returns:
            HRV in milliseconds (typical range: 20-80 ms)
        """
        try:
            # Ensure sufficient data
            if len(signal_data) < fs * 10:  # Need at least 10 seconds
                return 40.0  # Default physiological value
            
            # Apply bandpass filter only if not already preprocessed
            if is_preprocessed:
                filtered = signal_data
            else:
                sos = signal.butter(4, [0.8, 3.0], btype='band', fs=fs, output='sos')
                filtered = signal.sosfilt(sos, signal_data)
            
            # Detect peaks (R-peaks in ECG terminology, or pulse peaks)
            # Minimum distance = 0.4s (150 bpm max)
            min_distance = int(fs * 0.4)
            peaks, properties = signal.find_peaks(
                filtered,
                distance=min_distance,
                prominence=np.std(filtered) * 0.3  # Adaptive threshold
            )
            
            if len(peaks) < 3:
                return 40.0  # Need at least 3 peaks for 2 intervals
            
            # Calculate RR intervals in milliseconds
            rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
            
            # Filter out physiologically impossible intervals
            # Normal RR: 300-2000 ms (corresponding to 30-200 bpm)
            valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
            
            if len(valid_rr) < 2:
                return 40.0
            
            # Compute RMSSD: sqrt(mean(diff(RR)^2))
            successive_diffs = np.diff(valid_rr)
            rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
            
            # Clip to physiological range
            rmssd = np.clip(rmssd, 10, 150)
            
            return rmssd
            
        except Exception as e:
            logger.warning(f"HRV computation failed: {e}")
            return 40.0  # Safe default
    

    

    def _sample_roi_signals(
        self,
        face_frames: List[np.ndarray]
    ) -> Optional[Dict]:
        """
        Sample mean R, G, B from three face ROIs: forehead, left cheek, right cheek.
        Returns dict of {name: (R_arr, G_arr, B_arr)} or None if no valid data.
        """
        rois: Dict[str, tuple] = {
            "forehead": ([], [], []),
            "left":     ([], [], []),
            "right":    ([], [], []),
        }
        # ROI boxes: (row_start%, row_end%, col_start%, col_end%)
        roi_boxes = {
            "forehead": (0.05, 0.35, 0.20, 0.80),
            "left":     (0.50, 0.75, 0.55, 0.85),
            "right":    (0.50, 0.75, 0.15, 0.45),
        }
        for frame in face_frames:
            if frame is None or len(frame.shape) < 3:
                continue
            h, w = frame.shape[:2]
            if h < 10 or w < 10:
                continue
            for name, (r0, r1, c0, c1) in roi_boxes.items():
                patch = frame[int(h*r0):int(h*r1), int(w*c0):int(w*c1)]
                if patch.size == 0:
                    continue
                # OpenCV is BGR
                rois[name][0].append(float(np.mean(patch[:, :, 2])))
                rois[name][1].append(float(np.mean(patch[:, :, 1])))
                rois[name][2].append(float(np.mean(patch[:, :, 0])))

        result = {}
        for name, (rs, gs, bs) in rois.items():
            if len(rs) >= 30:
                result[name] = (np.array(rs), np.array(gs), np.array(bs))
        return result if result else None

    def _chrom_signal_from_roi(
        self,
        r_raw: np.ndarray,
        g_raw: np.ndarray,
        b_raw: np.ndarray,
        fps: float,
        window_sec: float = 2.0
    ) -> np.ndarray:
        """
        CHROM algorithm — De Haan & Jeanne (IEEE TBME, 2013).

        Per sliding 2-second Hann-windowed segment:
          Rn = R / mean(R);  Gn = G / mean(G);  Bn = B / mean(B)
          X = 3*Rn - 2*Gn   (coefficients sum to 1 → specular glare cancels)
          Y = 1.5*Rn + Gn - 1.5*Bn (same property)
          S = X - alpha * Y  where alpha = std(X) / std(Y)  (motion shadow cancels)
        Overlap-adds windows with Hann weighting into a full-length pulse signal.
        """
        n = len(r_raw)
        win = int(fps * window_sec)
        step = max(1, win // 2)
        pulse = np.zeros(n)
        weight = np.zeros(n)
        hann = np.hanning(win)

        for start in range(0, n - win + 1, step):
            end = start + win
            rn = r_raw[start:end] / (np.mean(r_raw[start:end]) + 1e-6)
            gn = g_raw[start:end] / (np.mean(g_raw[start:end]) + 1e-6)
            bn = b_raw[start:end] / (np.mean(b_raw[start:end]) + 1e-6)

            X = 3.0 * rn - 2.0 * gn          # sum of coefficients = 1
            Y = 1.5 * rn + gn - 1.5 * bn     # sum of coefficients = 1
            alpha_chrom = np.std(X) / (np.std(Y) + 1e-6)
            S = X - alpha_chrom * Y

            pulse[start:end] += S * hann
            weight[start:end] += hann

        valid = weight > 1e-6
        pulse[valid] /= weight[valid]
        # Bandpass 0.7–3.5 Hz (42–210 bpm) — slightly wider for robustness
        return self.preprocess_signal(pulse, fps, lowcut=0.7, highcut=3.5)

    def _pearson_r(self, a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation coefficient between two equal-length arrays."""
        if len(a) != len(b) or len(a) < 2:
            return 0.0
        a_z = a - np.mean(a)
        b_z = b - np.mean(b)
        denom = np.std(a_z) * np.std(b_z) * len(a)
        return float(np.sum(a_z * b_z) / denom) if denom > 1e-9 else 0.0

    def _extract_from_rppg(
        self,
        face_frames: List[np.ndarray],
        fps: float,
        biomarker_set: BiomarkerSet
    ) -> None:
        """
        Extract cardiovascular metrics from face video using CHROM rPPG.

        Replaces the old POS algorithm with:
          - Multi-ROI sampling (forehead + 2 cheeks)
          - CHROM chrominance projection (immune to glare & motion shadows)
          - Pearson cross-correlation fusion (discard noisy ROIs)
          - Cleaner pulse waveform fed into compute_proper_hrv for true HRV
        """
        min_frames = int(fps * 10)
        if len(face_frames) < min_frames:
            logger.warning(f"CHROM rPPG: {len(face_frames)} frames < {min_frames} needed. Skipping.")
            return

        try:
            # 1. Sample RGB time-series from each ROI
            roi_data = self._sample_roi_signals(face_frames)
            if not roi_data:
                logger.warning("CHROM rPPG: no valid ROI data from face frames.")
                return

            # 2. Compute CHROM pulse waveform per ROI
            roi_signals: Dict[str, np.ndarray] = {}
            for name, (r, g, b) in roi_data.items():
                sig = self._chrom_signal_from_roi(r, g, b, fps)
                if sig is not None and len(sig) > 0:
                    roi_signals[name] = sig

            if not roi_signals:
                logger.warning("CHROM rPPG: all ROI signals failed preprocessing.")
                return

            # 3. Multi-ROI Pearson fusion
            min_len = min(len(s) for s in roi_signals.values())
            aligned = {k: v[:min_len] for k, v in roi_signals.items()}
            names = list(aligned.keys())
            signals = [aligned[n] for n in names]

            if len(signals) == 1:
                fused = signals[0]
                n_used = 1
            else:
                # For each ROI, compute its mean Pearson r against peers
                avg_r = []
                for i, s in enumerate(signals):
                    peers = [signals[j] for j in range(len(signals)) if j != i]
                    avg_r.append(float(np.mean([self._pearson_r(s, p) for p in peers])))

                good_idx = [i for i, r in enumerate(avg_r) if r >= 0.4]
                if not good_idx:
                    good_idx = [int(np.argmax(avg_r))]  # take best even if low

                fused = np.mean([signals[i] for i in good_idx], axis=0)
                n_used = len(good_idx)
                logger.info(
                    f"CHROM rPPG: {n_used}/{len(names)} ROIs passed Pearson ≥0.4: "
                    f"{[names[i] for i in good_idx]}"
                )

            # 4. Signal quality gate
            is_valid, cf = self.validate_signal(fused, fps)
            quality = self.compute_signal_quality(fused, fps)
            base_confidence = 0.88 * quality * cf   # ceiling slightly above old POS 0.85

            if quality < 0.35:
                logger.warning(f"CHROM rPPG: quality {quality:.2f} too low — skipping biomarkers.")
                return

            # 5. Heart rate via FFT + parabolic interpolation
            n = len(fused)
            freqs = np.fft.fftfreq(n, d=1.0 / fps)
            fft_vals = np.abs(np.fft.fft(fused))

            cardiac_mask = (freqs > 0.8) & (freqs < 3.0)
            if not np.any(cardiac_mask):
                logger.warning("CHROM rPPG: no cardiac frequency band — skipping.")
                return

            cardiac_freqs = freqs[cardiac_mask]
            cardiac_power = fft_vals[cardiac_mask]
            peak_idx = int(np.argmax(cardiac_power))

            if 0 < peak_idx < len(cardiac_power) - 1:
                a_p = cardiac_power[peak_idx - 1]
                b_p = cardiac_power[peak_idx]
                c_p = cardiac_power[peak_idx + 1]
                denom_p = a_p - 2.0 * b_p + c_p
                if abs(denom_p) > 1e-9:
                    p_off = 0.5 * (a_p - c_p) / denom_p
                    peak_freq = cardiac_freqs[peak_idx] + p_off * (cardiac_freqs[1] - cardiac_freqs[0])
                else:
                    peak_freq = cardiac_freqs[peak_idx]
            else:
                peak_freq = cardiac_freqs[peak_idx]

            hr = float(np.clip(abs(peak_freq) * 60.0, 40.0, 200.0))

            self._add_biomarker(
                biomarker_set,
                name="heart_rate",
                value=hr,
                unit="bpm",
                confidence=float(np.clip(base_confidence, 0.30, 0.95)),
                normal_range=(60, 100),
                description=f"Heart rate from CHROM rPPG (webcam, {n_used} ROI{'s' if n_used > 1 else ''})"
            )

            # 6. HRV (RMSSD) — uses the clean CHROM pulse as peak-detection input
            hrv = self.compute_proper_hrv(fused, fps, is_preprocessed=True)
            self._add_biomarker(
                biomarker_set,
                name="hrv_rmssd",
                value=hrv,
                unit="ms",
                confidence=float(np.clip(base_confidence * 0.90, 0.22, 0.90)),
                normal_range=(20, 80),
                description="HRV (RMSSD) from CHROM rPPG beat-to-beat intervals"
            )

            logger.info(
                f"CHROM rPPG ✅  HR={hr:.1f}bpm  HRV={hrv:.1f}ms  "
                f"quality={quality:.2f}  ROIs={n_used}/{len(names)}  frames={len(face_frames)}"
            )

        except Exception as e:
            logger.error(f"CHROM rPPG extraction failed: {e}", exc_info=True)

    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:

        """
        Extract cardiovascular biomarkers.
        
        Expected data keys (in priority order):
        - vital_signs: Dict with direct vital measurements (highest confidence)
        - radar_data: 60GHz Radar breathing/heartbeat data (high confidence)
        - face_frames: List of face RGB frames for rPPG (85% accuracy)
        - ris_data: RIS signal array (75% confidence)
        - pose_sequence: Motion data for chest micro-motion (50% confidence)
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        # Priority 1: Direct vital signs (highest confidence ~95%)
        if "vital_signs" in data:
            self._extract_from_vitals(data["vital_signs"], biomarker_set)
            
        # Priority 1.5: Radar Data (Hardware Sensor)
        radar_hr_found = False
        if "radar_data" in data and "heart_rate" in data["radar_data"].get("radar", {}):
            radar = data["radar_data"]["radar"]
            self._add_biomarker(
                biomarker_set,
                name="heart_rate",
                value=float(radar["heart_rate"]),
                unit="bpm",
                confidence=0.90,
                normal_range=(60, 100),
                description="Heart rate from 60GHz Radar"
            )
            radar_hr_found = True
        elif "systems" in data:
             # Check for pre-processed radar heart rate
            for sys in data["systems"]:
                if sys.get("system") == "cardiovascular":
                    for bm in sys.get("biomarkers", []):
                        if bm["name"] == "heart_rate_radar":
                            self._add_biomarker(
                                biomarker_set,
                                name="heart_rate",
                                value=bm["value"],
                                unit="bpm",
                                confidence=0.90,
                                normal_range=(60, 100),
                                description="Heart rate from 60GHz Radar"
                            )
                            radar_hr_found = True
        
        # Quality-based cascading: Extract from all available sources, select best
        candidate_sources = []
        
        # Priority 2: rPPG from face frames (potential 85% accuracy)
        if "face_frames" in data:
            fps = data.get("fps", 30.0)
            temp_set = self._create_biomarker_set()
            self._extract_from_rppg(data["face_frames"], fps, temp_set)
            # Get quality from HR biomarker confidence
            hr_bm = next((bm for bm in temp_set.biomarkers if "heart_rate" in bm.name), None)
            if hr_bm:
                candidate_sources.append(("rppg", hr_bm.confidence, temp_set))
        
        # Priority 3: RIS-derived metrics (potential 75% confidence)
        if not radar_hr_found and "ris_data" in data:
            temp_set = self._create_biomarker_set()
            self._extract_from_ris(data["ris_data"], temp_set)
            hr_bm = next((bm for bm in temp_set.biomarkers if bm.name == "heart_rate"), None)
            if hr_bm:
                candidate_sources.append(("ris", hr_bm.confidence, temp_set))
        
        # Priority 4: Motion-derived (potential 50% confidence)
        if not radar_hr_found and "pose_sequence" in data:
            temp_set = self._create_biomarker_set()
            self._extract_from_motion(data["pose_sequence"], temp_set)
            hr_bm = next((bm for bm in temp_set.biomarkers if bm.name == "heart_rate"), None)
            if hr_bm and hr_bm.confidence > 0.4:
                candidate_sources.append(("motion", hr_bm.confidence, temp_set))
        
        # Select best quality source
        if candidate_sources and not radar_hr_found:
            # Sort by confidence (quality), select highest
            candidate_sources.sort(key=lambda x: x[1], reverse=True)
            best_source, best_quality, best_set = candidate_sources[0]
            logger.info(f"Selected {best_source} as primary source (quality={best_quality:.2f})")
            
            # Merge best source
            for bm in best_set.biomarkers:
                biomarker_set.add(bm)
            
            # Add secondary sources with renamed biomarkers
            for source_name, quality, temp_set in candidate_sources[1:]:
                for bm in temp_set.biomarkers:
                    if bm.name == "heart_rate":
                        bm.name = f"heart_rate_{source_name}"
                    biomarker_set.add(bm)
        
        # No fallback: Simulated values removed
        
        # NEW: Extract thermal asymmetry from thermal camera (ESP32)
        if "thermal_data" in data:
            self._extract_from_thermal(data["thermal_data"], biomarker_set)
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_thermal(
        self,
        thermal_data: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract cardiovascular biomarkers from thermal camera data."""
        
        # NOTE: thermal_asymmetry removed — MLX90640 32×24 resolution produces
        # cheek_asymmetry values of 3-5°C due to head pose noise, making it
        # unreliable as a CVD perfusion indicator at this sensor resolution.
        
        # Cheek temperatures for context
        left_cheek = thermal_data.get('left_cheek_temp')
        right_cheek = thermal_data.get('right_cheek_temp')
        if left_cheek is not None and right_cheek is not None:
            avg_cheek_temp = (left_cheek + right_cheek) / 2
            self._add_biomarker(
                biomarker_set,
                name="facial_perfusion_temp",
                value=float(avg_cheek_temp),
                unit="celsius",
                confidence=0.80,
                normal_range=(33.0, 36.0),
                description="Average cheek temperature (peripheral perfusion)"
            )
    
    def _extract_from_vitals(
        self,
        vitals: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract from direct vital sign measurements."""
        
        # Heart rate — only add if the key is actually present in vitals
        hr = vitals.get("heart_rate", vitals.get("heart_rate_bpm"))
        if hr is None:
            logger.warning("_extract_from_vitals: no heart_rate key found in vitals; skipping HR biomarker")
        else:
            self._add_biomarker(
                biomarker_set,
                name="heart_rate",
                value=float(hr),
                unit="bpm",
                confidence=0.95,
                normal_range=(60, 100),
                description="Resting heart rate"
            )
        
        # HRV if available
        if "hrv" in vitals or "rmssd" in vitals:
            hrv = vitals.get("hrv", vitals.get("rmssd"))
            self._add_biomarker(
                biomarker_set,
                name="hrv_rmssd",
                value=float(hrv),
                unit="ms",
                confidence=0.90,
                normal_range=(20, 80),
                description="Heart rate variability (RMSSD)"
            )
        elif hr is not None:
            # Only estimate HRV from HR if we actually have a real HR value.
            # This is a rough population-level estimate — confidence is low.
            estimated_hrv = self._estimate_hrv_from_hr(float(hr))
            self._add_biomarker(
                biomarker_set,
                name="hrv_rmssd",
                value=estimated_hrv,
                unit="ms",
                confidence=0.35,
                normal_range=(20, 80),
                description="HRV estimated from HR (population model, low confidence — no measured HRV available)"
            )
        else:
            logger.warning("_extract_from_vitals: no HR or HRV data available; skipping HRV biomarker")
        
        # Blood pressure if available
        if "systolic_bp" in vitals:
            self._add_biomarker(
                biomarker_set,
                name="systolic_bp",
                value=float(vitals["systolic_bp"]),
                unit="mmHg",
                confidence=0.95,
                normal_range=(90, 120),
                description="Systolic blood pressure"
            )
        if "diastolic_bp" in vitals:
            self._add_biomarker(
                biomarker_set,
                name="diastolic_bp",
                value=float(vitals["diastolic_bp"]),
                unit="mmHg",
                confidence=0.95,
                normal_range=(60, 80),
                description="Diastolic blood pressure"
            )
    
    def _extract_from_ris(
        self,
        ris_data: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract cardiovascular metrics from RIS bioimpedance data."""
        
        if ris_data.ndim == 1:
            ris_data = ris_data.reshape(-1, 1)
        
        # Use thoracic channels (0-7) for cardiac signal
        thoracic_signal = np.mean(ris_data[:, :min(8, ris_data.shape[1])], axis=1)
        
        # Preprocess and analyze cardiac signal
        if len(thoracic_signal) > 100:
            # Validate signal quality
            is_valid, confidence_factor = self.validate_signal(
                thoracic_signal, self.sample_rate, min_duration_sec=5.0
            )
            
            # Preprocess the signal
            processed = self.preprocess_signal(thoracic_signal, self.sample_rate)
            
            # Analyze for HR and HRV
            hr, hrv = self._analyze_cardiac_signal(processed)
            
            # Adjust confidence based on signal quality
            quality = self.compute_signal_quality(processed, self.sample_rate)
            base_confidence = 0.75 * quality * confidence_factor
        else:
            logger.warning("_extract_from_ris: RIS signal too short for cardiac analysis; skipping biomarkers")
            return
        
        self._add_biomarker(
            biomarker_set,
            name="heart_rate",
            value=hr,
            unit="bpm",
            confidence=0.75,
            normal_range=(60, 100),
            description="Heart rate derived from RIS"
        )
        
        self._add_biomarker(
            biomarker_set,
            name="hrv_rmssd",
            value=hrv,
            unit="ms",
            confidence=0.65,
            normal_range=(20, 80),
            description="HRV derived from RIS beat-to-beat intervals"
        )
        
        # Chest impedance for fluid status
        mean_impedance = float(np.mean(thoracic_signal))
        self._add_biomarker(
            biomarker_set,
            name="thoracic_impedance",
            value=mean_impedance,
            unit="ohms",
            confidence=0.70,
            normal_range=(400, 600),
            description="Mean thoracic bioimpedance (fluid proxy)"
        )
    
    def _extract_from_motion(
        self,
        pose_sequence: List[np.ndarray],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract cardiac proxies from chest micro-motion."""
        
        pose_array = np.array(pose_sequence)
        
        if pose_array.shape[0] < 30 or pose_array.shape[1] < 12:
            return
        
        # Chest landmarks: shoulders (11, 12)
        left_shoulder = pose_array[:, 11, :2]
        right_shoulder = pose_array[:, 12, :2]
        chest_center = (left_shoulder + right_shoulder) / 2
        
        # Analyze vertical micro-motion (breathing + cardiac)
        vertical_motion = chest_center[:, 1]
        
        # High-pass filter to isolate cardiac from respiratory
        if len(vertical_motion) > 60:
            # Simple differentiation as high-pass
            micro_motion = np.diff(vertical_motion)
            micro_motion_std = float(np.std(micro_motion))
        else:
            micro_motion_std = 0.002
        
        self._add_biomarker(
            biomarker_set,
            name="chest_micro_motion",
            value=micro_motion_std,
            unit="normalized_amplitude",
            confidence=0.55,
            normal_range=(0.001, 0.01),
            description="Chest wall micro-motion amplitude (cardiac proxy)"
        )
        
        # Estimate HR from motion frequency — only add if extraction succeeded
        hr = self._estimate_hr_from_motion(vertical_motion)
        if hr is not None:
            self._add_biomarker(
                biomarker_set,
                name="heart_rate",
                value=hr,
                unit="bpm",
                confidence=0.50,
                normal_range=(60, 100),
                description="Heart rate estimated from chest motion"
            )
        else:
            logger.warning("_extract_from_motion: could not estimate HR from motion signal; skipping biomarker")
    
    def _analyze_cardiac_signal(self, signal_data: np.ndarray) -> tuple:
        """
        Analyze signal for heart rate and HRV with parabolic interpolation.
        
        Returns:
            Tuple of (heart_rate_bpm, hrv_rmssd_ms)
        """
        # FFT for dominant frequency
        n = len(signal_data)
        freqs = np.fft.fftfreq(n, d=1 / self.sample_rate)
        fft_vals = np.abs(np.fft.fft(signal_data))
        
        # Look in cardiac range (0.8-3 Hz)
        cardiac_mask = (freqs > 0.8) & (freqs < 3.0)
        
        if not np.any(cardiac_mask):
            return 72.0, 45.0
        
        cardiac_freqs = freqs[cardiac_mask]
        cardiac_power = fft_vals[cardiac_mask]
        
        # Dominant frequency with parabolic interpolation for sub-bin accuracy
        peak_idx = np.argmax(cardiac_power)
        
        # Parabolic interpolation if peak is not at boundaries
        if 0 < peak_idx < len(cardiac_power) - 1:
            alpha = cardiac_power[peak_idx - 1]
            beta = cardiac_power[peak_idx]
            gamma = cardiac_power[peak_idx + 1]
            # Parabolic peak offset
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            # Interpolated frequency
            peak_freq = cardiac_freqs[peak_idx] + p * (cardiac_freqs[1] - cardiac_freqs[0])
        else:
            peak_freq = cardiac_freqs[peak_idx]
        
        hr = float(abs(peak_freq) * 60)  # Convert Hz to bpm
        hr = np.clip(hr, 40, 180)
        
        # Use proper HRV computation with peak detection (signal already preprocessed)
        hrv = self.compute_proper_hrv(signal_data, self.sample_rate, is_preprocessed=True)
        
        return hr, hrv
    
    def _estimate_hr_from_motion(self, motion_signal: np.ndarray, fps: float = 30.0) -> Optional[float]:
        """
        Estimate heart rate from motion signal using FFT peak detection.

        Args:
            motion_signal: Chest motion signal
            fps: Video frame rate (default 30)

        Returns:
            Estimated heart rate in bpm, or None if extraction fails.
        """
        if len(motion_signal) < fps * 5:  # Need at least 5 seconds
            logger.warning("_estimate_hr_from_motion: motion signal too short; cannot estimate HR")
            return None

        try:
            # Preprocess the signal with bandpass filter
            processed = self.preprocess_signal(motion_signal, fps, lowcut=0.8, highcut=3.0)

            # FFT for frequency analysis
            n = len(processed)
            freqs = np.fft.fftfreq(n, d=1/fps)
            fft_vals = np.abs(np.fft.fft(processed))

            # Find peak in cardiac range (0.8-3 Hz = 48-180 bpm)
            cardiac_mask = (freqs > 0.8) & (freqs < 3.0)

            if not np.any(cardiac_mask):
                logger.warning("_estimate_hr_from_motion: no cardiac frequency peak found")
                return None

            cardiac_freqs = freqs[cardiac_mask]
            cardiac_power = fft_vals[cardiac_mask]

            # Get dominant frequency
            peak_idx = np.argmax(cardiac_power)
            peak_freq = cardiac_freqs[peak_idx]
            hr = float(abs(peak_freq) * 60)

            return float(np.clip(hr, 50, 120))

        except Exception as e:
            logger.warning(f"_estimate_hr_from_motion: FFT failed: {e}")
            return None
    
    def _estimate_hrv_from_hr(self, hr: float) -> float:
        """
        Estimate HRV from heart rate using a deterministic population-level model.

        WARNING: This is a rough empirical approximation only. It must NOT be used
        as a substitute for measured HRV. Confidence must be kept low (≤0.35).
        Random noise has been intentionally removed — adding noise would hallucinate
        precision that does not exist.

        The inverse HR-HRV relationship is well-documented in literature
        (e.g., Bigger et al. 1992), but individual variation is very high.
        """
        # Deterministic inverse relationship: higher HR → lower HRV
        base_hrv = 60.0 - 0.5 * (hr - 60.0)
        return float(np.clip(base_hrv, 15.0, 90.0))
                           
                           
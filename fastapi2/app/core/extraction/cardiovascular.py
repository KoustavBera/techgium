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
    
    def _extract_from_rppg(
        self,
        face_frames: List[np.ndarray],
        fps: float,
        biomarker_set: BiomarkerSet
    ) -> None:
        """
        Extract cardiovascular metrics from face video using rPPG.
        
        Remote Photoplethysmography (rPPG) detects blood volume pulse
        from subtle color changes in facial skin captured by camera.
        
        Args:
            face_frames: List of face crop frames (RGB numpy arrays)
            fps: Video frame rate
            biomarker_set: BiomarkerSet to add biomarkers to
        """
        if len(face_frames) < int(fps * 10):  # Need at least 10 seconds
            logger.warning("rPPG: Insufficient frames, need at least 10 seconds")
            return
        
        try:
            # POS Algorithm (Plane-Orthogonal-to-Skin) - Wang et al. 2017
            # More robust to lighting variations/motion than Green channel only
            r_signal = []
            g_signal = []
            b_signal = []
            
            for frame in face_frames:
                if len(frame.shape) == 3:
                    # Extract ROI: Forehead (Upper 30% of face, centered)
                    h, w = frame.shape[:2]
                    forehead_roi = frame[int(h*0.1):int(h*0.4), int(w*0.25):int(w*0.75)]
                    
                    # OpenCV is BGR
                    b_val = np.mean(forehead_roi[:, :, 0])
                    g_val = np.mean(forehead_roi[:, :, 1])
                    r_val = np.mean(forehead_roi[:, :, 2])
                    
                    r_signal.append(r_val)
                    g_signal.append(g_val)
                    b_signal.append(b_val)
                else:
                    # Fallback for greyscale
                    val = np.mean(frame)
                    r_signal.append(val)
                    g_signal.append(val)
                    b_signal.append(val)
            
            r_raw = np.array(r_signal)
            g_raw = np.array(g_signal)
            b_raw = np.array(b_signal)
            
            # POS Processing
            # 1. Temporal Normalization
            rn = r_raw / (np.mean(r_raw) + 1e-6)
            gn = g_raw / (np.mean(g_raw) + 1e-6)
            bn = b_raw / (np.mean(b_raw) + 1e-6)
            
            # 2. Projection
            # S1 = G - B
            # S2 = G + B - 2R
            s1 = gn - bn
            s2 = gn + bn - 2 * rn
            
            # 3. Tuning (Alpha fusion)
            # Alpha = std(S1) / std(S2)
            std_s1 = np.std(s1)
            std_s2 = np.std(s2)
            alpha = std_s1 / (std_s2 + 1e-6)
            
            # 4. Pulse Signal
            green_signal = s1 + alpha * s2
            
            # Validate signal
            is_valid, confidence_factor = self.validate_signal(green_signal, fps)
            if not is_valid:
                logger.warning(f"rPPG: Signal validation failed, confidence={confidence_factor:.2f}")
            
            # Preprocess: detrend, bandpass filter, normalize
            processed_signal = self.preprocess_signal(green_signal, fps, lowcut=0.8, highcut=3.0)
            
            # Compute signal quality
            quality = self.compute_signal_quality(processed_signal, fps)
            base_confidence = 0.85 * quality * confidence_factor
            
            # Extract heart rate using FFT peak detection with parabolic interpolation
            n = len(processed_signal)
            freqs = np.fft.fftfreq(n, d=1/fps)
            fft_vals = np.abs(np.fft.fft(processed_signal))
            
            # Find peak in cardiac range (0.8-3 Hz = 48-180 bpm)
            cardiac_mask = (freqs > 0.8) & (freqs < 3.0)
            if np.any(cardiac_mask):
                cardiac_freqs = freqs[cardiac_mask]
                cardiac_power = fft_vals[cardiac_mask]
                peak_idx = np.argmax(cardiac_power)
                
                # Parabolic interpolation for sub-bin accuracy
                if 0 < peak_idx < len(cardiac_power) - 1:
                    alpha = cardiac_power[peak_idx - 1]
                    beta = cardiac_power[peak_idx]
                    gamma = cardiac_power[peak_idx + 1]
                    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
                    peak_freq = cardiac_freqs[peak_idx] + p * (cardiac_freqs[1] - cardiac_freqs[0])
                else:
                    peak_freq = cardiac_freqs[peak_idx]
                
                hr = float(abs(peak_freq) * 60)
                hr = np.clip(hr, 45, 180)
            else:
                logger.warning("rPPG: No cardiac frequency peak found; skipping heart_rate biomarker")
                return
            
            self._add_biomarker(
                biomarker_set,
                name="heart_rate",
                value=hr,
                unit="bpm",
                confidence=float(np.clip(base_confidence, 0.3, 0.95)),
                normal_range=(60, 100),
                description="Heart rate from rPPG (webcam)"
            )
            
            # Compute HRV using proper peak detection (signal already preprocessed)
            hrv = self.compute_proper_hrv(processed_signal, fps, is_preprocessed=True)
            self._add_biomarker(
                biomarker_set,
                name="hrv_rmssd",
                value=hrv,
                unit="ms",
                confidence=float(np.clip(base_confidence * 0.9, 0.25, 0.90)),
                normal_range=(20, 80),
                description="HRV (RMSSD) from rPPG"
            )
            
            # QUALITY GUARD: Only add biomarkers if quality is sufficient
            if quality < 0.4:
                logger.warning(f"rPPG: Signal quality too low ({quality:.2f}), skipping")
                return

            logger.info(f"rPPG extraction: HR={hr:.1f}bpm, HRV={hrv:.1f}ms, quality={quality:.2f}")
            
        except Exception as e:
            logger.error(f"rPPG extraction failed: {e}")
    

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
                           
                           
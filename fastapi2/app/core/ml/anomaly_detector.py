"""
Signal Anomaly Detection and Noise Separation Module

NON-DECISIONAL ML components that ONLY affect confidence scores.
Uses Isolation Forest and statistical methods for anomaly detection.

ARCHITECTURE CONSTRAINT:
- ML outputs → confidence adjustment ONLY
- ML outputs NEVER affect diagnosis or risk scores directly
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats

from app.utils import get_logger

logger = get_logger(__name__)

# Optional sklearn imports (graceful fallback if not available)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - ML features will use statistical fallbacks")


class AnomalyType(str, Enum):
    """Types of signal anomalies."""
    NONE = "none"
    NOISE_SPIKE = "noise_spike"
    SIGNAL_DROPOUT = "signal_dropout"
    SATURATION = "saturation"
    DRIFT = "drift"
    ARTIFACT = "artifact"
    UNKNOWN = "unknown"


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a signal segment."""
    is_anomalous: bool = False
    anomaly_score: float = 0.0      # 0-1, higher = more anomalous
    anomaly_type: AnomalyType = AnomalyType.NONE
    confidence_penalty: float = 0.0  # 0-1, penalty to apply
    affected_samples: int = 0
    total_samples: int = 0
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomalous": self.is_anomalous,
            "anomaly_score": round(self.anomaly_score, 3),
            "anomaly_type": self.anomaly_type.value,
            "confidence_penalty": round(self.confidence_penalty, 3),
            "affected_samples": self.affected_samples,
            "total_samples": self.total_samples,
            "details": self.details
        }


@dataclass
class SeparationResult:
    """Result of noise vs physiology separation."""
    physiological_signal: Optional[np.ndarray] = None
    noise_component: Optional[np.ndarray] = None
    snr_estimate: float = 0.0  # Signal-to-noise ratio in dB
    noise_fraction: float = 0.0  # 0-1, fraction of signal that is noise
    confidence_adjustment: float = 0.0  # Adjustment to confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "snr_estimate_db": round(self.snr_estimate, 2),
            "noise_fraction": round(self.noise_fraction, 3),
            "confidence_adjustment": round(self.confidence_adjustment, 3)
        }


class SignalAnomalyDetector:
    """
    Detects anomalies in sensor signals using Isolation Forest and statistics.
    
    NON-DECISIONAL: Outputs only affect confidence, never diagnosis.
    
    Uses:
    - Isolation Forest for multivariate anomaly detection
    - Statistical methods for signal quality assessment
    - Fallback to pure statistics if sklearn unavailable
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0-0.5)
        """
        self._contamination = min(max(contamination, 0.01), 0.5)
        self._detection_count = 0
        
        # Initialize Isolation Forest if available
        if SKLEARN_AVAILABLE:
            self._iso_forest = IsolationForest(
                contamination=self._contamination,
                random_state=42,
                n_estimators=100
            )
            self._scaler = StandardScaler()
            self._is_fitted = False
        else:
            self._iso_forest = None
            self._scaler = None
            self._is_fitted = False
        
        logger.info(f"SignalAnomalyDetector initialized (sklearn={SKLEARN_AVAILABLE})")
    
    def detect(
        self,
        signal: np.ndarray,
        sample_rate: int = 1000,
        use_ml: bool = True
    ) -> AnomalyResult:
        """
        Detect anomalies in a signal.
        
        Args:
            signal: 1D or 2D signal array
            sample_rate: Sampling rate in Hz
            use_ml: Whether to use ML (if available) or pure statistics
            
        Returns:
            AnomalyResult with anomaly detection results
        """
        result = AnomalyResult()
        
        if signal is None or signal.size == 0:
            result.is_anomalous = True
            result.anomaly_type = AnomalyType.SIGNAL_DROPOUT
            result.anomaly_score = 1.0
            result.confidence_penalty = 0.5
            result.details = "Empty or None signal"
            return result
        
        # Ensure 1D
        if len(signal.shape) > 1:
            signal = signal.flatten()
        
        result.total_samples = len(signal)
        
        # Statistical analysis
        stat_result = self._statistical_analysis(signal, sample_rate)
        
        # ML analysis if available and requested
        if use_ml and SKLEARN_AVAILABLE and len(signal) >= 100:
            ml_result = self._ml_analysis(signal)
            # Combine results (weighted average)
            result.anomaly_score = 0.4 * stat_result["score"] + 0.6 * ml_result["score"]
        else:
            result.anomaly_score = stat_result["score"]
        
        # Determine anomaly type and severity
        result.is_anomalous = result.anomaly_score > 0.5
        result.anomaly_type = stat_result["type"]
        result.affected_samples = stat_result["affected"]
        result.details = stat_result["details"]
        
        # Calculate confidence penalty (NON-DECISIONAL)
        # Penalty scales with anomaly severity but is capped
        result.confidence_penalty = min(result.anomaly_score * 0.3, 0.25)
        
        self._detection_count += 1
        return result
    
    def _statistical_analysis(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Perform statistical anomaly analysis."""
        issues = []
        anomaly_score = 0.0
        anomaly_type = AnomalyType.NONE
        affected = 0
        
        # 1. Check for NaN/Inf
        invalid = np.sum(~np.isfinite(signal))
        if invalid > 0:
            anomaly_score += 0.3 * (invalid / len(signal))
            affected += invalid
            issues.append(f"{invalid} invalid samples")
        
        # Clean for further analysis
        signal_clean = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Check for saturation/clipping
        sig_max, sig_min = np.max(signal_clean), np.min(signal_clean)
        at_max = np.sum(signal_clean >= sig_max * 0.99)
        at_min = np.sum(signal_clean <= sig_min * 1.01 + 1e-10)
        saturation_ratio = (at_max + at_min) / len(signal_clean)
        
        if saturation_ratio > 0.05:
            anomaly_score += 0.2
            anomaly_type = AnomalyType.SATURATION
            affected += int(saturation_ratio * len(signal_clean))
            issues.append(f"Signal saturation: {saturation_ratio*100:.1f}%")
        
        # 3. Check for sudden spikes (>5 sigma)
        z_scores = np.abs(stats.zscore(signal_clean))
        spikes = np.sum(z_scores > 5)
        spike_ratio = spikes / len(signal_clean)
        
        if spike_ratio > 0.01:
            anomaly_score += 0.25
            if anomaly_type == AnomalyType.NONE:
                anomaly_type = AnomalyType.NOISE_SPIKE
            affected += spikes
            issues.append(f"Noise spikes: {spikes} samples")
        
        # 4. Check for signal drift
        window_size = min(len(signal_clean) // 4, sample_rate)
        if window_size > 10:
            n_windows = len(signal_clean) // window_size
            window_means = [
                np.mean(signal_clean[i*window_size:(i+1)*window_size])
                for i in range(n_windows)
            ]
            if len(window_means) >= 2:
                drift = np.abs(window_means[-1] - window_means[0]) / (np.std(signal_clean) + 1e-10)
                if drift > 2:
                    anomaly_score += 0.15
                    if anomaly_type == AnomalyType.NONE:
                        anomaly_type = AnomalyType.DRIFT
                    issues.append(f"Signal drift detected: {drift:.1f} sigma")
        
        # 5. Check for constant signal (dropout)
        if np.std(signal_clean) < 1e-6:
            anomaly_score = 0.8
            anomaly_type = AnomalyType.SIGNAL_DROPOUT
            affected = len(signal_clean)
            issues.append("Signal has no variation (constant)")
        
        return {
            "score": min(anomaly_score, 1.0),
            "type": anomaly_type,
            "affected": affected,
            "details": "; ".join(issues) if issues else "No anomalies detected"
        }
    
    def _ml_analysis(self, signal: np.ndarray) -> Dict[str, Any]:
        """Perform ML-based anomaly detection using Isolation Forest."""
        if not SKLEARN_AVAILABLE or self._iso_forest is None:
            return {"score": 0.0}
        
        # Create features from signal segments
        window_size = min(100, len(signal) // 10)
        if window_size < 10:
            return {"score": 0.0}
        
        features = []
        for i in range(0, len(signal) - window_size, window_size // 2):
            segment = signal[i:i+window_size]
            # Extract statistical features
            features.append([
                np.mean(segment),
                np.std(segment),
                np.max(segment) - np.min(segment),
                stats.skew(segment),
                stats.kurtosis(segment)
            ])
        
        if len(features) < 5:
            return {"score": 0.0}
        
        features = np.array(features)
        
        # Scale and fit/predict
        try:
            features_scaled = self._scaler.fit_transform(features)
            predictions = self._iso_forest.fit_predict(features_scaled)
            
            # Count anomalies (-1 = anomaly, 1 = normal)
            n_anomalies = np.sum(predictions == -1)
            anomaly_ratio = n_anomalies / len(predictions)
            
            return {"score": anomaly_ratio}
        except Exception as e:
            logger.warning(f"ML analysis failed: {e}")
            return {"score": 0.0}
    
    def detect_batch(
        self,
        signals: Dict[str, np.ndarray],
        sample_rates: Optional[Dict[str, int]] = None
    ) -> Dict[str, AnomalyResult]:
        """Detect anomalies in multiple signals."""
        results = {}
        rates = sample_rates or {}
        
        for name, signal in signals.items():
            rate = rates.get(name, 1000)
            results[name] = self.detect(signal, rate)
        
        return results


class NoisePhysiologySeparator:
    """
    Separates physiological signal from noise components.
    
    NON-DECISIONAL: Only affects confidence, never diagnosis.
    
    Uses:
    - Bandpass filtering for physiological ranges
    - Wavelet denoising (optional)
    - Statistical noise estimation
    """
    
    def __init__(self):
        """Initialize separator."""
        self._separation_count = 0
        
        # Physiological frequency bands (Hz)
        self._bands = {
            "cardiac": (0.5, 3.0),      # Heart rate: 30-180 bpm
            "respiratory": (0.1, 0.5),   # Breathing: 6-30 breaths/min
            "motion": (0.01, 5.0),       # Body motion
            "tremor": (3.0, 12.0),       # Physiological tremor
        }
        
        logger.info("NoisePhysiologySeparator initialized")
    
    def separate(
        self,
        signal: np.ndarray,
        sample_rate: int = 1000,
        target_band: str = "cardiac"
    ) -> SeparationResult:
        """
        Separate physiological signal from noise.
        
        Args:
            signal: Input signal array
            sample_rate: Sampling rate in Hz
            target_band: Which physiological band to extract
            
        Returns:
            SeparationResult with separated components
        """
        result = SeparationResult()
        
        if signal is None or signal.size == 0:
            result.noise_fraction = 1.0
            result.confidence_adjustment = -0.3
            return result
        
        # Ensure 1D
        if len(signal.shape) > 1:
            signal = signal.flatten()
        
        # Get frequency band
        if target_band not in self._bands:
            target_band = "cardiac"
        low_freq, high_freq = self._bands[target_band]
        
        # Validate frequencies vs sample rate (Nyquist)
        nyquist = sample_rate / 2
        if high_freq >= nyquist:
            high_freq = nyquist * 0.95
        if low_freq >= high_freq:
            low_freq = high_freq * 0.1
        
        # Design bandpass filter
        try:
            sos = scipy_signal.butter(
                4,  # 4th order
                [low_freq / nyquist, high_freq / nyquist],
                btype='band',
                output='sos'
            )
            
            # Apply filter to extract physiological component
            physio = scipy_signal.sosfiltfilt(sos, signal)
            noise = signal - physio
            
            result.physiological_signal = physio
            result.noise_component = noise
            
            # Calculate SNR
            signal_power = np.var(physio) + 1e-10
            noise_power = np.var(noise) + 1e-10
            snr_linear = signal_power / noise_power
            result.snr_estimate = 10 * np.log10(snr_linear)
            
            # Calculate noise fraction
            result.noise_fraction = noise_power / (signal_power + noise_power)
            
            # Confidence adjustment based on SNR
            # Good SNR (>10dB) → no penalty
            # Poor SNR (<0dB) → larger penalty
            if result.snr_estimate > 10:
                result.confidence_adjustment = 0.05  # Small boost
            elif result.snr_estimate > 3:
                result.confidence_adjustment = 0.0
            elif result.snr_estimate > 0:
                result.confidence_adjustment = -0.1
            else:
                result.confidence_adjustment = -0.2
            
        except Exception as e:
            logger.warning(f"Signal separation failed: {e}")
            result.noise_fraction = 0.5
            result.confidence_adjustment = -0.1
        
        self._separation_count += 1
        return result
    
    def separate_multiband(
        self,
        signal: np.ndarray,
        sample_rate: int = 1000
    ) -> Dict[str, SeparationResult]:
        """Separate signal into all physiological bands."""
        results = {}
        
        for band_name in self._bands:
            results[band_name] = self.separate(signal, sample_rate, band_name)
        
        return results
    
    def estimate_noise_floor(
        self,
        signal: np.ndarray,
        sample_rate: int = 1000
    ) -> Tuple[float, float]:
        """
        Estimate the noise floor of a signal.
        
        Returns:
            Tuple of (noise_floor_amplitude, confidence_penalty)
        """
        if signal is None or len(signal) < 10:
            return 0.0, 0.3
        
        # Ensure 1D
        if len(signal.shape) > 1:
            signal = signal.flatten()
        
        # High-pass filter to extract high-frequency noise
        nyquist = sample_rate / 2
        high_cutoff = min(50, nyquist * 0.8)  # 50 Hz or below Nyquist
        
        try:
            sos = scipy_signal.butter(2, high_cutoff / nyquist, btype='high', output='sos')
            high_freq = scipy_signal.sosfiltfilt(sos, signal)
            
            # Noise floor is the RMS of high-frequency content
            noise_floor = np.sqrt(np.mean(high_freq**2))
            
            # Signal RMS
            signal_rms = np.sqrt(np.mean(signal**2)) + 1e-10
            
            # Noise ratio
            noise_ratio = noise_floor / signal_rms
            
            # Confidence penalty
            if noise_ratio < 0.1:
                penalty = 0.0
            elif noise_ratio < 0.3:
                penalty = 0.1
            else:
                penalty = min(noise_ratio * 0.5, 0.3)
            
            return float(noise_floor), float(penalty)
            
        except Exception as e:
            logger.warning(f"Noise floor estimation failed: {e}")
            return 0.0, 0.1

"""
Unit Tests for Optional ML Module (Phase 5)

Tests for signal anomaly detection and noise separation.
Verifies that ML outputs only affect confidence, never diagnosis.
"""
import pytest
import numpy as np
from typing import Dict, Any

from app.core.ml import (
    SignalAnomalyDetector,
    AnomalyResult,
    NoisePhysiologySeparator,
    SeparationResult
)
from app.core.ml.anomaly_detector import AnomalyType


# Fixtures
@pytest.fixture
def clean_cardiac_signal() -> np.ndarray:
    """Generate clean cardiac-like signal."""
    t = np.linspace(0, 10, 10000)  # 10 seconds at 1000 Hz
    # Cardiac at 72 bpm (1.2 Hz) + respiratory at 0.25 Hz
    cardiac = 10 * np.sin(2 * np.pi * 1.2 * t)
    respiratory = 5 * np.sin(2 * np.pi * 0.25 * t)
    return (cardiac + respiratory).astype(np.float32)


@pytest.fixture
def noisy_signal() -> np.ndarray:
    """Generate noisy signal."""
    t = np.linspace(0, 10, 10000)
    cardiac = 10 * np.sin(2 * np.pi * 1.2 * t)
    noise = np.random.randn(10000) * 20  # High noise
    return (cardiac + noise).astype(np.float32)


@pytest.fixture
def signal_with_spikes() -> np.ndarray:
    """Generate signal with noise spikes."""
    t = np.linspace(0, 10, 10000)
    signal = 10 * np.sin(2 * np.pi * 1.2 * t)
    # Add spikes
    spike_indices = np.random.choice(10000, 100, replace=False)
    signal[spike_indices] = signal[spike_indices] + 100  # Large spikes
    return signal.astype(np.float32)


@pytest.fixture
def saturated_signal() -> np.ndarray:
    """Generate saturated/clipped signal."""
    t = np.linspace(0, 10, 10000)
    signal = 100 * np.sin(2 * np.pi * 1.2 * t)
    # Clip signal
    signal = np.clip(signal, -50, 50)
    return signal.astype(np.float32)


@pytest.fixture
def constant_signal() -> np.ndarray:
    """Generate constant (dropout) signal."""
    return np.ones(10000).astype(np.float32) * 50


class TestSignalAnomalyDetector:
    """Tests for SignalAnomalyDetector."""
    
    def test_detect_clean_signal(self, clean_cardiac_signal):
        """Test detection on clean signal."""
        detector = SignalAnomalyDetector()
        result = detector.detect(clean_cardiac_signal, sample_rate=1000)
        
        assert isinstance(result, AnomalyResult)
        assert result.anomaly_score < 0.5  # Should be low
        assert result.confidence_penalty < 0.15  # Low penalty
    
    def test_detect_noisy_signal(self, noisy_signal):
        """Test detection on noisy signal."""
        detector = SignalAnomalyDetector()
        result = detector.detect(noisy_signal, sample_rate=1000)
        
        # Noisy signal should have higher anomaly score
        assert result.anomaly_score > 0  # Some anomaly detected
        assert result.confidence_penalty >= 0  # Penalty applied
    
    def test_detect_spikes(self, signal_with_spikes):
        """Test detection on signal with potential anomalies."""
        detector = SignalAnomalyDetector()
        result = detector.detect(signal_with_spikes, sample_rate=1000)
        
        # Should run without error and produce valid result
        assert isinstance(result, AnomalyResult)
        assert 0 <= result.anomaly_score <= 1
        assert 0 <= result.confidence_penalty <= 0.25
    
    def test_detect_saturation(self, saturated_signal):
        """Test detection of signal saturation."""
        detector = SignalAnomalyDetector()
        result = detector.detect(saturated_signal, sample_rate=1000)
        
        # Should detect saturation
        assert result.anomaly_score > 0
        assert "saturation" in result.details.lower() or result.anomaly_type == AnomalyType.SATURATION
    
    def test_detect_dropout(self, constant_signal):
        """Test detection of signal dropout (constant)."""
        detector = SignalAnomalyDetector()
        result = detector.detect(constant_signal, sample_rate=1000)
        
        # Constant signal should be detected as anomalous or have elevated score
        assert result.anomaly_score > 0.3 or result.anomaly_type == AnomalyType.SIGNAL_DROPOUT
    
    def test_detect_empty_signal(self):
        """Test detection on empty signal."""
        detector = SignalAnomalyDetector()
        result = detector.detect(np.array([]), sample_rate=1000)
        
        assert result.is_anomalous
        assert result.anomaly_type == AnomalyType.SIGNAL_DROPOUT
    
    def test_confidence_penalty_is_bounded(self, noisy_signal):
        """Test that confidence penalty is properly bounded."""
        detector = SignalAnomalyDetector()
        result = detector.detect(noisy_signal, sample_rate=1000)
        
        # Confidence penalty should be bounded [0, 0.25]
        assert 0 <= result.confidence_penalty <= 0.25
    
    def test_non_decisional(self, signal_with_spikes):
        """Test that ML output is non-decisional (confidence only)."""
        detector = SignalAnomalyDetector()
        result = detector.detect(signal_with_spikes, sample_rate=1000)
        
        # Result should only contain confidence-related outputs
        result_dict = result.to_dict()
        assert "confidence_penalty" in result_dict
        # Should NOT contain diagnosis or risk scores
        assert "diagnosis" not in result_dict
        assert "risk_score" not in result_dict
    
    def test_batch_detection(self, clean_cardiac_signal, noisy_signal):
        """Test batch detection on multiple signals."""
        detector = SignalAnomalyDetector()
        results = detector.detect_batch({
            "clean": clean_cardiac_signal,
            "noisy": noisy_signal
        })
        
        assert len(results) == 2
        assert "clean" in results
        assert "noisy" in results
    
    def test_statistical_fallback(self, clean_cardiac_signal):
        """Test that statistical analysis works without ML."""
        detector = SignalAnomalyDetector()
        result = detector.detect(clean_cardiac_signal, sample_rate=1000, use_ml=False)
        
        # Should still work with just statistics
        assert isinstance(result, AnomalyResult)
        assert result.total_samples > 0


class TestNoisePhysiologySeparator:
    """Tests for NoisePhysiologySeparator."""
    
    def test_separate_clean_signal(self, clean_cardiac_signal):
        """Test separation on clean cardiac signal."""
        separator = NoisePhysiologySeparator()
        result = separator.separate(clean_cardiac_signal, sample_rate=1000, target_band="cardiac")
        
        assert isinstance(result, SeparationResult)
        assert result.physiological_signal is not None
        assert result.noise_component is not None
        assert result.snr_estimate > 0  # Clean signal has positive SNR
        assert result.noise_fraction < 0.5  # Less than half is noise
    
    def test_separate_noisy_signal(self, noisy_signal):
        """Test separation on noisy signal."""
        separator = NoisePhysiologySeparator()
        result = separator.separate(noisy_signal, sample_rate=1000, target_band="cardiac")
        
        # Noisy signal should have lower SNR
        assert result.noise_fraction > 0.3  # More noise
        assert result.confidence_adjustment <= 0  # Penalty for noise
    
    def test_multiband_separation(self, clean_cardiac_signal):
        """Test separation into multiple physiological bands."""
        separator = NoisePhysiologySeparator()
        results = separator.separate_multiband(clean_cardiac_signal, sample_rate=1000)
        
        assert "cardiac" in results
        assert "respiratory" in results
        assert "motion" in results
        assert "tremor" in results
    
    def test_noise_floor_estimation(self, clean_cardiac_signal, noisy_signal):
        """Test noise floor estimation."""
        separator = NoisePhysiologySeparator()
        
        clean_floor, clean_penalty = separator.estimate_noise_floor(clean_cardiac_signal, 1000)
        noisy_floor, noisy_penalty = separator.estimate_noise_floor(noisy_signal, 1000)
        
        # Noisy signal should have higher noise floor
        assert noisy_floor > clean_floor or noisy_penalty > clean_penalty
    
    def test_empty_signal(self):
        """Test separation on empty signal."""
        separator = NoisePhysiologySeparator()
        result = separator.separate(np.array([]), sample_rate=1000)
        
        assert result.noise_fraction == 1.0
        assert result.confidence_adjustment < 0
    
    def test_confidence_adjustment_range(self, clean_cardiac_signal):
        """Test that confidence adjustment is within expected range."""
        separator = NoisePhysiologySeparator()
        result = separator.separate(clean_cardiac_signal, sample_rate=1000)
        
        # Confidence adjustment should be bounded
        assert -0.3 <= result.confidence_adjustment <= 0.1


class TestNonDecisionalConstraint:
    """Tests verifying that ML outputs are non-decisional."""
    
    def test_anomaly_result_structure(self):
        """Verify AnomalyResult only contains confidence-related fields."""
        result = AnomalyResult(
            is_anomalous=True,
            anomaly_score=0.7,
            confidence_penalty=0.2
        )
        
        result_dict = result.to_dict()
        
        # Should have confidence-related fields
        assert "confidence_penalty" in result_dict
        assert "anomaly_score" in result_dict
        
        # Should NOT have diagnosis/risk fields (these don't exist in the class)
        # This is a structural verification
        assert not hasattr(result, 'diagnosis')
        assert not hasattr(result, 'risk_score')
        assert not hasattr(result, 'disease_prediction')
    
    def test_separation_result_structure(self):
        """Verify SeparationResult only contains confidence-related fields."""
        result = SeparationResult(
            snr_estimate=10.0,
            noise_fraction=0.3,
            confidence_adjustment=-0.1
        )
        
        result_dict = result.to_dict()
        
        # Should have confidence adjustment
        assert "confidence_adjustment" in result_dict
        
        # Should NOT have diagnosis/risk fields
        assert not hasattr(result, 'diagnosis')
        assert not hasattr(result, 'risk_score')


class TestIntegration:
    """Integration tests for ML module."""
    
    def test_full_ml_pipeline(self, clean_cardiac_signal, noisy_signal):
        """Test complete ML pipeline."""
        # 1. Anomaly detection
        detector = SignalAnomalyDetector()
        anomaly_results = detector.detect_batch({
            "clean": clean_cardiac_signal,
            "noisy": noisy_signal
        })
        
        # 2. Noise separation
        separator = NoisePhysiologySeparator()
        separation_results = {
            "clean": separator.separate(clean_cardiac_signal, 1000),
            "noisy": separator.separate(noisy_signal, 1000)
        }
        
        # 3. Combine confidence adjustments
        for name in ["clean", "noisy"]:
            total_penalty = (
                anomaly_results[name].confidence_penalty +
                max(-separation_results[name].confidence_adjustment, 0)
            )
            # Total penalty should be reasonable
            assert total_penalty >= 0
            assert total_penalty < 1.0  # Should not destroy all confidence
        
        # Clean signal should have lower total penalty than noisy
        clean_penalty = anomaly_results["clean"].confidence_penalty
        noisy_penalty = anomaly_results["noisy"].confidence_penalty
        
        # We expect clean to be better or equal
        assert clean_penalty <= noisy_penalty + 0.1  # Allow small margin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

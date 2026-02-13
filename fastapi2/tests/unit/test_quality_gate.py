
import pytest
from unittest.mock import MagicMock, patch
from app.services.screening import ScreeningService
from app.core.validation.signal_quality import ModalityQualityScore, Modality

@pytest.mark.asyncio
async def test_quality_gate_low_quality():
    """Test that quality_score below 0.5 triggers rejection."""
    service = ScreeningService()
    
    # Mock assessor to return low quality
    mock_score = ModalityQualityScore(modality=Modality.CAMERA, overall_quality=0.3)
    service.quality_assessor.assess_camera = MagicMock(return_value=mock_score)
    
    # Raw data payload
    data = {"camera": {"frames": [1, 2, 3]}}
    
    score = await service.assess_data_quality(data, [])
    assert score == 0.3
    assert score < 0.5

@pytest.mark.asyncio
async def test_quality_gate_high_quality():
    """Test that quality_score above 0.5 passes."""
    service = ScreeningService()
    
    # Mock assessor to return high quality
    mock_score = ModalityQualityScore(modality=Modality.CAMERA, overall_quality=0.8)
    service.quality_assessor.assess_camera = MagicMock(return_value=mock_score)
    
    # Raw data payload
    data = {"camera": {"frames": [1, 2, 3]}}
    
    score = await service.assess_data_quality(data, [])
    assert score == 0.8
    assert score >= 0.5

@pytest.mark.asyncio
async def test_quality_gate_fallback():
    """Test fallback to biomarker confidence when raw data is missing."""
    service = ScreeningService()
    
    # Mock system input with biomarkers
    systems_input = [
        {
            "system": "cardiovascular",
            "biomarkers": [
                {"name": "hr", "value": 70, "confidence": 0.9},
                {"name": "rr", "value": 16, "confidence": 0.7}
            ]
        }
    ]
    
    score = await service.assess_data_quality(None, systems_input)
    assert score == 0.8 # Average of 0.9 and 0.7

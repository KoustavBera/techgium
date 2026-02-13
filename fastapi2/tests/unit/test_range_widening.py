import pytest
from app.core.validation.biomarker_plausibility import BiomarkerPlausibilityValidator
from app.core.extraction.base import Biomarker, BiomarkerSet, PhysiologicalSystem

def test_thermal_asymmetry_widened_range():
    """Test that thermal_asymmetry now accepts up to 3.5°C deviation."""
    validator = BiomarkerPlausibilityValidator()
    
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    biomarker_set.add(Biomarker(
        name="thermal_asymmetry",
        value=2.5,  # Previously would fail, now should pass
        unit="celsius",
        confidence=0.9,
        normal_range=(0.0, 3.5)
    ))
    
    result = validator.validate(biomarker_set)
    
    # Should pass physiological validation (2.5 < 3.5 upper bound)
    violations = [v for v in result.violations if v.biomarker_name == "thermal_asymmetry"]
    assert len(violations) == 0 or all(v.severity < 0.8 for v in violations)

def test_skin_temperature_environmental_tolerance():
    """Test that skin_temperature now tolerates cooler readings from cold environments."""
    validator = BiomarkerPlausibilityValidator()
    
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    biomarker_set.add(Biomarker(
        name="skin_temperature",
        value=33.5,  # Previously would fail at 33.5°C (below 35.0), now should pass
        unit="celsius",
        confidence=0.85,
        normal_range=(33.0, 39.0)
    ))
    
    result = validator.validate(biomarker_set)
    
    # Should pass physiological validation (33.5 is within 33.0-39.0)
    violations = [v for v in result.violations if v.biomarker_name == "skin_temperature"]
    assert len(violations) == 0 or all(v.severity < 0.8 for v in violations)

def test_inflammation_index_widened_upper_bound():
    """Test that inflammation_index tolerates higher values from ambient temperature."""
    validator = BiomarkerPlausibilityValidator()
    
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    biomarker_set.add(Biomarker(
        name="inflammation_index",
        value=18.0,  # Previously would flag at 16%, now should pass up to 20%
        unit="percent",
        confidence=0.7,
        normal_range=(0.0, 20.0)
    ))
    
    result = validator.validate(biomarker_set)
    
    # Should pass physiological validation (18.0 < 20.0)
    violations = [v for v in result.violations if v.biomarker_name == "inflammation_index"]
    assert len(violations) == 0 or all(v.severity < 0.8 for v in violations)

def test_skin_redness_webcam_noise_tolerance():
    """Test that skin_redness tolerates webcam sensor noise at lower/higher ends."""
    validator = BiomarkerPlausibilityValidator()
    
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    
    # Test lower bound (previously 0.1, now 0.05)
    biomarker_set.add(Biomarker(
        name="skin_redness",
        value=0.07,
        unit="normalized",
        confidence=0.8,
        normal_range=(0.05, 0.95)
    ))
    
    result_low = validator.validate(biomarker_set)
    violations_low = [v for v in result_low.violations if v.biomarker_name == "skin_redness"]
    assert len(violations_low) == 0 or all(v.severity < 0.8 for v in violations_low)
    
    # Test upper bound (previously 0.9, now 0.95)
    biomarker_set2 = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    biomarker_set2.add(Biomarker(
        name="skin_redness",
        value=0.93,
        unit="normalized",
        confidence=0.8,
        normal_range=(0.05, 0.95)
    ))
    
    result_high = validator.validate(biomarker_set2)
    violations_high = [v for v in result_high.violations if v.biomarker_name == "skin_redness"]
    assert len(violations_high) == 0 or all(v.severity < 0.8 for v in violations_high)

def test_color_uniformity_lighting_tolerance():
    """Test that color_uniformity accepts lower values from non-ideal lighting."""
    validator = BiomarkerPlausibilityValidator()
    
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    biomarker_set.add(Biomarker(
        name="color_uniformity",
        value=0.18,  # Previously would flag at 0.2, now should pass down to 0.15
        unit="score",
        confidence=0.75,
        normal_range=(0.15, 1.0)
    ))
    
    result = validator.validate(biomarker_set)
    
    # Should pass physiological validation (0.18 > 0.15 lower bound)
    violations = [v for v in result.violations if v.biomarker_name == "color_uniformity"]
    assert len(violations) == 0 or all(v.severity < 0.8 for v in violations)

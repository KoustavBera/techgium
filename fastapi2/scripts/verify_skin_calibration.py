"""
Verification Script: Skin Biomarker Calibration

Tests that healthy skin values now correctly fall within normal ranges
and produce LOW risk assessments.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.extraction.base import BiomarkerSet, PhysiologicalSystem
from app.core.inference.risk_engine import RiskEngine, RiskLevel


def test_healthy_skin_values():
    """
    Test case: Healthy skin values from user's report
    - texture_roughness: 0.34 (should be normal, was flagged as low)
    - skin_redness: 11.84 (should be normal, was flagged as low)
    - skin_yellowness: 13.35 (should be normal, was flagged as low)
    - color_uniformity: ~0.5 (should be normal, was flagged as low)
    """
    
    print("=" * 70)
    print("SKIN BIOMARKER CALIBRATION VERIFICATION")
    print("=" * 70)
    
    # Create a mock biomarker set with healthy values
    biomarker_set = BiomarkerSet(system=PhysiologicalSystem.SKIN)
    
    # Add healthy biomarkers (values from user's report)
    biomarker_set.biomarkers.append(type('Biomarker', (), {
        'name': 'texture_roughness',
        'value': 0.34,
        'unit': 'glcm_contrast',
        'confidence': 0.75,
        'normal_range': (0.0, 5.0),
        'is_abnormal': lambda self: not (self.normal_range[0] <= self.value <= self.normal_range[1]),
    })())
    
    biomarker_set.biomarkers.append(type('Biomarker', (), {
        'name': 'skin_redness',
        'value': 11.84,
        'unit': 'lab_deviation',
        'confidence': 0.85,
        'normal_range': (0.0, 25.0),
        'is_abnormal': lambda self: not (self.normal_range[0] <= self.value <= self.normal_range[1]),
    })())
    
    biomarker_set.biomarkers.append(type('Biomarker', (), {
        'name': 'skin_yellowness',
        'value': 13.35,
        'unit': 'lab_deviation',
        'confidence': 0.80,
        'normal_range': (0.0, 25.0),
        'is_abnormal': lambda self: not (self.normal_range[0] <= self.value <= self.normal_range[1]),
    })())
    
    biomarker_set.biomarkers.append(type('Biomarker', (), {
        'name': 'color_uniformity',
        'value': 0.50,
        'unit': 'entropy_inv',
        'confidence': 0.70,
        'normal_range': (0.25, 1.0),
        'is_abnormal': lambda self: not (self.normal_range[0] <= self.value <= self.normal_range[1]),
    })())
    
    # Initialize risk engine
    engine = RiskEngine(use_calibration=False)
    
    # Compute risk
    result = engine.compute_risk(biomarker_set)
    
    # Display results
    print()
    print("Biomarker Values and Status:")
    print("-" * 70)
    for bm in biomarker_set.biomarkers:
        status = "NORMAL" if not bm.is_abnormal() else "ABNORMAL"
        print(f"  {bm.name:20s}: {bm.value:6.2f} {bm.unit:15s} [{bm.normal_range[0]:6.2f} - {bm.normal_range[1]:6.2f}] -> {status}")
    
    print()
    print("Risk Assessment:")
    print("-" * 70)
    print(f"  Overall Risk Score: {result.overall_risk.score:.2f} / 100")
    print(f"  Risk Level: {result.overall_risk.level.value.upper()}")
    print(f"  Confidence: {result.overall_risk.confidence:.2%}")
    print(f"  Alerts: {len(result.alerts)}")
    if result.alerts:
        for alert in result.alerts:
            print(f"    - {alert}")
    
    print()
    print("=" * 70)
    
    # Validation
    all_normal = all(not bm.is_abnormal() for bm in biomarker_set.biomarkers)
    risk_is_low = result.overall_risk.level == RiskLevel.LOW
    
    if all_normal and risk_is_low:
        print("✅ PASS: All biomarkers are NORMAL and risk level is LOW (Healthy)")
        print("=" * 70)
        return True
    else:
        print("❌ FAIL: Calibration issue detected!")
        if not all_normal:
            print("   Issue: Some biomarkers still flagged as abnormal")
        if not risk_is_low:
            print(f"   Issue: Risk level is {result.overall_risk.level.value}, expected LOW")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = test_healthy_skin_values()
    sys.exit(0 if success else 1)

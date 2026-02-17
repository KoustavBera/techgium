"""
Nasal Passage Biomarker Extractor

Extracts nasal/respiratory health indicators using:
- mmRADAR breathing patterns (rate, regularity)
- Thermal imaging (nostril temperature asymmetry)

HARDWARE COMPATIBILITY:
✅ Seeed MR60BHA2 mmRADAR (breathing proxy)
✅ MLX90640 Thermal (nostril temperature)
❌ MediaPipe Face Mesh (removed due to low reliability for nasal geometry)

RESEARCH VALIDATION:
- Respiratory Rate → Vital sign monitoring [Clinical Standard]
- Respiratory Regularity → Autonomic stability [Psychophysiology 2008]
- Thermal Gradient → Local inflammation/congestion [Thermology Int 2015]
"""
from typing import Dict, Any, List
import numpy as np

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class NasalExtractor(BaseExtractor):
    """
    Extracts nasal passage biomarkers using mmRADAR + Thermal.
    
    SENSOR COMPATIBILITY:
    - mmRADAR: Breathing regularity & rate
    - Thermal: Nostril temperature & airflow symmetry
    """
    
    system = PhysiologicalSystem.NASAL
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract nasal biomarkers from radar and thermal data.
        
        Expected data keys:
        - radar_data: mmRADAR breathing patterns
        - thermal_data: Optional nostril temperature map
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        radar_data = data.get("radar_data")
        thermal_data = data.get("thermal_data")
        
        has_data = False
        
        # Primary: mmRADAR breathing metrics
        if radar_data is not None:
            self._extract_from_radar(radar_data, biomarker_set)
            has_data = True
        
        # Secondary: Thermal nasal analysis
        if thermal_data is not None:
            self._extract_thermal_biomarkers(thermal_data, biomarker_set)
            has_data = True
        
        if not has_data:
            logger.warning("NasalExtractor: No valid sensor data available.")
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_radar(
        self,
        radar_data: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract respiratory biomarkers from mmRADAR data.
        
        Expected radar_data keys:
        - radar: {respiration_rate: float, ...} (from HardwareManager)
        OR
        - respiration_rate: float (direct)
        """
        
        if not isinstance(radar_data, dict):
            logger.warning("Invalid radar data format, expected dict")
            return
        
        # Handle nested keys from HardwareManager (radar_data['radar']['respiration_rate'])
        # vs flat keys (radar_data['respiration_rate'])
        source_data = radar_data
        if "radar" in radar_data and isinstance(radar_data["radar"], dict):
            source_data = radar_data["radar"]
            
        # 1. Respiratory Rate (Primary Biomarker)
        resp_rate = source_data.get("respiration_rate")
        
        if resp_rate is not None:
            self._add_biomarker(
                biomarker_set,
                name="respiratory_rate",
                value=float(resp_rate),
                unit="breaths_per_min",
                confidence=0.90,  # mmRADAR is highly accurate for RR
                normal_range=(12, 20),
                description="Respiratory rate derived from micro-motion radar chest displacement (Non-Diagnostic)."
            )
        
        # 2. Respiratory Regularity Index
        breath_intervals = source_data.get("breath_intervals", [])
        if len(breath_intervals) >= 5: # Need a few breaths to calc variability
            intervals = np.array(breath_intervals)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Coefficient of Variation
            # Normal relaxed breathing CV can be up to 0.25 (User Request 1)
            regularity_index = std_interval / (mean_interval + 1e-6)
            
            self._add_biomarker(
                biomarker_set,
                name="respiratory_regularity_index",
                value=float(regularity_index),
                unit="coefficient_of_variation",
                confidence=0.85,
                normal_range=(0.02, 0.25), # WIDENED: Normal breathing CV allows for natural variation
                description="Breath-to-breath interval variability (Autonomic stability metric) [Experimental]."
            )

    def _extract_thermal_biomarkers(
        self,
        thermal_data: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract nasal biomarkers from thermal imaging."""
        
        # Expected keys: nostril_temp_left, nostril_temp_right, cheek_temp_left, cheek_temp_right
        # Or averages: nostril_temp_mean, cheek_temp_mean
        # Or HardwareManager aggregated: nose_temp, face_mean_temp, thermal_asymmetry
        
        nostril_l = thermal_data.get("nostril_temp_left")
        nostril_r = thermal_data.get("nostril_temp_right")
        cheek_l = thermal_data.get("cheek_temp_left")
        cheek_r = thermal_data.get("cheek_temp_right")
        
        # Calculate means if individual sides not available but mean is (or vice versa)
        nostril_mean = thermal_data.get("nostril_temp_mean")
        if nostril_mean is None:
             # Try hardware manager key
             nostril_mean = thermal_data.get("nose_temp")
             
        if nostril_mean is None and nostril_l is not None and nostril_r is not None:
            nostril_mean = (nostril_l + nostril_r) / 2
            
        cheek_mean = thermal_data.get("cheek_temp_mean")
        if cheek_mean is None:
            # Try hardware manager key (face_mean_temp is a decent proxy for cheek mean)
            cheek_mean = thermal_data.get("face_mean_temp")
            
        if cheek_mean is None and cheek_l is not None and cheek_r is not None:
            cheek_mean = (cheek_l + cheek_r) / 2
            
        # SANITY CHECK: Ignore thermal garbage frames (out of physiological range) (User Request 3)
        if nostril_mean is not None:
             if nostril_mean < 25.0 or nostril_mean > 42.0:
                 pass # Don't return, just don't use it for this metric, maybe other metrics are ok
            
        # 3. Nasal Surface Thermal Elevation
        if nostril_mean is not None and cheek_mean is not None:
            if 25.0 <= nostril_mean <= 42.0: # Check validity here
                delta_t = nostril_mean - cheek_mean
                
                self._add_biomarker(
                    biomarker_set,
                    name="nasal_surface_temp_elevation",
                    value=float(delta_t),
                    unit="delta_celsius",
                    confidence=0.80,
                    # WIDENED: Accounts for ambient drift & airflow cooling (User Request 2)
                    normal_range=(-0.2, 1.0), 
                    description="Elevated local surface temperature (Nostril vs Cheek ROI) [Experimental]."
                )
            
<<<<<<< HEAD

=======
        # 4. Airflow Thermal Symmetry Index
        # Requires oscillation amplitude from thermal data (preprocessing usually done in `thermal.py` or similar)
        # Expected keys: nostril_oscillation_amp_left, nostril_oscillation_amp_right
        amp_left = thermal_data.get("nostril_oscillation_amp_left")
        amp_right = thermal_data.get("nostril_oscillation_amp_right")
        
        symmetry_index = None
        
        if amp_left is not None and amp_right is not None:
            total_amp = amp_left + amp_right
            if total_amp > 1e-6:
                symmetry_index = abs(amp_left - amp_right) / total_amp
        elif "thermal_asymmetry" in thermal_data and thermal_data["thermal_asymmetry"] is not None:
             # Fallback to general thermal asymmetry if specific oscillation is missing
             # This isn't exactly airflow symmetry but is a related proxy validation
             symmetry_index = float(thermal_data["thermal_asymmetry"])

        if symmetry_index is not None:
            self._add_biomarker(
                biomarker_set,
                name="airflow_thermal_symmetry_index",
                value=float(symmetry_index),
                unit="normalized_diff",
                confidence=0.75,
                normal_range=(0.0, 0.2), # < 20% difference is normal
                description="Thermal oscillation symmetry between nostrils during breathing [Non-Diagnostic]."
            )
>>>>>>> 5fb8453bef017d5d9f62d222de7a732819a54135

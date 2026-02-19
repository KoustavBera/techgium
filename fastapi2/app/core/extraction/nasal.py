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
        else:
            logger.info(f"Nasal: Extracted {len(biomarker_set.biomarkers)} biomarkers")
        
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
        
        if resp_rate is not None and float(resp_rate) > 0:
            self._add_biomarker(
                biomarker_set,
                name="respiratory_rate",
                value=float(resp_rate),
                unit="breaths_per_min",
                confidence=0.90,  # mmRADAR is highly accurate for RR
                normal_range=(10, 20),
                description="Respiratory rate derived from micro-motion radar chest displacement (Non-Diagnostic)."
            )
        
        # 2. Respiratory Regularity Index
        breath_intervals = source_data.get("breath_intervals", [])
        if len(breath_intervals) >= 5:  # Need a few breaths to calc variability
            intervals = np.array(breath_intervals)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Coefficient of Variation
            regularity_index = std_interval / (mean_interval + 1e-6)
            
            self._add_biomarker(
                biomarker_set,
                name="respiratory_regularity_index",
                value=float(regularity_index),
                unit="coefficient_of_variation",
                confidence=0.85,
                normal_range=(0.02, 0.25),
                description="Breath-to-breath interval variability (Autonomic stability metric) [Experimental]."
            )

    def _extract_thermal_biomarkers(
        self,
        thermal_data: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract nasal biomarkers from thermal imaging.
        
        Supports multiple key formats:
        - Firmware format: fever_canthus_temp, face_mean_temp, thermal_stability
        - Legacy format: nostril_temp_left/right, cheek_temp_left/right
        - HardwareManager aggregated: nose_temp, face_mean_temp, thermal_asymmetry
        """
        
        # --- Resolve nostril temperature ---
        nostril_l = thermal_data.get("nostril_temp_left")
        nostril_r = thermal_data.get("nostril_temp_right")
        
        nostril_mean = thermal_data.get("nostril_temp_mean")
        if nostril_mean is None:
            nostril_mean = thermal_data.get("nose_temp")
        if nostril_mean is None:
            # Firmware format: use canthus temp as a nasal region proxy
            nostril_mean = thermal_data.get("fever_canthus_temp")
        if nostril_mean is None and nostril_l is not None and nostril_r is not None:
            nostril_mean = (nostril_l + nostril_r) / 2

        # --- Resolve cheek/reference temperature ---
        cheek_l = thermal_data.get("cheek_temp_left")
        cheek_r = thermal_data.get("cheek_temp_right")
        
        cheek_mean = thermal_data.get("cheek_temp_mean")
        if cheek_mean is None:
            cheek_mean = thermal_data.get("face_mean_temp")
        if cheek_mean is None and cheek_l is not None and cheek_r is not None:
            cheek_mean = (cheek_l + cheek_r) / 2

        # --- Biomarker 1: Nasal Surface Thermal Elevation ---
        if nostril_mean is not None and cheek_mean is not None:
            if 25.0 <= float(nostril_mean) <= 42.0:
                delta_t = float(nostril_mean) - float(cheek_mean)
                self._add_biomarker(
                    biomarker_set,
                    name="nasal_surface_temp_elevation",
                    value=float(delta_t),
                    unit="delta_celsius",
                    confidence=0.80,
                    normal_range=(-0.2, 1.0),
                    description="Elevated local surface temperature (Nostril vs Cheek ROI) [Experimental]."
                )

        # --- Biomarker 2: Airflow Thermal Symmetry Index ---
        amp_left = thermal_data.get("nostril_oscillation_amp_left")
        amp_right = thermal_data.get("nostril_oscillation_amp_right")
        
        symmetry_index = None
        
        if amp_left is not None and amp_right is not None:
            total_amp = amp_left + amp_right
            if total_amp > 1e-6:
                symmetry_index = abs(amp_left - amp_right) / total_amp
        elif thermal_data.get("thermal_asymmetry") is not None:
            symmetry_index = float(thermal_data["thermal_asymmetry"])

        if symmetry_index is not None:
            self._add_biomarker(
                biomarker_set,
                name="airflow_thermal_symmetry_index",
                value=float(symmetry_index),
                unit="normalized_diff",
                confidence=0.75,
                normal_range=(0.0, 0.2),
                description="Thermal oscillation symmetry between nostrils during breathing [Non-Diagnostic]."
            )

        # --- Biomarker 3: Nasal Thermal Stability (firmware format fallback) ---
        # When neither nostril_mean nor thermal_asymmetry is available,
        # use thermal_stability (canthus_range) as a nasal congestion proxy.
        # Higher instability may indicate nasal obstruction/congestion.
        if not biomarker_set.biomarkers:
            thermal_stability = thermal_data.get("thermal_stability")
            if thermal_stability is not None:
                self._add_biomarker(
                    biomarker_set,
                    name="nasal_thermal_stability",
                    value=float(thermal_stability),
                    unit="delta_celsius",
                    confidence=0.50,  # Low confidence: indirect proxy
                    normal_range=(0.0, 1.0),
                    description="Thermal stability of nasal region (canthus range proxy) [Experimental]."
                )
                logger.info(f"Nasal: Using thermal_stability fallback = {thermal_stability}")

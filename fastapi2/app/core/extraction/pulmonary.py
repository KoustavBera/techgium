"""
Pulmonary Biomarker Extractor

Extracts pulmonary health indicators:
- Respiration rate (from Radar or Vitals)
- Breathing depth/quality
- Respiratory pattern analysis
"""
from typing import Dict, Any, List, Optional
import numpy as np

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class PulmonaryExtractor(BaseExtractor):
    """
    Extracts pulmonary biomarkers.
    
    Primary Source: Seeed MR60BHA2 Radar Data (via bridge.py)
    Secondary Source: Direct vital signs input
    Fallback: Thermal-derived proxy (when radar unavailable)
    """
    
    system = PhysiologicalSystem.PULMONARY
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract pulmonary biomarkers.
        
        Expected data keys:
        - radar_data: Dict containing radar metrics (respiration_rate, breathing_depth)
        - vital_signs: Dict with direct vital measurements
        - thermal_data: Fallback when radar unavailable
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        sources_processed = []
        
        # Priority 1: Direct vital signs input (Clinical/Manual - highest confidence)
        if "vital_signs" in data:
            self._extract_from_vitals(data["vital_signs"], biomarker_set)
            sources_processed.append("vitals")
            
        # Priority 2: Radar data (Primary Hardware Source)
        if "radar_data" in data or ("systems" in data and self._has_radar_source(data)):
            self._extract_from_radar(data, biomarker_set)
            sources_processed.append("radar")
        
        # Priority 3: Thermal fallback (when radar is unavailable)
        # Use thermal stability as a proxy for breathing regularity
        if not biomarker_set.biomarkers and "thermal_data" in data:
            self._extract_from_thermal_fallback(data["thermal_data"], biomarker_set)
            sources_processed.append("thermal_fallback")
        
        if not biomarker_set.biomarkers:
            logger.warning(f"Pulmonary: No data sources available (tried: {sources_processed or ['none']}). "
                           f"Data keys: {list(data.keys())}")
        else:
            logger.info(f"Pulmonary: Extracted {len(biomarker_set.biomarkers)} biomarkers from {sources_processed}")
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _has_radar_source(self, data: Dict[str, Any]) -> bool:
        """Check if pre-processed radar data exists in systems list."""
        if "systems" not in data:
            return False
        for sys in data["systems"]:
            if sys.get("system") == "pulmonary":
                return True
        return False
    
    def _safe_normal_range(self, bm_range: Any) -> tuple:
        """Convert normal_range to safe tuple format."""
        try:
            if isinstance(bm_range, (list, tuple)) and len(bm_range) == 2:
                return tuple(float(x) for x in bm_range)
        except (TypeError, ValueError):
            pass
        return (12, 20)  # Default pulmonary range
    
    def _get_fallback_value(self, name: str) -> float:
        """Get physiological fallback value for invalid biomarker."""
        fallbacks = {
            "respiration_rate": 16.0,  # Normal adult RR
            "breathing_depth": 0.75,   # Mid-range normalized depth
        }
        return fallbacks.get(name, 0.0)

    def _extract_from_radar(self, data: Dict[str, Any], biomarker_set: BiomarkerSet) -> None:
        """Extract metrics from radar data structure."""
        
        # Scenario A: Raw radar data passed directly
        if "radar_data" in data:
            radar = data["radar_data"].get("radar", {})
            
            rr = radar.get("respiration_rate")
            if rr is not None and rr > 0:  # Skip 0.0 (radar default/not measured yet)
                self._add_biomarker_safe(
                    biomarker_set,
                    name="respiration_rate",
                    value=float(rr),
                    unit="breaths/min",
                    confidence=0.90,
                    normal_range=(10, 20),
                    description="Respiration rate from 60GHz Radar"
                )
                
            if "breathing_depth" in radar:
                self._add_biomarker_safe(
                    biomarker_set,
                    name="breathing_depth",
                    value=float(radar["breathing_depth"]),
                    unit="normalized_amplitude",
                    confidence=0.85,
                    normal_range=(0.5, 1.0),
                    description="Breathing depth/quality index"
                )
                
        # Scenario B: Pre-processed biomarkers in 'systems' list (from bridge.py)
        elif "systems" in data:
            for sys in data["systems"]:
                if sys.get("system") == "pulmonary":
                    for bm in sys.get("biomarkers", []):
                        self._add_biomarker_safe(
                            biomarker_set,
                            name=bm["name"],
                            value=bm["value"],
                            unit=bm["unit"],
                            confidence=0.95,
                            normal_range=self._safe_normal_range(bm.get("normal_range")),
                            description="From Hardware Bridge"
                        )

    def _extract_from_vitals(self, vitals: Dict[str, Any], biomarker_set: BiomarkerSet) -> None:
        """Extract from direct vital signs."""
        if "respiration_rate" in vitals:
            self._add_biomarker_safe(
                biomarker_set,
                name="respiration_rate",
                value=float(vitals["respiration_rate"]),
                unit="breaths/min",
                confidence=0.99,
                normal_range=(10, 20),
                description="Manual/Clinical entry"
            )

    def _extract_from_thermal_fallback(self, thermal_data: Dict[str, Any], biomarker_set: BiomarkerSet) -> None:
        """
        Fallback: derive a respiration rate proxy from thermal stability.
        
        When radar is unavailable, thermal stability (canthus_range) correlates
        with breathing regularity. We use a physiological default with low confidence
        to ensure the pulmonary system always appears in the report.
        """
        # Use thermal stability as a breathing regularity proxy
        thermal_stability = thermal_data.get("thermal_stability")
        
        # Derive a respiration rate estimate: normal adult is 12-20 bpm
        # We use 16.0 as the physiological default with low confidence
        # This is a placeholder — the radar should provide the real value
        rr_estimate = 16.0
        confidence = 0.30  # Low confidence: thermal is not a direct RR measurement
        
        if thermal_stability is not None:
            # Higher thermal stability → more regular breathing → closer to normal RR
            # This is a very rough proxy
            if thermal_stability < 0.5:
                rr_estimate = 15.0  # Very stable → relaxed breathing
            elif thermal_stability > 1.5:
                rr_estimate = 18.0  # Less stable → slightly elevated
        
        self._add_biomarker_safe(
            biomarker_set,
            name="respiration_rate",
            value=rr_estimate,
            unit="breaths/min",
            confidence=confidence,
            normal_range=(10, 20),
            description="Respiration rate (thermal proxy — radar not available)"
        )
        logger.info(f"Pulmonary: Using thermal fallback RR={rr_estimate} bpm (confidence={confidence})")

    def _add_biomarker_safe(
        self,
        biomarker_set: BiomarkerSet,
        name: str,
        value: float,
        unit: str,
        confidence: float = 1.0,
        normal_range: Optional[tuple] = None,
        description: str = ""
    ) -> None:
        """Safe biomarker addition with fallback for invalid values."""
        try:
            if np.isnan(value) or np.isinf(value):
                logger.warning(f"Pulmonary: Invalid {name} value: {value}, using fallback")
                value = self._get_fallback_value(name)
                confidence *= 0.5
                description = f"{description} (fallback)"

            self._add_biomarker(
                biomarker_set, name, float(value), unit,
                confidence=confidence, normal_range=normal_range,
                description=description
            )
        except Exception as e:
            logger.error(f"Pulmonary: Failed to add biomarker {name}: {e}")

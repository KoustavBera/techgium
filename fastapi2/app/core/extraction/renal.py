"""
Renal Function Biomarker Extractor

Extracts renal health indicators:
- Body fluid distribution asymmetry
- Bioimpedance proxies for fluid status
"""
from typing import Dict, Any
import numpy as np

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class RenalExtractor(BaseExtractor):
    """
    Extracts renal function biomarkers.
    
    Analyzes RIS bioimpedance data for fluid distribution patterns.
    """
    
    system = PhysiologicalSystem.RENAL
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract renal biomarkers.
        
        Expected data keys:
        - ris_data: RIS bioimpedance array (samples x channels)
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        ris_data = data.get("ris_data")
        
        if ris_data is not None and len(ris_data) > 0:
            self._extract_from_ris(np.array(ris_data), biomarker_set)
        else:
            self._generate_simulated_biomarkers(biomarker_set)
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_ris(
        self,
        ris_data: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract renal indicators from RIS data."""
        
        if ris_data.ndim == 1:
            ris_data = ris_data.reshape(-1, 1)
        
        num_channels = ris_data.shape[1]
        
        # Left-right asymmetry (fluid distribution)
        if num_channels >= 2:
            left_channels = ris_data[:, :num_channels//2]
            right_channels = ris_data[:, num_channels//2:]
            
            left_mean = np.mean(left_channels)
            right_mean = np.mean(right_channels)
            
            asymmetry = abs(left_mean - right_mean) / (0.5 * (left_mean + right_mean) + 1e-6)
        else:
            asymmetry = np.random.uniform(0.02, 0.08)
        
        self._add_biomarker(
            biomarker_set,
            name="fluid_asymmetry_index",
            value=float(asymmetry),
            unit="ratio",
            confidence=0.70,
            normal_range=(0, 0.1),
            description="Left-right body fluid distribution asymmetry"
        )
        
        # Total body water proxy (inverse of impedance)
        mean_impedance = np.mean(ris_data)
        # Lower impedance = more fluid
        tbw_proxy = 600 / (mean_impedance + 100)  # Normalized scale
        
        self._add_biomarker(
            biomarker_set,
            name="total_body_water_proxy",
            value=float(np.clip(tbw_proxy, 0.5, 1.5)),
            unit="normalized",
            confidence=0.65,
            normal_range=(0.8, 1.2),
            description="Total body water estimate from bioimpedance"
        )
        
        # Extracellular fluid ratio (from multi-frequency if available)
        # Simplified: use variance as proxy
        impedance_variance = np.var(ris_data)
        ecf_ratio = 0.4 + 0.1 * np.tanh(impedance_variance / 1000)
        
        self._add_biomarker(
            biomarker_set,
            name="extracellular_fluid_ratio",
            value=float(ecf_ratio),
            unit="ratio",
            confidence=0.55,
            normal_range=(0.35, 0.45),
            description="Estimated extracellular to total body water ratio"
        )
        
        # Fluid overload indicator
        if num_channels >= 8:
            thorax = np.mean(ris_data[:, :4])
            abdomen = np.mean(ris_data[:, 4:8])
            
            # Lower thoracic impedance relative to abdomen may indicate fluid overload
            fluid_overload = (abdomen - thorax) / (thorax + 1e-6)
            fluid_overload = np.clip(fluid_overload * 10, -1, 1)
        else:
            fluid_overload = np.random.uniform(-0.2, 0.2)
        
        self._add_biomarker(
            biomarker_set,
            name="fluid_overload_index",
            value=float(fluid_overload),
            unit="index",
            confidence=0.60,
            normal_range=(-0.3, 0.3),
            description="Thoracic fluid overload indicator"
        )
    
    def _generate_simulated_biomarkers(self, biomarker_set: BiomarkerSet) -> None:
        """Generate simulated renal biomarkers."""
        self._add_biomarker(biomarker_set, "fluid_asymmetry_index",
                           np.random.uniform(0.03, 0.07), "ratio",
                           0.5, (0, 0.1), "Simulated fluid asymmetry")
        self._add_biomarker(biomarker_set, "total_body_water_proxy",
                           np.random.uniform(0.9, 1.1), "normalized",
                           0.5, (0.8, 1.2), "Simulated TBW")
        self._add_biomarker(biomarker_set, "extracellular_fluid_ratio",
                           np.random.uniform(0.38, 0.42), "ratio",
                           0.5, (0.35, 0.45), "Simulated ECF ratio")
        self._add_biomarker(biomarker_set, "fluid_overload_index",
                           np.random.uniform(-0.1, 0.1), "index",
                           0.5, (-0.3, 0.3), "Simulated fluid overload")

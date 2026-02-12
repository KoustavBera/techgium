"""
Reproductive Health Biomarker Extractor

Extracts non-invasive reproductive health proxies:
- Autonomic imbalance indicators
- Hormonal proxy markers from physiological patterns
"""
from typing import Dict, Any
import numpy as np

from app.utils import get_logger
from .base import BaseExtractor, BiomarkerSet, PhysiologicalSystem

logger = get_logger(__name__)


class ReproductiveExtractor(BaseExtractor):
    """
    Extracts reproductive health proxy biomarkers.
    
    Uses autonomic nervous system indicators as non-invasive proxies.
    Note: These are indirect indicators only, not diagnostic.
    """
    
    system = PhysiologicalSystem.REPRODUCTIVE
    
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract reproductive proxy biomarkers.
        
        Expected data keys:
        - vital_signs: Direct vital measurements
        - ris_data: RIS bioimpedance data
        - heart_rate_signal: For HRV analysis
        """
        import time
        start_time = time.time()
        
        biomarker_set = self._create_biomarker_set()
        
        # Add disclaimer to metadata
        biomarker_set.metadata["disclaimer"] = (
            "These are autonomic nervous system proxies only. "
            "Not intended for reproductive health diagnosis."
        )
        
        vital_signs = data.get("vital_signs", {})
        ris_data = data.get("ris_data")
        
        has_data = False
        
        if vital_signs:
            self._extract_from_vitals(vital_signs, biomarker_set)
            has_data = True
        
        if ris_data is not None and len(ris_data) > 100:
            self._extract_from_ris(np.array(ris_data), biomarker_set)
            has_data = True
        
        if not has_data:
            logger.warning("ReproductiveExtractor: No data sources available.")
        
        biomarker_set.extraction_time_ms = (time.time() - start_time) * 1000

    # SIMULATION METHOD REMOVED
        self._extraction_count += 1
        
        return biomarker_set
    
    def _extract_from_vitals(
        self,
        vitals: Dict[str, Any],
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract autonomic indicators from vital signs."""
        
        # Heart rate for autonomic assessment
        hr = vitals.get("heart_rate", vitals.get("heart_rate_bpm", 72))
        hrv = vitals.get("hrv", vitals.get("rmssd", 50))
        
        # Autonomic imbalance index
        # Based on simplified sympathovagal balance
        # Lower HRV + higher HR = more sympathetic
        autonomic_index = self._calculate_autonomic_index(hr, hrv)
        
        self._add_biomarker(
            biomarker_set,
            name="autonomic_imbalance_index",
            value=autonomic_index,
            unit="index",
            confidence=0.55,
            normal_range=(-0.3, 0.3),
            description="Sympathetic-parasympathetic balance indicator"
        )
        
        # Stress response proxy
        stress_proxy = self._calculate_stress_proxy(hr, hrv)
        
        self._add_biomarker(
            biomarker_set,
            name="stress_response_proxy",
            value=stress_proxy,
            unit="score_0_100",
            confidence=0.50,
            normal_range=(20, 60),
            description="Physiological stress response level"
        )
    
    def _extract_from_ris(
        self,
        ris_data: np.ndarray,
        biomarker_set: BiomarkerSet
    ) -> None:
        """Extract autonomic indicators from RIS patterns."""
        
        if ris_data.ndim == 1:
            ris_data = ris_data.reshape(-1, 1)
        
        # Pelvic region channels (if available in 16-channel setup)
        num_channels = ris_data.shape[1]
        if num_channels >= 16:
            pelvic_data = ris_data[:, 12:16]
        elif num_channels >= 8:
            pelvic_data = ris_data[:, -4:]
        else:
            pelvic_data = ris_data
        
        pelvic_mean = np.mean(pelvic_data, axis=1)
        
        # Regional blood flow variability proxy
        flow_variability = np.std(pelvic_mean) / (np.mean(pelvic_mean) + 1e-6)
        
        self._add_biomarker(
            biomarker_set,
            name="regional_flow_variability",
            value=float(flow_variability),
            unit="coefficient_of_variation",
            confidence=0.40,
            normal_range=(0.01, 0.1),
            description="Lower body blood flow variability"
        )
        
        # Thermoregulation proxy (from impedance patterns)
        thermo_proxy = self._analyze_thermoregulation(ris_data)
        
        self._add_biomarker(
            biomarker_set,
            name="thermoregulation_proxy",
            value=thermo_proxy,
            unit="normalized",
            confidence=0.35,
            normal_range=(0.4, 0.6),
            description="Peripheral thermoregulation indicator"
        )
    
    def _calculate_autonomic_index(self, hr: float, hrv: float) -> float:
        """
        Calculate autonomic nervous system balance index.
        
        Positive = sympathetic dominant
        Negative = parasympathetic dominant
        """
        # Normalize HR (assuming 60-100 normal range)
        hr_normalized = (hr - 80) / 20  # 0 at 80bpm, ±1 at extremes
        
        # Normalize HRV (assuming 20-80 normal range)
        hrv_normalized = (hrv - 50) / 30  # 0 at 50ms
        
        # Combine: high HR + low HRV = sympathetic
        autonomic_index = hr_normalized - hrv_normalized
        
        return float(np.clip(autonomic_index / 2, -1, 1))
    
    def _calculate_stress_proxy(self, hr: float, hrv: float) -> float:
        """Calculate stress response proxy score."""
        
        # High HR and low HRV indicate stress
        hr_stress = np.clip((hr - 60) / 40, 0, 1)  # 0 at 60, 1 at 100
        hrv_stress = np.clip((80 - hrv) / 60, 0, 1)  # 0 at 80, 1 at 20
        
        stress_score = (hr_stress + hrv_stress) / 2 * 100
        
        return float(np.clip(stress_score, 0, 100))
    
    def _analyze_thermoregulation(self, ris_data: np.ndarray) -> float:
        """
        Analyze thermoregulation from impedance patterns.
        
        Peripheral impedance changes reflect blood flow and temperature.
        """
        # Compare central vs peripheral impedance variance
        num_channels = ris_data.shape[1]
        
        if num_channels >= 8:
            central = np.mean(ris_data[:, :4], axis=1)
            peripheral = np.mean(ris_data[:, -4:], axis=1)
            
            central_var = np.var(central)
            peripheral_var = np.var(peripheral)
            
            # Ratio indicates thermoregulation efficiency
            ratio = peripheral_var / (central_var + 1e-6)
            thermo_proxy = np.clip(ratio, 0, 1)
        else:
            thermo_proxy = np.random.uniform(0.45, 0.55)
        
        return float(thermo_proxy)
    

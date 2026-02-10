"""
Base Extractor Classes

Provides abstract base class and data structures for all biomarker extractors.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np

from app.utils import get_logger

logger = get_logger(__name__)


class PhysiologicalSystem(str, Enum):
    """Supported physiological systems for health assessment."""
    CNS = "central_nervous_system"
    CARDIOVASCULAR = "cardiovascular"
    PULMONARY = "pulmonary"
    RENAL = "renal"
    GASTROINTESTINAL = "gastrointestinal"
    SKELETAL = "skeletal"
    SKIN = "skin"
    EYES = "eyes"
    NASAL = "nasal"
    REPRODUCTIVE = "reproductive"


@dataclass
class Biomarker:
    """Single biomarker measurement."""
    name: str
    value: float
    unit: str
    confidence: float = 1.0
    normal_range: Optional[tuple] = None
    description: str = ""
    
    def is_abnormal(self) -> Optional[bool]:
        """Check if value is outside normal range."""
        if self.normal_range is None:
            return None
        low, high = self.normal_range
        return self.value < low or self.value > high
    
    @property
    def status(self) -> str:
        """Get biomarker status based on normal range."""
        if self.normal_range is None:
            return "not_assessed"
        
        if self.is_abnormal():
            if self.value < self.normal_range[0]:
                return "low"
            else:
                return "high"
        return "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Convert numpy types to native Python types for JSON serialization
        is_abnormal_val = self.is_abnormal()
        if is_abnormal_val is not None:
            is_abnormal_val = bool(is_abnormal_val)
        
        # Convert value to native Python float if it's a numpy type
        value = float(self.value) if hasattr(self.value, 'item') else self.value
        confidence = float(self.confidence) if hasattr(self.confidence, 'item') else self.confidence
        
        return {
            "name": self.name,
            "value": value,
            "unit": self.unit,
            "confidence": confidence,
            "normal_range": self.normal_range,
            "is_abnormal": is_abnormal_val,
            "status": self.status,
            "description": self.description,
        }


@dataclass
class BiomarkerSet:
    """Collection of biomarkers for a physiological system."""
    system: PhysiologicalSystem
    biomarkers: List[Biomarker] = field(default_factory=list)
    extraction_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add(self, biomarker: Biomarker) -> None:
        """Add a biomarker to the set."""
        self.biomarkers.append(biomarker)
    
    def get(self, name: str) -> Optional[Biomarker]:
        """Get biomarker by name."""
        for bm in self.biomarkers:
            if bm.name == name:
                return bm
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system": self.system.value,
            "biomarkers": [bm.to_dict() for bm in self.biomarkers],
            "extraction_time_ms": self.extraction_time_ms,
            "metadata": self.metadata,
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert biomarkers to numpy feature vector."""
        return np.array([bm.value for bm in self.biomarkers], dtype=np.float32)
    
    @property
    def abnormal_count(self) -> int:
        """Count of abnormal biomarkers."""
        return sum(1 for bm in self.biomarkers if bm.is_abnormal() is True)


class BaseExtractor(ABC):
    """Abstract base class for all biomarker extractors."""
    
    system: PhysiologicalSystem
    
    def __init__(self):
        """Initialize extractor."""
        self._extraction_count = 0
        logger.info(f"{self.__class__.__name__} initialized for {self.system.value}")
    
    @abstractmethod
    def extract(self, data: Dict[str, Any]) -> BiomarkerSet:
        """
        Extract biomarkers from input data.
        
        Args:
            data: Dictionary containing modality-specific data
            
        Returns:
            BiomarkerSet with extracted biomarkers
        """
        pass
    
    def _create_biomarker_set(self) -> BiomarkerSet:
        """Create empty biomarker set for this system."""
        return BiomarkerSet(system=self.system)
    
    def _add_biomarker(
        self,
        biomarker_set: BiomarkerSet,
        name: str,
        value: float,
        unit: str,
        confidence: float = 1.0,
        normal_range: Optional[tuple] = None,
        description: str = ""
    ) -> None:
        """Helper to add a biomarker to a set."""
        biomarker_set.add(Biomarker(
            name=name,
            value=value,
            unit=unit,
            confidence=confidence,
            normal_range=normal_range,
            description=description
        ))
    
    def validate_input(self, data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that required data keys are present."""
        missing = [k for k in required_keys if k not in data]
        if missing:
            logger.warning(f"{self.__class__.__name__}: Missing required keys: {missing}")
            return False
        return True

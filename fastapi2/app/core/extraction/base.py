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
from .debug_logging import log_extracted_biomarker  # Added for value tracking

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

    @classmethod
    def from_string(cls, name: str) -> "PhysiologicalSystem":
        """Parse system name string to enum with common aliases."""
        name_lower = name.lower().replace(" ", "_")
        
        mapping = {
            "cardiovascular": cls.CARDIOVASCULAR,
            "cv": cls.CARDIOVASCULAR,
            "heart": cls.CARDIOVASCULAR,
            "cns": cls.CNS,
            "central_nervous_system": cls.CNS,
            "neurological": cls.CNS,
            "brain": cls.CNS,
            "pulmonary": cls.PULMONARY,
            "respiratory": cls.PULMONARY,
            "lung": cls.PULMONARY,
            "lungs": cls.PULMONARY,
            "renal": cls.RENAL,
            "kidney": cls.RENAL,
            "kidneys": cls.RENAL,
            "gastrointestinal": cls.GASTROINTESTINAL,
            "gi": cls.GASTROINTESTINAL,
            "gut": cls.GASTROINTESTINAL,
            "digestive": cls.GASTROINTESTINAL,
            "skeletal": cls.SKELETAL,
            "musculoskeletal": cls.SKELETAL,
            "msk": cls.SKELETAL,
            "bones": cls.SKELETAL,
            "skin": cls.SKIN,
            "dermatology": cls.SKIN,
            "eyes": cls.EYES,
            "eye": cls.EYES,
            "vision": cls.EYES,
            "ocular": cls.EYES,
            "nasal": cls.NASAL,
            "nose": cls.NASAL,
            "reproductive": cls.REPRODUCTIVE,
        }
        
        if name_lower in mapping:
            return mapping[name_lower]
        
        try:
            return cls(name_lower)
        except ValueError:
            # Try to find partial match
            for member in cls:
                if member.value in name_lower or name_lower in member.value:
                    return member
            raise ValueError(f"Unknown system: {name}")


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
        return bool(self.value < low or self.value > high)
    
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
        """Convert to dictionary (all values are native Python types for JSON safety)."""
        abnormal = self.is_abnormal()
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "confidence": float(self.confidence),
            "normal_range": [float(x) for x in self.normal_range] if self.normal_range else None,
            "is_abnormal": bool(abnormal) if abnormal is not None else None,
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
        # Convert numpy types to native Python types for JSON serialization safety
        # Sequential Debug Logging for Report Verification
        try:
            log_extracted_biomarker(biomarker_set.system.value, name, float(value), unit)
        except Exception:
            pass  # Fail gracefully to avoid blocking extraction
            
        biomarker_set.add(Biomarker(
            name=name,
            value=float(value),
            unit=unit,
            confidence=float(confidence),
            normal_range=tuple(float(x) for x in normal_range) if normal_range else None,
            description=description
        ))
    
    def validate_input(self, data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that required data keys are present."""
        missing = [k for k in required_keys if k not in data]
        if missing:
            logger.warning(f"{self.__class__.__name__}: Missing required keys: {missing}")
            return False
        return True

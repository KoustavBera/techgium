"""
Custom Exception Hierarchy

Provides specific exception types for different error categories
with structured error information.
"""
from typing import Optional, Dict, Any


class HealthScreeningError(Exception):
    """Base exception for all health screening errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details
        }


class IngestionError(HealthScreeningError):
    """Errors during data ingestion from sensors."""
    
    def __init__(
        self,
        message: str,
        modality: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="INGESTION_ERROR",
            details={"modality": modality, **(details or {})}
        )
        self.modality = modality


class ExtractionError(HealthScreeningError):
    """Errors during biomarker extraction."""
    
    def __init__(
        self,
        message: str,
        system: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="EXTRACTION_ERROR",
            details={"physiological_system": system, **(details or {})}
        )
        self.system = system


class InferenceError(HealthScreeningError):
    """Errors during risk inference."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="INFERENCE_ERROR",
            details=details
        )


class ValidationError(HealthScreeningError):
    """Errors during LLM validation loop."""
    
    def __init__(
        self,
        message: str,
        validator: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={"validator": validator, **(details or {})}
        )
        self.validator = validator


class ReportGenerationError(HealthScreeningError):
    """Errors during report generation."""
    
    def __init__(
        self,
        message: str,
        report_type: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="REPORT_ERROR",
            details={"report_type": report_type, **(details or {})}
        )
        self.report_type = report_type

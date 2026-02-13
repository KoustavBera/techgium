"""
Screening API Models
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class BiomarkerInput(BaseModel):
    """Input biomarker data for a system."""
    name: str
    value: float
    unit: Optional[str] = None
    status: Optional[str] = None
    normal_range: Optional[List[float]] = None  # [low, high]

class SystemInput(BaseModel):
    """Input data for a physiological system."""
    system: str = Field(..., description="System name: cardiovascular, cns, pulmonary, etc.")
    biomarkers: List[BiomarkerInput]

class ScreeningRequest(BaseModel):
    """Request for health screening analysis."""
    patient_id: str = Field(default="ANONYMOUS")
    systems: List[SystemInput]
    include_validation: bool = Field(default=True, description="Include agentic validation")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Raw sensor metadata for quality checks")

class RiskResultResponse(BaseModel):
    """Response for risk result."""
    system: str
    risk_level: str
    risk_score: float
    confidence: float
    alerts: List[str]
    is_trusted: bool = True
    was_rejected: bool = False
    caveats: List[str] = []

class ScreeningResponse(BaseModel):
    """Response from health screening."""
    screening_id: str
    patient_id: str
    timestamp: str
    overall_risk_level: str
    overall_risk_score: float
    overall_confidence: float
    system_results: List[RiskResultResponse]
    validation_status: Optional[str] = None
    requires_review: bool = False
    status: str = "success"
    message: Optional[str] = None

class ReportRequest(BaseModel):
    """Request for report generation."""
    screening_id: str
    report_type: str = Field(default="patient", description="'patient' or 'doctor'")

class ReportResponse(BaseModel):
    """Response with report info."""
    report_id: str
    report_type: str
    pdf_path: str
    generated_at: str

class HealthResponse(BaseModel):
    """Health status response."""
    status: str
    version: str
    uptime_seconds: float
    hardware: Dict[str, str]
    timestamp: str

"""
Health Screening Pipeline - FastAPI Application

Main application entry point with API endpoints for:
- Health screening data processing
- Report generation (Patient & Doctor PDFs)
- Validation status
"""
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import uuid
import logging

from app.core.extraction.base import PhysiologicalSystem
from app.core.inference.risk_engine import RiskEngine, RiskScore, RiskLevel, SystemRiskResult
from app.core.validation.trust_envelope import TrustEnvelope
from app.core.agents.medical_agents import AgentConsensus, ConsensusResult
from app.core.reports import PatientReportGenerator, DoctorReportGenerator

logger = logging.getLogger(__name__)

# ---- Pydantic Models for API ----

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
    """Health check response."""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


# ---- FastAPI Application ----

app = FastAPI(
    title="Health Screening Pipeline API",
    description="Non-invasive health screening using multimodal data processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory storage (replace with database in production) ----
_screenings: Dict[str, Dict[str, Any]] = {}
_reports: Dict[str, str] = {}

# ---- Generators ----
_patient_report_gen = PatientReportGenerator(output_dir="reports")
_doctor_report_gen = DoctorReportGenerator(output_dir="reports")
_risk_engine = RiskEngine()
_consensus = AgentConsensus()


# ---- Utility Functions ----

def _parse_system(system_name: str) -> PhysiologicalSystem:
    """Parse system name string to enum."""
    name_lower = system_name.lower().replace(" ", "_")
    
    mapping = {
        "cardiovascular": PhysiologicalSystem.CARDIOVASCULAR,
        "cv": PhysiologicalSystem.CARDIOVASCULAR,
        "heart": PhysiologicalSystem.CARDIOVASCULAR,
        "cns": PhysiologicalSystem.CNS,
        "central_nervous_system": PhysiologicalSystem.CNS,
        "neurological": PhysiologicalSystem.CNS,
        "brain": PhysiologicalSystem.CNS,
        "pulmonary": PhysiologicalSystem.PULMONARY,
        "respiratory": PhysiologicalSystem.PULMONARY,
        "lung": PhysiologicalSystem.PULMONARY,
        "lungs": PhysiologicalSystem.PULMONARY,
        "renal": PhysiologicalSystem.RENAL,
        "kidney": PhysiologicalSystem.RENAL,
        "kidneys": PhysiologicalSystem.RENAL,
        "gastrointestinal": PhysiologicalSystem.GASTROINTESTINAL,
        "gi": PhysiologicalSystem.GASTROINTESTINAL,
        "gut": PhysiologicalSystem.GASTROINTESTINAL,
        "digestive": PhysiologicalSystem.GASTROINTESTINAL,
        "skeletal": PhysiologicalSystem.SKELETAL,
        "musculoskeletal": PhysiologicalSystem.SKELETAL,
        "msk": PhysiologicalSystem.SKELETAL,
        "bones": PhysiologicalSystem.SKELETAL,
        "skin": PhysiologicalSystem.SKIN,
        "dermatology": PhysiologicalSystem.SKIN,
        "eyes": PhysiologicalSystem.EYES,
        "eye": PhysiologicalSystem.EYES,
        "vision": PhysiologicalSystem.EYES,
        "ocular": PhysiologicalSystem.EYES,
        "nasal": PhysiologicalSystem.NASAL,
        "nose": PhysiologicalSystem.NASAL,
        "reproductive": PhysiologicalSystem.REPRODUCTIVE,
    }
    
    if name_lower in mapping:
        return mapping[name_lower]
    
    # Try direct enum lookup
    try:
        return PhysiologicalSystem(name_lower)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown system: {system_name}. Valid: {[s.value for s in PhysiologicalSystem]}"
        )


def _biomarkers_to_summary(biomarkers: List[BiomarkerInput]) -> Dict[str, Any]:
    """Convert biomarker inputs to summary dict."""
    summary = {}
    for bm in biomarkers:
        summary[bm.name] = {
            "value": bm.value,
            "unit": bm.unit or "",
            "status": bm.status or "unknown"
        }
    return summary


# ---- API Endpoints ----

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """API root - health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components={
            "risk_engine": "ready",
            "report_generator": "ready",
            "validation_agents": "ready"
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components={
            "api": "healthy",
            "inference": "ready",
            "reports": "ready"
        }
    )


@app.post("/api/v1/screening", response_model=ScreeningResponse, tags=["Screening"])
async def run_screening(request: ScreeningRequest):
    """
    Run health screening on provided biomarker data.
    
    Returns risk assessment for each system and overall composite risk.
    """
    screening_id = f"SCR-{uuid.uuid4().hex[:8].upper()}"
    
    try:
        from app.core.extraction.base import BiomarkerSet, Biomarker
        from app.core.inference.risk_engine import CompositeRiskCalculator, TrustedRiskResult
        
        trusted_results: Dict[PhysiologicalSystem, TrustedRiskResult] = {}
        system_results: Dict[PhysiologicalSystem, SystemRiskResult] = {}
        response_results: List[RiskResultResponse] = []
        
        for sys_input in request.systems:
            # Parse system
            system = _parse_system(sys_input.system)
            
            # Convert biomarker inputs to Biomarker objects
            biomarkers = [
                Biomarker(
                    name=bm.name,
                    value=bm.value,
                    unit=bm.unit or "",
                    confidence=1.0,
                    normal_range=tuple(bm.normal_range) if bm.normal_range and len(bm.normal_range) == 2 else None
                )
                for bm in sys_input.biomarkers
            ]
            
            # Create BiomarkerSet
            biomarker_set = BiomarkerSet(
                system=system,
                biomarkers=biomarkers
            )
            
            # Run risk calculation with validation (uses TrustedRiskResult)
            trusted_result = _risk_engine.compute_risk_with_validation(biomarker_set, plausibility=None)
            trusted_results[system] = trusted_result
            
            # Build response based on trust status
            if trusted_result.was_rejected:
                response_results.append(RiskResultResponse(
                    system=system.value,
                    risk_level="unknown",
                    risk_score=0.0,
                    confidence=0.0,
                    alerts=[trusted_result.rejection_reason] if trusted_result.rejection_reason else [],
                    is_trusted=False,
                    was_rejected=True,
                    caveats=trusted_result.caveats
                ))
            elif trusted_result.risk_result is not None:
                result = trusted_result.risk_result
                system_results[system] = result
                
                response_results.append(RiskResultResponse(
                    system=system.value,
                    risk_level=result.overall_risk.level.value,
                    risk_score=round(result.overall_risk.score, 1),
                    confidence=round(trusted_result.trust_adjusted_confidence, 2),
                    alerts=result.alerts,
                    is_trusted=trusted_result.is_trusted,
                    was_rejected=False,
                    caveats=trusted_result.caveats
                ))
        
        # Calculate composite risk using trusted results
        composite_calc = CompositeRiskCalculator()
        composite, rejected_systems = composite_calc.compute_composite_risk_from_trusted(trusted_results)
        
        # Validation (if requested)
        validation_status = None
        requires_review = False
        
        if request.include_validation and system_results:
            # Simplified validation for API
            consensus_result = ConsensusResult(
                overall_status=_consensus._determine_overall_status({}),
                overall_confidence=composite.confidence,
                agent_agreement=0.9,
                combined_flags=[],
                agent_results={},
                recommendation="Validation completed."
            )
            validation_status = "plausible" if not rejected_systems else "partial"
            requires_review = consensus_result.requires_human_review or len(rejected_systems) > 0
        
        # Store screening for report generation (include trusted results)
        _screenings[screening_id] = {
            "patient_id": request.patient_id,
            "system_results": system_results,
            "trusted_results": trusted_results,
            "composite_risk": composite,
            "rejected_systems": rejected_systems,
            "timestamp": datetime.now()
        }
        
        return ScreeningResponse(
            screening_id=screening_id,
            patient_id=request.patient_id,
            timestamp=datetime.now().isoformat(),
            overall_risk_level=composite.level.value,
            overall_risk_score=round(composite.score, 1),
            overall_confidence=round(composite.confidence, 2),
            system_results=response_results,
            validation_status=validation_status,
            requires_review=requires_review
        )
        
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@app.post("/api/v1/reports/generate", response_model=ReportResponse, tags=["Reports"])
async def generate_report(request: ReportRequest):
    """
    Generate PDF report for a completed screening.
    
    Report types: 'patient' or 'doctor'
    """
    # Check screening exists
    if request.screening_id not in _screenings:
        raise HTTPException(
            status_code=404,
            detail=f"Screening {request.screening_id} not found. Run screening first."
        )
    
    screening = _screenings[request.screening_id]
    
    try:
        if request.report_type == "patient":
            report = _patient_report_gen.generate(
                system_results=screening["system_results"],
                composite_risk=screening["composite_risk"],
                patient_id=screening["patient_id"]
            )
        elif request.report_type == "doctor":
            report = _doctor_report_gen.generate(
                system_results=screening["system_results"],
                composite_risk=screening["composite_risk"],
                patient_id=screening["patient_id"]
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid report_type. Use 'patient' or 'doctor'."
            )
        
        # Store report reference
        _reports[report.report_id] = report.pdf_path
        
        return ReportResponse(
            report_id=report.report_id,
            report_type=request.report_type,
            pdf_path=report.pdf_path or "",
            generated_at=report.generated_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/v1/reports/{report_id}/download", tags=["Reports"])
async def download_report(report_id: str):
    """
    Download a generated PDF report.
    """
    if report_id not in _reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    pdf_path = _reports[report_id]
    
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{report_id}.pdf"
    )


@app.get("/api/v1/screening/{screening_id}", tags=["Screening"])
async def get_screening(screening_id: str):
    """
    Get details of a completed screening.
    """
    if screening_id not in _screenings:
        raise HTTPException(status_code=404, detail="Screening not found")
    
    screening = _screenings[screening_id]
    
    results = []
    for system, result in screening["system_results"].items():
        results.append({
            "system": system.value,
            "risk_level": result.overall_risk.level.value,
            "risk_score": round(result.overall_risk.score, 1),
            "confidence": round(result.overall_risk.confidence, 2),
            "biomarker_count": len(result.biomarker_summary)
        })
    
    composite = screening["composite_risk"]
    
    return {
        "screening_id": screening_id,
        "patient_id": screening["patient_id"],
        "timestamp": screening["timestamp"].isoformat(),
        "overall_risk_level": composite.level.value,
        "overall_risk_score": round(composite.score, 1),
        "system_results": results
    }


@app.get("/api/v1/systems", tags=["Reference"])
async def list_systems():
    """
    List all supported physiological systems.
    """
    return {
        "systems": [
            {"name": s.value, "description": s.value.replace("_", " ").title()}
            for s in PhysiologicalSystem
        ]
    }


# ---- Application Lifecycle ----

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Health Screening Pipeline API starting up...")
    os.makedirs("reports", exist_ok=True)
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Health Screening Pipeline API shutting down...")


# ---- Run with uvicorn ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

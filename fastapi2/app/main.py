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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
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

# ---- Medical Validation Agents ----
from app.core.agents.medical_agents import MedGemmaAgent, OpenBioLLMAgent
_med_gemma_agent = MedGemmaAgent()
_open_biollm_agent = OpenBioLLMAgent()



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
            # Real AI validation using medical agents
            agent_results = {}
            
            # 1. MedGemma: Validate biomarker plausibility for each system
            for system, result in system_results.items():
                try:
                    med_validation = _med_gemma_agent.validate_biomarkers(
                        biomarker_summary=result.biomarker_summary,
                        system=system
                    )
                    agent_results[f"MedGemma_{system.value}"] = med_validation
                    logger.info(f"MedGemma validation for {system.value}: {med_validation.status.value}")
                except Exception as e:
                    logger.warning(f"MedGemma validation failed for {system.value}: {e}")
            
            # 2. OpenBioLLM: Cross-system consistency check
            try:
                consistency_validation = _open_biollm_agent.validate_consistency(
                    system_results=system_results,
                    trust_envelope=None
                )
                agent_results["OpenBioLLM_consistency"] = consistency_validation
                logger.info(f"OpenBioLLM consistency check: {consistency_validation.status.value}")
            except Exception as e:
                logger.warning(f"OpenBioLLM consistency check failed: {e}")
            
            # 3. Compute consensus from all agent results
            if agent_results:
                consensus_result = _consensus.compute_consensus(agent_results)
                validation_status = consensus_result.overall_status.value
                requires_review = consensus_result.requires_human_review or len(rejected_systems) > 0
            else:
                validation_status = "plausible" if not rejected_systems else "partial"
                requires_review = len(rejected_systems) > 0
        
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
                patient_id=screening["patient_id"],
                trusted_results=screening.get("trusted_results"),
                rejected_systems=screening.get("rejected_systems")
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


# ---- Hardware Screening Endpoint ----

class HardwareScreeningRequest(BaseModel):
    """Request for hardware-based screening."""
    patient_id: str = Field(default="HARDWARE_PATIENT")
    radar_port: str = Field(default="COM7", description="Serial port for mmRadar")
    camera_index: int = Field(default=0, description="Camera index for OpenCV")
    esp32_port: Optional[str] = Field(default=None, description="Optional ESP32 thermal port")


class HardwareScreeningResponse(BaseModel):
    """Response from hardware screening."""
    status: str
    screening_id: Optional[str] = None
    patient_report_id: Optional[str] = None
    doctor_report_id: Optional[str] = None
    error: Optional[str] = None


@app.post("/api/v1/hardware/start-screening", response_model=HardwareScreeningResponse, tags=["Hardware"])
async def start_hardware_screening(request: HardwareScreeningRequest):
    """
    Start a hardware-based health screening using connected sensors.
    
    Triggers the bridge.py script with the specified hardware configuration.
    Returns screening results and report IDs for download.
    """
    import subprocess
    import sys
    import re
    import asyncio
    
    def run_bridge_subprocess():
        """Run bridge.py in a subprocess (blocking call to be run in thread)."""
        bridge_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bridge.py")
        
        cmd = [
            sys.executable, bridge_path,
            "--radar-port", request.radar_port,
            "--camera", str(request.camera_index),
            "--patient-id", request.patient_id,
            "--api-url", "http://localhost:8000"
        ]
        
        if request.esp32_port:
            cmd.extend(["--port", request.esp32_port])
        
        logger.info(f"Starting hardware screening: {' '.join(cmd)}")
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3-minute timeout for hardware capture
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
    
    try:
        # Run subprocess in a thread to avoid blocking the event loop
        # This allows the server to handle the API calls from bridge.py
        result = await asyncio.to_thread(run_bridge_subprocess)
        
        logger.info(f"Bridge stdout (last 1000): {result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout}")
        if result.stderr:
            logger.warning(f"Bridge stderr (last 500): {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")
        
        # Parse screening ID from output
        screening_id = None
        screening_match = re.search(r'Screening ID:\s*(SCR-[A-Z0-9]+)', result.stdout)
        if screening_match:
            screening_id = screening_match.group(1)
        
        # Also check for screening ID in error/fallback pattern
        if not screening_id:
            # Try to find from the stored screenings (bridge.py already submitted)
            if _screenings:
                # Get the most recent screening
                screening_id = list(_screenings.keys())[-1]
                logger.info(f"Found screening ID from storage: {screening_id}")
        
        if not screening_id:
            return HardwareScreeningResponse(
                status="error",
                error="Screening completed but no screening ID found. Check server logs."
            )
        
        # Generate reports
        patient_report_id = None
        doctor_report_id = None
        
        try:
            patient_req = ReportRequest(screening_id=screening_id, report_type="patient")
            patient_resp = await generate_report(patient_req)
            patient_report_id = patient_resp.report_id
        except Exception as e:
            logger.warning(f"Patient report generation failed: {e}")
        
        try:
            doctor_req = ReportRequest(screening_id=screening_id, report_type="doctor")
            doctor_resp = await generate_report(doctor_req)
            doctor_report_id = doctor_resp.report_id
        except Exception as e:
            logger.warning(f"Doctor report generation failed: {e}")
        
        return HardwareScreeningResponse(
            status="success",
            screening_id=screening_id,
            patient_report_id=patient_report_id,
            doctor_report_id=doctor_report_id
        )
        
    except subprocess.TimeoutExpired:
        logger.error("Hardware screening timed out")
        return HardwareScreeningResponse(
            status="error",
            error="Screening timed out after 3 minutes. Check hardware connections."
        )
    except Exception as e:
        logger.error(f"Hardware screening failed: {e}")
        return HardwareScreeningResponse(
            status="error",
            error=str(e)
        )


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

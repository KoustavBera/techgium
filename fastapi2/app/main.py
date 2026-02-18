"""
Health Screening Pipeline - FastAPI Application (Unified Architecture)

Main application entry point with API endpoints for:
- Health screening data processing
- Report generation (Patient & Doctor PDFs)
- Validation status
- Hardware management (Camera, Radar, Thermal)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import json
from starlette.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage

# Medical Agent Imports
try:
    from agent.agent_graph import build_graph, load_model
    from agent.nodes import set_llm, set_token_callback
    AGENT_AVAILABLE = True
except ImportError:
    logger.warning("Medical Agent packages not found. Doctor chat will be disabled.")
    AGENT_AVAILABLE = False


# Load environment variables
load_dotenv()
import uuid
import logging

from app.core.extraction.base import PhysiologicalSystem
from app.core.inference.risk_engine import RiskEngine, RiskScore, RiskLevel, SystemRiskResult
from app.core.validation.trust_envelope import TrustEnvelope
from app.core.agents.medical_agents import AgentConsensus, ConsensusResult
from app.core.reports import PatientReportGenerator, DoctorReportGenerator
from app.core.hardware.manager import HardwareManager, HardwareConfig
from app.services.screening import ScreeningService

logger = logging.getLogger(__name__)

from app.models.screening import (
    BiomarkerInput,
    SystemInput,
    ScreeningRequest,
    RiskResultResponse,
    ScreeningResponse,
    ReportRequest,
    ReportResponse,
    HealthResponse
)

class DoctorChatRequest(BaseModel):
    """Request for medical doctor chat interaction."""
    query: str
    patient_id: Optional[str] = "GUEST"

# ---- Hardware Manager Singleton ----
_hw_manager = HardwareManager()


# ---- Application Lifespan ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage hardware lifecycle: startup → yield → shutdown."""
    # Initialize service
    app.state.screening_service = _screening_service
    
    # Initialize hardware (camera, radar, thermal)
    config = HardwareConfig(
        camera_index=0,
        radar_port=os.environ.get("RADAR_PORT", "COM7"),
        esp32_port=os.environ.get("ESP32_PORT", "COM6"),
    )
    await _hw_manager.startup(config, screening_service=_screening_service)
    
    # Initialize Medical Agent (Chiranjeevi)
    if AGENT_AVAILABLE:
        try:
            logger.info("Initializing Medical Agent (Chiranjeevi)...")
            llm = await run_in_threadpool(load_model)
            # LLM is now set in load_model() for all modules
            set_llm(llm)
            app.state.medical_agent = build_graph()
            logger.info("Medical Agent ready with Trust Envelope™")
        except Exception as e:
            logger.error(f"Failed to load Medical Agent: {e}")
            app.state.medical_agent = None
    else:
        app.state.medical_agent = None

    logger.info("API ready to accept requests")
    yield
    
    # Shutdown hardware
    await _hw_manager.shutdown()
    logger.info("Health Screening Pipeline API shut down.")


# ---- FastAPI Application ----

app = FastAPI(
    title="Health Screening Pipeline API",
    description="Non-invasive health screening using multimodal data processing (Unified Architecture)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Serve Frontend ----
frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
if not frontend_path.exists():
    # Fallback to local directory if parent path fails
    frontend_path = Path("frontend")

app.mount("/frontend", StaticFiles(directory=str(frontend_path)), name="frontend")


# ---- In-memory storage (replace with database in production) ----
_screenings: Dict[str, Dict[str, Any]] = {}
_reports: Dict[str, str] = {}
START_TIME = datetime.now()

# ---- Generators ----
_patient_report_gen = PatientReportGenerator(output_dir="reports")
_doctor_report_gen = DoctorReportGenerator(output_dir="reports")
_risk_engine = RiskEngine()
_consensus = AgentConsensus()

# ---- Multi-LLM Interpreter (Gemini + GPT-OSS + II-Medical) ----
from app.core.llm.multi_llm_interpreter import MultiLLMInterpreter
_multi_llm_interpreter = MultiLLMInterpreter()

# ---- Unified Services ----
_screening_service = ScreeningService(risk_engine=_risk_engine, interpreter=_multi_llm_interpreter)



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
    status = _hw_manager.get_sensor_status()
    hardware_status = {
        "radar": status.get("radar", {}).get("status", "unknown"),
        "thermal": status.get("thermal", {}).get("status", "unknown"),
        "camera": status.get("camera", {}).get("status", "unknown")
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=(datetime.now() - START_TIME).total_seconds(),
        hardware=hardware_status
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    status = _hw_manager.get_sensor_status()
    hardware_status = {
        "radar": status.get("radar", {}).get("status", "unknown"),
        "thermal": status.get("thermal", {}).get("status", "unknown"),
        "camera": status.get("camera", {}).get("status", "unknown")
    }

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=(datetime.now() - START_TIME).total_seconds(),
        hardware=hardware_status
    )


@app.post("/api/v1/screening", response_model=ScreeningResponse, tags=["Screening"])
async def run_screening(request: ScreeningRequest):
    """
    Run health screening on provided biomarker data.
    """
    try:
        # Prepare service input
        systems_input = []
        for sys_input in request.systems:
            systems_input.append({
                "system": sys_input.system,
                "biomarkers": [bm.dict() for bm in sys_input.biomarkers]
            })
        
        # NEW Phase 1.4: Minimum Data Quality Gate
        # Check overall data quality BEFORE processing risk
        quality_score = await _screening_service.assess_data_quality(request.data, systems_input)
        
        if quality_score < 0.5:
            logger.warning(f"Screening aborted: Insufficient data quality ({quality_score:.2f})")
            return ScreeningResponse(
                screening_id=f"REJ-{datetime.now().strftime('%H%M%S')}",
                patient_id=request.patient_id,
                timestamp=datetime.now().isoformat(),
                overall_risk_level="unknown",
                overall_risk_score=0.0,
                overall_confidence=quality_score,
                system_results=[],
                status="INSUFFICIENT_DATA",
                message="Screening aborted due to poor data quality. Please ensure face is clearly visible, lighting is sufficient, and you remain stable during capture."
            )

        # Call service
        result = await _screening_service.process_screening(
            patient_id=request.patient_id,
            systems_input=systems_input,
            include_validation=request.include_validation
        )
        
        # Store for report generation
        _screenings[result["screening_id"]] = {
            "patient_id": result["patient_id"],
            "system_results": result["system_results_internal"],
            "trusted_results": result["trusted_results"],
            "composite_risk": result["composite_risk"],
            "rejected_systems": result["rejected_systems"],
            "timestamp": result["timestamp"]
        }
        
        # Map to response model
        response_results = [RiskResultResponse(**r) for r in result["system_results"]]
        
        return ScreeningResponse(
            screening_id=result["screening_id"],
            patient_id=result["patient_id"],
            timestamp=result["timestamp"].isoformat(),
            overall_risk_level=result["overall_risk_level"],
            overall_risk_score=result["overall_risk_score"],
            overall_confidence=result["overall_confidence"],
            system_results=response_results,
            validation_status=result["validation_status"],
            requires_review=result["requires_review"]
        )
        
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")
        
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
    radar_port: str = Field(default="COM6", description="Serial port for mmRadar")
    camera_index: int = Field(default=0, description="Camera index for OpenCV")
    esp32_port: Optional[str] = Field(default="COM5", description="Optional ESP32 thermal port")


class HardwareScreeningResponse(BaseModel):
    """Response from hardware screening."""
    status: str
    screening_id: Optional[str] = None
    patient_report_id: Optional[str] = None
    doctor_report_id: Optional[str] = None
    error: Optional[str] = None


@app.get("/api/v1/hardware/status", tags=["Hardware"])
async def get_hardware_status():
    """
    Check status of connected hardware (live from HardwareManager).
    """
    status = _hw_manager.get_sensor_status()
    return {
        "radar": status["radar"]["status"] == "connected",
        "thermal": status["thermal"]["status"] == "connected",
        "camera": status["camera"]["status"] == "connected",
        "details": status
    }


@app.post("/api/v1/hardware/start-screening", response_model=HardwareScreeningResponse, tags=["Hardware"])
async def start_hardware_screening(request: HardwareScreeningRequest):
    """
    Start a hardware-based health screening using HardwareManager.
    
    Launches background scan: face capture → body capture → extraction → risk assessment.
    Poll /api/v1/hardware/scan-status for progress.
    """
    started = _hw_manager.start_scan(
        patient_id=request.patient_id,
        screenings_dict=_screenings,
    )
    
    if not started:
        return HardwareScreeningResponse(
            status="error",
            error="A scan is already in progress. Wait for it to complete."
        )
    
    return HardwareScreeningResponse(
        status="started",
        screening_id=None,  # Will be available via /scan-status when complete
    )


@app.get("/api/v1/hardware/scan-status", tags=["Hardware"])
async def get_scan_status():
    """
    Poll scan progress. Returns state, phase, message, progress %, and result IDs.
    
    States: idle, running, complete, error
    Phases: IDLE, INITIALIZING, FACE_ANALYSIS, BODY_ANALYSIS, PROCESSING, COMPLETE, ERROR
    """
    return _hw_manager.get_scan_status()


# ---- Doctor Chat Agent Endpoint ----

@app.post("/api/v1/doctor/chat", tags=["Doctor"])
async def doctor_chat(request: DoctorChatRequest):
    """
    Direct chat with Dr. Chiranjeevi using the LangGraph agent.
    Returns a streamed response of medical advice/responses with status updates.
    """
    if not app.state.medical_agent:
        raise HTTPException(
            status_code=503, 
            detail="Medical Agent is not initialized or model file is missing."
        )

    async def stream_generator():
        queue = asyncio.Queue()
        
        # Callback to put status events into the queue
        def on_status(status_dict: dict):
            # We use loop.call_soon_threadsafe because the agent runs in a threadpool
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "status",
                **status_dict
            })
        
        # Callback to put tokens into the queue
        def on_token(token: str):
            # We use loop.call_soon_threadsafe because the agent runs in a threadpool
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "token",
                "token": token
            })

        # Function to run the agent in a thread
        def run_agent():
            from agent.nodes import set_status_callback
            set_status_callback(on_status)
            set_token_callback(on_token)
            try:
                app.state.medical_agent.invoke(
                    {"messages": [HumanMessage(content=request.query)]},
                    config={"configurable": {"thread_id": request.patient_id or "GUEST"}}
                )
            finally:
                set_status_callback(None)  # Cleanup callback
                set_token_callback(None)  # Cleanup callback
                # Signal completion
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop = asyncio.get_event_loop()
        # Start agent in threadpool
        agent_task = asyncio.create_task(run_in_threadpool(run_agent))

        while True:
            event = await queue.get()
            if event is None: # None is our end-of-stream signal
                break
            # Yield as Server-Sent Event (SSE)
            yield f"data: {json.dumps(event)}\n\n"
        
        await agent_task

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# ---- Sensor Status & Live Camera Feed Endpoints ----


@app.get("/api/v1/hardware/sensor-status", tags=["Hardware"])
async def check_sensor_status():
    """
    Get live sensor connection status from HardwareManager.
    
    No port probing needed — reads from the already-connected hardware.
    """
    status = _hw_manager.get_sensor_status()
    return {
        "camera": status["camera"],
        "esp32": status["thermal"],  # Keep 'esp32' key for frontend compat
        "radar": status["radar"],
    }


@app.get("/api/v1/hardware/video-feed", tags=["Hardware"])
async def video_feed():
    """
    Live MJPEG video stream from HardwareManager's continuous capture.
    
    Use in an <img> tag: <img src="/api/v1/hardware/video-feed">
    The camera is owned by HardwareManager — no conflicts.
    """
    return StreamingResponse(
        _hw_manager.get_video_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


# ---- Run with uvicorn ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

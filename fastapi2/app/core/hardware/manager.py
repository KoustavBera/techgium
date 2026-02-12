"""
HardwareManager — Unified Kiosk-Style Singleton.

Owns all sensor hardware (Camera, Radar, Thermal) and provides:
- Continuous MJPEG video stream for the frontend
- On-demand scan orchestration (face capture → body capture → extraction → API)
- Real-time sensor status reporting

Replaces the old bridge.py subprocess model with in-process hardware control.
"""

import asyncio
import json
import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Generator
import logging

import numpy as np
import cv2

from app.core.hardware.drivers import (
    CameraCapture,
    RadarReader,
    ESP32Reader,
    SERIAL_AVAILABLE,
)

# Extractors (same ones used by DataFusion in bridge.py)
from app.core.extraction.cardiovascular import CardiovascularExtractor
from app.core.extraction.cns import CNSExtractor
from app.core.extraction.skeletal import SkeletalExtractor
from app.core.extraction.skin import SkinExtractor
from app.core.extraction.pulmonary import PulmonaryExtractor
from app.core.extraction.renal import RenalExtractor
from app.core.extraction.eyes import EyeExtractor
from app.core.extraction.nasal import NasalExtractor
from app.core.extraction.base import BiomarkerSet

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class HardwareConfig:
    """Configuration for the unified hardware manager."""
    # Camera
    camera_index: int = 0
    camera_fps: int = 30
    face_capture_seconds: int = 20
    body_capture_seconds: int = 20
    
    # Serial ports
    radar_port: str = "COM7"
    radar_baud: int = 115200
    esp32_port: str = "COM6"
    esp32_baud: int = 115200
    
    # MJPEG quality
    jpeg_quality: int = 80


# ==============================================================================
# CUSTOM JSON ENCODER
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==============================================================================
# HARDWARE MANAGER SINGLETON
# ==============================================================================

class HardwareManager:
    """
    Singleton that owns all hardware and provides video streaming + analysis.
    
    Lifecycle:
        1. startup() — opens camera, connects serial sensors
        2. get_video_stream() — yields MJPEG frames continuously
        3. start_scan() — runs analysis pipeline in background thread
        4. shutdown() — releases all hardware
    """
    
    _instance: Optional["HardwareManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.config = HardwareConfig()
        
        # Hardware handles
        self.camera: Optional[CameraCapture] = None
        self.radar: Optional[RadarReader] = None
        self.thermal: Optional[ESP32Reader] = None
        
        # Video stream state
        self._latest_frame_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._stream_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Visual feedback state
        self._current_distance_warning: Optional[str] = None
        self._enable_overlays = True  # Enable face mesh and pose skeleton drawing
        
        # Scan state
        self._scan_active = False
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_status: Dict[str, Any] = {
            "state": "idle",
            "phase": "IDLE",
            "message": "Ready for screening",
            "progress": 0,
            "screening_id": None,
            "patient_report_id": None,
            "doctor_report_id": None,
            "user_warnings": {
                "distance_warning": None,  # "too_close", "too_far", or None
                "face_detected": True,
                "pose_detected": True,
            },
        }
        self._scan_lock = threading.Lock()
        
        # Recording state (Broadcast Model)
        self._recording_buffer: List[Dict[str, Any]] = []  # Stores {'frame': np.ndarray, 'timestamp': float}
        self._recording_active = False
        self._recording_lock = threading.Lock()
        
        # Extractors (same set as DataFusion)
        self._extractors_initialized = False
        self.cv_extractor = None
        self.cns_extractor = None
        self.skeletal_extractor = None
        self.skin_extractor = None
        self.pulmonary_extractor = None
        self.renal_extractor = None
        self.eye_extractor = None
        self.nasal_extractor = None
        
        logger.info("HardwareManager singleton created")
    
    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------
    
    async def startup(self, config: Optional[HardwareConfig] = None):
        """Initialize all hardware. Call during FastAPI lifespan startup."""
        if config:
            self.config = config
        
        logger.info("=" * 60)
        logger.info("HardwareManager starting up (Unified Architecture)")
        logger.info("=" * 60)
        
        # 1. Camera
        try:
            self.camera = CameraCapture(
                camera_index=self.config.camera_index,
                fps=self.config.camera_fps
            )
            if self.camera.open():
                logger.info(f"✅ Camera {self.config.camera_index} opened")
            else:
                logger.warning("⚠️ Camera not available — video feed will be empty")
                self.camera = None
        except Exception as e:
            logger.error(f"Camera init failed: {e}")
            self.camera = None
        
        # 2. Radar (Seeed MR60BHA2)
        if SERIAL_AVAILABLE:
            try:
                self.radar = RadarReader(
                    port=self.config.radar_port,
                    baud=self.config.radar_baud
                )
                if self.radar.connect():
                    self.radar.start_reading()
                    logger.info(f"✅ Radar connected on {self.config.radar_port}")
                else:
                    logger.warning(f"⚠️ Radar not available on {self.config.radar_port}")
                    self.radar = None
            except Exception as e:
                logger.error(f"Radar init failed: {e}")
                self.radar = None
        else:
            logger.warning("⚠️ pyserial not installed — radar disabled")
            self.radar = None
        
        # 3. Thermal (ESP32 + MLX90640)
        if SERIAL_AVAILABLE:
            try:
                self.thermal = ESP32Reader(
                    port=self.config.esp32_port,
                    baud=self.config.esp32_baud
                )
                if self.thermal.connect():
                    self.thermal.start_reading()
                    logger.info(f"✅ Thermal connected on {self.config.esp32_port}")
                else:
                    logger.warning(f"⚠️ Thermal not available on {self.config.esp32_port}")
                    self.thermal = None
            except Exception as e:
                logger.error(f"Thermal init failed: {e}")
                self.thermal = None
        else:
            self.thermal = None
        
        # 4. Initialize extractors
        self._init_extractors()
        
        # 5. Start continuous capture thread
        if self.camera:
            self._running = True
            self._stream_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="hw-stream"
            )
            self._stream_thread.start()
            logger.info("✅ Continuous capture thread started")
        
        logger.info("HardwareManager startup complete")
    
    async def shutdown(self):
        """Release all hardware. Call during FastAPI lifespan shutdown."""
        logger.info("HardwareManager shutting down...")
        
        self._running = False
        
        # Wait for stream thread
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        
        # Wait for scan thread
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_active = False
            self._scan_thread.join(timeout=5.0)
        
        # Release hardware (try/finally for robustness)
        try:
            if self.camera:
                self.camera.close()
                self.camera = None
        except Exception as e:
            logger.error(f"Camera cleanup error: {e}")
        
        try:
            if self.radar:
                self.radar.disconnect()
                self.radar = None
        except Exception as e:
            logger.error(f"Radar cleanup error: {e}")
        
        try:
            if self.thermal:
                self.thermal.disconnect()
                self.thermal = None
        except Exception as e:
            logger.error(f"Thermal cleanup error: {e}")
        
        logger.info("HardwareManager shutdown complete")
    
    # ------------------------------------------------------------------
    # EXTRACTORS
    # ------------------------------------------------------------------
    
    def _init_extractors(self):
        """Initialize all biomarker extractors (same set as DataFusion in bridge.py)."""
        if self._extractors_initialized:
            return
        
        self.cv_extractor = CardiovascularExtractor()
        self.cns_extractor = CNSExtractor()
        self.skeletal_extractor = SkeletalExtractor()
        self.skin_extractor = SkinExtractor()
        self.pulmonary_extractor = PulmonaryExtractor()
        self.renal_extractor = RenalExtractor()
        self.eye_extractor = EyeExtractor(
            sample_rate=30.0,
            frame_width=1280,
            frame_height=720
        )
        self.nasal_extractor = NasalExtractor()
        self._extractors_initialized = True
        logger.info("✅ All 8 extractors initialized")
    
    # ------------------------------------------------------------------
    # CONTINUOUS CAPTURE (Video Stream)
    # ------------------------------------------------------------------
    
    def _capture_loop(self):
        """Background thread: continuously reads frames, draws overlays, and encodes to JPEG."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        
        while self._running and self.camera:
            frame = self.camera.read_frame()
            if frame is not None:
                # Apply server-side overlays
                if self._enable_overlays:
                    # Consolidated MediaPipe Pass (Optimization)
                    results = self.camera.detect_all(frame)
                    
                    # 1. Overlay based on Phase (Logic Restoration)
                    current_phase = self._scan_status.get("phase", "IDLE")
                    if current_phase == "FACE_ANALYSIS":
                        frame = self.camera.draw_face_mesh_on_frame(frame, results)
                    elif current_phase == "BODY_ANALYSIS":
                        frame = self.camera.draw_pose_skeleton_on_frame(frame, results)
                    
                    # 2. Calculate and draw distance warnings
                    warning_type, face_width = self.camera.calculate_face_distance(frame, results)
                    if warning_type:
                        frame = self.camera.add_distance_warning_overlay(frame, warning_type)
                    
                    # Update status for frontend polling
                    self._current_distance_warning = warning_type
                    self._update_scan_status(
                        user_warnings={
                            "distance_warning": warning_type,
                            "face_detected": face_width is not None,
                            "pose_detected": results.get("pose") is not None and results["pose"].pose_landmarks is not None,
                        }
                    )
                
                # Encode to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                if ret:
                    with self._frame_lock:
                        self._latest_frame_jpeg = jpeg.tobytes()
                
                # Record frame if scan is active
                if self._recording_active:
                    with self._recording_lock:
                        self._recording_buffer.append({
                            'frame': frame.copy(),
                            'timestamp': time.time()
                        })
            else:
                time.sleep(0.01)  # Brief sleep if no frame
    
    def get_video_stream(self) -> Generator[bytes, None, None]:
        """
        Yield MJPEG frames for StreamingResponse.
        
        Usage in FastAPI:
            return StreamingResponse(
                hw_manager.get_video_stream(),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )
        """
        while self._running:
            with self._frame_lock:
                frame_data = self._latest_frame_jpeg
            
            if frame_data:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    frame_data +
                    b'\r\n'
                )
            
            time.sleep(1.0 / 30)  # ~30 FPS target
    
    # ------------------------------------------------------------------
    # SENSOR STATUS
    # ------------------------------------------------------------------
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Return live connection status of all sensors."""
        result = {
            "camera": {
                "status": "disconnected",
                "detail": "Not initialized"
            },
            "radar": {
                "status": "disconnected",
                "detail": "Not initialized"
            },
            "thermal": {
                "status": "disconnected",
                "detail": "Not initialized"
            }
        }
        
        # Camera
        if self.camera and self.camera.cap and self.camera.cap.isOpened():
            w = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result["camera"] = {
                "status": "connected",
                "detail": f"Camera {self.config.camera_index} — {w}x{h} resolution"
            }
        
        # Radar
        if self.radar and self.radar.serial_conn and self.radar.serial_conn.is_open:
            latest = self.radar.get_latest_data()
            hr = latest.get('radar', {}).get('heart_rate', 0) if latest else 0
            result["radar"] = {
                "status": "connected",
                "detail": f"Seeed MR60BHA2 on {self.config.radar_port}" + (f" (HR: {hr} bpm)" if hr else "")
            }
        
        # Thermal
        if self.thermal and self.thermal.serial_conn and self.thermal.serial_conn.is_open:
            latest = self.thermal.get_latest_data()
            temp = latest.get('thermal', {}).get('core_regions', {}).get('canthus_mean', 0) if latest else 0
            result["thermal"] = {
                "status": "connected",
                "detail": f"ESP32 MLX90640 on {self.config.esp32_port}" + (f" ({temp:.1f}°C)" if temp else "")
            }
        
        return result
    
    # ------------------------------------------------------------------
    # SCAN ORCHESTRATION
    # ------------------------------------------------------------------
    
    def get_scan_status(self) -> Dict[str, Any]:
        """Return current scan progress for frontend polling."""
        with self._scan_lock:
            return self._scan_status.copy()
    
    def _update_scan_status(self, **kwargs):
        """Thread-safe status update."""
        with self._scan_lock:
            self._scan_status.update(kwargs)

    def _wait_for_alignment(self, target_phase: str):
        """Busy-wait until user is correctly positioned for the phase."""
        logger.info(f"⏳ Waiting for alignment in {target_phase}...")
        
        # Initial wait for first frame processing
        time.sleep(1.0)
        
        while self._scan_active:
            warning = self._current_distance_warning
            
            # Achieving alignment: No distance warning AND correct detection present
            with self._scan_lock:
                user_warnings = self._scan_status.get("user_warnings", {})
            
            alignment_met = False
            if target_phase == "FACE_ANALYSIS":
                if warning is None and user_warnings.get("face_detected"):
                    alignment_met = True
            elif target_phase == "BODY_ANALYSIS":
                if warning is None and user_warnings.get("pose_detected"):
                    alignment_met = True
            
            if alignment_met:
                logger.info(f"✅ Alignment achieved for {target_phase}")
                break
            
            # Guide user through status message
            msg = "Ready. Please look at the camera."
            if warning == "too_close":
                msg = "MOVE BACK - You are too close."
            elif warning == "too_far":
                msg = "MOVE CLOSER - You are too far."
            elif target_phase == "BODY_ANALYSIS" and not user_warnings.get("pose_detected"):
                msg = "STEP BACK - Full body must be visible."
            elif target_phase == "FACE_ANALYSIS" and not user_warnings.get("face_detected"):
                msg = "LOOK HERE - Face not detected."

            self._update_scan_status(message=msg)
            time.sleep(0.2)
    
    def start_scan(self, patient_id: str, screenings_dict: Dict, app_internals: Dict = None) -> bool:
        """
        Launch analysis in a background thread.
        
        Args:
            patient_id: Patient identifier
            screenings_dict: Reference to main.py's _screenings dict for direct injection
            app_internals: Dict with references to _risk_engine, _multi_llm_interpreter etc.
        
        Returns:
            True if scan started, False if one is already running
        """
        if self._scan_active:
            logger.warning("Scan already in progress")
            return False
        
        logger.info(f"Initiating scan for patient: {patient_id}")
        self._scan_active = True
        self._update_scan_status(
            state="running",
            phase="INITIALIZING",
            message="Preparing sensors...",
            progress=5,
            screening_id=None,
            patient_report_id=None,
            doctor_report_id=None,
        )
        
        logger.info("Launching scan thread...")
        logger.info(f"Target function: {self._run_scan}")
        self._scan_thread = threading.Thread(
            target=self._run_scan,
            args=(patient_id, screenings_dict, app_internals),
            daemon=True,
            name="hw-scan"
        )
        logger.info("Starting thread...")
        self._scan_thread.start()
        logger.info(f"Thread started: {self._scan_thread.is_alive()}")
        logger.info("Scan thread launched successfully")
        return True
    
    def _run_scan(self, patient_id: str, screenings_dict: Dict, app_internals: Dict = None):
        """
        Full scanning pipeline (runs in background thread).
        
        Mirrors bridge.py HardwareBridge.run_single_screening() logic:
        1. Clear sensor queues
        2. Phase 1: Face capture (face ROIs + FaceMesh landmarks)
        3. Phase 2: Body capture (pose landmarks)
        4. Aggregate radar + thermal data
        5. Run all 8 extractors
        6. POST to internal screening endpoint for risk assessment
        7. Auto-generate reports
        """
        import httpx
        
        try:
            logger.info("=" * 60)
            logger.info(f"Starting scan for patient: {patient_id} (THREAD STARTED)")
            logger.info("=" * 60)
            
            radar_data = None
            esp32_data = None
            face_frames = []
            face_landmarks_sequence = []
            pose_sequence = []
            
            # Clear sensor queues to sync with video start
            if self.radar:
                logger.info("Clearing radar queue...")
                while not self.radar.data_queue.empty():
                    try:
                        self.radar.data_queue.get_nowait()
                    except:
                        break
                logger.info("Radar queue cleared.")
                
            if self.thermal:
                logger.info("Clearing thermal queue...")
                while not self.thermal.data_queue.empty():
                    try:
                        self.thermal.data_queue.get_nowait()
                    except:
                        break
                logger.info("Thermal queue cleared.")
            
            # ============================================================
            # Phase 1: Face capture
            # ============================================================
            logger.info("Checking if camera is available for Phase 1...")
            if self.camera:
                logger.info("Camera found. Updating status to FACE_ANALYSIS...")
                self._update_scan_status(
                    phase="FACE_ANALYSIS",
                    message="Analyzing facial features...",
                    progress=10,
                )
                
                # Wait for user to be correctly positioned
                self._wait_for_alignment("FACE_ANALYSIS")
                
                # Start recording
                with self._recording_lock:
                    self._recording_buffer = []
                    self._recording_active = True
                
                # Wait for duration
                time.sleep(self.config.face_capture_seconds)
                
                # Stop recording
                self._recording_active = False
                with self._recording_lock:
                     # Copy buffer to local list
                    raw_captures = list(self._recording_buffer)
                
                logger.info(f"Captured {len(raw_captures)} face frames from broadcast")
                
                # Post-process frames (extract ROIs and landmarks)
                processed_data = []
                for item in raw_captures:
                     frame = item['frame']
                     processed_data.append({
                        'roi': self.camera.extract_face_roi(frame),
                        'landmarks': self.camera.extract_face_landmarks(frame)
                     })

                face_frames = [d['roi'] for d in processed_data if d.get('roi') is not None]
                face_landmarks_sequence = [d['landmarks'] for d in processed_data if d.get('landmarks') is not None]
                
                logger.info(f"Extracted {len(face_frames)} face ROIs, {len(face_landmarks_sequence)} landmark sets")
            
            self._update_scan_status(progress=40)
            
            # ============================================================
            # Phase 2: Body capture
            # ============================================================
            # ============================================================
            # Phase 2: Body capture
            # ============================================================
            if self.camera:
                self._update_scan_status(
                    phase="BODY_ANALYSIS",
                    message="Analyzing posture and movement...",
                    progress=45,
                )
                
                logger.info(f"🚶 Phase 2: Body capture ({self.config.body_capture_seconds}s)")
                logger.info(f"🚶 Phase 2: Body capture ({self.config.body_capture_seconds}s)")
                
                # Wait for user to be correctly positioned
                self._wait_for_alignment("BODY_ANALYSIS")
                
                # Start recording
                with self._recording_lock:
                    self._recording_buffer = []
                    self._recording_active = True
                
                # Wait for duration
                time.sleep(self.config.body_capture_seconds)
                
                # Stop recording
                self._recording_active = False
                with self._recording_lock:
                    raw_captures = list(self._recording_buffer)
                
                logger.info(f"Captured {len(raw_captures)} body frames from broadcast")
                
                # Post-process frames
                pose_sequence = []
                for item in raw_captures:
                    pose = self.camera.extract_pose_from_frame(item['frame'])
                    if pose is not None:
                        pose_sequence.append(pose)

                logger.info(f"Extracted {len(pose_sequence)} pose frames")
            
            self._update_scan_status(progress=70)
            
            # ============================================================
            # Data Aggregation
            # ============================================================
            self._update_scan_status(
                phase="PROCESSING",
                message="Processing sensor data...",
                progress=75,
            )
            
            # Aggregate Radar
            radar_data = self._aggregate_radar()
            
            # Aggregate Thermal
            esp32_data = self._aggregate_thermal()
            
            # ============================================================
            # Build screening request (run all 8 extractors)
            # ============================================================
            self._update_scan_status(
                message="Running biomarker extraction...",
                progress=80,
            )
            
            request_payload = self._build_screening_request(
                patient_id=patient_id,
                radar_data=radar_data,
                esp32_data=esp32_data,
                face_frames=face_frames if face_frames else None,
                pose_sequence=pose_sequence if pose_sequence else None,
                face_landmarks_sequence=face_landmarks_sequence if face_landmarks_sequence else None,
                fps=self.config.camera_fps
            )
            
            logger.info(f"Screening request: {len(request_payload['systems'])} systems")
            for sys_data in request_payload['systems']:
                logger.info(f"  - {sys_data['system']}: {len(sys_data['biomarkers'])} biomarkers")
            
            # ============================================================
            # Submit to internal API (reuse all risk engine + LLM logic)
            # ============================================================
            self._update_scan_status(
                message="Computing risk assessment...",
                progress=85,
            )
            
            screening_id = None
            patient_report_id = None
            doctor_report_id = None
            
            # Use httpx to call our own screening endpoint
            try:
                with httpx.Client(base_url="http://localhost:8000", timeout=60.0) as client:
                    # POST screening
                    resp = client.post("/api/v1/screening", json=request_payload)
                    resp.raise_for_status()
                    result = resp.json()
                    screening_id = result.get("screening_id")
                    
                    logger.info(f"✅ Screening completed: {screening_id}")
                    logger.info(f"   Overall risk: {result.get('overall_risk_level')} ({result.get('overall_risk_score')})")
                    
                    # Generate reports
                    self._update_scan_status(
                        message="Generating reports...",
                        progress=90,
                    )
                    
                    try:
                        pr = client.post("/api/v1/reports/generate", json={
                            "screening_id": screening_id,
                            "report_type": "patient"
                        })
                        if pr.status_code == 200:
                            patient_report_id = pr.json().get("report_id")
                            logger.info(f"Patient report: {patient_report_id}")
                    except Exception as e:
                        logger.warning(f"Patient report failed: {e}")
                    
                    try:
                        dr = client.post("/api/v1/reports/generate", json={
                            "screening_id": screening_id,
                            "report_type": "doctor"
                        })
                        if dr.status_code == 200:
                            doctor_report_id = dr.json().get("report_id")
                            logger.info(f"Doctor report: {doctor_report_id}")
                    except Exception as e:
                        logger.warning(f"Doctor report failed: {e}")
                    
            except Exception as e:
                logger.error(f"Internal API call failed: {e}")
                self._update_scan_status(
                    state="error",
                    phase="ERROR",
                    message=f"API error: {str(e)}",
                    progress=0,
                )
                self._scan_active = False
                return
            
            # ============================================================
            # Done!
            # ============================================================
            self._update_scan_status(
                state="complete",
                phase="COMPLETE",
                message="Screening complete!",
                progress=100,
                screening_id=screening_id,
                patient_report_id=patient_report_id,
                doctor_report_id=doctor_report_id,
            )
            
            logger.info("✅ Scan pipeline complete")
        
        except Exception as e:
            logger.error(f"Scan failed: {e}", exc_info=True)
            self._update_scan_status(
                state="error",
                phase="ERROR",
                message=f"Scan error: {str(e)}",
                progress=0,
            )
        finally:
            self._scan_active = False
    
    # ------------------------------------------------------------------
    # DATA AGGREGATION (from bridge.py HardwareBridge.run_single_screening)
    # ------------------------------------------------------------------
    
    def _aggregate_radar(self) -> Optional[Dict]:
        """Aggregate radar readings from queue into averaged values."""
        if not self.radar:
            return None
        
        items = []
        while not self.radar.data_queue.empty():
            try:
                items.append(self.radar.data_queue.get_nowait())
            except queue.Empty:
                break
        
        if not items:
            latest = self.radar.get_latest_data()
            if latest:
                logger.warning("No radar data in queue — using last known")
            return latest
        
        avg_hr = int(sum(i['radar']['heart_rate'] for i in items) / len(items))
        avg_resp = round(sum(i['radar']['respiration_rate'] for i in items) / len(items), 1)
        
        result = items[-1].copy()
        result['radar']['heart_rate'] = avg_hr
        result['radar']['respiration_rate'] = avg_resp
        
        logger.info(f"Aggregated {len(items)} radar samples. Avg HR: {avg_hr}, Avg Resp: {avg_resp}")
        return result
    
    def _aggregate_thermal(self) -> Optional[Dict]:
        """Aggregate thermal readings from queue into averaged values."""
        if not self.thermal:
            logger.warning("⚠️ Thermal not initialized — no thermal data")
            return None
        
        items = []
        while not self.thermal.data_queue.empty():
            try:
                items.append(self.thermal.data_queue.get_nowait())
            except queue.Empty:
                break
        
        if not items:
            latest = self.thermal.get_latest_data()
            if latest:
                logger.warning("No thermal data in queue — using last known")
            return latest
        
        logger.info(f"📊 Collected {len(items)} thermal samples")
        
        # Detect format
        is_firmware = 'core_regions' in items[0].get('thermal', {})
        
        def get_val(item, category, field_name, default=0.0):
            return item.get('thermal', {}).get(category, {}).get(field_name, default)
        
        if is_firmware:
            canthus_valid = [get_val(i, 'core_regions', 'canthus_mean') for i in items 
                           if get_val(i, 'core_regions', 'canthus_mean') > 25.0]
            neck_valid = [get_val(i, 'core_regions', 'neck_mean') for i in items 
                         if get_val(i, 'core_regions', 'neck_mean') > 25.0]
            stability_vals = [get_val(i, 'stability_metrics', 'canthus_range') for i in items]
            asymmetry_vals = [get_val(i, 'symmetry', 'cheek_asymmetry') for i in items]
            gradient_vals = [get_val(i, 'gradients', 'forehead_nose_gradient') for i in items]
            
            avg_canthus = round(sum(canthus_valid)/len(canthus_valid), 2) if canthus_valid else 0.0
            avg_neck = round(sum(neck_valid)/len(neck_valid), 2) if neck_valid else 0.0
            avg_stability = round(sum(stability_vals)/len(stability_vals), 2) if stability_vals else 0.0
            avg_asymmetry = round(sum(asymmetry_vals)/len(asymmetry_vals), 3) if asymmetry_vals else 0.0
            avg_gradient = round(sum(gradient_vals)/len(gradient_vals), 2) if gradient_vals else 0.0
            
            result = {
                'timestamp': items[-1].get('timestamp', 0),
                'thermal': {
                    'core_regions': {'canthus_mean': avg_canthus, 'neck_mean': avg_neck},
                    'stability_metrics': {'canthus_range': avg_stability},
                    'symmetry': {'cheek_asymmetry': avg_asymmetry},
                    'gradients': {'forehead_nose_gradient': avg_gradient}
                }
            }
            logger.info(
                f"Aggregated thermal (firmware). Canthus: {avg_canthus}°C, "
                f"Neck: {avg_neck}°C, Stability: {avg_stability}"
            )
            return result
        else:
            # Legacy format
            neck_temps = [get_val(i, 'fever', 'neck_temp') for i in items 
                         if get_val(i, 'fever', 'neck_temp') > 0]
            avg_neck = sum(neck_temps)/len(neck_temps) if neck_temps else 0.0
            stress_vals = [get_val(i, 'autonomic', 'stress_gradient') for i in items]
            avg_stress = sum(stress_vals)/len(stress_vals) if stress_vals else 0.0
            
            esp32_data = items[-1].copy()
            if 'fever' in esp32_data.get('thermal', {}):
                esp32_data['thermal']['fever']['neck_temp'] = round(avg_neck, 2)
            if 'autonomic' in esp32_data.get('thermal', {}):
                esp32_data['thermal']['autonomic']['stress_gradient'] = round(avg_stress, 2)
            
            logger.info(f"Aggregated thermal (legacy). Avg Neck: {avg_neck:.1f}")
            return esp32_data
    
    # ------------------------------------------------------------------
    # EXTRACTION (ported from DataFusion.build_screening_request)
    # ------------------------------------------------------------------
    
    def _transform_biomarker_set(self, bm_set: BiomarkerSet) -> Dict[str, Any]:
        """Convert BiomarkerSet to API-compatible system dictionary."""
        return {
            "system": bm_set.system.value,
            "biomarkers": [bm.to_dict() for bm in bm_set.biomarkers]
        }
    
    def _build_screening_request(
        self,
        patient_id: str,
        radar_data: Optional[Dict] = None,
        esp32_data: Optional[Dict] = None,
        face_frames: Optional[List[np.ndarray]] = None,
        pose_sequence: Optional[List[np.ndarray]] = None,
        face_landmarks_sequence: Optional[List[np.ndarray]] = None,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """Build complete screening request using all 8 extractors.
        
        This is a direct port of DataFusion.build_screening_request() from bridge.py.
        """
        systems = []
        
        raw_data_context = {"fps": fps, "patient_id": patient_id}
        
        if radar_data:
            raw_data_context["radar_data"] = radar_data
        
        if face_landmarks_sequence:
            raw_data_context["face_landmarks_sequence"] = face_landmarks_sequence
        
        # Flatten thermal data (supports firmware + legacy formats)
        if esp32_data and 'thermal' in esp32_data:
            thermal = esp32_data['thermal']
            is_firmware_format = 'core_regions' in thermal
            
            if is_firmware_format:
                core = thermal.get('core_regions', {})
                stability = thermal.get('stability_metrics', {})
                symmetry = thermal.get('symmetry', {})
                gradients = thermal.get('gradients', {})
                
                canthus_mean = core.get('canthus_mean')
                neck_mean = core.get('neck_mean')
                face_max = core.get('face_max')
                canthus_range = stability.get('canthus_range')
                
                face_mean_temp = None
                if canthus_mean is not None and neck_mean is not None:
                    face_mean_temp = (canthus_mean + neck_mean) / 2.0
                elif canthus_mean is not None:
                    face_mean_temp = canthus_mean
                elif neck_mean is not None:
                    face_mean_temp = neck_mean
                
                inflammation_pct = None
                if canthus_range is not None:
                    inflammation_pct = min(10.0, max(0.0, (canthus_range - 0.8) * 8.33))
                
                raw_data_context['thermal_data'] = {
                    'fever_neck_temp': neck_mean,
                    'fever_canthus_temp': canthus_mean,
                    'fever_face_max': face_max,
                    'thermal_stability': canthus_range,
                    'inflammation_pct': inflammation_pct,
                    'face_mean_temp': face_mean_temp,
                    'thermal_asymmetry': symmetry.get('cheek_asymmetry'),
                    'left_cheek_temp': None,
                    'right_cheek_temp': None,
                    'diabetes_canthus_temp': canthus_mean,
                    'diabetes_risk_flag': (canthus_mean is not None and canthus_mean < 35.5),
                    'stress_gradient': abs(gradients.get('forehead_nose_gradient', 0)) if gradients.get('forehead_nose_gradient') is not None else None,
                    'nose_temp': None,
                    'forehead_temp': None,
                }
            else:
                raw_data_context['thermal_data'] = {
                    'fever_neck_temp': thermal.get('fever', {}).get('neck_temp'),
                    'fever_canthus_temp': thermal.get('fever', {}).get('canthus_temp'),
                    'thermal_stability': None,
                    'inflammation_pct': thermal.get('inflammation', {}).get('hot_pixel_pct'),
                    'face_mean_temp': thermal.get('inflammation', {}).get('face_mean_temp'),
                    'thermal_asymmetry': thermal.get('cardiovascular', {}).get('thermal_asymmetry'),
                    'left_cheek_temp': thermal.get('cardiovascular', {}).get('left_cheek_temp'),
                    'right_cheek_temp': thermal.get('cardiovascular', {}).get('right_cheek_temp'),
                    'diabetes_canthus_temp': thermal.get('diabetes', {}).get('canthus_temp'),
                    'diabetes_risk_flag': thermal.get('diabetes', {}).get('risk_flag', 0) == 1,
                    'stress_gradient': thermal.get('autonomic', {}).get('stress_gradient'),
                    'nose_temp': thermal.get('autonomic', {}).get('nose_temp'),
                    'forehead_temp': thermal.get('autonomic', {}).get('forehead_temp'),
                }
            
            raw_data_context['esp32_data'] = esp32_data
        elif esp32_data:
            raw_data_context['esp32_data'] = esp32_data
            
        if face_frames:
            raw_data_context["face_frames"] = face_frames
        if pose_sequence:
            raw_data_context["pose_sequence"] = pose_sequence
        
        # Run all 8 extractors (same order as bridge.py DataFusion)
        extractor_map = [
            ("Pulmonary", self.pulmonary_extractor),
            ("Cardiovascular", self.cv_extractor),
            ("Skin", self.skin_extractor),
            ("CNS", self.cns_extractor),
            ("Skeletal", self.skeletal_extractor),
            ("Renal", self.renal_extractor),
            ("Eyes", self.eye_extractor),
            ("Nasal", self.nasal_extractor),
        ]
        
        current_progress = 80
        step_increment = 5 / len(extractor_map)  # Spread 5% progress across extractors

        for i, (name, extractor) in enumerate(extractor_map):
            try:
                # Update status for frontend feedback
                self._update_scan_status(
                    message=f"Analyzing {name} system...",
                    progress=int(current_progress + (i * step_increment))
                )
                
                result = extractor.extract(raw_data_context)
                if result.biomarkers:
                    systems.append(self._transform_biomarker_set(result))
            except Exception as e:
                logger.error(f"{name} extraction failed: {e}")
        
        return {
            "patient_id": patient_id,
            "include_validation": True,
            "systems": systems
        }

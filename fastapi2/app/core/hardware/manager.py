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
<<<<<<< HEAD
=======
from app.core.extraction.renal import RenalExtractor
>>>>>>> 5fb8453bef017d5d9f62d222de7a732819a54135
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
    face_capture_seconds: int = 10
    body_capture_seconds: int = 10
    
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
        elif hasattr(obj, 'isoformat'): # Handle datetime and similar
            return obj.isoformat()
        elif hasattr(obj, 'model_dump'): # Pydantic v2
            return obj.model_dump()
        elif hasattr(obj, 'dict'): # Pydantic v1
            return obj.dict()
        elif hasattr(obj, '__dict__'): # Generic objects
            return obj.__dict__
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
        
        # Service and Event Loop
        self.screening_service = None
        self.loop = None
        
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
        self._enable_overlays = True
        
        # Inference State
        self._last_inference_results = {
            "face_mesh": None, "pose": None, "face_det": None
        }
        self._frame_counter = 0
        self._inference_interval = 2  # Run inference every 2nd frame (15 FPS inference, 30 FPS video)
        
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
<<<<<<< HEAD
=======
        self.renal_extractor = None
>>>>>>> 5fb8453bef017d5d9f62d222de7a732819a54135
        self.eye_extractor = None
        self.nasal_extractor = None
        
        logger.info("HardwareManager singleton created")
    
    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------
    
    async def startup(self, config: Optional[HardwareConfig] = None, screening_service = None):
        """Initialize all hardware. Call during FastAPI lifespan startup."""
        if config:
            self.config = config
        
        self.screening_service = screening_service
        self.loop = asyncio.get_running_loop()
        
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
<<<<<<< HEAD
=======
        self.renal_extractor = RenalExtractor()
>>>>>>> 5fb8453bef017d5d9f62d222de7a732819a54135
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
                self._frame_counter += 1
                
                # Determine active phase and required models
                current_phase = self._scan_status.get("phase", "IDLE")
                active_models = []
                
                # Phase-Gating Logic
                if current_phase == "FACE_ANALYSIS":
                    active_models = ["face_mesh"]
                elif current_phase == "BODY_ANALYSIS":
                    active_models = ["pose"]
                elif current_phase == "IDLE" or current_phase == "INITIALIZING":
                     # In IDLE, we only need face_det for distance warning
                     # But we can skip it most of the time to save power
                     active_models = ["face_det"]
                else:
                    active_models = [] # PROCESSING or ERROR -> No AI needed
                
                # Frame Skipping for Inference (Throttle during active scans)
                current_phase = self._scan_status.get("phase", "IDLE")
                is_scanning = current_phase in ["FACE_ANALYSIS", "BODY_ANALYSIS"]
                
                # Use a much higher interval (lower FPS) for overlays during active recording
                dynamic_interval = 10 if is_scanning else self._inference_interval
                should_run_inference = (self._frame_counter % dynamic_interval == 0)
                
                if should_run_inference and active_models:
                    # Run selective inference
                    self._last_inference_results = self.camera.detect_all(frame, active_models)
                
                # Apply server-side overlays using LAST KNOWN results (for smoothness)
                if self._enable_overlays:
                    results = self._last_inference_results
                    
                    # 1. Overlay based on Phase
                    if current_phase == "FACE_ANALYSIS":
                        frame = self.camera.draw_face_mesh_on_frame(frame, results)
                    elif current_phase == "BODY_ANALYSIS":
                        frame = self.camera.draw_pose_skeleton_on_frame(frame, results)
                    
                    # 2. Calculate and draw distance warnings (Throttle distance check too? No, fast feedback is good)
                    # We reuse the same analysis results. If we skipped inference, we use old results.
                    
                    warning_type = self._current_distance_warning
                    face_detected = False

                    # Limit distance warning check to IDLE/FACE phases
                    if current_phase in ["IDLE", "INITIALIZING", "FACE_ANALYSIS"]:
                        warning_type, face_width = self.camera.calculate_face_distance(frame, results)
                        if warning_type:
                            # Distance warning is now handled by status message only
                            pass
                        
                        # Update status for frontend polling
                        self._current_distance_warning = warning_type
                        face_detected = face_width is not None
                    elif current_phase == "BODY_ANALYSIS":
                        # Clear distance warning so it doesn't interfere
                        warning_type = None
                        self._current_distance_warning = None

                    self._update_scan_status(
                        user_warnings={
                            "distance_warning": warning_type,
                            "face_detected": face_detected,
                            "pose_detected": results.get("pose") is not None and results["pose"].pose_landmarks is not None,
                        }
                    )
                
                # Encode to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                if ret:
                    with self._frame_lock:
                        self._latest_frame_jpeg = jpeg.tobytes()
                
                # Record frame if scan is active (HANDLED BY DRIVER Thread for 30FPS)
                # We no longer need to append here; driver does it in background
                pass
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

    def _wait_for_alignment(self, target_phase: str, timeout: int = None):
        """
        Busy-wait until user is correctly positioned for the phase.
        If timeout is provided and reached, proceeds anyway with a warning.
        """
        logger.info(f"⏳ Waiting for alignment in {target_phase} (timeout={timeout}s)...")
        
        # Initial wait for first frame processing
        time.sleep(1.0)
        
        start_time = time.time()
        
        while self._scan_active:
            # Timeout Check
            if timeout and (time.time() - start_time > timeout):
                logger.warning(f"Alignment timeout for {target_phase}. Proceeding anyway.")
                self._update_scan_status(message="Proceeding with partial alignment...")
                time.sleep(1.0) # Show message briefly
                break

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
            msg = "Positioning correct. Stay still."
            if warning == "too_close":
                msg = "Please move back slightly for a better view."
            elif warning == "too_far":
                msg = "Please move closer to the camera."
            elif target_phase == "BODY_ANALYSIS" and not user_warnings.get("pose_detected"):
                msg = "Please step back so your full body is visible."
            elif target_phase == "FACE_ANALYSIS" and not user_warnings.get("face_detected"):
                msg = "Face not detected. Please look directly at the camera."

            self._update_scan_status(message=msg)
            time.sleep(0.2)

    def _countdown(self, seconds: int, message_prefix: str, phase: str = None):
        """Helper to update status with a visible countdown timer."""
        for i in range(seconds, 0, -1):
            if not self._scan_active: 
                break
            self._update_scan_status(
                message=f"{message_prefix} {i}s...",
                phase=phase if phase else self._scan_status.get("phase")
            )
            time.sleep(1.0)
    
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

    def _get_calibration_data(self, duration_seconds: int = 5) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """Collect raw data for baseline calibration."""
        thermal_frames = []
        rgb_frames = []
        
        # Start recording for RGB
        with self._recording_lock:
            self._recording_buffer = []
            self._recording_active = True
            
        # Collect for duration
        start_t = time.time()
        while time.time() - start_t < duration_seconds:
            if not self._scan_active: break
            
            # Collect thermal
            if self.thermal:
                while not self.thermal.data_queue.empty():
                    try:
                        thermal_frames.append(self.thermal.data_queue.get_nowait())
                    except:
                        break
            
            time.sleep(0.1) # 10Hz sampling
            
        self._recording_active = False
        with self._recording_lock:
            rgb_frames = [item['frame'] for item in self._recording_buffer]
            
        return thermal_frames, rgb_frames
    
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
            # Phase 0: Vitals Collection (NEW)
            # ============================================================
            self._update_scan_status(
                phase="VITALS_COLLECTION",
                message="Collecting vitals... Please sit still.",
                progress=7,
            )
            logger.info("🎯 Phase 0: Vitals Collection (10s)")
            
            # Wait for user to be stationary
            time.sleep(2)
            
            # Capture vitals for 10 seconds
            self._countdown(10, "Collecting Vitals...", "VITALS_COLLECTION")
            
            # Aggregate vitals immediately while user is stationary
            logger.info("📊 Aggregating vitals from Phase 0...")
            radar_data = self._aggregate_radar()
            esp32_data = self._aggregate_thermal()
            
            # ============================================================
            # Phase 0.5: Baseline Calibration
            # ============================================================
            self._update_scan_status(
                phase="BASELINE_CALIBRATION",
                message="Calibrating environment...",
                progress=15,
            )
            logger.info("🎯 Phase 0.5: Baseline Calibration (5s)")
            
            # Wait for initial alignment if possible, but don't block too long
            self._wait_for_alignment("FACE_ANALYSIS", timeout=3)
            
            thermal_baseline_frames, rgb_baseline_frames = self._get_calibration_data(duration_seconds=5)
            
            session_baseline = None
            if self.skin_extractor:
                try:
                    # Filter and flatten thermal frames for baseline
                    system_thermal = []
                    for raw in thermal_baseline_frames:
                        # Extract the inner 'thermal' part if it exists (standard ESP32Reader format)
                        # We need 'fever_face_max' and 'background_temp'
                        # Flatten it like _build_screening_request does
                        t_data = raw.get('thermal', {})
                        core = t_data.get('core_regions', {})
                        stability = t_data.get('stability_metrics', {})
                        
                        system_thermal.append({
                            'fever_face_max': core.get('face_max'),
                            'background_temp': t_data.get('ambient_temp') or 25.0
                        })
                    
                    session_baseline = self.skin_extractor.capture_session_baseline(
                        thermal_frames=system_thermal,
                        rgb_frames=rgb_baseline_frames
                    )
                except Exception as e:
                    logger.error(f"Baseline capture failed: {e}")
            
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
                
                # Preparation Timer (10s)
                self._countdown(10, "Get Ready...", "FACE_ANALYSIS")
                
                # Start recording (DRIVER LEVEL)
                self.camera.start_recording()
                
                # Capture Timer (10s)
                self._countdown(self.config.face_capture_seconds, "Scanning Face...", "FACE_ANALYSIS")
                
                # Stop recording (DRIVER LEVEL)
                raw_captures = self.camera.stop_recording()
                
                logger.info(f"Captured {len(raw_captures)} face frames from hardware driver")
                
                # Post-process frames (CONSOLIDATED extraction: ROI + Landmarks)
                processed_data = []
                for item in raw_captures:
                     frame = item['frame']
                     roi, landmarks = self.camera.extract_face_features(frame)
                     if roi is not None:
                         processed_data.append({'roi': roi, 'landmarks': landmarks})

                face_frames = [d['roi'] for d in processed_data]
                face_landmarks_sequence = [d['landmarks'] for d in processed_data if d['landmarks'] is not None]
                logger.info(f"Extracted {len(face_frames)} face ROIs, {len(face_landmarks_sequence)} landmark sets (CONSOLIDATED)")
                
                # Validation: Warn if faces detected but landmarks missing
                if len(face_frames) > 0 and len(face_landmarks_sequence) == 0:
                    logger.warning(f"⚠️ Face ROIs detected ({len(face_frames)}) but NO FaceMesh landmarks extracted. Eye blink detection will fail.")
                elif len(face_frames) > 0 and len(face_landmarks_sequence) < len(face_frames) * 0.5:
                    logger.warning(f"⚠️ Low landmark detection rate: {len(face_landmarks_sequence)}/{len(face_frames)} frames ({100*len(face_landmarks_sequence)/len(face_frames):.1f}%)")
            
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
                
                # Wait for user to be correctly positioned (max 5 seconds, then proceed)
                # This makes it "less strict" - it warns but doesn't block forever
                self._wait_for_alignment("BODY_ANALYSIS", timeout=5)
                
                # Preparation Timer (10s)
                self._countdown(10, "Get Ready...", "BODY_ANALYSIS")
                
                # Start recording (DRIVER LEVEL)
                self.camera.start_recording()
                
                # Capture Timer (10s)
                self._countdown(self.config.body_capture_seconds, "Scanning Body...", "BODY_ANALYSIS")
                
                # Stop recording (DRIVER LEVEL)
                raw_captures = self.camera.stop_recording()
                
                logger.info(f"Captured {len(raw_captures)} body frames from hardware driver")
                
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
            
            # Aggregate Radar (Only if not already done in Phase 0)
            if radar_data is None:
                radar_data = self._aggregate_radar()
            
            # Aggregate Thermal (Only if not already done in Phase 0)
            if esp32_data is None:
                esp32_data = self._aggregate_thermal()
            
            # ============================================================
            # Build screening request (run all 8 extractors)
            # ============================================================
            self._update_scan_status(
                phase="PROCESSING",
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
                fps=self.config.camera_fps,
                session_baseline=session_baseline # NEW
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
            
            # ============================================================
            # Submit to ScreeningService (DIRECT CALL - no network overhead)
            # ============================================================
            self._update_scan_status(
                message="Computing risk assessment...",
                progress=85,
            )
            
            screening_id = None
            patient_report_id = None
            doctor_report_id = None
            
            if self.screening_service and self.loop:
                try:
                    # Run async service call in event loop from background thread
                    future = asyncio.run_coroutine_threadsafe(
                        self.screening_service.process_screening(
                            patient_id=patient_id, 
                            systems_input=request_payload["systems"],
                            include_validation=True
                        ),
                        self.loop
                    )
                    result = future.result(timeout=120)  # Wait for result
                    screening_id = result.get("screening_id")
                    
                    # Store in reports dict (mimicking main.py behavior)
                    # We need access to the reports dict from main.py
                    # For now, we update the scan status, and main.py can pick it up
                    
                    logger.info(f"✅ Screening completed via service: {screening_id}")
                    
                    # Update local screenings storage (directly injecting into main.py dict)
                    from app.main import _screenings
                    _screenings[screening_id] = {
                        "patient_id": result["patient_id"],
                        "system_results": result["system_results_internal"],
                        "trusted_results": result["trusted_results"],
                        "composite_risk": result["composite_risk"],
                        "rejected_systems": result["rejected_systems"],
                        "timestamp": result["timestamp"]
                    }
                    
                    # Generate reports DIRECTLY using generators (instead of HTTP calls)
                    self._update_scan_status(message="Generating reports...", progress=90)
                    
                    # Import here to avoid circular dependencies
                    from app.main import _reports, _patient_report_gen, _doctor_report_gen
                    
                    try:
                        p_report = _patient_report_gen.generate(
                            system_results=result["system_results_internal"],
                            composite_risk=result["composite_risk"],
                            patient_id=result["patient_id"],
                            trusted_results=result["trusted_results"],
                            rejected_systems=result["rejected_systems"]
                        )
                        patient_report_id = p_report.report_id
                        _reports[patient_report_id] = p_report.pdf_path
                        logger.info(f"Patient report generated: {patient_report_id}")
                    except Exception as e:
                        logger.warning(f"Patient report generation failed: {e}")

                    try:
                        d_report = _doctor_report_gen.generate(
                            system_results=result["system_results_internal"],
                            composite_risk=result["composite_risk"],
                            patient_id=result["patient_id"]
                        )
                        doctor_report_id = d_report.report_id
                        _reports[doctor_report_id] = d_report.pdf_path
                        logger.info(f"Doctor report generated: {doctor_report_id}")
                    except Exception as e:
                        logger.warning(f"Doctor report generation failed: {e}")
                        
                except Exception as e:
                    logger.error(f"ScreeningService call failed: {e}")
                    raise e
            else:
                logger.error("ScreeningService not available in HardwareManager")
                raise Exception("ScreeningService not available")
            
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
                trust_metadata=result.get("trust_metadata")  # NEW: Expose trust metadata
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
        
        # Calculate medians for physiological stability (Replaced mean)
        hr_values = [i['radar']['heart_rate'] for i in items]
        rr_values = [i['radar']['respiration_rate'] for i in items]
        
        avg_hr = int(np.median(hr_values)) if hr_values else 0
        avg_resp = round(float(np.median(rr_values)), 1) if rr_values else 0.0
        
        result = items[-1].copy()
        result['radar']['heart_rate'] = avg_hr
        result['radar']['respiration_rate'] = avg_resp
        
        logger.info(f"   ✅ Collected {len(items)} Radar samples")
        logger.info(f"   🔍 Radar Logic:")
        logger.info(f"      - Input HR: {hr_values}")
        logger.info(f"      - Median HR: {avg_hr}")
        logger.info(f"      - Input RR: {rr_values}")
        logger.info(f"      - Median RR: {avg_resp}")
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
            face_max_valid = [get_val(i, 'core_regions', 'face_max') for i in items 
                           if get_val(i, 'core_regions', 'face_max') > 25.0]
            stability_vals = [get_val(i, 'stability_metrics', 'canthus_range') for i in items]
            asymmetry_vals = [get_val(i, 'symmetry', 'cheek_asymmetry') for i in items]
            gradient_vals = [get_val(i, 'gradients', 'forehead_nose_gradient') for i in items]
            
            # Use median for core temperatures and metrics (more robust to noise)
            avg_canthus = round(float(np.median(canthus_valid)), 2) if canthus_valid else 0.0
            avg_neck = round(float(np.median(neck_valid)), 2) if neck_valid else 0.0
            avg_face_max = round(float(np.median(face_max_valid)), 2) if face_max_valid else 0.0
            avg_stability = round(float(np.median(stability_vals)), 2) if stability_vals else 0.0
            avg_asymmetry = round(float(np.median(asymmetry_vals)), 3) if asymmetry_vals else 0.0
            avg_gradient = round(float(np.median(gradient_vals)), 2) if gradient_vals else 0.0
            
            result = {
                'timestamp': items[-1].get('timestamp', 0),
                'thermal': {
                    'core_regions': {
                        'canthus_mean': avg_canthus, 
                        'neck_mean': avg_neck,
                        'face_max': avg_face_max
                    },
                    'stability_metrics': {'canthus_range': avg_stability},
                    'symmetry': {'cheek_asymmetry': avg_asymmetry},
                    'gradients': {'forehead_nose_gradient': avg_gradient}
                }
            }
            logger.info(
                f"Aggregated thermal (firmware - Median). Canthus: {avg_canthus}°C, "
                f"Neck: {avg_neck}°C, Stability: {avg_stability}"
            )
            return result
        else:
            # Legacy format aggregation
            avg_neck = float(np.median(neck_temps)) if neck_temps else 0.0
            stress_vals = [get_val(i, 'autonomic', 'stress_gradient') for i in items]
            avg_stress = float(np.median(stress_vals)) if stress_vals else 0.0
            
            esp32_data = items[-1].copy()
            if 'fever' in esp32_data.get('thermal', {}):
                esp32_data['thermal']['fever']['neck_temp'] = round(avg_neck, 2)
            if 'autonomic' in esp32_data.get('thermal', {}):
                esp32_data['thermal']['autonomic']['stress_gradient'] = round(avg_stress, 2)
            
            logger.info(f"Aggregated thermal (legacy - Median). Avg Neck: {avg_neck:.1f}")
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
        fps: float = 30.0,
        session_baseline: Optional[Any] = None # NEW
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
        
        if session_baseline: # NEW
            raw_data_context["session_baseline"] = session_baseline
        
        # Run all 8 extractors (same order as bridge.py DataFusion)
        extractor_map = [
            ("Pulmonary", self.pulmonary_extractor),
            ("Cardiovascular", self.cv_extractor),
            ("Skin", self.skin_extractor),
            ("CNS", self.cns_extractor),
            ("Skeletal", self.skeletal_extractor),
<<<<<<< HEAD
=======
            ("Renal", self.renal_extractor),
>>>>>>> 5fb8453bef017d5d9f62d222de7a732819a54135
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

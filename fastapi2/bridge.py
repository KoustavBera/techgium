"""
Hardware Bridge - Split-USB Architecture Integration

This module provides seamless integration between hardware sensors and
the FastAPI health screening pipeline using a Split-USB topology.

SPLIT-USB ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                    LAPTOP / PC                                  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                      bridge.py                           │   │
│   │                                                          │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│   │  │ CameraCapture│  │ RadarReader  │  │ ESP32Reader  │   │   │
│   │  │ (OpenCV)     │  │ (Serial)     │  │ (Serial)     │   │   │
│   │  │ Webcam       │  │ COM_A        │  │ COM_B        │   │   │
│   │  │ Video Data   │  │ Breathing HR │  │ Thermal Data │   │   │
│   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │   │
│   │         │                 │                 │            │   │
│   │         └─────────────────┴─────────────────┘            │   │
│   │                           │                              │   │
│   │                    DataFusion                            │   │
│   │                           │                              │   │
│   │                           ▼                              │   │
│   │               POST /api/v1/screening                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                      ▲                          │
│                                      │ USB                      │
│   ┌──────────────────────────────────┴────────────────────┐     │
│   │                4-PORT USB 2.0 HUB                      │     │
│   └───┬─────────────────┬─────────────────┬───────────────┘     │
│       │                 │                 │                      │
│       ▼                 ▼                 ▼                      │
│   ┌────────┐     ┌────────────┐    ┌────────────┐               │
│   │ Webcam │     │ Seeed Radar│    │ ESP32      │               │
│   │        │     │ MR60BHA2   │    │ NodeMCU    │               │
│   │        │     │ (USB/COM_A)│    │ (USB/COM_B)│               │
│   └────────┘     └────────────┘    └─────┬──────┘               │
│                                          │ I2C                   │
│                                          ▼                       │
│                                    ┌────────────┐               │
│                                    │ MLX90640   │               │
│                                    │ Thermal    │               │
│                                    └────────────┘               │
└─────────────────────────────────────────────────────────────────┘

Key Design Points:
- Webcam connects directly to laptop via USB Hub (Video to OpenCV)
- Seeed Radar Kit connects via USB Hub (Serial/COM_A - Breathing & HR)
- ESP32 NodeMCU connects via USB Hub (Serial/COM_B - Thermal only)
- MLX90640 thermal camera connects to ESP32 via I2C

Usage:
    python bridge.py --radar-port COM7 --port COM6 --camera 0
    python bridge.py --simulate  # Test without any hardware
"""

import asyncio
import json
import re
import time
import threading
import queue
import struct
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import argparse

import sys
import os

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import numpy as np
import cv2
import requests

# Core Extractors for unified logic
from app.core.extraction.cardiovascular import CardiovascularExtractor
from app.core.extraction.cns import CNSExtractor
from app.core.extraction.skeletal import SkeletalExtractor
from app.core.extraction.skin import SkinExtractor
from app.core.extraction.pulmonary import PulmonaryExtractor
from app.core.extraction.renal import RenalExtractor
from app.core.extraction.eyes import EyeExtractor
from app.core.extraction.nasal import NasalExtractor
from app.core.extraction.base import BiomarkerSet


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

# Optional serial import (may not be available on all systems)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Run: pip install pyserial")

# Optional MediaPipe import
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Camera features limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class BridgeConfig:
    """Configuration for the hardware bridge (Split-USB Architecture)."""
    
    # Serial - ESP32 NodeMCU (Thermal Data)
    serial_port: str = "COM6"  # Windows: COM6, Linux: /dev/ttyUSB1
    serial_baud: int = 115200
    
    # Serial - Seeed Radar Kit (Breathing/Heartbeat Data)
    radar_port: str = "COM7"  # Windows: COM7, Linux: /dev/ttyUSB0
    radar_baud: int = 115200
    
    # Camera
    camera_index: int = 0
    camera_fps: int = 30
    face_capture_seconds: int = 20
    body_capture_seconds: int = 20
    
    # API
    api_url: str = "http://localhost:8000"
    api_timeout: int = 30
    
    # Timing
    sensor_poll_interval: float = 0.05  # 20 Hz
    data_fusion_interval: float = 1.0   # 1 Hz API calls


# ==============================================================================
# BASE SERIAL READER
# ==============================================================================


# ==============================================================================
# BASE SERIAL READER
# ==============================================================================

class BaseSerialReader:
    """Base class for serial port readers."""
    
    def __init__(self, port: str, baud: int = 115200, name: str = "Device"):
        self.port = port
        self.baud = baud
        self.name = name
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False
        self.data_queue: queue.Queue = queue.Queue(maxsize=100)
        self.last_data: Dict[str, Any] = {}
        
    def connect(self) -> bool:
        """Connect to serial port."""
        if not SERIAL_AVAILABLE:
            logger.error("pyserial not installed")
            return False
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=1.0
            )
            logger.info(f"Connected to {self.name} on {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to connect to {self.name} on {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info(f"Disconnected from {self.name}")
    
    def read_loop(self):
        """Background thread to read serial data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement read_loop")

    def start_reading(self):
        """Start background reading thread."""
        thread = threading.Thread(target=self.read_loop, daemon=True)
        thread.start()
        return thread
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get most recent sensor data."""
        return self.last_data if self.last_data else None


# ==============================================================================
# ESP32 SERIAL READER (Thermal Data)
# ==============================================================================

class ESP32Reader(BaseSerialReader):
    """Reads thermal biomarker data from ESP32 NodeMCU via serial port."""
    
    def __init__(self, port: str, baud: int = 115200):
        super().__init__(port, baud, name="ESP32 NodeMCU (Thermal MLX90640)")
    
    def read_loop(self):
        """Background thread to read JSON thermal data from ESP32."""
        self.running = True
        
        while self.running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                
                if not line or not line.startswith('{'):
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Skip error messages from ESP32
                    if 'error' in data:
                        logger.warning(f"ESP32 thermal error: {data.get('error')}")
                        continue
                    
                    # Validate required structure
                    if 'thermal' not in data:
                        continue
                    
                    thermal = data['thermal']
                    
                    # Validate metadata (face detection quality check)
                    if 'metadata' in thermal:
                        if thermal['metadata'].get('face_detected', 0) == 0:
                            logger.debug("Face not detected in thermal frame")
                            # We might still want to capture data if we are debugging, 
                            # but for strict screening we might skip. 
                            # For now, let's allow it but log it.
                            # continue 
                            pass
                        
                        valid_rois = thermal['metadata'].get('valid_rois', 0)
                        if valid_rois < 5:  # Need at least 5 ROIs for reliable data
                            logger.debug(f"Low quality frame: only {valid_rois} valid ROIs")
                            continue
                    
                    # Store complete data
                    data['received_at'] = time.time()
                    self.last_data = data
                    
                    # Queue management
                    if self.data_queue.full():
                        self.data_queue.get()
                    self.data_queue.put(data)
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"ESP32 JSON parse error: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"ESP32 read error: {e}")
                time.sleep(0.1)


# ==============================================================================
# RADAR SERIAL READER (Seeed MR60BHA2)
# ==============================================================================

class RadarReader(BaseSerialReader):
    """Reads breathing and heartbeat data from Seeed MR60BHA2 Radar Kit."""
    
    def __init__(self, port: str, baud: int = 115200):
        super().__init__(port, baud, name="Seeed Radar Kit (MR60BHA2)")
    
    def read_loop(self):
        """Background thread to read serial data (Text-based) from Radar."""
        self.running = True
        
        while self.running and self.serial_conn and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode(errors="ignore").strip()
                    
                    if not line:
                        continue
                    
                    # Parse text-based output using regex (matches actual hardware)
                    hr_match = re.search(r'heart rate.*?([\d.]+)\s*bpm', line.lower())
                    rr_match = re.search(r'respiratory rate.*?([\d.]+)', line.lower())
                    
                    # Update last_data with any new values found
                    if hr_match or rr_match:
                        if not self.last_data or 'radar' not in self.last_data:
                            self.last_data = {
                                'timestamp': int(time.time()),
                                'radar': {
                                    'respiration_rate': 0.0,
                                    'heart_rate': 0,
                                    'presence_detected': True
                                }
                            }
                        
                        if hr_match:
                            heart_rate = float(hr_match.group(1))
                            # Validate physiological range
                            if 40 <= heart_rate <= 180:
                                self.last_data['radar']['heart_rate'] = int(heart_rate)
                                self.last_data['timestamp'] = int(time.time())
                                logger.debug(f"Heart Rate: {heart_rate} bpm")
                        
                        if rr_match:
                            resp_rate = float(rr_match.group(1))
                            # Validate physiological range
                            if 5 <= resp_rate <= 40:
                                self.last_data['radar']['respiration_rate'] = round(resp_rate, 1)
                                self.last_data['timestamp'] = int(time.time())
                                logger.debug(f"Respiration Rate: {resp_rate} bpm")
                        
                        # Update queue with latest data
                        self.last_data['received_at'] = time.time()
                        if self.data_queue.full():
                            self.data_queue.get()
                        self.data_queue.put(self.last_data.copy())
                else:
                    time.sleep(0.01)
                        
            except Exception as e:
                logger.error(f"Radar read error: {e}")
                time.sleep(0.1)
    
    def parse_radar_text(self, line: str) -> Optional[Dict]:
        """Parse text-based output from Seeed MR60BHA2 radar.
        
        Text format (actual hardware output):
        - "heart rate ... XX.XXXXX bpm"
        - "respiratory rate ... XX.XXXXX bpm"
        
        Note: This method is deprecated - parsing is now done inline in read_loop()
        for better real-time performance. Kept for backward compatibility.
        """
        hr_match = re.search(r'heart rate.*?([\d.]+)\s*bpm', line.lower())
        rr_match = re.search(r'respiratory rate.*?([\d.]+)', line.lower())
        
        if not (hr_match or rr_match):
            return None
        
        result = {
            'timestamp': int(time.time()),
            'radar': {
                'respiration_rate': 0.0,
                'heart_rate': 0,
                'presence_detected': True
            }
        }
        
        if hr_match:
            heart_rate = float(hr_match.group(1))
            if 40 <= heart_rate <= 180:
                result['radar']['heart_rate'] = int(heart_rate)
        
        if rr_match:
            resp_rate = float(rr_match.group(1))
            if 5 <= resp_rate <= 40:
                result['radar']['respiration_rate'] = round(resp_rate, 1)
        
        return result




# ==============================================================================
# CAMERA CAPTURE (rPPG + Pose)
# ==============================================================================

class CameraCapture:
    """Captures face and body frames for biomarker extraction."""
    
    def __init__(self, camera_index: int = 0, fps: int = 30):
        self.camera_index = camera_index
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
        # MediaPipe components
        self.face_detector = None
        self.pose_landmarker = None
        self.face_landmarker = None
        
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe using legacy solutions API (compatible with all versions)."""
        if not MEDIAPIPE_AVAILABLE:
            return
            
        try:
            # Legacy Face Detection (same as used successfully elsewhere)
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0 = short range (2m), 1 = full range (5m)
                min_detection_confidence=0.5
            )
            logger.info("✅ FaceDetector (Legacy) initialized")
        except Exception as e:
            logger.warning(f"Face detector init failed: {e}")
        
        try:
            # Legacy Pose Detection (same API as mp.solutions.face_mesh used in SkinExtractor)
            self.pose_landmarker = mp.solutions.pose.Pose(
                static_image_mode=False,  # Video mode for smoothing
                model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✅ PoseLandmarker (Legacy) initialized")
        except Exception as e:
            logger.warning(f"Pose landmarker init failed: {e}")

        try:
            # FaceMesh (for Eyes & Nasal) - 468/478 landmarks
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # Enables iris tracking (478 landmarks)
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✅ FaceMesh initialized (478 landmarks)")
        except Exception as e:
            logger.warning(f"FaceMesh init failed: {e}")
    
    def open(self) -> bool:
        """Open camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info(f"Camera {self.camera_index} opened")
        return True
    
    def close(self):
        """Close camera."""
        if self.cap:
            self.cap.release()
            logger.info("Camera closed")
    
    def capture_and_process_frames(self, duration_seconds: int, process_func: Optional[callable] = None) -> Tuple[List[Any], int]:
        """Capture frames and optionally process them on the fly to save memory."""
        processed_data = []
        target_frames = duration_seconds * self.fps
        frames_captured = 0
        
        start_time = time.time()
        while frames_captured < target_frames:
            ret, frame = self.cap.read()
            if ret:
                frames_captured += 1
                if process_func:
                    # Process immediately and store result (e.g., small crop or landmarks)
                    result = process_func(frame)
                    if result is not None:
                        processed_data.append(result)
                else:
                    # Store raw frame (legacy mode - risky for long durations)
                    processed_data.append(frame)
            else:
                break
                
            # Maintain frame rate
            elapsed = time.time() - start_time
            expected = frames_captured / self.fps
            if elapsed < expected:
                time.sleep(expected - elapsed)
        
        return processed_data, frames_captured
    
    def capture_frames(self, duration_seconds: int) -> Tuple[List[np.ndarray], List[float]]:
        """Legacy capture wrapper."""
        frames, _ = self.capture_and_process_frames(duration_seconds)
        # Generate timestamps retroactively for legacy support
        timestamps = [i/self.fps for i in range(len(frames))] 
        return frames, timestamps
    
    def extract_face_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract face ROI from a single frame using legacy API."""
        if not self.face_detector:
             # Fallback
            fh, fw = frame.shape[:2]
            return frame[fh//4:3*fh//4, fw//4:3*fw//4]
            
        try:
            # CRITICAL: OpenCV captures in BGR, MediaPipe Legacy API expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_detector.process(rgb_frame)
            
            if result.detections:
                detection = result.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                # Convert normalized coordinates to pixels
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # Add padding
                pad = int(max(box_w, box_h) * 0.2)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w, x + box_w + pad), min(h, y + box_h + pad)
                # Return the crop from ORIGINAL BGR frame (for cv2 compatibility downstream)
                return frame[y1:y2, x1:x2]
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            
        fh, fw = frame.shape[:2]
        return frame[fh//4:3*fh//4, fw//4:3*fw//4]

    def extract_face_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Batch extraction (legacy)."""
        return [self.extract_face_roi(f) for f in frames]
    
    def extract_pose_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose landmarks from a single frame using legacy API."""
        if not self.pose_landmarker:
            return None  # Silent fail - will log first occurrence only
            
        try:
            # CRITICAL: OpenCV captures in BGR, MediaPipe Legacy API expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose_landmarker.process(rgb_frame)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                landmark_array = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in landmarks
                ])
                
                # VALIDATE SHAPE (33 landmarks x 4 values)
                if landmark_array.shape == (33, 4):
                    return landmark_array
                else:
                    logger.warning(f"Invalid pose shape: {landmark_array.shape}")
            # else: No pose detected in this frame (normal for some frames)
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
        return None

    def extract_pose_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Batch extraction (legacy)."""
        # Note: This still iterates a list, but checks shape via extract_pose_from_frame
        sequence = []
        for frame in frames:
            pose = self.extract_pose_from_frame(frame)
            if pose is not None:
                sequence.append(pose)
        return sequence

    def extract_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468-point FaceMesh landmarks for Eyes & Nasal analysis."""
        if not hasattr(self, 'face_mesh') or self.face_mesh is None:
            return None
            
        try:
            # Color conversion critical for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb_frame)
            
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                # Convert to numpy array (x, y, z, visibility=1.0)
                # FaceMesh landmarks don't provide visibility score, defaulting to 1.0
                return np.array([
                    [lm.x, lm.y, lm.z, 1.0]
                    for lm in landmarks
                ])
        except Exception as e:
            logger.debug(f"FaceMesh error: {e}")
        return None


# ==============================================================================
# DATA FUSION & API CLIENT
# ==============================================================================

class DataFusion:
    """Fuses sensor data from multiple sources using core extractors and sends to API."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.screening_endpoint = f"{api_url}/api/v1/screening"
        self.report_endpoint = f"{api_url}/api/v1/reports/generate"
        
        # Initialize Core Extractors
        # Using default sample rates (can be tuned if specialized hardware config used)
        self.cv_extractor = CardiovascularExtractor()
        self.cns_extractor = CNSExtractor()
        self.skeletal_extractor = SkeletalExtractor()
        self.skin_extractor = SkinExtractor()
        self.pulmonary_extractor = PulmonaryExtractor()
        self.renal_extractor = RenalExtractor()
        
        # New Extractors (Augmented with FaceMesh)
        self.eye_extractor = EyeExtractor(
            sample_rate=30.0,
            frame_width=1280,
            frame_height=720
        )
        self.nasal_extractor = NasalExtractor()
        
        logger.info("Core Extractors initialized in DataFusion")
        
    def _transform_biomarker_set(self, bm_set: BiomarkerSet) -> Dict[str, Any]:
        """Convert BiomarkerSet to API-compatible system dictionary."""
        return {
            "system": bm_set.system.value,
            "biomarkers": [bm.to_dict() for bm in bm_set.biomarkers]
        }

    def build_screening_request(
        self,
        patient_id: str,
        radar_data: Optional[Dict] = None,
        esp32_data: Optional[Dict] = None,
        face_frames: Optional[List[np.ndarray]] = None,
        pose_sequence: Optional[List[np.ndarray]] = None,
        face_landmarks_sequence: Optional[List[np.ndarray]] = None,  # NEW: For Eyes/Nasal
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """Build complete screening request using unified core extractors."""
        systems = []
        
        # Aggregate all raw data into a single context for extractors
        # This allows cross-modality fusion if supported by extractors in future
        raw_data_context = {
            "fps": fps,
            "patient_id": patient_id
        }
        
        if radar_data:
            raw_data_context["radar_data"] = radar_data
        
        if face_landmarks_sequence:
            raw_data_context["face_landmarks_sequence"] = face_landmarks_sequence  # For Eyes/Nasal
        
        # Flatten ESP32 thermal data for distribution to multiple extractors
        # Supports BOTH real firmware format (core_regions/stability_metrics/symmetry/gradients)
        # and legacy simulation format (fever/inflammation/cardiovascular/diabetes/autonomic)
        if esp32_data and 'thermal' in esp32_data:
            thermal = esp32_data['thermal']
            
            # Detect firmware format: real hardware uses 'core_regions'
            is_firmware_format = 'core_regions' in thermal
            
            if is_firmware_format:
                # REAL FIRMWARE FORMAT from esp32_thermal_bridge.ino
                core = thermal.get('core_regions', {})
                stability = thermal.get('stability_metrics', {})
                symmetry = thermal.get('symmetry', {})
                gradients = thermal.get('gradients', {})
                
                canthus_mean = core.get('canthus_mean')
                neck_mean = core.get('neck_mean')
                face_max = core.get('face_max')  # NEW
                canthus_range = stability.get('canthus_range')
                
                # Calculate face_mean_temp from available core temperatures
                face_mean_temp = None
                if canthus_mean is not None and neck_mean is not None:
                    face_mean_temp = (canthus_mean + neck_mean) / 2.0
                elif canthus_mean is not None:
                    face_mean_temp = canthus_mean
                elif neck_mean is not None:
                    face_mean_temp = neck_mean
                
                # Estimate inflammation from thermal stability (high variance may indicate hot spots)
                # Normal canthus_range is < 0.8°C, higher values suggest inflammation
                inflammation_pct = None
                if canthus_range is not None:
                    # Convert range to percentage scale (0-10%)
                    # 0.8°C range = 0%, 2.0°C range = 10%
                    inflammation_pct = min(10.0, max(0.0, (canthus_range - 0.8) * 8.33))
                
                raw_data_context['thermal_data'] = {
                    # For SkinExtractor
                    'fever_neck_temp': neck_mean,
                    'fever_canthus_temp': canthus_mean,
                    'fever_face_max': face_max,  # NEW
                    'thermal_stability': canthus_range,
                    'inflammation_pct': inflammation_pct,  # Calculated from thermal variance
                    'face_mean_temp': face_mean_temp,      # Calculated from available temps
                    # For CardiovascularExtractor
                    'thermal_asymmetry': symmetry.get('cheek_asymmetry'),
                    'left_cheek_temp': None,    # Only asymmetry delta available
                    'right_cheek_temp': None,
                    # For RenalExtractor (diabetes/microcirculation)
                    'diabetes_canthus_temp': canthus_mean,
                    'diabetes_risk_flag': (canthus_mean is not None and canthus_mean < 35.5),
                    # For CNSExtractor (autonomic stress)
                    'stress_gradient': abs(gradients.get('forehead_nose_gradient', 0)) if gradients.get('forehead_nose_gradient') is not None else None,
                    'nose_temp': None,          # Only gradient available
                    'forehead_temp': None,
                }
            else:
                # LEGACY SIMULATION FORMAT (fever/inflammation/cardiovascular/etc.)
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
            
            raw_data_context['esp32_data'] = esp32_data  # Keep original too
        elif esp32_data:
            raw_data_context['esp32_data'] = esp32_data
            
        if face_frames:
            raw_data_context["face_frames"] = face_frames
        if pose_sequence:
            raw_data_context["pose_sequence"] = pose_sequence
            
        # 1. Pulmonary (Radar)
        try:
            pulmonary_set = self.pulmonary_extractor.extract(raw_data_context)
            if pulmonary_set.biomarkers:
                 systems.append(self._transform_biomarker_set(pulmonary_set))
        except Exception as e:
            logger.error(f"Pulmonary extraction failed: {e}")

        # 2. Cardiovascular (Radar + rPPG)
        try:
            # Note: CardiovascularExtractor can use radar_data AND face_frames
            cv_set = self.cv_extractor.extract(raw_data_context)
            if cv_set.biomarkers:
                systems.append(self._transform_biomarker_set(cv_set))
        except Exception as e:
            logger.error(f"Cardiovascular extraction failed: {e}")

        # 3. Skin (Thermal)
        try:
            skin_set = self.skin_extractor.extract(raw_data_context)
            if skin_set.biomarkers:
                systems.append(self._transform_biomarker_set(skin_set))
        except Exception as e:
            logger.error(f"Skin extraction failed: {e}")

        # 4. CNS (Motion/Gait)
        try:
            cns_set = self.cns_extractor.extract(raw_data_context)
            if cns_set.biomarkers:
                systems.append(self._transform_biomarker_set(cns_set))
        except Exception as e:
             logging.error(f"CNS extraction failed: {e}")

        # 5. Skeletal (Pose)
        try:
            skeletal_set = self.skeletal_extractor.extract(raw_data_context)
            if skeletal_set.biomarkers:
                systems.append(self._transform_biomarker_set(skeletal_set))
        except Exception as e:
             logging.error(f"Skeletal extraction failed: {e}")

        # 6. Renal (Thermal - Diabetes/Microcirculation)
        try:
            renal_set = self.renal_extractor.extract(raw_data_context)
            if renal_set.biomarkers:
                systems.append(self._transform_biomarker_set(renal_set))
        except Exception as e:
            logger.error(f"Renal extraction failed: {e}")

        # 7. Eyes (FaceMesh)
        try:
            eye_set = self.eye_extractor.extract(raw_data_context)
            if eye_set.biomarkers:
                systems.append(self._transform_biomarker_set(eye_set))
        except Exception as e:
            logger.error(f"Eye extraction failed: {e}")

        # 8. Nasal (FaceMesh + Radar + Thermal)
        try:
            nasal_set = self.nasal_extractor.extract(raw_data_context)
            if nasal_set.biomarkers:
                systems.append(self._transform_biomarker_set(nasal_set))
        except Exception as e:
            logger.error(f"Nasal extraction failed: {e}")
        
        return {
            "patient_id": patient_id,
            "include_validation": True,
            "systems": systems
        }
    
    def send_screening(self, request: Dict[str, Any]) -> Optional[Dict]:
        """Send screening request to API."""
        try:
            response = requests.post(
                self.screening_endpoint,
                json=request,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Screening completed: {result.get('screening_id')}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def generate_report(self, screening_id: str, report_type: str = "patient") -> Optional[Dict]:
        """Generate report for completed screening."""
        try:
            response = requests.post(
                self.report_endpoint,
                json={"screening_id": screening_id, "report_type": report_type},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Report generation failed: {e}")
            return None


# ==============================================================================
# MAIN BRIDGE CONTROLLER
# ==============================================================================

class HardwareBridge:
    """Main controller for Split-USB hardware-software integration."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.radar_reader: Optional[RadarReader] = None
        self.esp32_reader: Optional[ESP32Reader] = None
        self.camera: Optional[CameraCapture] = None
        self.data_fusion = DataFusion(config.api_url)
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize all hardware connections."""
        success = True
        
        # Initialize Seeed Radar Kit reader (COM_A)
        if SERIAL_AVAILABLE:
            self.radar_reader = RadarReader(
                port=self.config.radar_port,
                baud=self.config.radar_baud
            )
            if not self.radar_reader.connect():
                logger.warning("Seeed Radar Kit not connected - continuing without radar")
                self.radar_reader = None
        
        # Initialize ESP32 reader (COM_B - Thermal)
        if SERIAL_AVAILABLE:
            self.esp32_reader = ESP32Reader(
                port=self.config.serial_port,
                baud=self.config.serial_baud
            )
            if not self.esp32_reader.connect():
                logger.warning("ESP32 (Thermal) not connected - continuing without thermal")
                self.esp32_reader = None
        
        # Initialize camera
        self.camera = CameraCapture(
            camera_index=self.config.camera_index,
            fps=self.config.camera_fps
        )
        if not self.camera.open():
            logger.warning("Camera not available - continuing without camera")
            self.camera = None
        
        return success
    
    def run_single_screening(self, patient_id: str = "WALKTHROUGH_PATIENT") -> Optional[Dict]:
        """Run a complete screening session with Split-USB Architecture."""
        logger.info("=" * 60)
        logger.info("Starting health screening session (Split-USB Architecture)")
        logger.info("=" * 60)
        
        radar_data = None
        esp32_data = None
        face_frames = []
        pose_sequence = []
        
        # Start Seeed Radar Kit data collection (COM_A)
        if self.radar_reader:
            logger.info("Starting Seeed Radar Kit data collection (COM_A)...")
            self.radar_reader.start_reading()
        
        # Start ESP32 thermal data collection (COM_B)
        if self.esp32_reader:
            logger.info("Starting ESP32 thermal data collection (COM_B)...")
            self.esp32_reader.start_reading()
        
        # Phase 1: Face capture (10s)
        if self.camera:
            logger.info(f"\n📷 Phase 1: Face capture ({self.config.face_capture_seconds}s)")
            logger.info("Please look directly at the camera (close-up)...")
            
            # Clear sensor queues to sync with video start
            if self.radar_reader:
                with self.radar_reader.data_queue.mutex:
                    self.radar_reader.data_queue.queue.clear()
            if self.esp32_reader:
                with self.esp32_reader.data_queue.mutex:
                    self.esp32_reader.data_queue.queue.clear()
            
            # Use on-the-fly skin ROI extraction AND FaceMesh landmarks
            # Capture both face ROIs (for Skin) and FaceMesh landmarks (for Eyes/Nasal)
            captured_data, count = self.camera.capture_and_process_frames(
                duration_seconds=self.config.face_capture_seconds,
                process_func=lambda f: {
                    'roi': self.camera.extract_face_roi(f),
                    'landmarks': self.camera.extract_face_landmarks(f)
                }
            )
            
            # Unpack results
            face_frames = [d['roi'] for d in captured_data if d['roi'] is not None]
            face_landmarks_sequence = [d['landmarks'] for d in captured_data if d['landmarks'] is not None]
            
            logger.info(f"Captured {len(face_frames)} face frames and {len(face_landmarks_sequence)} landmark sets (from {count} raw)")
        
        # Phase 2: Body capture for gait/posture (10s)
        if self.camera:
            logger.info(f"\n🚶 Phase 2: Body capture ({self.config.body_capture_seconds}s)")
            logger.info("Please walk naturally or stand for posture analysis (full body)...")
            
            # Use on-the-fly pose extraction
            pose_sequence, count = self.camera.capture_and_process_frames(
                duration_seconds=self.config.body_capture_seconds,
                process_func=self.camera.extract_pose_from_frame
            )
            logger.info(f"Extracted {len(pose_sequence)} pose frames (from {count} raw)")
        
        # ---------------------------------------------------------
        # DATA AGGREGATION (Sync with video duration)
        # ---------------------------------------------------------
        
        # Aggregate Radar Data
        if self.radar_reader:
            radar_items = []
            while not self.radar_reader.data_queue.empty():
                radar_items.append(self.radar_reader.data_queue.get())
            
            if radar_items:
                # Calculate averages for physiological stability
                avg_hr = int(sum(i['radar']['heart_rate'] for i in radar_items) / len(radar_items))
                avg_resp = round(sum(i['radar']['respiration_rate'] for i in radar_items) / len(radar_items), 1)
                
                # Use the latest item as the base structure, but inject averages
                radar_data = radar_items[-1]
                radar_data['radar']['heart_rate'] = avg_hr
                radar_data['radar']['respiration_rate'] = avg_resp
                
                logger.info(f"aggregated {len(radar_items)} radar samples. Avg HR: {avg_hr}, Avg Resp: {avg_resp}")
            else:
                 # Fallback to last known if queue empty (unlikely with clear)
                radar_data = self.radar_reader.get_latest_data()
                logger.warning("No radar data collected during video - using last known")

        # Aggregate ESP32 Thermal Data
        if self.esp32_reader:
            thermal_items = []
            while not self.esp32_reader.data_queue.empty():
                thermal_items.append(self.esp32_reader.data_queue.get())
            
            logger.info(f"📊 Collected {len(thermal_items)} thermal samples from ESP32 queue")
            
            if thermal_items:
                # Average key thermal metrics across all collected frames.
                # Supports REAL firmware format (core_regions/stability_metrics/symmetry/gradients)
                
                def get_val(item, category, field, default=0.0):
                    """Extract deep value safely from thermal JSON."""
                    return item.get('thermal', {}).get(category, {}).get(field, default)

                # Detect format from first item
                is_firmware = 'core_regions' in thermal_items[0].get('thermal', {})
                
                if is_firmware:
                    # REAL FIRMWARE: average all 5 ROI measurements
                    canthus_vals = [get_val(i, 'core_regions', 'canthus_mean') for i in thermal_items]
                    neck_vals = [get_val(i, 'core_regions', 'neck_mean') for i in thermal_items]
                    stability_vals = [get_val(i, 'stability_metrics', 'canthus_range') for i in thermal_items]
                    asymmetry_vals = [get_val(i, 'symmetry', 'cheek_asymmetry') for i in thermal_items]
                    gradient_vals = [get_val(i, 'gradients', 'forehead_nose_gradient') for i in thermal_items]
                    
                    # Filter out zeros for temp values (sensor may not have face lock)
                    canthus_valid = [v for v in canthus_vals if v > 25.0]
                    neck_valid = [v for v in neck_vals if v > 25.0]
                    
                    avg_canthus = round(sum(canthus_valid)/len(canthus_valid), 2) if canthus_valid else 0.0
                    avg_neck = round(sum(neck_valid)/len(neck_valid), 2) if neck_valid else 0.0
                    avg_stability = round(sum(stability_vals)/len(stability_vals), 2) if stability_vals else 0.0
                    avg_asymmetry = round(sum(asymmetry_vals)/len(asymmetry_vals), 3) if asymmetry_vals else 0.0
                    avg_gradient = round(sum(gradient_vals)/len(gradient_vals), 2) if gradient_vals else 0.0
                    
                    # Build aggregated data in firmware format
                    esp32_data = {
                        'timestamp': thermal_items[-1].get('timestamp', 0),
                        'thermal': {
                            'core_regions': {
                                'canthus_mean': avg_canthus,
                                'neck_mean': avg_neck
                            },
                            'stability_metrics': {
                                'canthus_range': avg_stability
                            },
                            'symmetry': {
                                'cheek_asymmetry': avg_asymmetry
                            },
                            'gradients': {
                                'forehead_nose_gradient': avg_gradient
                            }
                        }
                    }
                    
                    logger.info(
                        f"Aggregated {len(thermal_items)} thermal samples (firmware). "
                        f"Canthus: {avg_canthus}°C, Neck: {avg_neck}°C, "
                        f"Stability: {avg_stability}, Asymmetry: {avg_asymmetry}, "
                        f"Gradient: {avg_gradient}"
                    )
                else:
                    # LEGACY FORMAT: average fever.neck_temp and autonomic.stress_gradient
                    neck_temps = [get_val(i, 'fever', 'neck_temp') for i in thermal_items if get_val(i, 'fever', 'neck_temp') > 0]
                    avg_neck = sum(neck_temps)/len(neck_temps) if neck_temps else 0.0
                    stress_vals = [get_val(i, 'autonomic', 'stress_gradient') for i in thermal_items]
                    avg_stress = sum(stress_vals)/len(stress_vals) if stress_vals else 0.0
                    
                    esp32_data = thermal_items[-1]
                    if 'fever' in esp32_data.get('thermal', {}):
                        esp32_data['thermal']['fever']['neck_temp'] = round(avg_neck, 2)
                    if 'autonomic' in esp32_data.get('thermal', {}):
                        esp32_data['thermal']['autonomic']['stress_gradient'] = round(avg_stress, 2)
                    
                    logger.info(f"Aggregated {len(thermal_items)} thermal samples (legacy). Avg Neck: {avg_neck:.1f}, Avg Stress: {avg_stress:.1f}")
            else:
                esp32_data = self.esp32_reader.get_latest_data()
                logger.warning("No thermal data collected during video - using last known")
        else:
            logger.warning("⚠️ ESP32 reader not initialized - NO THERMAL DATA AVAILABLE")
        
        # Log final data status
        logger.info("\n📊 Final Data Summary:")
        logger.info(f"  Radar data: {'✓ Available' if radar_data else '✗ Not available'}")
        logger.info(f"  ESP32 data: {'✓ Available' if esp32_data else '✗ Not available'}")
        logger.info(f"  Face frames: {len(face_frames) if face_frames else 0}")
        logger.info(f"  Pose sequence: {len(pose_sequence) if pose_sequence else 0}")
        
        if esp32_data:
            logger.info(f"  Thermal data keys: {list(esp32_data.get('thermal', {}).keys())}")
        
        # Build and send screening request
        logger.info("\n📊 Processing biomarkers and sending to API...")
        
        request = self.data_fusion.build_screening_request(
            patient_id=patient_id,
            radar_data=radar_data,
            esp32_data=esp32_data,
            face_frames=face_frames if face_frames else None,
            pose_sequence=pose_sequence if pose_sequence else None,
            fps=self.config.camera_fps
        )
        
        logger.info(f"Screening request: {len(request['systems'])} systems")
        for sys in request['systems']:
            logger.info(f"  - {sys['system']}: {len(sys['biomarkers'])} biomarkers")
        
        result = self.data_fusion.send_screening(request)
        
        if result:
            logger.info("\n✅ Screening completed!")
            logger.info(f"Screening ID: {result.get('screening_id')}")
            logger.info(f"Overall Risk: {result.get('overall_risk_level')} ({result.get('overall_risk_score')})")
            
            # Generate reports
            screening_id = result.get('screening_id')
            if screening_id:
                patient_report = self.data_fusion.generate_report(screening_id, "patient")
                if patient_report:
                    logger.info(f"Patient report: {patient_report.get('pdf_path')}")
                
                doctor_report = self.data_fusion.generate_report(screening_id, "doctor")
                if doctor_report:
                    logger.info(f"Doctor report: {doctor_report.get('pdf_path')}")
        else:
            logger.error("❌ Screening failed")
        
        return result
    
    def cleanup(self):
        """Cleanup all resources."""
        if self.radar_reader:
            self.radar_reader.disconnect()
        if self.esp32_reader:
            self.esp32_reader.disconnect()
        if self.camera:
            self.camera.close()
        logger.info("Hardware bridge cleanup complete")


# ==============================================================================
# SIMULATED DATA (for testing without hardware)
# ==============================================================================

def generate_simulated_radar_data() -> Dict[str, Any]:
    """Generate realistic simulated Seeed Radar Kit data."""
    return {
        "timestamp": int(time.time()),
        "radar": {
            "respiration_rate": round(float(np.random.uniform(12, 18)), 1),
            "heart_rate": int(np.random.uniform(65, 85)),
            "breathing_depth": round(float(np.random.uniform(0.5, 0.9)), 2),
            "presence_detected": True
        }
    }


def generate_simulated_esp32_data() -> Dict[str, Any]:
    """Generate realistic simulated ESP32 thermal data.
    
    Matches the REAL firmware JSON schema from esp32_thermal_bridge.ino.
    Uses core_regions/stability_metrics/symmetry/gradients structure.
    """
    # Realistic ROI temperatures for a healthy face
    canthus_mean = round(float(np.random.uniform(31.5, 33.0)), 2)  # Inner eye corner
    neck_mean = round(float(np.random.uniform(29.0, 31.0)), 2)      # Neck region
    canthus_range = round(float(np.random.uniform(0.3, 0.7)), 2)    # Temporal stability
    cheek_asymmetry = round(float(np.random.uniform(0.1, 0.3)), 3)  # Left-right delta
    forehead_nose_gradient = round(float(np.random.uniform(-2.5, -1.0)), 2)  # Typically negative
    
    return {
        "timestamp": int(time.time()),
        "thermal": {
            "core_regions": {
                "canthus_mean": canthus_mean,
                "neck_mean": neck_mean
            },
            "stability_metrics": {
                "canthus_range": canthus_range
            },
            "symmetry": {
                "cheek_asymmetry": cheek_asymmetry
            },
            "gradients": {
                "forehead_nose_gradient": forehead_nose_gradient
            }
        }
    }


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    """Main entry point for hardware bridge (Split-USB Architecture)."""
    parser = argparse.ArgumentParser(
        description="Hardware Bridge for Health Screening (Split-USB Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bridge.py --radar-port COM6 --port COM5 --camera 0
  python bridge.py --simulate
  python bridge.py --simulate --patient-id PATIENT_123

Split-USB Architecture:
  - Radar (Seeed MR60BHA2): --radar-port (default: COM7)
  - ESP32 Thermal: --port (default: COM6)
  - Webcam: --camera (default: 0)
        """
    )
    parser.add_argument("--radar-port", default="COM7", 
                        help="Serial port for Seeed Radar Kit (COM_A)")
    parser.add_argument("--port", default="COM6", 
                        help="Serial port for ESP32 NodeMCU thermal (COM_B)")
    parser.add_argument("--camera", type=int, default=0, 
                        help="Camera index for OpenCV")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                        help="FastAPI server URL")
    parser.add_argument("--simulate", action="store_true", 
                        help="Use simulated sensor data (no hardware)")
    parser.add_argument("--patient-id", default="WALKTHROUGH_001", 
                        help="Patient identifier")
    args = parser.parse_args()
    
    config = BridgeConfig(
        radar_port=args.radar_port,
        serial_port=args.port,
        camera_index=args.camera,
        api_url=args.api_url
    )
    
    bridge = HardwareBridge(config)
    
    try:
        print("\n" + "=" * 60)
        print("  HEALTH SCREENING HARDWARE BRIDGE")
        print("  Split-USB Architecture")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Radar Port (Seeed):  {config.radar_port}")
        print(f"  ESP32 Port (Thermal): {config.serial_port}")
        print(f"  Camera Index:         {config.camera_index}")
        print(f"  API URL:              {config.api_url}")
        
        if args.simulate:
            print("\n[SIMULATION MODE] - Using generated sensor data")
            
            # Create mock data for both sensors
            simulated_radar = generate_simulated_radar_data()
            simulated_thermal = generate_simulated_esp32_data()
            
            print(f"\nSimulated Radar Data:")
            print(json.dumps(simulated_radar, indent=2, cls=NumpyEncoder))
            print(f"\nSimulated Thermal Data:")
            print(json.dumps(simulated_thermal, indent=2, cls=NumpyEncoder))
            
            # Send directly using DataFusion
            fusion = DataFusion(config.api_url)
            request = fusion.build_screening_request(
                patient_id=args.patient_id,
                radar_data=simulated_radar,
                esp32_data=simulated_thermal
            )
            
            print(f"\nAPI Request ({len(request['systems'])} systems):")
            for sys in request['systems']:
                print(f"  - {sys['system']}: {len(sys['biomarkers'])} biomarkers")
            
            result = fusion.send_screening(request)
            
            if result:
                print(f"\n✅ Screening ID: {result.get('screening_id')}")
                print(f"Risk Level: {result.get('overall_risk_level')}")
            else:
                print("\n[X] Screening failed - Is the API server running?")
        else:
            bridge.initialize()
            bridge.run_single_screening(patient_id=args.patient_id)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
    
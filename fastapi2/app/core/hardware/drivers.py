"""
Hardware Drivers for Chiranjeevi Health Screening.

Contains sensor driver classes ported from bridge.py for use within
the unified HardwareManager architecture.

Drivers:
- BaseSerialReader: Abstract serial port reader with threaded read loop
- ESP32Reader: Thermal data from ESP32 NodeMCU + MLX90640
- RadarReader: Heart rate & respiration from Seeed MR60BHA2
- CameraCapture: OpenCV capture + MediaPipe FaceMesh/Pose extraction
"""

import json
import re
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Tuple
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Optional serial import
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial not installed. Run: pip install pyserial")

# Optional MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Camera features limited.")


# ==============================================================================
# BASE SERIAL READER
# ==============================================================================

class BaseSerialReader:
    """Base class for serial port readers."""
    
    def __init__(self, port: str, baud: int = 115200, name: str = "Device"):
        self.port = port
        self.baud = baud
        self.name = name
        self.serial_conn = None
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
                        valid_rois = thermal['metadata'].get('valid_rois', 0)
                        if valid_rois < 5:
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
                            if 40 <= heart_rate <= 180:
                                self.last_data['radar']['heart_rate'] = int(heart_rate)
                                self.last_data['timestamp'] = int(time.time())
                        
                        if rr_match:
                            resp_rate = float(rr_match.group(1))
                            if 5 <= resp_rate <= 40:
                                self.last_data['radar']['respiration_rate'] = round(resp_rate, 1)
                                self.last_data['timestamp'] = int(time.time())
                        
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
        """Parse text-based output from Seeed MR60BHA2 radar (deprecated, kept for compat)."""
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
        self.face_mesh = None
        
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe using legacy solutions API."""
        if not MEDIAPIPE_AVAILABLE:
            return
            
        try:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            logger.info("✅ FaceDetector (Legacy) initialized")
        except Exception as e:
            logger.warning(f"Face detector init failed: {e}")
        
        try:
            self.pose_landmarker = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✅ PoseLandmarker (Legacy) initialized")
        except Exception as e:
            logger.warning(f"Pose landmarker init failed: {e}")

        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
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
            self.cap = None
            logger.info("Camera closed")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera. Thread-safe."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
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
                    result = process_func(frame)
                    if result is not None:
                        processed_data.append(result)
                else:
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
        timestamps = [i/self.fps for i in range(len(frames))] 
        return frames, timestamps
    
    def extract_face_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract face ROI from a single frame using legacy API."""
        if not self.face_detector:
            fh, fw = frame.shape[:2]
            return frame[fh//4:3*fh//4, fw//4:3*fw//4]
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_detector.process(rgb_frame)
            
            if result.detections:
                detection = result.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                pad = int(max(box_w, box_h) * 0.2)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w, x + box_w + pad), min(h, y + box_h + pad)
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
            return None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose_landmarker.process(rgb_frame)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                landmark_array = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in landmarks
                ])
                
                if landmark_array.shape == (33, 4):
                    return landmark_array
                else:
                    logger.warning(f"Invalid pose shape: {landmark_array.shape}")
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
        return None

    def extract_pose_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Batch extraction (legacy)."""
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb_frame)
            
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                return np.array([
                    [lm.x, lm.y, lm.z, 1.0]
                    for lm in landmarks
                ])
        except Exception as e:
            logger.debug(f"FaceMesh error: {e}")
        return None
    
    def detect_all(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Consolidate all MediaPipe processing into a single RGB conversion pass.
        Returns a dictionary containing raw MediaPipe result objects.
        """
        results = {
            "face_mesh": None,
            "pose": None,
            "face_det": None
        }
        if not MEDIAPIPE_AVAILABLE:
            return results
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.face_mesh:
                results["face_mesh"] = self.face_mesh.process(rgb_frame)
            if self.pose_landmarker:
                results["pose"] = self.pose_landmarker.process(rgb_frame)
            if self.face_detector:
                results["face_det"] = self.face_detector.process(rgb_frame)
                
        except Exception as e:
            logger.debug(f"Consolidated detection error: {e}")
            
        return results

    # ==================================================================
    # SERVER-SIDE DRAWING UTILITIES (for MJPEG stream overlays)
    # ==================================================================
    
    def calculate_face_distance(self, frame: np.ndarray, results: Optional[Dict] = None) -> Tuple[Optional[str], Optional[float]]:
        """
        Calculate user distance from camera based on face width.
        Uses results if provided, otherwise runs detector.
        """
        det_result = results["face_det"] if results and "face_det" in results else None
        
        if not det_result and self.face_detector:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det_result = self.face_detector.process(rgb_frame)
            except: pass

        if det_result and det_result.detections:
            detection = det_result.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            
            face_width_px = bbox.width * w
            face_ratio = bbox.width
            
            if face_ratio > 0.5:
                return "too_close", face_width_px
            elif face_ratio < 0.15:
                return "too_far", face_width_px
            else:
                return None, face_width_px
        
        return None, None
    
    def draw_face_mesh_on_frame(self, frame: np.ndarray, results: Optional[Dict] = None) -> np.ndarray:
        """
        Draw face mesh overlay on frame. Uses results if provided.
        """
        mesh_result = results["face_mesh"] if results and "face_mesh" in results else None
        
        if not mesh_result and self.face_mesh:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mesh_result = self.face_mesh.process(rgb_frame)
            except: pass
            
        if mesh_result and mesh_result.multi_face_landmarks:
            try:
                h, w = frame.shape[:2]
                landmarks = mesh_result.multi_face_landmarks[0].landmark
                
                # Draw key contours (face oval, lips, eyes)
                face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                
                for i in range(len(face_oval)):
                    start_idx = face_oval[i]
                    end_idx = face_oval[(i + 1) % len(face_oval)]
                    start, end = landmarks[start_idx], landmarks[end_idx]
                    pt1 = (int(start.x * w), int(start.y * h))
                    pt2 = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                
                left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                
                for eye_indices in [left_eye, right_eye]:
                    for i in range(len(eye_indices)):
                        start_idx, end_idx = eye_indices[i], eye_indices[(i + 1) % len(eye_indices)]
                        start, end = landmarks[start_idx], landmarks[end_idx]
                        pt1 = (int(start.x * w), int(start.y * h))
                        pt2 = (int(end.x * w), int(end.y * h))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 1)
                
                lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
                       402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311,
                       310, 415, 308]
                
                for i in range(len(lips)):
                    start_idx, end_idx = lips[i], lips[(i + 1) % len(lips)]
                    start, end = landmarks[start_idx], landmarks[end_idx]
                    pt1 = (int(start.x * w), int(start.y * h))
                    pt2 = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, pt1, pt2, (0, 128, 255), 1)
            except Exception as e:
                logger.debug(f"Face mesh drawing error: {e}")
        
        return frame
    
    def draw_pose_skeleton_on_frame(self, frame: np.ndarray, results: Optional[Dict] = None) -> np.ndarray:
        """
        Draw pose skeleton overlay on frame. Uses results if provided.
        """
        pose_result = results["pose"] if results and "pose" in results else None
        
        if not pose_result and self.pose_landmarker:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = self.pose_landmarker.process(rgb_frame)
            except: pass
            
        if pose_result and pose_result.pose_landmarks:
            try:
                h, w = frame.shape[:2]
                landmarks = pose_result.pose_landmarks.landmark
                
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 7),
                    (0, 4), (4, 5), (5, 6), (6, 8),
                    (9, 10),
                    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                    (11, 23), (12, 24), (23, 24),
                    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
                ]
                
                for start_idx, end_idx in connections:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_lm, end_lm = landmarks[start_idx], landmarks[end_idx]
                        if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                            continue
                        pt1 = (int(start_lm.x * w), int(start_lm.y * h))
                        pt2 = (int(end_lm.x * w), int(end_lm.y * h))
                        cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
                
                for idx, lm in enumerate(landmarks):
                    if lm.visibility > 0.5:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            except Exception as e:
                logger.debug(f"Pose skeleton drawing error: {e}")
        
        return frame
    
    def add_distance_warning_overlay(self, frame: np.ndarray, warning_type: str) -> np.ndarray:
        """
        Add visual warning overlay to frame based on distance check.
        
        Args:
            frame: Input frame
            warning_type: "too_close" or "too_far"
        
        Returns: Modified frame with warning overlay
        """
        h, w = frame.shape[:2]
        
        if warning_type == "too_close":
            text = "TOO CLOSE - Move Back"
            color = (0, 0, 255)  # Red
            cv2.rectangle(frame, (10, 10), (w - 10, 80), color, 3)
        elif warning_type == "too_far":
            text = "TOO FAR - Move Closer"
            color = (0, 165, 255)  # Orange
            cv2.rectangle(frame, (10, 10), (w - 10, 80), color, 3)
        else:
            return frame
        
        # Draw text
        cv2.putText(
            frame, text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )
        
        return frame

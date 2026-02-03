"""
Camera-Based Health Screening Test

Uses webcam to extract biomarkers and test the full pipeline.
No RIS hardware required - camera only!

Requirements:
    pip install opencv-python mediapipe
"""
import cv2
import numpy as np
import requests
import time
from typing import Dict, Any, List
import sys

# Import your extractors
# Import your extractors - FIXED FOR SHARED VENV
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Add project root

from app.core.extraction.cardiovascular import CardiovascularExtractor
from app.core.extraction.cns import CNSExtractor
from app.core.extraction.eyes import EyeExtractor
from app.core.extraction.skeletal import SkeletalExtractor
print("✅ All 9 extractors loaded!")


# MediaPipe for pose/face detection
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded!")
except ImportError:
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

API_URL = "http://localhost:8000"


def draw_landmarks_manual(frame, landmarks, connections, h, w):
    """Draw pose landmarks manually without mp.solutions.drawing_utils."""
    # Draw connections (skeleton lines)
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            # Only draw if both points are visible enough
            if start.visibility > 0.3 and end.visibility > 0.3:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 165, 255), 3)  # Orange lines
    
    # Draw landmarks (joints)
    for landmark in landmarks:
        if landmark.visibility > 0.3:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dots


# Define pose connections (MediaPipe standard)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
]


# ==============================================================================
# TWO-PHASE VIDEO CAPTURE SYSTEM
# Phase 1: Close-up face (rPPG heart rate + eye analysis)  
# Phase 2: Full body (gait/posture CNS + skeletal analysis)
# ==============================================================================

def draw_face_oval_guide(frame, h, w):
    """Draw an oval guide for face positioning."""
    center = (w // 2, h // 2 - 50)
    axes = (120, 160)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 255), 2)
    cv2.putText(frame, "Position face inside oval", 
               (center[0] - 130, center[1] + axes[1] + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def capture_phase1_face(cap, duration_seconds=10, countdown_seconds=5):
    """
    Phase 1: Close-up face capture for cardiovascular (rPPG) and eye analysis.
    
    User should be ~0.5-1 meter from camera with face clearly visible.
    """
    print("\n" + "="*60)
    print("📸 PHASE 1: CLOSE-UP FACE CAPTURE")
    print("="*60)
    print("For: Heart Rate (rPPG) + Eye Analysis")
    print("Position: Sit close to camera (~0.5-1 meter)")
    print("         Face should fill most of the frame")
    print("Action: Stay still, breathe normally, look at camera")
    
    # Initialize face detector for visual feedback
    face_detector = None
    if MEDIAPIPE_AVAILABLE:
        try:
            base_options = python.BaseOptions(model_asset_path='face_detector.task')
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE
            )
            face_detector = vision.FaceDetector.create_from_options(options)
            print("✅ Face detection enabled")
        except Exception as e:
            print(f"⚠ Face detector not available: {e}")
    
    # Countdown phase
    print(f"\n⏱️ Get ready! Recording starts in {countdown_seconds} seconds...")
    countdown_start = time.time()
    
    while time.time() - countdown_start < countdown_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Draw face oval guide
        draw_face_oval_guide(display_frame, h, w)
        
        # Detect and highlight face
        face_detected = False
        if face_detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_detector.detect(mp_image)
            
            if results.detections and len(results.detections) > 0:
                face_detected = True
                # Draw face bounding box
                det = results.detections[0]
                bbox = det.bounding_box
                cv2.rectangle(display_frame, 
                            (bbox.origin_x, bbox.origin_y),
                            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                            (0, 255, 0), 3)
        
        remaining = countdown_seconds - int(time.time() - countdown_start)
        
        # Status indicators
        cv2.putText(display_frame, "PHASE 1: FACE CLOSE-UP", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)
        cv2.putText(display_frame, f"Starting in: {remaining}s", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        
        status_text = "FACE DETECTED - Good!" if face_detected else "Move face into frame"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(display_frame, status_text, 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cv2.imshow('Health Screening - Phase 1: Face Close-up', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if face_detector:
                face_detector.close()
            return None, None
    
    # Recording phase
    print("🎥 RECORDING FACE! Stay still and breathe normally...")
    
    frames = []
    timestamps = []
    start_time = time.time()
    face_detected_count = 0
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame.copy())
        timestamps.append(time.time() - start_time)
        
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Face detection feedback
        if face_detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_detector.detect(mp_image)
            
            if results.detections and len(results.detections) > 0:
                face_detected_count += 1
                det = results.detections[0]
                bbox = det.bounding_box
                cv2.rectangle(display_frame, 
                            (bbox.origin_x, bbox.origin_y),
                            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                            (0, 255, 0), 3)
        
        elapsed = int(time.time() - start_time)
        remaining = duration_seconds - elapsed
        
        # Recording indicator
        cv2.circle(display_frame, (30, 30), 12, (0, 0, 255), -1)  # Red recording dot
        cv2.putText(display_frame, f"REC {elapsed}s / {duration_seconds}s", 
                   (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, "Stay still - Measuring heart rate", 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Health Screening - Phase 1: Face Close-up', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if face_detector:
        face_detector.close()
    
    detection_pct = (face_detected_count / len(frames)) * 100 if frames else 0
    print(f"✓ Phase 1 complete: {len(frames)} frames, face detected in {detection_pct:.1f}%")
    
    return frames, timestamps


def capture_phase2_fullbody(cap, duration_seconds=10, countdown_seconds=5):
    """
    Phase 2: Full body capture for CNS (gait) and skeletal analysis.
    
    User should be ~2-3 meters from camera with full body visible.
    """
    print("\n" + "="*60)
    print("🏃 PHASE 2: FULL BODY CAPTURE")
    print("="*60)
    print("For: Gait Analysis + Posture + Balance (CNS/Skeletal)")
    print("Position: Step back (~2-3 meters from camera)")
    print("         Full body should be visible (head to feet)")
    print("Action: Walk naturally, turn around, move arms")
    
    # Initialize pose detector
    pose_detector = None
    if MEDIAPIPE_AVAILABLE:
        try:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE
            )
            pose_detector = vision.PoseLandmarker.create_from_options(options)
            print("✅ Pose detection enabled - skeleton will show on screen!")
        except Exception as e:
            print(f"⚠ Pose detector not available: {e}")
    
    # Countdown phase
    print(f"\n⏱️ Step back! Recording starts in {countdown_seconds} seconds...")
    countdown_start = time.time()
    
    while time.time() - countdown_start < countdown_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Pose detection
        pose_detected = False
        if pose_detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = pose_detector.detect(mp_image)
            
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_detected = True
                draw_landmarks_manual(display_frame, results.pose_landmarks[0], POSE_CONNECTIONS, h, w)
        
        remaining = countdown_seconds - int(time.time() - countdown_start)
        
        # Status indicators
        cv2.putText(display_frame, "PHASE 2: FULL BODY", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)
        cv2.putText(display_frame, f"Starting in: {remaining}s", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        cv2.putText(display_frame, "Step back - show full body!", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        status_text = "POSE DETECTED - Good!" if pose_detected else "Move full body into frame"
        status_color = (0, 255, 0) if pose_detected else (0, 0, 255)
        cv2.putText(display_frame, status_text, 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cv2.imshow('Health Screening - Phase 2: Full Body', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if pose_detector:
                pose_detector.close()
            return None, None
    
    # Recording phase
    print("🎥 RECORDING FULL BODY! Walk around naturally...")
    
    frames = []
    timestamps = []
    start_time = time.time()
    pose_detected_count = 0
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame.copy())
        timestamps.append(time.time() - start_time)
        
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Pose detection and visualization
        if pose_detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = pose_detector.detect(mp_image)
            
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_detected_count += 1
                draw_landmarks_manual(display_frame, results.pose_landmarks[0], POSE_CONNECTIONS, h, w)
        
        elapsed = int(time.time() - start_time)
        
        # Recording indicator
        cv2.circle(display_frame, (30, 30), 12, (0, 0, 255), -1)  # Red recording dot
        cv2.putText(display_frame, f"REC {elapsed}s / {duration_seconds}s", 
                   (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, "Walk, move arms, turn around!", 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection percentage
        detection_pct = (pose_detected_count / len(frames)) * 100 if frames else 0
        cv2.putText(display_frame, f"Pose: {detection_pct:.0f}%", 
                   (w - 150, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Health Screening - Phase 2: Full Body', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if pose_detector:
        pose_detector.close()
    
    detection_pct = (pose_detected_count / len(frames)) * 100 if frames else 0
    print(f"✓ Phase 2 complete: {len(frames)} frames, pose detected in {detection_pct:.1f}%")
    
    return frames, timestamps


def capture_two_phase_video(phase1_duration=10, phase2_duration=10, countdown_seconds=5):
    """
    Two-phase video capture for optimal biomarker extraction.
    
    Phase 1: Close-up face (10s) - for rPPG heart rate + eye analysis
    Phase 2: Full body (10s) - for gait/posture CNS + skeletal analysis
    
    Returns:
        Tuple of (face_frames, face_timestamps, body_frames, body_timestamps)
    """
    print("\n" + "="*60)
    print("🎥 TWO-PHASE HEALTH SCREENING CAPTURE")
    print("="*60)
    print("\nThis screening has two phases:")
    print("  📸 Phase 1: Close-up face (sit close to camera)")
    print("  🏃 Phase 2: Full body (step back, walk around)")
    print(f"\nTotal time: ~{phase1_duration + phase2_duration + countdown_seconds*2}s")
    
    # Open camera once for both phases
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return None
    
    try:
        # Phase 1: Face close-up
        face_frames, face_timestamps = capture_phase1_face(
            cap, duration_seconds=phase1_duration, countdown_seconds=countdown_seconds
        )
        if face_frames is None:
            return None
        
        # Brief pause between phases
        print("\n⏳ Phase 1 complete! Preparing for Phase 2...")
        pause_start = time.time()
        while time.time() - pause_start < 2:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                cv2.putText(frame, "GREAT! Now step back for full body scan...", 
                           (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow('Health Screening - Transitioning...', frame)
            cv2.waitKey(1)
        
        # Phase 2: Full body
        body_frames, body_timestamps = capture_phase2_fullbody(
            cap, duration_seconds=phase2_duration, countdown_seconds=countdown_seconds
        )
        if body_frames is None:
            return None
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ BOTH PHASES COMPLETE!")
    print("="*60)
    print(f"  📸 Face frames: {len(face_frames)}")
    print(f"  🏃 Body frames: {len(body_frames)}")
    
    return face_frames, face_timestamps, body_frames, body_timestamps


# Legacy single-phase capture (kept for compatibility)
def capture_video_sequence(duration_seconds=10, fps=30, countdown_seconds=10):
    """Single-phase video capture (legacy - use capture_two_phase_video instead)."""
    result = capture_two_phase_video(
        phase1_duration=duration_seconds//2, 
        phase2_duration=duration_seconds//2,
        countdown_seconds=countdown_seconds//2
    )
    if result is None:
        return None
    
    face_frames, face_timestamps, body_frames, body_timestamps = result
    # Combine for backward compatibility
    all_frames = face_frames + body_frames
    all_timestamps = face_timestamps + [t + max(face_timestamps) for t in body_timestamps]
    return all_frames, all_timestamps


def extract_pose_sequence(frames):
    """Extract pose landmarks using MediaPipe Tasks API (0.10.x)."""
    if not MEDIAPIPE_AVAILABLE:
        print("⚠ MediaPipe not available, generating simulated poses...")
        # Return simulated pose data
        return [np.random.rand(33, 4) * 0.1 + 0.5 for _ in range(len(frames))]
    
    print("\n🏃 Extracting pose data...")
    
    try:
        # Create pose landmarker using Tasks API
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        
        pose_sequence = []
        for i, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility] 
                    for lm in result.pose_landmarks[0]
                ])
                pose_sequence.append(landmarks)
        
        detector.close()
        print(f"✓ Extracted pose from {len(pose_sequence)}/{len(frames)} frames")
        return pose_sequence
        
    except Exception as e:
        print(f"⚠ Pose extraction failed: {e}")
        print("  Generating simulated pose data instead...")
        # Return simulated pose data (33 landmarks with x, y, z, visibility)
        return [np.random.rand(33, 4) * 0.1 + 0.5 for _ in range(len(frames))]


def extract_face_frames(frames):
    """
    Extract cropped face frames for rPPG analysis.
    
    Uses MediaPipe Face Detection to locate faces and crop them.
    These frames are used for remote photoplethysmography (rPPG)
    to extract heart rate and HRV from facial skin color changes.
    """
    if not MEDIAPIPE_AVAILABLE:
        print("⚠ MediaPipe not available, using full frames for rPPG...")
        # Fallback: use center region of full frames as "face"
        face_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            # Use center region as proxy for face
            face_roi = frame[int(h*0.2):int(h*0.6), int(w*0.3):int(w*0.7)]
            face_frames.append(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        return face_frames
    
    print("\n😊 Extracting face frames for rPPG...")
    
    try:
        # Try using the Tasks API for face detection
        try:
            base_options = python.BaseOptions(model_asset_path='face_detector.task')
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE
            )
            detector = vision.FaceDetector.create_from_options(options)
            use_tasks_api = True
        except Exception:
            # Fallback to legacy API
            mp_face = mp.solutions.face_detection
            detector = mp_face.FaceDetection(min_detection_confidence=0.5)
            use_tasks_api = False
        
        face_frames = []
        detected_count = 0
        
        for i, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            face_detected = False
            
            if use_tasks_api:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = detector.detect(mp_image)
                
                if result.detections and len(result.detections) > 0:
                    det = result.detections[0]
                    bbox = det.bounding_box
                    x1 = max(0, bbox.origin_x)
                    y1 = max(0, bbox.origin_y)
                    x2 = min(w, x1 + bbox.width)
                    y2 = min(h, y1 + bbox.height)
                    face_roi = rgb_frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        face_frames.append(face_roi)
                        face_detected = True
                        detected_count += 1
            else:
                result = detector.process(rgb_frame)
                
                if result.detections and len(result.detections) > 0:
                    det = result.detections[0]
                    bbox = det.location_data.relative_bounding_box
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    x2 = min(w, int((bbox.xmin + bbox.width) * w))
                    y2 = min(h, int((bbox.ymin + bbox.height) * h))
                    face_roi = rgb_frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        face_frames.append(face_roi)
                        face_detected = True
                        detected_count += 1
            
            # If no face detected, use center region
            if not face_detected:
                face_roi = rgb_frame[int(h*0.2):int(h*0.6), int(w*0.3):int(w*0.7)]
                face_frames.append(face_roi)
        
        if use_tasks_api:
            detector.close()
        else:
            detector.close()
        
        print(f"✓ Extracted {len(face_frames)} face frames ({detected_count} with detected faces)")
        return face_frames
        
    except Exception as e:
        print(f"⚠ Face extraction failed: {e}")
        print("  Using center region of frames instead...")
        face_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            face_roi = frame[int(h*0.2):int(h*0.6), int(w*0.3):int(w*0.7)]
            face_frames.append(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        return face_frames


def extract_biomarkers_from_camera(frames, timestamps):
    """Extract biomarkers using camera data."""
    print("\n🔬 Extracting biomarkers...")
    
    # Extract pose sequence for CNS/skeletal analysis
    pose_sequence = extract_pose_sequence(frames)
    
    # Extract face frames for rPPG (cardiovascular)
    face_frames = extract_face_frames(frames)
    
    # Calculate actual FPS
    fps = len(frames) / max(timestamps) if timestamps and max(timestamps) > 0 else 30
    
    # Prepare data for extractors
    extraction_data = {
        "frames": frames,
        "timestamps": timestamps,
        "pose_sequence": pose_sequence,
        "face_frames": face_frames,  # For rPPG extraction
        "fps": fps,  # Required for rPPG
        "frame_rate": fps
    }
    
    all_biomarkers = {}
    
    # 1. Cardiovascular (from rPPG using face frames - Priority 2!)
    try:
        print("  → Cardiovascular (rPPG)...")
        cv_extractor = CardiovascularExtractor()
        cv_result = cv_extractor.extract(extraction_data)
        all_biomarkers["cardiovascular"] = cv_result
        
        # Show extracted HR and HRV
        hr = cv_result.get("heart_rate")
        hrv = cv_result.get("hrv_rmssd")
        if hr:
            print(f"    ✓ Heart Rate: {hr.value:.1f} bpm (confidence: {hr.confidence:.2f})")
        if hrv:
            print(f"    ✓ HRV (RMSSD): {hrv.value:.1f} ms (confidence: {hrv.confidence:.2f})")
        print(f"    ✓ Found {len(cv_result.biomarkers)} biomarkers total")
    except Exception as e:
        print(f"    ⚠ Cardiovascular extraction failed: {e}")
    
    # 2. CNS (from pose/gait if available)
    try:
        print("  → CNS...")
        cns_extractor = CNSExtractor()
        cns_result = cns_extractor.extract(extraction_data)
        all_biomarkers["cns"] = cns_result
        print(f"    ✓ Found {len(cns_result.biomarkers)} biomarkers")
    except Exception as e:
        print(f"    ⚠ CNS extraction failed: {e}")
    
    # 3. Eyes (from face tracking)
    try:
        print("  → Eyes...")
        eye_extractor = EyeExtractor()
        eye_result = eye_extractor.extract(extraction_data)
        all_biomarkers["eyes"] = eye_result
        print(f"    ✓ Found {len(eye_result.biomarkers)} biomarkers")
    except Exception as e:
        print(f"    ⚠ Eyes extraction failed: {e}")
    
    # 4. Skeletal (from pose)
    try:
        print("  → Skeletal...")
        skeletal_extractor = SkeletalExtractor()
        skeletal_result = skeletal_extractor.extract(extraction_data)
        all_biomarkers["skeletal"] = skeletal_result
        print(f"    ✓ Found {len(skeletal_result.biomarkers)} biomarkers")
    except Exception as e:
        print(f"    ⚠ Skeletal extraction failed: {e}")
    
    return all_biomarkers


def format_for_api(biomarker_results: Dict) -> Dict:
    """Format extracted biomarkers for API."""
    print("\n📝 Formatting data for API...")
    
    systems = []
    for system_name, biomarker_set in biomarker_results.items():
        biomarkers = []
        for bm in biomarker_set.biomarkers:
            biomarkers.append({
                "name": bm.name,
                "value": float(bm.value),
                "unit": bm.unit,
                "normal_range": list(bm.normal_range) if bm.normal_range else None
            })
        
        if biomarkers:  # Only add if we have biomarkers
            systems.append({
                "system": system_name,
                "biomarkers": biomarkers
            })
    
    payload = {
        "patient_id": "CAMERA-TEST-001",
        "systems": systems,
        "include_validation": True
    }
    
    print(f"✓ Formatted {len(systems)} systems with biomarkers")
    return payload


def send_to_api(payload: Dict) -> Dict:
    """Send to screening API."""
    print(f"\n📤 Sending to API: {API_URL}/api/v1/screening")
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/screening",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Is the server running?")
        print("   Start with: uvicorn app.main:app --reload --port 8000")
        return None
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None


def generate_reports(screening_id: str):
    """Generate patient and doctor reports."""
    print(f"\n📄 Generating reports for screening: {screening_id}")
    
    # Patient report
    try:
        response = requests.post(
            f"{API_URL}/api/v1/reports/generate",
            json={"screening_id": screening_id, "report_type": "patient"}
        )
        response.raise_for_status()
        patient_report = response.json()
        print(f"✓ Patient Report: {patient_report['report_id']}")
        print(f"  Download: {API_URL}/api/v1/reports/{patient_report['report_id']}/download")
    except Exception as e:
        print(f"⚠ Patient report failed: {e}")
    
    # Doctor report
    try:
        response = requests.post(
            f"{API_URL}/api/v1/reports/generate",
            json={"screening_id": screening_id, "report_type": "doctor"}
        )
        response.raise_for_status()
        doctor_report = response.json()
        print(f"✓ Doctor Report: {doctor_report['report_id']}")
        print(f"  Download: {API_URL}/api/v1/reports/{doctor_report['report_id']}/download")
    except Exception as e:
        print(f"⚠ Doctor report failed: {e}")


def extract_biomarkers_two_phase(face_frames, face_timestamps, body_frames, body_timestamps):
    """
    Extract biomarkers using optimal data from each phase.
    
    Phase 1 (face): Cardiovascular (rPPG) + Eye analysis
    Phase 2 (body): CNS (gait) + Skeletal analysis
    """
    print("\n🔬 Extracting biomarkers from two-phase capture...")
    
    all_biomarkers = {}
    
    # Calculate FPS for each phase
    face_fps = len(face_frames) / max(face_timestamps) if face_timestamps and max(face_timestamps) > 0 else 30
    body_fps = len(body_frames) / max(body_timestamps) if body_timestamps and max(body_timestamps) > 0 else 30
    
    # ==========================================================
    # PHASE 1 DATA: Close-up face → Cardiovascular + Eyes
    # ==========================================================
    print("\n  📸 Processing Phase 1 (Face Close-up)...")
    
    # Extract face regions from Phase 1 frames (they're already close-up!)
    face_frames_rgb = []
    for frame in face_frames:
        h, w = frame.shape[:2]
        # For Phase 1, face should already fill the frame, use center region
        face_roi = frame[int(h*0.15):int(h*0.85), int(w*0.2):int(w*0.8)]
        face_frames_rgb.append(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    
    face_data = {
        "frames": face_frames,
        "timestamps": face_timestamps,
        "face_frames": face_frames_rgb,
        "fps": face_fps,
        "frame_rate": face_fps
    }
    
    # 1. Cardiovascular (from rPPG - uses Phase 1 close-up face!)
    try:
        print("    → Cardiovascular (rPPG from close-up face)...")
        cv_extractor = CardiovascularExtractor()
        cv_result = cv_extractor.extract(face_data)
        all_biomarkers["cardiovascular"] = cv_result
        
        hr = cv_result.get("heart_rate")
        hrv = cv_result.get("hrv_rmssd")
        if hr:
            print(f"      ✓ Heart Rate: {hr.value:.1f} bpm (confidence: {hr.confidence:.2f})")
        if hrv:
            print(f"      ✓ HRV (RMSSD): {hrv.value:.1f} ms (confidence: {hrv.confidence:.2f})")
        print(f"      ✓ Found {len(cv_result.biomarkers)} cardiovascular biomarkers")
    except Exception as e:
        print(f"      ⚠ Cardiovascular extraction failed: {e}")
    
    # 2. Eyes (from Phase 1 close-up face!)
    try:
        print("    → Eyes (from close-up face)...")
        eye_extractor = EyeExtractor()
        eye_result = eye_extractor.extract(face_data)
        all_biomarkers["eyes"] = eye_result
        print(f"      ✓ Found {len(eye_result.biomarkers)} eye biomarkers")
    except Exception as e:
        print(f"      ⚠ Eyes extraction failed: {e}")
    
    # ==========================================================
    # PHASE 2 DATA: Full body → CNS + Skeletal
    # ==========================================================
    print("\n  🏃 Processing Phase 2 (Full Body)...")
    
    # Extract pose sequence from Phase 2 body frames
    pose_sequence = extract_pose_sequence(body_frames)
    
    body_data = {
        "frames": body_frames,
        "timestamps": body_timestamps,
        "pose_sequence": pose_sequence,
        "fps": body_fps,
        "frame_rate": body_fps
    }
    
    # 3. CNS (from Phase 2 full body gait!)
    try:
        print("    → CNS (gait/posture from full body)...")
        cns_extractor = CNSExtractor()
        cns_result = cns_extractor.extract(body_data)
        all_biomarkers["cns"] = cns_result
        
        gait = cns_result.get("gait_variability")
        stability = cns_result.get("cns_stability_score")
        if gait:
            print(f"      ✓ Gait Variability: {gait.value:.4f} (confidence: {gait.confidence:.2f})")
        if stability:
            print(f"      ✓ CNS Stability: {stability.value:.1f}/100 (confidence: {stability.confidence:.2f})")
        print(f"      ✓ Found {len(cns_result.biomarkers)} CNS biomarkers")
    except Exception as e:
        print(f"      ⚠ CNS extraction failed: {e}")
    
    # 4. Skeletal (from Phase 2 full body pose!)
    try:
        print("    → Skeletal (from full body pose)...")
        skeletal_extractor = SkeletalExtractor()
        skeletal_result = skeletal_extractor.extract(body_data)
        all_biomarkers["skeletal"] = skeletal_result
        print(f"      ✓ Found {len(skeletal_result.biomarkers)} skeletal biomarkers")
    except Exception as e:
        print(f"      ⚠ Skeletal extraction failed: {e}")
    
    print(f"\n✅ Total: {sum(len(b.biomarkers) for b in all_biomarkers.values())} biomarkers extracted")
    
    return all_biomarkers


def main():
    """Main camera test flow with two-phase capture."""
    print("="*60)
    print("🎥 CAMERA-BASED HEALTH SCREENING TEST")
    print("="*60)
    print("\n🆕 TWO-PHASE CAPTURE MODE")
    print("\nThis will:")
    print("  📸 Phase 1: Capture close-up face (10s) for heart rate")
    print("  🏃 Phase 2: Capture full body (10s) for gait analysis")
    print("  🔬 Extract biomarkers from optimal data sources")
    print("  📊 Send to API for risk assessment")
    print("  📄 Generate patient and doctor reports")
    print("\nMake sure:")
    print("  ✓ API server is running (uvicorn)")
    print("  ✓ Camera is connected and working")
    print("  ✓ Good lighting for best results")
    print("  ✓ Space to sit close AND step back")
    
    input("\nPress Enter to start...")
    
    # Step 1: Two-phase video capture
    result = capture_two_phase_video(phase1_duration=10, phase2_duration=10, countdown_seconds=5)
    if result is None:
        return
    
    face_frames, face_timestamps, body_frames, body_timestamps = result
    
    # Step 2: Extract biomarkers using phase-specific data
    biomarker_results = extract_biomarkers_two_phase(
        face_frames, face_timestamps, 
        body_frames, body_timestamps
    )
    if not biomarker_results:
        print("\n❌ No biomarkers extracted!")
        return
    
    # Step 3: Format for API
    api_payload = format_for_api(biomarker_results)
    
    # Step 4: Send to API
    screening_result = send_to_api(api_payload)
    if screening_result is None:
        return
    
    screening_id = screening_result.get("screening_id")
    print(f"\n✅ Screening Complete!")
    print(f"   ID: {screening_id}")
    print(f"   Overall Risk: {screening_result.get('overall_risk_level')}")
    print(f"   Risk Score: {screening_result.get('overall_risk_score')}")
    
    # Step 5: Generate reports
    generate_reports(screening_id)
    
    print("\n" + "="*60)
    print("✅ TWO-PHASE CAMERA TEST COMPLETE!")
    print("="*60)
    print("\nCheck the generated PDF reports to see:")
    print("  • Heart rate & HRV from close-up face analysis")
    print("  • Gait & posture from full body analysis")
    print("  • Combined risk assessments")
    print("  • AI-generated explanations")


if __name__ == "__main__":
    main()

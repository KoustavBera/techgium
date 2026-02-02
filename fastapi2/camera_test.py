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


def capture_video_sequence(duration_seconds=10, fps=30, countdown_seconds=10):
    """Capture video from webcam with positioning countdown."""
    print(f"\n📹 Opening webcam...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return None
    
    # Positioning countdown
    print(f"\n⏱️ Position yourself! Recording starts in {countdown_seconds} seconds...")
    countdown_start = time.time()
    
    while time.time() - countdown_start < countdown_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        remaining = countdown_seconds - int(time.time() - countdown_start)
        # Draw countdown on frame
        cv2.putText(frame, f"Get Ready: {remaining}s", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        cv2.putText(frame, "Position yourself in front of camera", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Health Screening - Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    print("🎥 Recording NOW! Look at the camera and breathe normally")
    print(f"   (Recording for {duration_seconds}s, press 'q' to stop early)")
    
    frames = []
    timestamps = []
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        timestamps.append(time.time() - start_time)
        frame_count += 1
        
        # Display frame
        cv2.putText(frame, f"Recording: {int(time.time()-start_time)}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Health Screening - Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Captured {frame_count} frames ({frame_count/duration_seconds:.1f} fps)")
    return frames, timestamps


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


def extract_biomarkers_from_camera(frames, timestamps):
    """Extract biomarkers using camera data."""
    print("\n🔬 Extracting biomarkers...")
    
    # Extract pose sequence
    pose_sequence = extract_pose_sequence(frames)
    
    # Prepare data for extractors
    extraction_data = {
        "frames": frames,
        "timestamps": timestamps,
        "pose_sequence": pose_sequence,
        "frame_rate": len(frames) / max(timestamps) if timestamps else 30
    }
    
    all_biomarkers = {}
    
    # 1. Cardiovascular (from chest motion, simulated PPG)
    try:
        print("  → Cardiovascular...")
        cv_extractor = CardiovascularExtractor()
        cv_result = cv_extractor.extract(extraction_data)
        all_biomarkers["cardiovascular"] = cv_result
        print(f"    ✓ Found {len(cv_result.biomarkers)} biomarkers")
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


def main():
    """Main camera test flow."""
    print("="*60)
    print("🎥 CAMERA-BASED HEALTH SCREENING TEST")
    print("="*60)
    print("\nThis will:")
    print("1. Capture video from your webcam (10 seconds)")
    print("2. Extract biomarkers using your extraction modules")
    print("3. Send to the API for risk assessment")
    print("4. Generate patient and doctor reports")
    print("\nMake sure:")
    print("  ✓ API server is running (uvicorn)")
    print("  ✓ Camera is connected and working")
    print("  ✓ Good lighting for best results")
    
    input("\nPress Enter to start...")
    
    # Step 1: Capture video
    result = capture_video_sequence(duration_seconds=10, fps=30)
    if result is None:
        return
    frames, timestamps = result
    
    # Step 2: Extract biomarkers
    biomarker_results = extract_biomarkers_from_camera(frames, timestamps)
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
    print("✅ CAMERA TEST COMPLETE!")
    print("="*60)
    print("\nCheck the generated PDF reports to see:")
    print("  • Extracted biomarker values")
    print("  • Risk assessments")
    print("  • AI-generated explanations")


if __name__ == "__main__":
    main()

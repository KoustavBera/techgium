import asyncio
import json
import logging
import os
import sys
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Unified Hardware Manager
from app.core.hardware.manager import HardwareManager, HardwareConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("headless_runner")

# Output directory for raw data logs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class HeadlessRunner:
    """
    Headless implementation of the health screening pipeline.
    
    This class replicates the orchestration logic of the frontend + HardwareManager
    to execute a full scan while providing "glass box" visibility into data
    aggregation and median selection.
    """
    
    def __init__(self):
        self.manager = HardwareManager()
        self.patient_id = f"HEADLESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_log = {
            "patient_id": self.patient_id,
            "timestamp": datetime.now().isoformat(),
            "raw_sequences": {
                "radar": [],
                "thermal": [],
                "face_frames": [], # Meta only
                "pose_frames": []  # Meta only
            },
            "aggregation_logic": {}
        }

    async def initialize(self):
        """Initialize hardware."""
        print("\n[1/5] Initializing Hardware...")
        # Use existing config structure
        config = HardwareConfig(
            camera_index=0,
            radar_port=os.environ.get("RADAR_PORT", "COM7"),
            esp32_port=os.environ.get("ESP32_PORT", "COM6"),
            face_capture_seconds=10,
            body_capture_seconds=10
        )
        await self.manager.startup(config=config)
        
        # Verify connections
        status = self.manager.get_sensor_status()
        print(f"   - Camera: {status['camera']['status']}")
        print(f"   - Radar: {status['radar']['status']}")
        print(f"   - Thermal: {status['thermal']['status']}")
        
        if status['radar']['status'] != 'connected':
             logger.warning("⚠️ Radar not connected. Data will be missing.")
        if status['thermal']['status'] != 'connected':
             logger.warning("⚠️ Thermal camera not connected. Data will be missing.")

        print("\n[PROPOSED FIX] Clearing stale buffer data...")
        # Drain queues to ensure fresh data for the scan
        if self.manager.radar:
            while not self.manager.radar.data_queue.empty():
                self.manager.radar.data_queue.get_nowait()
        if self.manager.thermal:
            while not self.manager.thermal.data_queue.empty():
                self.manager.thermal.data_queue.get_nowait()
        print("   ✅ Queues cleared.")

    async def run_scan(self):
        """Execute the full scan pipeline with detailed logging."""
        try:
            # --- Phase 0: Vitals Collection (10s) ---
            print("\n   📡 PHASE 0: Vitals Collection (10s)")
            print("   Please sit still for accurate readings...")
            for i in range(10, 0, -1):
                print(f"   Collecting Vitals... {i}s", end='\r')
                await asyncio.sleep(1)
            print("   Collecting Vitals... Done!")
            
            # --- Phase 0.5: Baseline Calibration (5s) ---
            print("\n   ⏳ PHASE 0.5: Baseline Calibration (5s)")
            print("   Establishing environmental baseline...")
            for i in range(5, 0, -1):
                print(f"   Calibrating... {i}s", end='\r')
                await asyncio.sleep(1)
            print("   Calibrating... Done!")
            
            # --- Phase 1: Face Analysis (10s) ---
            print("\n   📸 PHASE 1: Face Analysis (10s)")
            print("   Please look at the camera...")
            
            if self.manager.camera:
                 self.manager.camera.start_recording()
            
            # We must MANUALLY collect data during this time for radar/thermal
            # matching what the frontend would do (it just waits), but the backend
            # manager accumulates it in queues.
            
            # Wait 10s
            for i in range(10, 0, -1):
                print(f"   Scanning... {i}s", end='\r')
                await asyncio.sleep(1)
            print("   Scanning... Done!")
            
            face_frames_meta = []
            if self.manager.camera:
                # Stop recording returns list of {'frame': np.array, 'timestamp': float}
                raw_captures = self.manager.camera.stop_recording()
                print(f"   ✅ Captured {len(raw_captures)} face frames")
                face_frames_meta = [{"timestamp": c['timestamp'], "shape": c['frame'].shape} for c in raw_captures]
                # We store the actual frames in the manager for processing, but log meta here
                self.session_log["raw_sequences"]["face_frames"] = face_frames_meta
                self.face_frames_data = raw_captures  # STORE FOR PROCESSING

            # --- Phase 2: Body Analysis (10s) ---
            print("\n   🚶 PHASE 2: Body Analysis (10s)")
            print("   Please show your full body...")
            
            if self.manager.camera:
                 self.manager.camera.start_recording()
            
            for i in range(10, 0, -1):
                print(f"   Scanning... {i}s", end='\r')
                await asyncio.sleep(1)
            print("   Scanning... Done!")
            
            if self.manager.camera:
                # Stop recording returns list of {'frame': np.array, 'timestamp': float}
                raw_captures = self.manager.camera.stop_recording()
                print(f"   ✅ Captured {len(raw_captures)} body frames")
                self.session_log["raw_sequences"]["pose_frames"] = [{"timestamp": c['timestamp']} for c in raw_captures]
                self.body_frames_data = raw_captures  # STORE FOR PROCESSING

            # --- Phase 3: Data Aggregation & Logic Viewing ---
            print("\n[3/5] Aggregating Sensor Data (GLASS BOX VIEW)...")
            
            # 1. Radar Aggregation
            radar_items = []
            if self.manager.radar:
                # Drain the queue manually to see the raw data
                while not self.manager.radar.data_queue.empty():
                    radar_items.append(self.manager.radar.data_queue.get_nowait())
                
                # Log raw sequence
                self.session_log["raw_sequences"]["radar"] = radar_items
                
                if radar_items:
                    print(f"   ✅ Collected {len(radar_items)} Radar samples")
                    
                    # REPLICATE LOGIC: Median Calculation
                    hr_values = [i['radar']['heart_rate'] for i in radar_items]
                    rr_values = [i['radar']['respiration_rate'] for i in radar_items]
                    
                    median_hr = int(np.median(hr_values))
                    median_rr = round(float(np.median(rr_values)), 1)
                    
                    self.session_log["aggregation_logic"]["radar"] = {
                        "heart_rate_sequence": hr_values,
                        "heart_rate_median": median_hr,
                        "respiration_rate_sequence": rr_values,
                        "respiration_rate_median": median_rr
                    }
                    
                    print(f"   🔍 Radar Logic:")
                    print(f"      - Input HR: {hr_values}")
                    print(f"      - Median HR: {median_hr}")
                else:
                    print("   ⚠️ No Radar data collected")

            # 2. Thermal Aggregation
            thermal_items = []
            if self.manager.thermal:
                while not self.manager.thermal.data_queue.empty():
                    thermal_items.append(self.manager.thermal.data_queue.get_nowait())
                    
                # Store full raw objects
                # Note: numpy arrays in there might need json conversion later
                self.session_log["raw_sequences"]["thermal"] = thermal_items
                
                if thermal_items:
                    print(f"   ✅ Collected {len(thermal_items)} Thermal samples")
                    
                    # Check format
                    is_firmware = 'core_regions' in thermal_items[0].get('thermal', {})
                    
                    if is_firmware:
                        # REPLICATE LOGIC: Firmware Format Median
                        def get_val(item, category, field_name):
                             return item.get('thermal', {}).get(category, {}).get(field_name)

                        canthus_vals = [get_val(i, 'core_regions', 'canthus_mean') for i in thermal_items]
                        # Filter valid (>25.0)
                        canthus_valid = [v for v in canthus_vals if v is not None and v > 25.0]
                        
                        median_canthus = round(float(np.median(canthus_valid)), 2) if canthus_valid else 0.0
                        
                        # Added: Face Max Calculation
                        face_max_vals = [get_val(i, 'core_regions', 'face_max') for i in thermal_items]
                        valid_face_max = [v for v in face_max_vals if v is not None and v > 25.0]
                        median_face_max = round(float(np.median(valid_face_max)), 2) if valid_face_max else 0.0

                        self.session_log["aggregation_logic"]["thermal"] = {
                            "type": "firmware",
                            "canthus_sequence": canthus_vals,
                            "valid_canthus": canthus_valid,
                            "median_canthus": median_canthus,
                            "face_max_sequence": face_max_vals,
                            "median_face_max": median_face_max
                        }
                        
                        print(f"   🔍 Thermal Logic (Firmware Mode):")
                        print(f"      - Input Canthus: {canthus_vals}")
                        print(f"      - Valid Canthus (>25C): {canthus_valid}")
                        print(f"      - Median Canthus: {median_canthus}")
                        print(f"      - Median Face Max: {median_face_max} (Used for Fever Check)")
                        
                    else:
                        # REPLICATE LOGIC: Legacy Format Median
                        neck_vals = [i.get('thermal', {}).get('fever', {}).get('neck_temp') for i in thermal_items]
                        valid_neck = [v for v in neck_vals if v is not None]
                        
                        median_neck = float(np.median(valid_neck)) if valid_neck else 0.0
                        
                        self.session_log["aggregation_logic"]["thermal"] = {
                            "type": "legacy",
                            "neck_sequence": neck_vals,
                            "median_neck": median_neck
                        }
                        
                        print(f"   🔍 Thermal Logic (Legacy Mode):")
                        print(f"      - Input Neck Temps: {neck_vals}")
                        print(f"      - Median Neck: {median_neck}")
                else:
                    print("   ⚠️ No Thermal data collected")

            # --- Phase 4: Full Processing (Verification) ---
            print("\n[4/5] Running Full Processing Pipeline (Extractors + Risk Engine)...")
            
            # We need to reconstruct the inputs for the ScreeningService
            # Logic taken from HardwareManager._run_scan
            
            # 1. Aggregate objects (we already have the raw items)
            # We need to recreate the aggregation result structure expected by _build_screening_request
            
            radar_aggregated = None
            if radar_items:
                # Re-use the aggregation logic from manager but using our captured items
                # We can't just call manager._aggregate_radar because it reads from queue (which is empty)
                # So we manually construct the dict
                hr_values = [i['radar']['heart_rate'] for i in radar_items]
                rr_values = [i['radar']['respiration_rate'] for i in radar_items]
                median_hr = int(np.median(hr_values))
                median_rr = round(float(np.median(rr_values)), 1)
                
                radar_aggregated = radar_items[-1].copy()
                radar_aggregated['radar']['heart_rate'] = median_hr
                radar_aggregated['radar']['respiration_rate'] = median_rr
            
            esp32_aggregated = None
            if thermal_items:
                # Re-use aggregation logic
                is_firmware = 'core_regions' in thermal_items[0].get('thermal', {})
                
                if is_firmware:
                    def get_val(item, category, field_name):
                         return item.get('thermal', {}).get(category, {}).get(field_name)

                    canthus_vals = [get_val(i, 'core_regions', 'canthus_mean') for i in thermal_items if get_val(i, 'core_regions', 'canthus_mean') > 25.0]
                    neck_vals = [get_val(j, 'core_regions', 'neck_mean') for j in thermal_items if get_val(j, 'core_regions', 'neck_mean') > 25.0]
                    face_max_vals = [get_val(k, 'core_regions', 'face_max') for k in thermal_items if get_val(k, 'core_regions', 'face_max') > 25.0]
                    
                    avg_canthus = round(float(np.median(canthus_vals)), 2) if canthus_vals else 0.0
                    avg_neck = round(float(np.median(neck_vals)), 2) if neck_vals else 0.0
                    avg_face_max = round(float(np.median(face_max_vals)), 2) if face_max_vals else 0.0
                    
                    # Store in structure
                    esp32_aggregated = thermal_items[-1].copy()
                    # We need to update the deeply nested values. 
                    # Simpler to just overwrite the relevant parts if we are sure of structure
                    if 'core_regions' in esp32_aggregated['thermal']:
                        esp32_aggregated['thermal']['core_regions']['canthus_mean'] = avg_canthus
                        esp32_aggregated['thermal']['core_regions']['neck_mean'] = avg_neck
                        esp32_aggregated['thermal']['core_regions']['face_max'] = avg_face_max
                else:
                    # Legacy
                    neck_vals = [i.get('thermal', {}).get('fever', {}).get('neck_temp') for i in thermal_items]
                    valid_neck = [v for v in neck_vals if v is not None]
                    median_neck = float(np.median(valid_neck)) if valid_neck else 0.0
                    
                    esp32_aggregated = thermal_items[-1].copy()
                    if 'fever' in esp32_aggregated.get('thermal', {}):
                        esp32_aggregated['thermal']['fever']['neck_temp'] = median_neck

            # 2. Extract Pose/Landmarks (Heavy Processing)
            print("   ⚙️ Processing Images (Body/Face)...")
            
            # We need actual frames to run extractors.
            # Only if we kept them. 
            # In Phase 1/2 we didn't store raw frames in python memory for long to save RAM, 
            # but manager.camera.stop_recording() returned them.
            # In this script, we need to capture and HOLD them for processing if we want real extraction.
            # The current implementation of run_scan above discarded the actual frames after logging meta.
            # I will modify Phase 1/2 to keep them.
            
            # (Self-correction: I need to edit Phase 1/2 in this same `multi_replace` or assume I edit them below)
            # I will assume I edit Phase 1/2 logic to store `self.face_frames_data` and `self.body_frames_data`
            
            processed_face_frames = []
            processed_face_landmarks = []
            processed_pose_sequence = []
            
            if hasattr(self, 'face_frames_data') and self.face_frames_data:
                print(f"   🔍 Extracting Face Features from {len(self.face_frames_data)} frames...")
                for item in self.face_frames_data:
                    frame = item['frame']
                    # Use manager's camera instance methods
                    if self.manager.camera:
                        roi, landmarks = self.manager.camera.extract_face_features(frame)
                        if roi is not None:
                            processed_face_frames.append(roi)
                        if landmarks is not None:
                            processed_face_landmarks.append(landmarks)
                            
            if hasattr(self, 'body_frames_data') and self.body_frames_data:
                print(f"   🔍 Extracting Body Pose from {len(self.body_frames_data)} frames...")
                for item in self.body_frames_data:
                     frame = item['frame']
                     if self.manager.camera:
                         pose = self.manager.camera.extract_pose_from_frame(frame)
                         if pose is not None:
                             processed_pose_sequence.append(pose)

            # 3. Build Request
            print("   ⚙️ Building Screening Request (Running 8 Extractors)...")
            # We call the manager's private method (glass box access)
            request_payload = self.manager._build_screening_request(
                patient_id=self.patient_id,
                radar_data=radar_aggregated,
                esp32_data=esp32_aggregated,
                face_frames=processed_face_frames if processed_face_frames else None,
                pose_sequence=processed_pose_sequence if processed_pose_sequence else None,
                face_landmarks_sequence=processed_face_landmarks if processed_face_landmarks else None,
                fps=30.0
            )
            
            # 4. Call Service
            print("   🧠 invoking ScreeningService (Risk Engine + LLM)...")
            from app.services.screening import ScreeningService
            
            # Instantiate service (headless)
            service = ScreeningService()
            
            # Run processing
            start_proc = time.time()
            result = await service.process_screening(
                patient_id=self.patient_id,
                systems_input=request_payload["systems"],
                include_validation=True
            )
            proc_time = time.time() - start_proc
            
            print(f"   ✅ Processing Complete in {proc_time:.1f}s")
            print(f"   📊 Result: {result['overall_risk_level']} (Score: {result['overall_risk_score']})")
            
            # Sanitize request_payload (remove raw image data)
            sanitized_payload = request_payload.copy()
            if "face_frames" in sanitized_payload:
                sanitized_payload["face_frames"] = f"<Removed {len(sanitized_payload['face_frames'])} frames>"
            if "systems" in sanitized_payload:
                # Keep systems output (biomarkers), it's the important part
                pass
                
            self.session_log["extracted_features"] = sanitized_payload
            self.session_log["result"] = result

            # --- Phase 5: Save Logs ---
            print("\n[5/5] Saving Detailed Logs...")
            log_path = os.path.join(OUTPUT_DIR, f"headless_log_{self.patient_id}.json")
            
            # Use custom encoder for numpy types
            from app.core.hardware.manager import NumpyEncoder
            try:
                print(f"   💾 Attempting to save to {log_path}...")
                with open(log_path, 'w') as f:
                    json.dump(self.session_log, f, indent=2, cls=NumpyEncoder)
                print(f"   ✅ Log saved to: {log_path}")
            except Exception as e:
                print(f"   ❌ FAILED TO SAVE LOG: {e}")
                logger.error(f"Save failed: {e}", exc_info=True)


        except Exception as e:
            logger.error(f"Scan failed: {e}", exc_info=True)
            
        finally:
            await self.manager.shutdown()

async def main():
    runner = HeadlessRunner()
    await runner.initialize()
    await runner.run_scan()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

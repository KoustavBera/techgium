import os
import sys
import time
import numpy as np
import logging
import queue
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.hardware.manager import HardwareManager
from app.utils import get_logger

# Configure logging to be very verbose for this script
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = get_logger("diagnostic")

def run_diagnostic():
    print("=" * 60)
    print("      HARDWARE SENSOR DIAGNOSTIC TOOL (RAW DATA MODE)")
    print("=" * 60)
    
    manager = HardwareManager()
    
    # Wait for drivers to stabilize
    print("\n[1/3] Initializing Hardware...")
    time.sleep(2)
    
    if not manager.camera or not manager.thermal or not manager.radar:
        print("❌ CRITICAL: Not all sensors are connected.")
        if not manager.thermal: print(" - Thermal Camera (COM6) MISSING")
        if not manager.radar: print(" - Seeed Radar (COM7) MISSING")
        return

    print("✅ All sensors connected.")
    
    patient_id = "DIAG_TEST_001"
    print(f"\n[2/3] Starting 10-second Raw Data Capture for {patient_id}...")
    
    # We will "spy" on the queues manually instead of just calling _run_scan
    # to see the RAW sequences before they are aggregated.
    
    thermal_raw_sequence = []
    radar_raw_sequence = []
    
    # Trigger a real scan in the manager to get the recording architecture going
    manager.initiate_screening(patient_id)
    
    start_time = time.time()
    duration = 15 # Capture for 15s to cover the whole 10s scan
    
    print("\n--- CAPTURING RAW SEQUENCES ---")
    while time.time() - start_time < duration:
        # Peak into Thermal Queue
        while not manager.thermal.data_queue.empty():
            try:
                item = manager.thermal.data_queue.get_nowait()
                t_val = item.get('thermal', {}).get('core_regions', {}).get('face_max')
                if t_val:
                    thermal_raw_sequence.append(t_val)
                    print(f"🌡️ Thermal Raw: {t_val:.2f}°C", end='\r')
            except: break
            
        # Peak into Radar Queue
        while not manager.radar.data_queue.empty():
            try:
                item = manager.radar.data_queue.get_nowait()
                hr = item.get('radar', {}).get('heart_rate')
                if hr:
                    radar_raw_sequence.append(hr)
                    print(f"💓 Radar HR: {hr} bpm", end='\r')
            except: break
            
        time.sleep(0.1)
    
    print("\n\n[3/3] Diagnostic Analysis of Raw Data")
    print("-" * 40)
    
    if thermal_raw_sequence:
        print(f"📊 THERMAL ANALYSIS ({len(thermal_raw_sequence)} samples):")
        print(f"   - Min:  {min(thermal_raw_sequence):.2f}°C")
        print(f"   - Max:  {max(thermal_raw_sequence):.2f}°C")
        print(f"   - Mean: {np.mean(thermal_raw_sequence):.2f}°C")
        print(f"   - Median: {np.median(thermal_raw_sequence):.2f}°C (Current Code uses this)")
        print(f"   - 95th Percentile: {np.percentile(thermal_raw_sequence, 95):.2f}°C (Proposed Optimization)")
        
        # Check for noise spikes
        std_dev = np.std(thermal_raw_sequence)
        print(f"   - Stability (StdDev): {std_dev:.3f} (Lower = Better)")
        if std_dev > 0.5:
            print("   ⚠️ WARNING: High thermal noise detected. Check for head movement or interference.")
    else:
        print("❌ NO THERMAL DATA CAPTURED. Check COM6 connection.")

    print("-" * 40)
    
    if radar_raw_sequence:
        print(f"📊 RADAR HRM ANALYSIS ({len(radar_raw_sequence)} samples):")
        print(f"   - Samples: {radar_raw_sequence}")
        print(f"   - Range: {min(radar_raw_sequence)} - {max(radar_raw_sequence)} bpm")
        print(f"   - Median: {np.median(radar_raw_sequence):.1f} bpm")
    else:
        print("❌ NO RADAR DATA CAPTURED. Check COM7 connection.")

    print("\n" + "=" * 60)
    print("      DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostic()

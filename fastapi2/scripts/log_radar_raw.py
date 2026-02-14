
import serial
import time
import re
import os

# Configuration: Default to COM7 for Radar (since COM6 is Thermal)
PORT = os.getenv("RADAR_PORT", "COM7")
BAUD = 115200

def main():
    print(f"Attempting connection to {PORT} at {BAUD}...")
    try:
        with serial.Serial(PORT, BAUD, timeout=1) as ser:
            print(f"Connected to {PORT}. Logging data for 20 seconds...")
            print("-" * 60)
            
            start_time = time.time()
            while time.time() - start_time < 20:
                if ser.in_waiting:
                    raw_line = ser.readline()
                    try:
                        decoded = raw_line.decode('utf-8', errors='ignore').strip()
                        if not decoded:
                            continue
                            
                        print(f"\nRAW BYTES: {raw_line}")
                        print(f"DECODED:   {decoded}")
                        
                        # Test current driver regex
                        hr_regex = r"heart rate'.*?sending state\s*([\d.]+)"
                        rr_regex = r"respiratory rate'.*?sending state\s*([\d.]+)"
                        
                        hr_match = re.search(hr_regex, decoded.lower())
                        rr_match = re.search(rr_regex, decoded.lower())
                        
                        if hr_match:
                            print(f"✅ HR MATCH: {hr_match.group(1)} bpm")
                        
                        if rr_match:
                            print(f"✅ RR MATCH: {rr_match.group(1)}")

                    except Exception as e:
                        print(f"Error processing line: {e}")
                else:
                    time.sleep(0.01)
                    
            print("-" * 60)
            print("Logging complete.")
            
    except serial.SerialException as e:
        print(f"CRITICAL: Could not connect to {PORT}. Error: {e}")
        print("Check if Arduino IDE or another app is open.")

if __name__ == "__main__":
    main()

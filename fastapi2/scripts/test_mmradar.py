import serial
import json
import time
import sys

# CONFIGURATION
PORT = "COM7"
BAUD = 115200
TIMEOUT = 1

def run_test():
    print(f"--- mmWave Radar Test on {PORT} ---")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        print(f"Successfully connected to {PORT}")
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {PORT}. {e}")
        sys.exit(1)

    print("Listening for data... (Press Ctrl+C to stop)\n")
    
    # Store latest values
    latest = {
        'heart_rate': None,
        'respiratory_rate': None,
        'illuminance': None,
        'last_update': time.time()
    }
    
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            # Parse ESPHome debug logs
            if "Real-time heart rate" in line and "Sending state" in line:
                try:
                    hr = float(line.split("Sending state")[1].split()[0])
                    latest['heart_rate'] = hr
                except:
                    pass
            
            elif "Real-time respiratory rate" in line and "Sending state" in line:
                try:
                    rr = float(line.split("Sending state")[1].split()[0])
                    # Apply -10% calibration correction for sensor error
                    rr_corrected = rr * 0.9
                    latest['respiratory_rate'] = rr_corrected
                except:
                    pass
            
            elif "Illuminance" in line and "Got illuminance" in line:
                try:
                    lux = float(line.split("illuminance=")[1].split("lx")[0])
                    latest['illuminance'] = lux
                except:
                    pass
            
            # Print summary every 2 seconds
            if time.time() - latest['last_update'] >= 2:
                print(f"\r[{time.strftime('%H:%M:%S')}] HR: {latest['heart_rate'] or 'N/A':>3} bpm | RR: {latest['respiratory_rate'] or 'N/A':>2} bpm | Light: {latest['illuminance'] or 'N/A':>5.1f} lx", end='', flush=True)
                latest['last_update'] = time.time()

    except KeyboardInterrupt:
        print("\n\nStopping test...")
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    run_test()

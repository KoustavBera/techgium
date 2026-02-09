import serial, re

ser = serial.Serial("COM7", 115200)

while True:
    line = ser.readline().decode(errors="ignore").strip()

    hr = re.search(r'heart rate.*?([\d.]+)\s*bpm', line.lower())
    rr = re.search(r'respiratory rate.*?([\d.]+)', line.lower())

    if hr:
        print(f"Heart Rate: {hr.group(1)} bpm")

    if rr:
        print(f"Respiration Rate: {rr.group(1)} bpm")
        
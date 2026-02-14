import re

line = "[D][sensor:092]: 'Real-time heart rate': Sending state 68.00000 with 0 decimals of accuracy"

# Current drivers.py regex
regex = r'heart rate.*?([\d.]+)\s*bpm'
match = re.search(regex, line.lower())
print(f"Driver Regex Match: {match.group(1) if match else 'FAIL'}")

# test_mmradar.py logic
if "Real-time heart rate" in line and "Sending state" in line:
    try:
        val = float(line.split("Sending state")[1].split()[0])
        print(f"Test Script Match: {val}")
    except:
        print("Test Script Parse Error")
else:
    print("Test Script Logic Skip")

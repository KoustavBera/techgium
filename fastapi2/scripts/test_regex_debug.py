import re

def test_regex():
    # Example ESPHome debug log lines
    lines = [
        "[11:17:05][D][sensor:094]: 'Real-time heart rate': Sending state 87.000000",
        "[11:17:06][D][sensor:094]: 'Real-time respiratory rate': Sending state 12.000000",
        "[11:01:36][D][sensor:117]: 'Real-time heart rate': Sending state 96.000000",
    ]
    
    # The regex currently in drivers.py
    regex_hr = r'heart rate.*?([\d.]+)'
    regex_rr = r'respiratory rate.*?([\d.]+)'
    
    print("Testing Regex Performance:")
    for line in lines:
        print(f"\nLine: {line}")
        hr_match = re.search(regex_hr, line.lower())
        rr_match = re.search(regex_rr, line.lower())
        
        if hr_match:
            print(f"  HR Match found: '{hr_match.group(0)}' -> {hr_match.group(1)}")
        if rr_match:
            print(f"  RR Match found: '{rr_match.group(0)}' -> {rr_match.group(1)}")

if __name__ == "__main__":
    test_regex()

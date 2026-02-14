from datetime import datetime

def log_extracted_biomarker(system, name, value, unit):
    """
    Log biomarker extraction value to server console in a sequential, structured format.
    This utility is intended for debugging to ensure report values match extraction.
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"🔍 [{timestamp}] [DEBUG] {system.upper():<15} | {name:<25}: {value:>8.3f} {unit}", flush=True)

# Radar Data Investigation (Constant Heart Rate)

## Observation
The user reported that the Radar Heart Rate (HR) appears constant (e.g., 62 bpm) in the headless logs, despite the sensor ostensibly sending variable values (e.g., 79, 76, 73 bpm) as seen in Arduino IDE. The Respiratory Rate (RR) was also missing from the aggregation log snippet.

## Hypotheses
1.  **Regex Mismatch**: The regex `r"heart rate'.*?sending state\s*([\d.]+)"` might fail to match the specific format of HR lines, possibly due to ANSI color codes, quotes, or whitespace variations.
2.  **Stale Data Persistence**: The `RadarReader` updates a shared `last_data` structure. If only RR updates are received (or matched), the `last_data` timestamp updates, and the old HR value is pushed to the queue repeatedly. This floods the aggregation window with a single stale HR value, leading to a constant median.
3.  **Port Mismatch**: `COM6` was identified as the Thermal Camera (ESP32). `COM7` is likely the Radar. If the wrong port was used, no data (or wrong data) would be read.

## Investigation Steps
1.  **Raw Logging**: Created `scripts/log_radar_raw.py` to capture raw serial bytes from `COM7`.
2.  **Regex Verification**: The script tests the driver's regex against the live data.
    - Result: RR lines match correctly (`✅ RR MATCH`).
    - Status of HR matches is verified in `radar_raw_log.txt`.

## Solution Plan
1.  **Refine Regex**: Ensure the regex accounts for all variations (color codes, quotes).
2.  **Data Expiry**: Modify `RadarReader` to clear or mark HR as "stale" if it hasn't been updated recently, rather than persisting the last known value indefinitely.
3.  **Separate Streams**: Alternatively, track HR and RR timestamps independently to avoid queueing stale mixed data.

## Missing RR in Logs
The "missing RR" in `[3/5] Aggregating Sensor Data` logs is likely because `HardwareManager._aggregate_radar` logging was focused only on HR. The data *is* being collected (as seen in the internal `raw_sequences` of the JSON log), but the debug printout just didn't show it.

## Final Resolution
1.  **Reference Bug**: The constant HR was caused by `self.last_data.copy()` being a **shallow copy**. All 84 queued items shared the same `['radar']` dictionary reference. When aggregation ran, it read the current value (e.g., 70) 84 times.
    - **Fix**: Implemented explicit deep copy of the inner dictionary in `drivers.py`.
2.  **JSON Crash**: The runner crashed on save because `SystemRiskResult` wasn't serializable.
    - **Fix**: Updated `NumpyEncoder` in `manager.py` to handle generic objects.
3.  **Visualization**: Added RR statistics to `HardwareManager` logs to confirm data presence.

# Chiranjeevi Health Screening: End-to-End Architecture Flow

This document outlines the complete journey of data, from the moment a sensor detects a signal to the final generation of a clinical-grade health report.

## The "Start Analysis" Execution Flow

When a user clicks **"Start Screening Flow"** in the frontend, a choreographed sequence of events occurs across the entire stack:

### 1. The Trigger (Frontend)
- **File**: `frontend/index.html` → `startScreening()`
- **Action**: The frontend sends a `POST /api/v1/hardware/start-screening` request to the backend with the Patient ID and sensor ports.
- **Visual**: The UI switches to the "Status" card and prepares the progress steps.

### 2. The Orchestrator (Backend API)
- **File**: `app/main.py` → `start_hardware_screening()`
- **Action**: The API calls `HardwareManager.start_scan()`. 
- **Threading**: Since health analysis takes ~25 seconds, the backend launches a **Background Thread** to run the scan so the API can return a "Started" message immediately without making the user wait.

### 3. The Hardware Pipeline (Manager)
- **File**: `app/core/hardware/manager.py` → `_run_scan_loop()`
- **Timeline**:
    *   **Phase: INITIALIZING**: Wakes up the Radar and Thermal sensors.
    *   **Phase: FACE_ANALYSIS (10s)**: Captures video frames for heart rate (rPPG) and skin analysis.
    *   **Phase: BODY_ANALYSIS (10s)**: Analyzes posture, skeletal symmetry, and gait.
    *   **Phase: PROCESSING**: 
        - Aggregates the 10 seconds of raw sensor data using **Median Filtering**.
        - Passes data to the 8 system extractors.
        - Calls the **Multi-LLM Pipeline** for medical interpretation.
        - Generates the final **PDF Report**.

### 4. Real-time Feedback (Polling)
- **Frontend Loop**: `pollScanStatus()` runs every 500ms, calling `GET /api/v1/hardware/scan-status`.
- **Backend Response**: The `HardwareManager` provides the current `% progress` and the current `phase` name.
- **Logic Restoration**: If the user is too close or far during the scan, the backend injects `user_warnings` into this status, which the frontend displays immediately as yellow alerts.

### 5. Completion
- **Action**: Once the backend thread finishes, it sets the state to `complete` and provides the `patient_report_id`.
- **Frontend**: The poll detects the `complete` status, stops the timer, and enables the **"Download Patient Report"** button.

---

**Associated Directory**: `app/core/hardware/`
**Primary File**: `manager.py` (`HardwareManager`)

*   **Continuous Capture**: When a scan begins, the `HardwareManager` (a singleton) starts capturing data from the Radar (heart/respiration) and Thermal ESP32 (temperatures).
*   **Data Aggregation**: Sensors provide continuous streams. Every 10 seconds, `HardwareManager._aggregate_radar()` and `_aggregate_thermal()` take the collected queue of data and perform **Median Filtering** (recently upgraded from Mean) to remove noise and outliers.
*   **Orchestration**: `HardwareManager.run_scan_pipeline()` manages the sequence:
    1.  **Face Capture**: (Camera/Drivers)
    2.  **Body/Vital Capture**: (Radar/Thermal)
    3.  **Data Fusion**: Gathering all points for analysis.

---

## 2. Biomarker Extraction
**Associated Directory**: `app/core/extraction/`
**Primary File**: `base.py` (Abstract Classes), `cardiovascular.py`, `pulmonary.py`, etc.

*   **The Extractors**: There are 8 specialized extraction modules (Pulmonary, Cardiovascular, Skin, CNS, Skeletal, Renal, Eyes, Nasal).
*   **The Process**: `HardwareManager._build_screening_request()` passes the raw aggregated data to each extractor's `.extract()` method.
*   **The Output**: Each extractor returns a `BiomarkerSet`. This contains raw numerical values (e.g., Heart Rate: 72 bpm) and their associated confidence scores.

---

## 3. Screening Service & Risk Engine
**Associated Directories**: `app/services/`, `app/core/inference/`
**Primary Files**: `screening.py`, `risk_engine.py`

*   **Service Orchestration**: `ScreeningService.process_screening()` receives the `BiomarkerSet` for all systems.
*   **Risk Calculation**: It calls `RiskEngine`.
    *   **Rule-Based Scoring**: `RiskEngine` compares biomarkers against clinical thresholds (normal ranges).
    *   **Weighting**: Different biomarkers are weighted based on their clinical importance (e.g., HRV is weighted higher for Cardiovascular health than thermal asymmetry).
    *   **Composite Risk**: `CompositeRiskCalculator` aggregates all 8 systems into one overall "Health Risk Score" (0-100).
*   **Validation Gating**: The `TrustEnvelope` (in `app/core/validation/`) checks signal quality and plausibility. If data is "physiologically impossible," the system can reject it or apply a confidence penalty.

---

## 4. Multi-LLM Interpretation (Sequential Quality Pipeline)
**Associated Directory**: `app/core/llm/`
**Primary File**: `multi_llm_interpreter.py`

When risk assessment is done, the system performs a high-accuracy, 3-phase LLM analysis:
1.  **Phase 1 (Gemini 1.5 Flash)**: Generates the primary medical summary, explanations, and actionable recommendations in JSON format.
2.  **Phase 2 (HF Medical Model 1 - "Intelligent Internet")**: Acts as a "Second Opinion." It reviews the Gemini output for clinical tone and correctness.
3.  **Phase 3 (HF Medical Model 2 - "GPT-OSS")**: The Arbiter. It looks at the primary report and the validator's feedback to make a final "Pass/Fail" decision. It can even correct minor summary details.

---

## 5. Report Generation
**Associated Directory**: `app/core/reports/`
**Primary File**: `patient_report.py`

*   **PDF Generation**: `EnhancedPatientReportGenerator` takes the risk scores, the LLM-generated interpretations, and the trust metadata.
*   **Styling**: It uses the `reportlab` library to build a "Pixel UI 2025" style PDF.
*   **Visuals**: It generates color-coded risk indicators (Pills) and a "Health Stats Chart" (Donut chart) showing the breakdown of system health.
*   **Output**: Saves the report to the `reports/` folder and returns the path to the frontend.

---

## Folder Summary Table

| Layer | Responsibility | Key Folder/File |
| :--- | :--- | :--- |
| **Frontend** | UI & User Interaction | `frontend/index.html` |
| **Bridge** | Hardware Control | `app/core/hardware/manager.py` |
| **Extraction** | Math & DSP on raw data | `app/core/extraction/*.py` |
| **Inference** | Clinical Scoring | `app/core/inference/risk_engine.py` |
| **Validation** | Trust & Quality Checks | `app/core/validation/trust_envelope.py` |
| **Brain** | LLM Interpretation | `app/core/llm/multi_llm_interpreter.py` |
| **Reports** | PDF Layouts | `app/core/reports/patient_report.py` |

# Health Screening Pipeline - Documentation

## Project Overview

A modular, multimodal health screening system that processes sensor data to assess health risks across 9 physiological systems.

---

## Module 1: Data Ingestion ✅

### Goals

Convert raw sensor/multimodal streams into synchronized time-series chunks ready for feature extraction.

### Sub-modules

| Component | File           | Purpose                                          | Status |
| --------- | -------------- | ------------------------------------------------ | ------ |
| Sync      | `sync.py`      | Time synchronization layer, DataPacket structure | ✅     |
| Camera    | `camera.py`    | OpenCV video frame extraction (+ simulation)     | ✅     |
| Motion    | `motion.py`    | MediaPipe pose estimation (+ simulation)         | ✅     |
| RIS       | `ris.py`       | Simulated radio impedance streams                | ✅     |
| Auxiliary | `auxiliary.py` | Heartbeat, thermal, CSV datasets                 | ✅     |

### Data Packet Structure

```python
@dataclass
class DataPacket:
    timestamp: float          # Unix timestamp (ms)
    modality: ModalityType    # 'camera', 'motion', 'ris', 'auxiliary'
    data: np.ndarray          # Raw data payload
    metadata: Dict[str, Any]  # Modality-specific metadata
    session_id: str           # Unique session identifier
    sequence_id: int          # Sequential packet number
```

### Verification Results ✅

- **31 unit tests passed** in 3.68s
- All modalities synchronized correctly
- Simulation fallbacks working when hardware unavailable

---

## Execution Log

### 2026-02-01 19:52

- Started Module 1: Data Ingestion implementation
- Created project structure and core utilities

### 2026-02-01 20:XX

- Completed all ingestion components
- Fixed deprecation warning in logging
- All 31 tests passed

### 2026-02-05 08:XX

- Integrated Hardware Extraction (Split-USB support)
- Created `PulmonaryExtractor` for Radar (respiration, depth)
- Updated `CardiovascularExtractor` to fuse Radar HR + rPPG
- Updated `SkinExtractor` to support MLX90640 Thermal data
- Verified with new unit tests ensuring hardware priority

---

## Module 2: Feature Extraction ✅

### Goals

Extract clinically relevant biomarkers from multimodal sensor data for 9 physiological systems.

### Components

| System         | File                  | Key Biomarkers                                         |
| -------------- | --------------------- | ------------------------------------------------------ |
| CNS            | `cns.py`              | gait_variability, posture_entropy, tremor              |
| Cardiovascular | `cardiovascular.py`   | heart_rate (radar+rPPG), hrv_rmssd, chest_micro_motion |
| Pulmonary      | `pulmonary.py`        | respiration_rate (radar), breathing_depth              |
| Renal          | `renal.py`            | fluid_asymmetry, total_body_water, ecf_ratio           |
| GI             | `gastrointestinal.py` | abdominal_rhythm, visceral_variance                    |
| Skeletal       | `skeletal.py`         | gait_symmetry, stance_stability, joint_rom             |
| Skin           | `skin.py`             | skin_temperature (thermal), texture_roughness, lesions |
| Eyes           | `eyes.py`             | blink_rate, gaze_stability, fixation                   |
| Nasal          | `nasal.py`            | breathing_regularity, respiratory_rate                 |
| Reproductive   | `reproductive.py`     | autonomic_imbalance, stress_proxy                      |

### Verification Results ✅

- **18 unit tests passed** in 3.24s
- All extractors support simulation fallback
- Biomarkers include confidence scores and normal ranges

---

## Module 3: Baseline AI Inference ✅

### Goals

Map biomarker feature vectors to preliminary risk scores with explanations.

### Components

| Component            | File             | Purpose                               |
| -------------------- | ---------------- | ------------------------------------- |
| RiskEngine           | `risk_engine.py` | Weighted biomarker-to-risk scoring    |
| ConfidenceCalibrator | `calibration.py` | Confidence adjustment and uncertainty |
| ExplanationGenerator | `explanation.py` | Human-readable risk explanations      |

### Key Classes

- `RiskScore` - Score (0-100), level, confidence, explanation
- `RiskLevel` - LOW/MODERATE/HIGH/CRITICAL enum
- `SystemRiskResult` - Per-system risk with sub-risks and alerts
- `CompositeRiskCalculator` - Weighted aggregate across systems

### Verification Results ✅

- **22 unit tests passed** in 3.63s
- All 9 systems have weighted scoring profiles
- Explanation generator supports multiple output formats

---

## Phase 4: Signal & Physiological Validity Layer ✅

### Goals

Validate that signals and biomarkers are physically and physiologically valid before interpretation.

### Components

| Component                      | File                          | Purpose                          |
| ------------------------------ | ----------------------------- | -------------------------------- |
| SignalQualityAssessor          | `signal_quality.py`           | Modality quality metrics (NO ML) |
| BiomarkerPlausibilityValidator | `biomarker_plausibility.py`   | Hard physiological limits        |
| CrossSystemConsistencyChecker  | `cross_system_consistency.py` | Inter-system agreement           |
| TrustEnvelopeAggregator        | `trust_envelope.py`           | Gates downstream interpretation  |

### Key Features

- Physics-based validation (NO ML/AI)
- Hard limits for impossible values
- Cross-system physiological consistency rules
- TrustEnvelope gates LLM interpretation

### Verification Results ✅

- **20 unit tests passed** in 4.05s
- All 4 modalities assessed (camera, motion, RIS, vitals)
- 6 cross-system consistency rules implemented

---

## Phase 5: Optional ML/DL (Non-Decisional) ✅

### Goals

ML/DL for signal quality assessment ONLY - outputs affect confidence, never diagnosis.

### Components

| Component                | File                  | Purpose                                         |
| ------------------------ | --------------------- | ----------------------------------------------- |
| SignalAnomalyDetector    | `anomaly_detector.py` | IsolationForest + statistical anomaly detection |
| NoisePhysiologySeparator | `anomaly_detector.py` | Bandpass filtering for noise separation         |

### Key Constraints

- Outputs → confidence_penalty ONLY
- Never affects diagnosis or risk scores
- Graceful fallback if sklearn unavailable

### Verification Results ✅

- **19 unit tests passed** in 6.47s
- Non-decisional constraint verified in tests

---

## Phase 6: LLM Interpretation (Gemini 1.5 Flash) ✅

### Goals

Explain already-computed risk scores using LLM - does NOT diagnose.

### Components

| Component               | File                   | Purpose                                          |
| ----------------------- | ---------------------- | ------------------------------------------------ |
| GeminiClient            | `gemini_client.py`     | API wrapper with rate limiting and mock fallback |
| RiskInterpreter         | `risk_interpreter.py`  | Explains pre-computed risks (non-decisional)     |
| MedicalContextGenerator | `context_generator.py` | Educational health content for each system       |

### Architecture Constraints

- LLM receives: SystemRiskResult, CompositeRiskResult, TrustEnvelope
- LLM DOES NOT see: Raw sensor data, raw biomarkers
- LLM outputs: Explanations, medical context, summaries
- LLM DOES NOT output: Diagnoses, risk scores, treatment decisions

### Verification Results ✅

- **22 unit tests passed** in 2.52s
- Non-decisional constraints verified architecturally and in tests

---

## Phase 7: Agentic Medical Validation ✅

### Goals

Multi-agent validation using medical LLMs via Hugging Face Inference API.
All agents are NON-DECISIONAL - they validate and flag, not diagnose.

### Components

| Component         | File                | Purpose                                     |
| ----------------- | ------------------- | ------------------------------------------- |
| HuggingFaceClient | `hf_client.py`      | HF Inference API wrapper with mock fallback |
| MedGemmaAgent     | `medical_agents.py` | Biomarker plausibility validation           |
| OpenBioLLMAgent   | `medical_agents.py` | Cross-system consistency checking           |
| AgentConsensus    | `medical_agents.py` | Multi-agent agreement aggregation           |

### Supported Models

- `google/medgemma-4b-it` (MedGemma)
- `aaditya/OpenBioLLM-Llama3-8B` (OpenBioLLM)
- `BioMistral/BioMistral-7B` (fallback)
- `epfl-llm/meditron-7b` (fallback)

### Architecture Constraints

- Agents receive: Pre-computed risk results, biomarker summaries
- Agents NEVER see: Raw sensor data
- Agents output: ValidationResult with flags, confidence, explanation
- Agents NEVER output: Diagnoses, treatments, prescriptions

### Verification Results ✅

- **23 unit tests passed** in 2.63s
- Non-decisional constraints verified in tests

---

## Phase 8: Report Generation ✅

### Goals

Generate downloadable PDF health screening reports for patients and healthcare professionals.

### Components

| Component              | File                | Purpose                                     |
| ---------------------- | ------------------- | ------------------------------------------- |
| PatientReportGenerator | `patient_report.py` | Color-coded, simple PDF for patients        |
| DoctorReportGenerator  | `doctor_report.py`  | Detailed clinical PDF with biomarker tables |
| PatientReport          | `patient_report.py` | Patient report data container               |
| DoctorReport           | `doctor_report.py`  | Clinical report data container              |

### Dependencies

- `reportlab` - PDF generation library

### Patient Report Features

- Color-coded risk indicators (green/amber/red)
- Simple, non-technical language
- System-by-system summary table
- Recommendations and caveats
- Disclaimer footer

### Doctor Report Features

- Executive summary with key metrics
- Trust envelope visualization
- Agentic validation results
- System-by-system analysis
- Detailed biomarker tables
- Alerts and flags section
- Clinical disclaimer

### Verification Results ✅

- **12 unit tests** (6 skipped in CI due to reportlab)
- PDF generation verified with reportlab
- Bytes generation for direct download supported

---

## Phase 9: Backend API ✅

### Goals

FastAPI REST API for health screening pipeline with PDF report generation.

### Endpoints

| Method | Endpoint                        | Purpose                |
| ------ | ------------------------------- | ---------------------- |
| GET    | `/`                             | Health check           |
| GET    | `/health`                       | API health status      |
| GET    | `/api/v1/systems`               | List supported systems |
| POST   | `/api/v1/screening`             | Run health screening   |
| GET    | `/api/v1/screening/{id}`        | Get screening details  |
| POST   | `/api/v1/reports/generate`      | Generate PDF report    |
| GET    | `/api/v1/reports/{id}/download` | Download PDF           |

### Features

- Pydantic request/response models
- CORS middleware enabled
- System name aliasing (e.g., "heart" → "cardiovascular")
- In-memory storage (replace with DB in production)

### How to Run

```bash
# Install dependencies
pip install pydantic-settings uvicorn

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access docs
# Swagger: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Verification Results ✅

- 167 unit tests passing
- API implementation complete

---

## Phase 10: End-to-End Demo ✅

### Demo Script

Created `demo.py` - comprehensive end-to-end test of the entire pipeline.

### Mock Data Generated

- **PPG Signal**: 300 samples (camera-based heart rate)
- **Audio Signal**: 160,000 samples (breathing sounds)
- **Accelerometer**: 500 samples x 3 axes (gait analysis)

### Pipeline Components Tested

1. ✅ **Module Loading** - All components imported successfully
2. ✅ **Mock Sensor Data** - PPG, audio, accelerometer signals generated
3. ✅ **Feature Extraction** - Biomarkers extracted for 3 systems (Cardiovascular, CNS, Pulmonary)
4. ✅ **Risk Inference** - System risks and composite risk calculated
5. ✅ **Trust Envelope** - Confidence and trust metrics computed
6. ✅ **Gemini LLM Interpretation** - Medical context generated via Gemini API
7. ✅ **Agent Validation** - Multi-agent consensus using HuggingFace models
8. ✅ **PDF Reports** - Both patient and doctor reports generated

### Verification Results ✅

- **Total execution time**: ~30 seconds
- **Gemini API calls**: Successfully made for medical interpretation
- **HuggingFace agents**: MedGemma & OpenBioLLM validation completed
- **PDFs generated**: 7 total reports in `./reports/` directory
  - Patient reports: Simple, color-coded, AI-enhanced
  - Doctor reports: Detailed with biomarker tables, trust envelope, and validation results

### Key Features Verified

- ✅ Non-decisional AI constraints maintained
- ✅ Medical interpretation from Gemini (educational context)
- ✅ Agent consensus validation (plausibility checks)
- ✅ PDF generation with reportlab
- ✅ Trust envelope calculation
- ✅ Multi-modal sensor integration

---

## ESP32 Thermal Camera Integration (MLX90640) ✅

### Overview

The ESP32 NodeMCU reads thermal imaging data from the MLX90640 sensor and transmits processed biomarker data via USB serial to the bridge.py application.

### Firmware JSON Output Format

```json
{
	"timestamp": 12345,
	"thermal": {
		"fever": { "canthus_temp": 36.4, "neck_temp": 36.8, "fever_risk": 0 },
		"diabetes": { "canthus_temp": 36.4, "risk_flag": 0 },
		"cardiovascular": {
			"thermal_asymmetry": 0.3,
			"left_cheek_temp": 35.2,
			"right_cheek_temp": 34.9
		},
		"inflammation": {
			"hot_pixel_pct": 3.2,
			"face_mean_temp": 35.0,
			"detected": 0
		},
		"autonomic": {
			"stress_gradient": 1.2,
			"forehead_temp": 35.5,
			"nose_temp": 34.3,
			"stress_flag": 0
		},
		"metadata": { "face_detected": 1, "valid_rois": 7 }
	}
}
```

### Biomarker Distribution (Option 1 - Physiologically Appropriate)

| Firmware Data                      | Target System  | Biomarker Name          | Clinical Significance     |
| ---------------------------------- | -------------- | ----------------------- | ------------------------- |
| `fever.neck_temp`                  | Skin           | skin_temperature        | Core body temp proxy      |
| `fever.canthus_temp`               | Skin           | skin_temperature_max    | Fever detection           |
| `inflammation.hot_pixel_pct`       | Skin           | inflammation_index      | Localized inflammation    |
| `cardiovascular.thermal_asymmetry` | Cardiovascular | thermal_asymmetry       | Blood perfusion imbalance |
| `diabetes.canthus_temp`            | Renal          | microcirculation_temp   | Microvascular dysfunction |
| `autonomic.stress_gradient`        | CNS            | thermal_stress_gradient | Sympathetic activation    |

### Data Flow

```
ESP32 + MLX90640
    → Serial USB (COM_B, 115200 baud)
    → bridge.py (ESP32Reader)
    → DataFusion (flattens to thermal_data)
    → Extractors (Skin, CV, CNS, Renal)
    → API /api/v1/screening
```

### Quality Checks

- `metadata.face_detected == 1` required
- `valid_rois >= 5` for reliable data
- Frames with low visibility automatically skipped

### Execution Log

- **2026-02-07**: Integrated ESP32 thermal firmware v2
- Added JSON parsing in ESP32Reader (bridge.py)
- Distributed biomarkers to appropriate physiological systems
- Added backward compatibility for old esp32_data format

---

## Bridge.py Hardware Integration ✅

### Overview

The `bridge.py` script implements the Split-USB architecture, connecting the webcam, Seeed Radar Kit (MR60BHA2), and ESP32 NodeMCU to the laptop via a USB hub.

### Components

| Component     | Purpose                               |
| ------------- | ------------------------------------- |
| CameraCapture | OpenCV webcam frame capture           |
| RadarReader   | Seeed MR60BHA2 binary protocol parser |
| ESP32Reader   | Thermal camera JSON reader            |
| DataFusion    | Aggregates all sensor data            |

### Radar Binary Protocol (MR60BHA2)

```
Header:  0x02 0x81 (2 bytes)
Padding: 2 bytes
Respiration Rate: 4 bytes (little-endian float)
Heart Rate: 4 bytes (little-endian float)
Total: 12 bytes per frame
```

### Execution Log

- **2026-02-09**: Fixed `parse_radar_binary` indentation (was at module level, now in `RadarReader` class)
- **2026-02-09**: Updated `generate_simulated_esp32_data` to match HARDWARE.md thermal biomarker structure
- Verified with syntax check and simulation mode

---

## Patient Report Improvements ✅

### Overview

Enhanced the PDF patient report generator with improved status color coding and cleaner visual presentation.

### Status Color Scheme

| Status       | Background Color          | Description       |
| ------------ | ------------------------- | ----------------- |
| Normal       | `#ECFDF5` (Mint)          | Healthy readings  |
| Above Normal | `#FFECD2` (Pastel Orange) | Elevated values   |
| Below Normal | `#FFECD2` (Pastel Orange) | Low values        |
| Not Assessed | `#F9FAFB` (Light Gray)    | Insufficient data |

### Execution Log

- **2026-02-09**: Fixed status color logic - reordered checks to match "Above/Below" before "Normal" to prevent false green coloring
- **2026-02-09**: Removed emoji icons (✓, ⚠, —) from status text for cleaner PDF rendering
- Verified with unhealthy patient simulation

---

## Deep Dive: Biomarker Extraction Logic

Each extraction module uses a combination of signal processing, computer vision, and physical modeling to derive health markers.

### 🫀 Cardiovascular (`cardiovascular.py`)
- **Heart Rate (rPPG)**: Tracks tiny color changes in facial skin (green channel) caused by the pulse. Uses FFT (Fast Fourier Transform) to find the dominant "beat" frequency.
- **Heart Rate (Radar)**: Uses 60GHz mmWave radar to detect rhythmic chest displacement with sub-millimeter precision.
- **HRV (Heart Rate Variability)**: Measures the millisecond-level variation between beats (RMSSD). High variability often indicates better autonomic health.
- **Thermal Asymmetry**: Compares temperatures between left and right facial regions to detect blood flow imbalances.

### 🧠 Central Nervous System (`cns.py`)
- **Gait Variability**: Uses "Zeni's method" to identify heel strikes from ankle vertical position. Calculates how consistent your stride timing is.
- **Postural Sway Complexity**: Calculates "Sample Entropy" of your body's center of mass motion. Higher complexity usually indicates a healthier, more adaptable balance system.
- **Tremor Analysis**: Uses spectral analysis (Welch PSD) on hand motion to identify specific frequency bands (e.g., 4-6 Hz) typical of different neurological conditions.

### 👁️ Eyes (`eyes.py`)
- **Blink Rate (EAR)**: Calculates the Eye Aspect Ratio (height-to-width) using 6 landmarks per eye. Detects a blink when the ratio drops sharply.
- **Gaze Stability**: Follows ISO 9241-3 standards to measure how steady your eyes remain during fixation using RMS velocity.
- **Symmetry**: Correlates the motion patterns of both eyes to ensure they are moving in unison.

### 🫁 Pulmonary (`pulmonary.py`)
- **Respiration Rate**: Extracted from mmWave radar signals that track the rise and fall of the chest.
- **Breathing Depth**: Analyzes the magnitude (amplitude) of the radar signal's chest displacement.

### 🛡️ Skin (`skin.py`)
- **Texture Roughness**: Uses GLCM (Gray Level Co-occurrence Matrix) to measure contrast in skin pixels, acting as a proxy for surface smoothness or roughness.
- **Skin Color (CIELab)**: Analyzes the 'a*' (green-red) and 'b*' (blue-yellow) color channels to detect redness (inflammation proxy) or yellowness (jaundice proxy).
- **Thermal "Hot Spots"**: Identifies localized inflammation by counting "hot pixels" in thermal camera data.

### 🦴 Skeletal (`skeletal.py`)
- **Bilateral Symmetry**: Compares the 3D range of motion between left and right joints (shoulders, elbows, knees).
- **Stance Stability**: Measures the sway of the hips/center-of-mass to assess physical balance and core stability.

### 🧪 Renal (`renal.py`)
- **Fluid Distribution Index**: Uses Bioimpedance (RIS) to assess how water is distributed across the body, looking for asymmetries.
- **Microcirculation (Diabetes Proxy)**: Monitors the temperature of the inner canthus (tear duct area) as a proxy for peripheral blood flow health.

### 🍔 Gastrointestinal (`gastrointestinal.py`)
- **Abdominal Rhythm**: Projects abdominal motion into the frequency domain to look for the slow, rhythmic patterns (0.05 Hz) typical of gut motility.
- **Visceral Variance**: Measures the "energy" or amount of movement in the abdominal area.

### 👃 Nasal (`nasal.py`)
- **Nostril Flare**: Measures the dilation of nostrils during breathing as a proxy for respiratory effort (work of breathing).
- **Nasal Cycle Balance**: Analyzes the alternating dominance of airflow between left and right nostrils using nostril area ratios.

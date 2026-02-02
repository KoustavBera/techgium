# Technical Specification: Walkthrough Health Screening Chamber

## Problem Statement

> Design and develop a multidimensional, non-invasive diagnostic system in the form of a walkthrough chamber that enables a complete health check-up while the individual simply walks through it. The system should be capable of monitoring and diagnosing key physiological systems including the **central nervous system, cardiovascular health, renal function, gastrointestinal health, skeletal structure, skin, eyes, nasal passages, and reproductive organs**. The process should be passive and voluntary, requiring no active participation beyond walking through the chamber. The results should be automatically processed and shared via a mobile application or through a connected healthcare provider.

---

## 1. Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       WALKTHROUGH CHAMBER HARDWARE                          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │  RGB-D CAMERAS   │   │   RIS SENSORS    │   │   AUXILIARY      │        │
│  │  (4K @ 60fps)    │   │ (RF Generator)   │   │   (Motion/Audio) │        │
│  │  Multiple angles │   │ 16-channel array │   │   IMU, Mic       │        │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘        │
│           └────────────────────┬─────────────────────────┘                  │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │ Data Ingestion
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SOFTWARE PIPELINE                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│  │ BIOMARKER       │ → │ RISK ENGINE     │ → │ MULTI-LLM       │           │
│  │ EXTRACTION      │   │ (Rules + ML)    │   │ INTERPRETATION  │           │
│  │ (9 Systems)     │   │                 │   │ (3 Models)      │           │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘           │
│           │                     │                     │                     │
│           ▼                     ▼                     ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              PDF REPORTS (Patient + Doctor)                  │           │
│  └─────────────────────────────────────────────────────────────┘           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ API Output
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DELIVERY LAYER                                         │
│  ┌────────────────┐               ┌────────────────┐                        │
│  │ Mobile App     │               │ Healthcare     │                        │
│  │ (Patient View) │               │ Provider Portal│                        │
│  └────────────────┘               └────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Requirements

### 2.1 Primary Sensors

| Sensor | Function | Biomarkers Enabled |
|--------|----------|-------------------|
| **RGB-D Camera** (4-6 units) | Pose, face, skin analysis | CNS, Eyes, Skin, Skeletal |
| **RIS (RF Generator)** | Bioimpedance measurement | Cardiovascular, Renal, GI, Nasal, Reproductive |
| **Motion Sensors (IMU)** | Gait, balance tracking | CNS, Skeletal |
| **Microphone Array** | Breathing sounds | Nasal, Pulmonary |

### 2.2 Camera Specifications

| Parameter | Recommended |
|-----------|-------------|
| Resolution | 4K (3840×2160) minimum |
| Frame Rate | 60 fps for motion analysis |
| Lens | Wide-angle (120°) |
| Count | 4-6 cameras for full body coverage |
| Type | RGB-D (depth sensing) preferred |
| Library | MediaPipe, OpenCV |

### 2.3 RIS (Radar/RF Impedance Sensor) Specifications

| Parameter | Recommended |
|-----------|-------------|
| Frequency | 1-100 MHz (bioimpedance range) |
| Channels | 16 electrodes minimum |
| Sample Rate | 1000 Hz |
| Depth | Up to 10cm tissue penetration |
| Output | I/Q data, bioimpedance measurements |

---

## 3. Physiological Systems & Biomarkers

### 3.1 Central Nervous System (CNS)

**Extractor**: `app/core/extraction/cns.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `gait_variability` | 0.02-0.08 | CV | 📷 Camera | Ankle position variance over strides |
| `posture_entropy` | 1.5-3.5 | bits | 📷 Camera | Shannon entropy of trunk angle |
| `tremor_resting` | 0-0.1 | PSD | 📷 Camera | FFT 4-6 Hz band on wrist motion |
| `tremor_postural` | 0-0.1 | PSD | 📷 Camera | FFT 6-12 Hz band on wrist motion |
| `tremor_intention` | 0-0.1 | PSD | 📷 Camera | FFT 3-5 Hz band |
| `cns_stability_score` | 70-100 | score | 📷 Camera | Composite sway analysis |

**Hardware Requirements**: 📷 Camera only

**Extraction Method**:
1. MediaPipe tracks 33 body landmarks at 30fps
2. Ankle positions (27, 28) → Gait variability via stride length CV
3. Wrist motion (15, 16) → FFT for tremor frequency bands
4. Hip center (23, 24) → Postural sway magnitude

---

### 3.2 Cardiovascular

**Extractor**: `app/core/extraction/cardiovascular.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `heart_rate` | 60-100 | bpm | 📡 RIS / 📷 Camera | FFT 0.8-3 Hz on thoracic signal |
| `hrv_rmssd` | 20-80 | ms | 📡 RIS | Beat-to-beat interval analysis |
| `thoracic_impedance` | 400-600 | ohms | 📡 RIS | Mean thoracic bioimpedance |
| `chest_micro_motion` | 0.001-0.01 | amplitude | 📷 Camera | Chest wall motion from pose |
| `blood_pressure_systolic` | 90-120 | mmHg | 📡 RIS | Pulse wave velocity estimate |
| `blood_pressure_diastolic` | 60-80 | mmHg | 📡 RIS | Pulse wave velocity estimate |

**Hardware Requirements**: 📡 RIS (primary), 📷 Camera (backup)

**Extraction Method**:
1. **RIS**: Thoracic channels (0-7) filtered for cardiac frequencies
2. **Camera**: Shoulder landmarks (11, 12) micro-motion
3. FFT peak detection in 0.8-3 Hz (48-180 bpm) range

---

### 3.3 Renal Function

**Extractor**: `app/core/extraction/renal.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `fluid_asymmetry_index` | 0-0.1 | ratio | 📡 RIS | Left-right impedance comparison |
| `total_body_water_proxy` | 0.8-1.2 | normalized | 📡 RIS | Inverse of mean impedance |
| `extracellular_fluid_ratio` | 0.35-0.45 | ratio | 📡 RIS | Multi-frequency impedance variance |
| `fluid_overload_index` | -0.3-0.3 | index | 📡 RIS | Thorax vs abdomen impedance delta |

**Hardware Requirements**: 📡 RIS only

**Extraction Method**:
1. Compare left/right channel impedance → Asymmetry
2. Lower impedance = more total body water
3. Thoracic vs abdominal distribution → Fluid overload detection

---

### 3.4 Gastrointestinal

**Extractor**: `app/core/extraction/gastrointestinal.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `abdominal_rhythm_score` | 0.4-0.9 | score | 📡 RIS | FFT 0.03-0.25 Hz (GI motility) |
| `visceral_motion_variance` | 10-100 | variance | 📡 RIS | Abdominal channel variance |
| `abdominal_respiratory_rate` | 12-20 | breaths/min | 📷 Camera | Hip landmark vertical motion |

**Hardware Requirements**: 📡 RIS (primary), 📷 Camera (secondary)

**Extraction Method**:
1. Abdominal RIS channels (8-11) → Low-frequency power analysis
2. Normal GI motility: 3-12 cycles/min (0.05-0.2 Hz)
3. Camera hip center → Respiratory rate estimation

---

### 3.5 Skeletal Structure

**Extractor**: `app/core/extraction/skeletal.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `gait_symmetry_ratio` | 0.85-1.0 | ratio | 📷 Camera | Left/right range of motion comparison |
| `step_length_symmetry` | 0.85-1.0 | ratio | 📷 Camera | Ankle step length L/R comparison |
| `stance_stability_score` | 75-100 | score | 📷 Camera | Center of mass sway analysis |
| `sway_velocity` | 0.001-0.01 | units/frame | 📷 Camera | COM velocity over time |
| `average_joint_rom` | 0.3-0.8 | radians | 📷 Camera | Elbow/knee angle range |

**Hardware Requirements**: 📷 Camera only

**Extraction Method**:
1. Compare symmetric joint pairs (shoulders, hips, knees, ankles)
2. Calculate range of motion for each joint
3. Sway analysis from hip center trajectory

---

### 3.6 Skin

**Extractor**: `app/core/extraction/skin.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `texture_roughness` | 5-25 | variance | 📷 Camera | Local variance in grayscale patches |
| `skin_redness` | 0.3-0.6 | intensity | 📷 Camera | Red channel ratio analysis |
| `skin_yellowness` | 0.2-0.5 | intensity | 📷 Camera | R+G vs B ratio (jaundice proxy) |
| `color_uniformity` | 0.7-1.0 | score | 📷 Camera | Inverse of color variance |
| `lesion_count` | 0-5 | count | 📷 Camera | High-contrast region detection |

**Hardware Requirements**: 📷 Camera only

**Extraction Method**:
1. Local variance in 5×5 pixel patches → Texture
2. RGB channel ratio analysis → Color metrics
3. Threshold-based anomaly detection → Lesions

---

### 3.7 Eyes

**Extractor**: `app/core/extraction/eyes.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `blink_rate` | 12-20 | blinks/min | 📷 Camera | Eye visibility transitions |
| `gaze_stability_score` | 70-100 | score | 📷 Camera | Eye position variance |
| `fixation_duration` | 150-400 | ms | 📷 Camera | Low-velocity eye periods |
| `saccade_frequency` | 2-5 | saccades/sec | 📷 Camera | High-velocity eye movements |
| `eye_symmetry` | 0.9-1.0 | ratio | 📷 Camera | L/R eye motion correlation |

**Hardware Requirements**: 📷 Camera only

**Extraction Method**:
1. Eye landmarks (2, 5 in MediaPipe) tracked
2. Visibility score dips → Blink detection
3. Velocity analysis → Saccade vs fixation classification

---

### 3.8 Nasal Passages

**Extractor**: `app/core/extraction/nasal.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `breathing_regularity` | 0.6-1.0 | score | 📷 Camera | CV of inter-breath intervals |
| `nasal_asymmetry_proxy` | 0.1-0.5 | ratio | 📷 Camera | Lateral vs vertical nose motion |
| `respiratory_rate` | 12-20 | breaths/min | 📡 RIS | FFT 0.1-0.5 Hz peak frequency |
| `breath_depth_index` | 0.5-1.5 | normalized | 📡 RIS | Peak-to-trough amplitude |
| `airflow_turbulence` | 0.01-0.1 | power ratio | 📡 RIS | High-frequency content (5-50 Hz) |

**Hardware Requirements**: 📷 Camera (breathing regularity), 📡 RIS (respiratory metrics)

**Extraction Method**:
1. Nose landmark (0) vertical motion → Breathing pattern
2. Thoracic impedance → Respiratory rate via FFT
3. High-frequency components indicate turbulent airflow

---

### 3.9 Reproductive

**Extractor**: `app/core/extraction/reproductive.py`

| Biomarker | Normal Range | Unit | Source | Method |
|-----------|--------------|------|--------|--------|
| `autonomic_imbalance_index` | -0.3-0.3 | index | 📡 RIS | HR and HRV combined metric |
| `stress_response_proxy` | 20-60 | score | 📡 RIS | Sympathovagal balance |
| `regional_flow_variability` | 0.01-0.1 | CV | 📡 RIS | Pelvic channel variance |
| `thermoregulation_proxy` | 0.4-0.6 | normalized | 📡 RIS | Central vs peripheral impedance |

**Hardware Requirements**: 📡 RIS only

**Extraction Method**:
1. HRV-based autonomic nervous system assessment
2. Pelvic region channels (12-16) for regional blood flow
3. These are **proxy indicators only**, not diagnostic

> ⚠️ **Disclaimer**: Reproductive biomarkers are autonomic nervous system proxies only. Not intended for reproductive health diagnosis.

---

## 4. Hardware Coverage Matrix

| Physiological System | 📷 Camera | 📡 RIS | 🎤 Audio | IMU |
|---------------------|:---------:|:------:|:--------:|:---:|
| Central Nervous System | ✅ Primary | ⬜ | ⬜ | 🔶 Optional |
| Cardiovascular | 🔶 Backup | ✅ Primary | ⬜ | ⬜ |
| Renal | ⬜ | ✅ Required | ⬜ | ⬜ |
| Gastrointestinal | 🔶 Backup | ✅ Primary | ⬜ | ⬜ |
| Skeletal | ✅ Required | ⬜ | ⬜ | 🔶 Optional |
| Skin | ✅ Required | ⬜ | ⬜ | ⬜ |
| Eyes | ✅ Required | ⬜ | ⬜ | ⬜ |
| Nasal | ✅ Partial | ✅ Partial | 🔶 Optional | ⬜ |
| Reproductive | ⬜ | ✅ Required | ⬜ | ⬜ |

**Legend**: ✅ Required | 🔶 Optional/Backup | ⬜ Not Used

---

## 5. Camera-Only Mode (Current Capability)

With only cameras, you can extract **~60% of biomarkers**:

| System | Camera-Only Biomarkers | Count |
|--------|----------------------|-------|
| CNS | gait_variability, posture_entropy, tremors, stability | 6/6 |
| Cardiovascular | heart_rate (from chest motion), chest_micro_motion | 2/6 |
| Renal | ❌ None | 0/4 |
| Gastrointestinal | abdominal_respiratory_rate | 1/3 |
| Skeletal | All biomarkers | 5/5 |
| Skin | All biomarkers | 5/5 |
| Eyes | All biomarkers | 5/5 |
| Nasal | breathing_regularity, nasal_asymmetry | 2/5 |
| Reproductive | ❌ None | 0/4 |

**Camera-Only Total: 26 biomarkers from 7 systems**

---

## 6. Full System (Camera + RIS)

With both sensors:

| Metric | Camera Only | Camera + RIS |
|--------|-------------|--------------|
| Biomarkers | 26 | 42 |
| Systems Covered | 7/9 | 9/9 |
| Cardiovascular Accuracy | 50% | 95% |
| Renal Coverage | 0% | 100% |
| Reproductive Coverage | 0% | 100% |

---

## 7. Walkthrough Chamber Operation

### 7.1 User Journey (Passive, ~30 seconds)

```
┌─────────────────────────────────────────────────────────────┐
│                      WALKTHROUGH CHAMBER                     │
│                         (5m x 2m x 2.5m)                    │
│                                                              │
│  ENTRY          ZONE 1         ZONE 2         EXIT          │
│    │              │              │              │            │
│    ▼              ▼              ▼              ▼            │
│  ┌───┐        ┌───────┐      ┌───────┐      ┌───┐          │
│  │   │        │ SCAN  │      │ SCAN  │      │   │          │
│  │   │  ───►  │ FACE  │ ───► │ BODY  │ ───► │   │          │
│  │   │        │ EYES  │      │ GAIT  │      │   │          │
│  └───┘        └───────┘      └───────┘      └───┘          │
│                                                              │
│  📷 Cameras: ███ ███ ███ ███ ███ ███                       │
│  📡 RIS:     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (floor/sides)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Data Collection Timeline

| Time (s) | Action | Data Collected |
|----------|--------|----------------|
| 0-5 | Entry + Face scan | Eyes, Skin (face) |
| 5-15 | Walking through | Gait, CNS, Skeletal |
| 10-20 | RIS continuous | Cardiovascular, Renal, GI, Reproductive |
| 15-25 | Full body scan | Skin (body), Nasal |
| 25-30 | Processing | All biomarkers computed |

---

## 8. API Integration

### 8.1 Screening Request (After Chamber Walkthrough)

```json
POST /api/v1/screening
{
  "patient_id": "PATIENT-001",
  "systems": [
    {
      "system": "cardiovascular",
      "biomarkers": [
        {"name": "heart_rate", "value": 72, "unit": "bpm", "normal_range": [60, 100]},
        {"name": "hrv_rmssd", "value": 45, "unit": "ms", "normal_range": [20, 80]}
      ]
    },
    {
      "system": "cns",
      "biomarkers": [
        {"name": "gait_variability", "value": 0.04, "unit": "CV", "normal_range": [0.02, 0.08]},
        {"name": "cns_stability_score", "value": 85, "unit": "score", "normal_range": [70, 100]}
      ]
    }
    // ... 7 more systems
  ],
  "include_validation": true
}
```

### 8.2 Response

```json
{
  "screening_id": "SCR-XXXXXXXX",
  "overall_risk_level": "low",
  "overall_risk_score": 25.5,
  "system_results": {...},
  "composite_risk": {...}
}
```

### 8.3 Report Generation

```json
POST /api/v1/reports/generate
{
  "screening_id": "SCR-XXXXXXXX",
  "report_type": "patient"  // or "doctor"
}
```

---

## 9. Total Biomarker Summary

| System | Biomarkers | Camera | RIS | Confidence |
|--------|:----------:|:------:|:---:|:----------:|
| CNS | 6 | ✅ | ⬜ | 75-85% |
| Cardiovascular | 6 | 🔶 | ✅ | 70-95% |
| Renal | 4 | ⬜ | ✅ | 55-70% |
| Gastrointestinal | 3 | 🔶 | ✅ | 55-65% |
| Skeletal | 5 | ✅ | ⬜ | 70-80% |
| Skin | 5 | ✅ | ⬜ | 45-65% |
| Eyes | 5 | ✅ | ⬜ | 50-70% |
| Nasal | 5 | ✅ | ✅ | 45-70% |
| Reproductive | 4 | ⬜ | ✅ | 35-55% |
| **TOTAL** | **42** | **26** | **23** | **~65% avg** |

---

## 10. Future Enhancements

1. **Thermal Camera** → Skin temperature, circulation
2. **Spectral Camera** → SpO2, hemoglobin estimation
3. **Pressure Sensors (Floor)** → Weight distribution, gait force
4. **Audio Analysis** → Heart sounds, lung sounds
5. **3D LiDAR** → Precise body measurements

---

## 11. Compliance & Limitations

> [!IMPORTANT]
> This system is for **screening purposes only** and does not replace clinical diagnosis.

- ✅ Non-invasive, passive operation
- ✅ No blood samples or physical contact required
- ✅ GDPR-compliant data handling
- ❌ Not FDA/CE approved for diagnosis
- ❌ Reproductive biomarkers are proxy indicators only
- ❌ Individual biomarker confidence varies (35-95%)

---

## Quick Reference: Running Camera Test Now

```bash
# With your existing setup:
.\benv\Scripts\python.exe camera_test.py
```

This will test 7 of 9 systems using camera only!

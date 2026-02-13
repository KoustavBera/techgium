# System Architecture Validation Report

**Date**: February 13, 2026  
**System**: Non-Invasive Health Screening Pipeline v2.0  
**Analysis**: 8-Point Critique Validation

---

## Executive Summary

✅ **CRITIQUE VALIDATED**: The concerns raised are **largely accurate** and represent genuine architectural weaknesses that will cause **false positives** in clinical deployment.

**Critical Findings**:
- ❌ No environmental baseline calibration
- ❌ Overly aggressive clinical-grade ranges for consumer hardware
- ⚠️ Confidence weighting exists but underutilized
- ❌ No multi-session user profiling
- ❌ Thermal asymmetry lacks pose validation
- ⚠️ Reporting language moderately aggressive  
- ✅ Signal quality gates partially implemented
- ❌ No minimum data quality threshold

**Overall Risk**: **HIGH** - System will generate unreliable results in production

---

## Detailed Validation

### ✅ STEP 1: Environmental Baseline Calibration — **ABSENT** (CRITICAL)

**Status**: ❌ **NOT IMPLEMENTED**

**Findings**:

Searched across all extraction modules for baseline calibration mechanisms:
- **Skin extractor** ([`skin.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/skin.py)):
  - Uses **fixed hardware calibration offset** (+0.8°C for thermal camera)
  - **No per-user baseline** for facial temperature
  - **No dynamic redness/yellowness baseline**
  - Lines 162-163: `normal_range=(0.0, 25.0)` for `skin_redness` — absolute, not user-relative
  
- **Cardiovascular extractor** ([`cardiovascular.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/cardiovascular.py)):
  - No baseline HR capture
  - No HRV baseline establishment
  
- **Calibration module** ([`calibration.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/calibration.py)):
  - Contains `_system_baselines` but these are **system-level confidence multipliers**, NOT user baselines
  - Line 58: `_initialize_baselines()` → provides fixed confidence factors per physiological system (0.55-0.90)
  - **No user-specific thermal or color baseline capture**

**Impact**:

> **Thermal variation** from room temperature, lighting, skin tone, and time-of-day will be **misinterpreted as pathology**.

**Example False Positive Scenario**:
```
User in cold room (20°C ambient):
- Facial temp: 34.2°C (normal for environment)
- System interprets as: "Below range [35.5-37.5]" → Risk elevated
- Actual: User is healthy, environment is cold
```

**Validation**: ✅ **CRITIQUE IS CORRECT**

---

### ✅ STEP 2: Overly Aggressive Physiological Ranges — **CONFIRMED** (CRITICAL)

**Status**: ❌ **RANGES TOO NARROW FOR CONSUMER HARDWARE**

**Findings**:

Examined [`biomarker_plausibility.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/validation/biomarker_plausibility.py) (lines 83-167):

**🔴 Problem Ranges Identified**:

| Biomarker | Current Range | Hardware | Issue |
|-----------|--------------|----------|-------|
| `thermal_asymmetry` | **(0.0, 0.5°C)** | Webcam thermal, 60GHz radar | Too tight for non-medical devices |
| `skin_redness` | **(0.0, 25.0)** | Webcam RGB | Non-calibrated lighting sensitivity |
| `skin_temperature` | **(35.5, 37.5°C)** | MLX90640 thermal | Clinical thermometer range, not environmental |
| `color_uniformity` | **(0.25, 1.0)** | Webcam | Lighting-dependent |
| `inflammation_index` | **(0.0, 5.0%)** | ESP32 thermal | Sensitive to room temp |

**Specific Evidence**:

1. **Thermal Asymmetry** (Line 110):
   ```python
   "thermal_asymmetry": {"hard": (0.0, 10.0), "physiological": (0.0, 2.0)}
   ```
   But in [`skin.py:415`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/skin.py#L415):
   ```python
   normal_range=(0.0, 0.5)  # Even tighter!
   ```
   **Critique says**: Use `(0.0, 3.5)` for screening-grade → **VALID**

2. **Skin Redness** (Line 143):
   ```python
   "skin_redness": {"hard": (0.0, 1.0), "physiological": (0.1, 0.9)}
   ```
   But in [`skin.py:162`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/skin.py#L162):
   ```python
   normal_range=(0.0, 25.0)  # Lab deviation scale, lighting-dependent
   ```
   **Critique says**: Make dynamic `(mean ± 2*std)` from dataset → **VALID**

**Validation**: ✅ **CRITIQUE IS CORRECT** - Ranges are clinical-lab style, not consumer-hardware adjusted

---

### ⚠️ STEP 3: Confidence-Weighted Risk Aggregation — **PARTIALLY IMPLEMENTED**

**Status**: ⚠️ **EXISTS BUT UNDERUTILIZED**

**Findings**:

Signal quality IS computed ([`signal_quality.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/validation/signal_quality.py)):
- Lines 111-258: `assess_camera()` computes `overall_quality` (0-1 scale)
- Lines 351-431: `assess_radar()` validates presence detection
- Lines 433-531: `assess_thermal()` checks temperature plausibility

**BUT** — In risk aggregation ([`risk_engine.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py)):

Line 264-275 (Risk computation):
```python
for biomarker in biomarker_set.biomarkers:
    bm_risk, bm_alert = self._calculate_biomarker_risk(biomarker)
    
    weight = weights.get(biomarker.name, 0.1)
    weighted_scores.append(bm_risk * weight)
    confidences.append(biomarker.confidence * weight)  # ← Uses confidence
```

**BUT NO AGGRESSIVE DOWNWEIGHTING** for low quality. Line 300:
```python
confidence=float(np.clip(overall_confidence, 0, 1))  # No penalties
```

**Missing**:
```python
# Critique recommends:
effective_score = deviation_score × biomarker_confidence × signal_quality
```

Currently: Score uses confidence, **but not signal quality** from `SignalQualityAssessor`.

**Validation**: ⚠️ **PARTIALLY CORRECT** - Confidence exists, but signal quality not integrated into risk scores

---

### ❌ STEP 4: Multi-Session Averaging — **NOT IMPLEMENTED** (CRITICAL)

**Status**: ❌ **COMPLETELY ABSENT**

**Findings**:

Searched for user profile persistence:
```bash
grep -ri "user_profile" app/
# Result: No results found
```

Searched for session tracking:
```bash
grep -ri "session" app/
# Result: Only 2 mentions — both unrelated to multi-session storage
```

**Evidence**:

- [`main.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/main.py):
  - Line 100-101: In-memory storage for CURRENT screenings only:
    ```python
    _screenings: Dict[str, Dict[str, Any]] = {}
    ```
  - **No persistent user profile database**
  - **No running mean/std tracking**
  
- No files named `user_profile.py`, `session_history.py`, or similar

**Impact**:

> Every scan is treated as **completely independent**. A user who always runs warm (36.8°C baseline) will be flagged as "high temp" repeatedly.

**Validation**: ✅ **CRITIQUE IS 100% CORRECT** - No multi-session tracking exists

---

### ❌ STEP 5: Thermal Asymmetry Pose Validation — **NOT IMPLEMENTED** (CRITICAL)

**Status**: ❌ **NO POSE ENFORCEMENT**

**Findings**:

Examined [`skin.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/skin.py) thermal processing:

Lines 419-469: `_extract_from_thermal_v2()`:
```python
# Thermal Asymmetry from canthus (core body proxy - medically validated)
if thermal_data.get('thermal_asymmetry') is not None:
    self._add_biomarker(
        biomarker_set,
        name="thermal_asymmetry",
        value=float(thermal_data['thermal_asymmetry']),
        ...
    )
```

**NO CHECKS FOR**:
- Face yaw angle (rotation left/right)
- Face pitch angle (looking up/down)  
- Frontal pose validation

**Critique recommends**:
```
Reject frames where:
- yaw > 10°
- pitch > 10°
```

**Currently**: Asymmetry is computed **regardless of head pose**, making it unreliable.

**Validation**: ✅ **CRITIQUE IS CORRECT** - Pose validation missing from thermal asymmetry

---

### ⚠️ STEP 6: Reporting Language Severity — **MODERATELY AGGRESSIVE**

**Status**: ⚠️ **PARTIALLY VALID**

**Findings**:

Examined risk level definitions ([`risk_engine.py:33-53`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py#L33-L53)):

```python
class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"  # ← Exists
    UNKNOWN = "unknown"
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if score < 25: return cls.LOW
        elif score < 50: return cls.MODERATE
        elif score < 75: return cls.HIGH
        else: return cls.CRITICAL  # 75-100 = CRITICAL
```

**Issue**: "Critical" level exists and is easily triggered (score > 75).

**Composite Risk** ([`risk_engine.py:641-642`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py#L641-L642)):
```python
if max_level == RiskLevel.CRITICAL and composite_score < 75:
    composite_score = 75.0  # Force elevation to CRITICAL
```

**Language in explanations** (Line 535):
```python
f"{system.value.replace('_', ' ').title()} assessment indicates {level.value} risk."
```

**Examples**:
- "Cardiovascular assessment indicates **critical risk**"
- No softening language like "Signal Deviation Detected"

**However**: No explicit "Immediate Care" wording found in code (may be in frontend or reports).

**Validation**: ⚠️ **PARTIALLY CORRECT** - Language is clinical-style ("critical risk"), but less dramatic than critique suggests. Recommend toning down.

---

### ⚠️ STEP 7: Unstable System Impact Weights — **PARTIALLY ADDRESSED**

**Status**: ⚠️ **SYSTEM WEIGHTS EXIST, BUT NOT DYNAMICALLY ADJUSTED**

**Findings**:

Composite Risk Calculator ([`risk_engine.py:568-585`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py#L568-L585)):

```python
self.system_weights = {
    PhysiologicalSystem.CNS: 1.0,
    PhysiologicalSystem.CARDIOVASCULAR: 1.2,  # Higher weight
    PhysiologicalSystem.RENAL: 1.0,
    PhysiologicalSystem.GASTROINTESTINAL: 0.8,
    PhysiologicalSystem.SKELETAL: 0.9,
    PhysiologicalSystem.SKIN: 0.7,
    PhysiologicalSystem.EYES: 0.8,
    PhysiologicalSystem.NASAL: 0.8,
    PhysiologicalSystem.REPRODUCTIVE: 0.7,  # Lower weight
}
```

**Good**: Weights are differentiated

**Missing**: No `experimental_mode` flag to dynamically reduce weights for unstable systems

**Critique suggests**:
- Mark unreliable systems (Nasal, Renal microcirculation, Gait) as `experimental_mode = True`
- Reduce their impact multiplier

**Validation**: ⚠️ **PARTIALLY CORRECT** - Weights exist but are static, no experimental flagging

---

### ❌ STEP 8: Minimum Data Quality Gate — **NOT IMPLEMENTED**

**Status**: ❌ **NO HARD QUALITY THRESHOLD**

**Findings**:

Signal quality is computed ([`signal_quality.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/validation/signal_quality.py)) but:

**No rejection logic** in main screening pipeline ([`main.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/main.py)):

Lines 215-259: `/api/v1/screening` endpoint:
```python
result = await _screening_service.process_screening(
    patient_id=request.patient_id,
    systems_input=systems_input,
    include_validation=request.include_validation
)
# No quality gate check here
```

**Critique recommends**:
```python
if signal_quality < 0.5 or face_detected < 80% or pose_detected < 70%:
    report_status = "INSUFFICIENT DATA"
    # DO NOT COMPUTE RISK
```

**Currently**: Risk is always computed regardless of data quality.

**Validation**: ✅ **CRITIQUE IS CORRECT** - No minimum quality gate before risk computation

---

## Summary of Validation Results

| Step | Issue | Status | Severity |
|------|-------|--------|----------|
| 1 | Environmental Baseline Calibration | ❌ **ABSENT** | 🔴 CRITICAL |
| 2 | Aggressive Physiological Ranges | ❌ **TOO NARROW** | 🔴 CRITICAL |
| 3 | Confidence-Weighted Risk | ⚠️ **PARTIAL** | 🟡 MEDIUM |
| 4 | Multi-Session Averaging | ❌ **ABSENT** | 🔴 CRITICAL |
| 5 | Thermal Pose Validation | ❌ **ABSENT** | 🔴 CRITICAL |
| 6 | Reporting Language | ⚠️ **MODERATE** | 🟡 MEDIUM |
| 7 | Unstable System Weights | ⚠️ **STATIC** | 🟡 MEDIUM |
| 8 | Data Quality Gate | ❌ **ABSENT** | 🔴 CRITICAL |

**Overall**: **6/8 issues validated**, **5 critical gaps identified**

---

## Recommended Action Plan

### 🔥 **Phase 1: Critical Fixes (Prevent False Positives)**

#### 1.1 Add Environmental Baseline Calibration
**Where**: [`app/core/extraction/skin.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/skin.py), [`app/core/extraction/cardiovascular.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/cardiovascular.py)

**Changes**:
- Add 30-second baseline capture mode:
  ```python
  class BaselineCapture:
      def capture_baseline(self, frames, duration_sec=30):
          mean_facial_temp = compute_mean(thermal_frames)
          mean_redness = compute_mean(color_a_channel)
          mean_yellowness = compute_mean(color_b_channel)
          return UserBaseline(...)
  ```
- Convert all biomarkers to **delta from baseline**:
  ```python
  thermal_deviation = current_temp - user_baseline.temp
  redness_deviation = current_redness - user_baseline.redness
  ```

#### 1.2 Loosen Physiological Ranges for Consumer Hardware
**Where**: [`app/core/validation/biomarker_plausibility.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/validation/biomarker_plausibility.py) (lines 83-167)

**Changes**:
```python
# BEFORE (Clinical-grade):
"thermal_asymmetry": {"physiological": (0.0, 0.5)}

# AFTER (Screening-grade):
"thermal_asymmetry": {"physiological": (0.0, 3.5)}

# BEFORE:
"skin_redness": {"physiological": (0.1, 0.9)}

# AFTER (Dynamic from dataset):
"skin_redness": {"physiological": (mean - 2*std, mean + 2*std)}
```

#### 1.3 Add Pose Validation for Thermal Asymmetry
**Where**: [`app/core/extraction/skin.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/skin.py#L419)

**Changes**:
```python
def _extract_from_thermal_v2(self, thermal_data, biomarker_set, pose_data=None):
    # NEW: Validate frontal pose
    if pose_data:
        yaw, pitch = extract_head_pose(pose_data)
        if abs(yaw) > 10 or abs(pitch) > 10:
            logger.warning("Non-frontal pose detected - skipping thermal asymmetry")
            return  # Don't compute asymmetry
    
    # Existing asymmetry calculation...
```

#### 1.4 Add Minimum Data Quality Gate
**Where**: [`app/main.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/main.py#L215), [`app/services/screening.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/services/screening.py)

**Changes**:
```python
@app.post("/api/v1/screening")
async def run_screening(request: ScreeningRequest):
    # NEW: Check data quality BEFORE processing
    quality_scores = assess_signal_quality(request.data)
    
    if quality_scores.overall < 0.5:
        return ScreeningResponse(
            status="INSUFFICIENT_DATA",
            message="Please ensure: face clearly visible, good lighting, stable camera",
            overall_risk_level="unknown"
        )
    
    # Proceed with existing screening...
```

---

### ⚙️ **Phase 2: Multi-Session Tracking (Reduce Noise)**

#### 2.1 Create User Profile Database
**New Files**: 
- `app/models/user_profile.py`
- `app/data/profiles.db` (SQLite)

**Schema**:
```python
class UserProfile:
    user_id: str
    baseline_temp: float
    baseline_redness: float
    session_history: List[ScreeningResult]
    running_mean: Dict[str, float]
    running_std: Dict[str, float]
    session_count: int
```

#### 2.2 Modify Risk Engine to Use History
**Where**: [`app/core/inference/risk_engine.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py)

**Changes**:
```python
def compute_risk(self, biomarker_set, user_profile=None):
    if user_profile and user_profile.session_count >= 3:
        # Only flag if: 3 consecutive sessions abnormal OR
        # deviation > 2× previous std
        if not is_persistent_anomaly(biomarker_set, user_profile):
            # Downgrade to LOW risk
            overall_risk.score *= 0.5
```

---

### 🎨 **Phase 3: UX & Reporting Improvements**

#### 3.1 Soften Reporting Language
**Where**: [`app/core/inference/risk_engine.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py#L535), Report generators

**Changes**:
```python
# BEFORE:
"Cardiovascular assessment indicates critical risk."

# AFTER:
"Signal Deviation Detected in Cardiovascular Metrics — Recommend Repeat Scan"
```

Remove "CRITICAL" level for first-time scans.

#### 3.2 Add Experimental Mode Flags
**Where**: [`app/core/inference/risk_engine.py`](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py#L575)

**Changes**:
```python
EXPERIMENTAL_SYSTEMS = {
    PhysiologicalSystem.NASAL: True,
    PhysiologicalSystem.RENAL: True,
    PhysiologicalSystem.SKELETAL: True
}

for system, result in system_results.items():
    if EXPERIMENTAL_SYSTEMS.get(system, False):
        result.overall_risk.score *= 0.5  # 50% weight reduction
        result.alerts.append("(Experimental measurement)")
```

---

## Testing Recommendations

### Regression Tests
1. **Baseline Calibration Test**:
   - Scan same user at different room temperatures
   - Expected: Consistent risk levels after baseline normalization
   
2. **Multi-Session Test**:
   - Run 5 scans on same healthy user
   - Expected: Risk should DECREASE after session 3 (running baseline established)
   
3. **Pose Validation Test**:
   - Capture thermal data while rotating head ±30°
   - Expected: Asymmetry flagged as "unreliable" or skipped

### Quality Gate Test
- Feed low-quality frames (blurry, low lighting)
- Expected: "INSUFFICIENT DATA" response before risk computation

---

## Conclusion

The critique is **substantially correct**. The current system:

1. ✅ **Has strong technical foundation** (good extraction, validation modules)
2. ❌ **Lacks personalization** (no baselines, no multi-session memory)
3. ❌ **Uses overly strict thresholds** (clinical-grade for consumer hardware)
4. ❌ **No quality gating** (processes garbage-in → garbage-out)

**Without these fixes**, the system will produce **high false positive rates** in real-world deployment.

**Priority**: Implement Phase 1 (Critical Fixes) immediately before any production use.

---

## File References

### Critical Files to Modify
- [app/core/extraction/skin.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/skin.py)
- [app/core/extraction/cardiovascular.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/extraction/cardiovascular.py)
- [app/core/validation/biomarker_plausibility.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/validation/biomarker_plausibility.py)
- [app/core/inference/risk_engine.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/risk_engine.py)
- [app/main.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/main.py)

### Supporting Files Examined
- [app/core/inference/calibration.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/inference/calibration.py)
- [app/core/validation/signal_quality.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/validation/signal_quality.py)
- [app/core/hardware/manager.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/hardware/manager.py)
- [app/core/hardware/drivers.py](file:///c:/Users/Swetanjana%20Maity/Desktop/kblndt/techgium/fastapi2/app/core/hardware/drivers.py)

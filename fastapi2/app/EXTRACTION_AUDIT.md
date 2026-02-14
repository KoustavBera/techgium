# Extraction Logic Audit & Fix Manual

**Date:** 2026-02-14  
**Scope:** `app/core/extraction/*`  
**Purpose:** Identify environmental sensitivities in current biomarker extraction and propose algorithmic fixes.

---

## 1. Skin & Thermal (`skin.py`, `manager.py`)

### Problem A: "Inflammation" & "Asymmetry" False Positives
**Symptom:** User reports "Inflammation" (10%) and "Thermal Asymmetry" (3.77°C) despite feeling healthy.  
**Root Cause:**
*   **Code:** `SkinExtractor._extract_from_thermal_v2` limits inflammation calculation to `(canthus_range - 0.8) * 8.33`.
*   **Defect:** "Temperature Stability" (`canthus_range`) is assumed to be biological voltage change. In reality, **Head Motion** moves the target (canthus) off the hot skin pixel to a cold background pixel, creating a massive delta (>3°C).
*   **Environmental Factor:** Movement, Hand-held camera instability.

**Fix: Motion Gating**
Calculate frame-to-frame optical flow or landmark velocity. If velocity > threshold, **discard the thermal frame**.
```python
# Pseudo-code Fix
if head_velocity > 0.5_pixels_per_frame:
    return previous_value  # Ignore this frame
else:
    process_thermal_frame()
```

### Problem B: "Redness" / "Yellowness" Artifacts
**Symptom:** High deviations (>12) in Skin Color analysis.  
**Root Cause:**
*   **Code:** `_analyze_skin_color_lab` converts RGB to CIELab and checks deviation.
*   **Defect:** The camera measures **Reflected Light**, not **Albedo** (Skin Pigment). Warm room lighting (2700K bulbs) adds yellow/red bias to the pixels.
*   **Environmental Factor:** Indoor lighting temperature.

**Fix: Automatic White Balance (AWB) Normalization**
Use a "Grey World" assumption or background subtraction relative to a neutral object to normalize the skin tone.
```python
# Pseudo-code Fix
avg_scene_color = np.mean(image, axis=(0,1))
correction_factor = [128/avg_scene_color[0], 128/avg_scene_color[1], 128/avg_scene_color[2]]
balanced_image = image * correction_factor
```

---

## 2. Eyes & Attention (`eyes.py`)

### Problem C: "Low Blink Rate" (Computer Vision Syndrome)
**Symptom:** Blink rate ~6/min (Normal 12-15).  
**Root Cause:**
*   **Code:** `_estimate_blink_rate_ear` counts blinks over time.
*   **Defect:** Logic ignores **Context**. When a user stares at a screen (the test interface) to align their face, blink rate physiologically drops by 50% (Computer Vision Syndrome). This is "Functionally Normal" but "Clinically Low."
*   **Environmental Factor:** Task focus / Screen staring.

**Fix: Context-Aware Ranges**
Adjust the "Normal" range dynamically based on the activity.
```python
# Pseudo-code Fix
if user_state == "watching_screen":
    normal_range = (5, 30)  # Relaxed lower bound
else:
    normal_range = (12, 30) # Clinical standard
```

---

## 3. Cardiovascular (`cardiovascular.py`)

### Problem D: rPPG Noise Signal
**Symptom:** Spiky or inconsistent Heart Rate confidence.  
**Root Cause:**
*   **Code:** `_extract_from_rppg` uses Green channel mean.
*   **Defect:** Fluctuations in ambient light (50Hz/60Hz flicker) or shadows from moving heads modify the Green intensity more than blood flow does.
*   **Environmental Factor:** AC Lighting mains hum, Shadows.

**Fix: Independent Component Analysis (ICA) or POS**
Move from single-channel Green mean to **Plane-Orthogonal-to-Skin (POS)** algorithm, which mathematically cancelling out the specular (lighting) component.
```python
# Pseudo-code Fix (POS Algorithm)
# Combine R, G, B channels to isolate melanin/hemoglobin checks, removing lighting intensity I.
X = 3*R - 2*G
Y = 1.5*R + G - 1.5*B
pulse_signal = X + alpha * Y
```

---

## 4. CNS & Skeletal (`cns.py`, `skeletal.py`)

### Problem E: "Rigid" Posture / Low Entropy
**Symptom:** CNS Entropy 0.38 (Low/Abnormal).  
**Root Cause:**
*   **Code:** `_calculate_posture_entropy` measures sway complexity.
*   **Defect:** Users often perform the test **Sitting Down**. A chair mechanically stabilizes the spine, removing the micro-sway required for this metric. The code assumes a **Standing** Romberg test.
*   **Environmental Factor:** Seated position.

**Fix: Posture Detection**
Detect "Seated" vs "Standing" using visible hip/knee landmarks or vertical bounding box aspect ratio.
```python
# Pseudo-code Fix
if is_likely_seated(pose_landmarks):
    biomarker_set.add_note("CNS Analysis Limited: Patient Seated")
    return  # Skip sway analysis
```

### Problem F: Depth Scaling
**Symptom:** Stance Stability score varies by distance.  
**Root Cause:**
*   **Code:** `skeletal.py` uses raw pixel coordinate standard deviation for sway.
*   **Defect:** A user 1 meter away has 2x the pixel sway of a user 2 meters away for the same physical movement.
*   **Environmental Factor:** Distance from camera.

**Fix: 3D Normalization / Iris Reference**
Normalize pixel/landmark movement by the **Inter-Pupillary Distance (IPD)** or **Face Width** to get relative units (e.g., "% of head width").
```python
# Pseudo-code Fix
face_width_px = abs(left_ear.x - right_ear.x)
normalized_sway = raw_sway_px / face_width_px
```

---

## Summary of Actionable Fixes

1.  [ ] **Implement Motion Gating** in `skin.py` to nullify thermal artifacts.
2.  [ ] **Add White Balance** in `skin.py` for color accuracy.
3.  [ ] **Correct Normal Ranges** in `eyes.py` for screen-viewing context.
4.  [ ] **Normalize Skeletal Metrics** by face width/IPD to remove distance artifacts.

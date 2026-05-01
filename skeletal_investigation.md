# Skeletal Analysis — Full Investigation Report

## Root Cause: 3-Layer Failure Chain

The Skeletal system appears in the report as "Low Risk — Healthy" but with an alert saying **"Plausibility validation failed: Required biomarker 'gait_symmetry_ratio' is missing"**. This single alert causes the entire system risk result to be **rejected** by the Trust Engine.

---

## Layer 1 — The Plausibility Validator Blocks the System

**File:** `app/core/validation/biomarker_plausibility.py` **line 256**

```python
PhysiologicalSystem.SKELETAL: ["gait_symmetry_ratio"],  # ← HARD requirement
```

The validator marks any skeletal result **invalid** (`severity=0.8`) if `gait_symmetry_ratio` is absent.
Severity ≥ 0.8 triggers a full rejection in the risk engine (`risk_engine.py` line 393):

```python
violation_msgs = [v.message for v in plausibility.violations if v.severity >= 0.8]
# → rejection_reason = "Plausibility validation failed: Required biomarker 'gait_symmetry_ratio' is missing"
```

**Since a kiosk user is always stationary, `gait_symmetry_ratio` is never produced → system always rejected.**

---

## Layer 2 — The Algorithm: What the Webcam Can and Cannot See

### What MediaPipe Pose gives us from a stationary kiosk user

| Signal | Available from Webcam? | Quality | Notes |
|---|---|---|---|
| Joint positions (33 landmarks) | ✅ Yes | Good | Works seated or standing |
| Postural sway (CoM oscillation) | ✅ Yes | Good | Shoulder/hip center over time |
| Joint range of motion (ROM) | ✅ Yes | Good | Works stationary |
| Bilateral symmetry (left vs right) | ✅ Yes | Good | Works stationary |
| Gait (walking stride cycle) | ❌ No | N/A | Requires subject to walk |
| Step length | ❌ No | N/A | Requires walking |

### Current algorithm problems

1. **gait_symmetry_ratio computed from bilateral ROM, not actual gait** — the function `_extract_gait_symmetry` computes the *range of motion symmetry* between left/right joints, which is valid when stationary! But it's gated behind `is_walking=True`, so it never runs.

2. **The bilateral symmetry computation in `_extract_gait_symmetry` works fine for a stationary person** — it compares `l_range vs r_range` from filtered joint trajectories. It doesn't actually require walking — it just needs joint motion variance, which exists even when breathing/shifting weight.

3. **Threshold is too strict** — `STATIONARY_THRESHOLD_PPS = 40.0` pixels/sec is calibrated for walking, but a kiosk user at 640px wide and normal breathing/micro-movement will produce 5-20 pps. The `gait_symmetry_ratio` would always be skipped.

4. **Missing: upper body symmetry** — The extractor only checks hips/ankles for gait, but upper-body joint symmetry (shoulders, elbows, wrists) would work perfectly for a seated subject.

---

## Layer 3 — Signal Quality from the Webcam

### Real issues a webcam introduces for pose:

1. **2D projection problem** — MediaPipe gives pseudo-3D (`z`) estimated from image scale, not true depth. This degrades ROM accuracy for joints that move in/out of plane.

2. **Landmark jitter** — At rest, MediaPipe landmarks jitter ±3-8 pixels. For sway analysis this is signal. The Butterworth filter at 0.1–2Hz helps, but the SNR is still low.

3. **Occlusion** — In a kiosk setting, the lower body (hips/knees/ankles) is often partially occluded by a desk or cut off by the camera frame. This is why ankle visibility fails and `step_length_symmetry` often doesn't extract.

4. **Scale ambiguity** — Subject distance from camera varies, so absolute pixel coordinates are meaningless. The shoulder-width normalization for sway is the right approach.

---

## Hackathon-Winning Fix Plan

### Fix 1: Remove `gait_symmetry_ratio` as hard-required → replace with `stance_stability_score`

Stance stability is **always extractable** from a stationary subject and is actually more clinically meaningful for a kiosk setting (falls risk, postural disorders).

**Change:** `biomarker_plausibility.py` line 256  
```python
# Before
PhysiologicalSystem.SKELETAL: ["gait_symmetry_ratio"],

# After
PhysiologicalSystem.SKELETAL: ["stance_stability_score"],
```

### Fix 2: Run bilateral symmetry even when stationary

The **bilateral symmetry computation** in `_extract_gait_symmetry` does NOT require walking — it measures the range of motion each joint has over the recording period, which works for any person. Rename it `_extract_bilateral_symmetry` and always run it. Only the *step-length* sub-computation needs walking.

**Change:** `skeletal.py` — remove the `is_walking` gate on `_extract_gait_symmetry`

### Fix 3: Add upper-body symmetry (works perfectly seated)

Add shoulder/elbow/wrist symmetry score — measures postural tilt and asymmetry even when fully seated. Very clinically meaningful (scoliosis screening, shoulder injury).

### Fix 4: Add a webcam-specific posture scoring biomarker

Use the shoulder-ear-hip alignment angle to produce a **posture_score** (head forward, spine curvature). This is a direct 2D measurement from the camera that is highly accurate and doesn't require depth.

### Fix 5: Add sway entropy (more discriminating than sway_velocity)

Sample Entropy of the CoM sway signal is a better biomarker than raw velocity — it separates "rigid/stiff" sway from "healthy oscillatory" sway. This is used in clinical balance assessment.

---

## Summary: What the Final System Produces From a Webcam

| Biomarker | Method | Requires Walking? | Clinical Meaning |
|---|---|---|---|
| `stance_stability_score` | CoM sway amplitude / shoulder-width | ❌ No | Falls risk, vestibular function |
| `sway_velocity` | CoM velocity over time | ❌ No | Postural control speed |
| `posture_score` | Ear-shoulder-hip alignment angle | ❌ No | Spinal alignment, forward head |
| `bilateral_symmetry` | Left vs right joint ROM variance | ❌ No | Musculoskeletal imbalance |
| `average_joint_rom` | Elbow + knee angle range | ❌ No | Joint stiffness, mobility |
| `gait_symmetry_ratio` | Left vs right stride (from walking) | ✅ Yes | Gait disorders, Parkinson's |

The `gait_symmetry_ratio` should be **computed when walking, reported as "not assessed" when stationary** — not used as a hard gate that rejects the entire system.

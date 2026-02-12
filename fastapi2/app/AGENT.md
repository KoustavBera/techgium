# Investigation Findings: UI Feature Regression & Distance Detection

## 1. Root Cause of Missing Features
The transition from a Client-Side (JS) architecture to a Unified Backend (Python) architecture resulted in the removal of direct camera access from the browser. 
- **Legacy (`legacy2.html`)**: Used MediaPipe in JS to draw overlays and detect distance locally.
- **Unified (`index.html`)**: Displaying an MJPEG stream from the backend. The frontend is now a "Dumb Display".

## 2. Issues with Current Distance Detection
- **Inefficient Processing**: The backend `HardwareManager` is performing three separate MediaPipe passes per frame (FaceMesh, Pose, and FaceDetection). This significantly reduces FPS and can lead to laggy or skipped detection events.
- **Threshold Discrepancy**: The distance detection in `drivers.py` uses a different metric (bounding box ratio) compared to the legacy JS version (inter-ocular distance).
- **Communication Lag**: Distance warnings are sent via the `/scan-status` polling endpoint (0.5s - 1.0s interval), which is significantly slower than the real-time feedback in the legacy JS version.

## 3. Recommended Fixes
- **Consolidate Processing**: Update `HardwareManager` to perform a single MediaPipe pass and share the results across overlays and distance calculations.
- **Server-Side Overlay Enhancement**: Improve the `add_distance_warning_overlay` to be more prominent, matching the visual style of the legacy UI.
- **Exposure of Distance Metrics**: Include the raw `face_ratio` or distance estimate in the `/scan-status` response to allow the frontend to provide smoother UI feedback.

## 4. Why Distance Detection is "Failing"
Preliminary environmental checks show MediaPipe is installed. However, the redundant processing likely causes the `_capture_loop` to fall behind real-time, making the feedback feel broken or non-responsive. Additionally, the `face_detector` (bounding box) is less accurate for distance than the `face_mesh` landmarks.

## [2026-02-12] Optimization: Smooth Video Processing

### Problem
User reported video lag. Investigation revealed that `app/core/hardware/drivers.py` runs heavy MediaPipe inference (FaceMesh + Pose) synchronously on every frame, blocking the video feed.

### Changes
#### `app/core/hardware/drivers.py`
- **Threaded Capture:** Implemented background thread in `CameraCapture` to continuously read frames. This prevents buffer buildup and ensures `read_frame()` returns the latest frame instantly.
- **Selective Inference:** Updated `detect_all` to accept `active_models` list. Now it only runs the models strictly required for the current phase, instead of all of them.

#### `app/core/hardware/manager.py`
- **Phase-Gating:** Updated `_capture_loop` to pass specific model requirements based on scan phase:
    - `IDLE`: No models (or just face detection for distance).
    - `FACE_ANALYSIS`: Only `face_mesh`.
    - `BODY_ANALYSIS`: Only `pose`.
- **Frame Skipping:** Implemented logic to run inference only every 3rd frame (configurable) to free up CPU cycles for video rendering.

## [2026-02-12] UI Enhancement: Countdown Timers

### Problem
User requested better feedback during the scan process. Static messages like "Analyzing..." don't give the user a sense of how long to hold still.

### Changes
#### `app/core/hardware/manager.py`
- **Countdown Logic:** Replaced `time.sleep()` with a `_countdown()` helper that updates `_scan_status` every second.
- **Workflow Update:** 
    - Added a 10s "Preparation" timer after alignment is achieved.
    - Added a 10s "Capture" timer during the actual data recording.
- **Config Update:** Reduced capture duration from 20s to 10s as requested.

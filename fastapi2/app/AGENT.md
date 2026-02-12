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

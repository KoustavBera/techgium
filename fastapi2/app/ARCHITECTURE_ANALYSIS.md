# Analysis of Architecture Migration Plan (`app/AGENT2.md`)

## 1. Codebase & Architecture Overview

The current codebase follows a **Split-Architecture** model as described in the plan:

*   **Frontend (`frontend/index.html`)**: Heavily relies on client-side MediaPipe (JS) for FaceMesh and Pose detection. It manages the camera stream directly.
*   **Bridge (`bridge.py`)**: A standalone script that connects to hardware (Camera, Radar, ESP32) only when triggered by the `/api/v1/hardware/start-screening` endpoint.
*   **Hand-off**: The user clicks "Start", the frontend captures a frame, stops the camera, and *then* the backend (`bridge.py`) attempts to open the camera. This "stop-then-start" gap is the source of the "camera freeze" or "busy" errors.
*   **Backend (`app/main.py`)**: Acts as a coordinator but currently spawns `bridge.py` as a subprocess. It has a `video_feed` endpoint, but it's not integrated with the main hardware flow.

## 2. Plan Validation (`app/AGENT2.md`)

The proposed "Unified Kiosk-Style" architecture in `AGENT2.md` is **Technically Sound** and addresses the core issues.

### ✅ Correct Assumptions
1.  **Hardware Ownership**: Moving hardware control to a persistent `HardwareManager` singleton in the backend is the correct solution to eliminate the "hand-off" race conditions.
2.  **Extraction Portability**: I verified `app/core/extraction/*.py`. The extractors (e.g., `CardiovascularExtractor`) are largely stateless and accept data dictionaries (`face_frames`, `ris_data`). They can be easily called by `HardwareManager` without `bridge.py`.
3.  **Removal of `bridge.py`**: `bridge.py` is essentially a runner script. Its logic (connect -> loop -> extract -> post) can be fully ported to `HardwareManager`.

### ⚠️ Risks & Refinements needed

#### A. The `video_feed` Conflict
*   **Current State**: `app/main.py` already contains a `/api/v1/hardware/video-feed` endpoint that creates a *new* `cv2.VideoCapture(0)` on every request.
*   **Risk**: If `HardwareManager` opens Camera 0 at startup (as planned), the existing `video_feed` endpoint in `main.py` will fail to open the camera (Device Busy).
*   **Correction**: The `video_feed` endpoint in `main.py` MUST be refactored to consume frames from `HardwareManager.latest_frame` instead of opening the camera itself.

#### B. MediaPipe Environment
*   **Observation**: `bridge.py` uses `mediapipe` Python package.
*   **Requirement**: Ensure the backend environment where `main.py` runs has `mediapipe` installed and working. Since `bridge.py` runs in the same environment, this is likely fine, but worth double-checking `requirements.txt`.

#### C. Frontend "Dumb" Mode
*   **Complexity**: `frontend/index.html` currently has significant logic for "Phases" (Face, Pose) driven by JS State.
*   **Challenge**: migrating this to a "dumb display" means the Backend must send status updates (e.g., "Move Closer", "Face Detected") via the `/scan_status` endpoint. The frontend polling logic needs to handle these UI hints that used to be instant in JS.

#### D. State Management
*   **Missing Detail**: `bridge.py` currently POSTs results to `/api/v1/screening`. In the new design, `HardwareManager` runs inside the API process. It should probably inject results directly into `_screenings` or call the internal service method, rather than making an HTTP request to itself (though an HTTP request is a valid/simple decoupling step).

## 3. Revised Migration Steps

1.  **Create `app/core/hardware/manager.py`**: Implement the Singleton.
    *   *Crucial*: Implement `get_latest_frame()` that returns the frame in a thread-safe way for the video feed.
2.  **Refactor `app/main.py`**:
    *   Initialize `HardwareManager` on startup.
    *   **Rewrite** `/video_feed` to use `HardwareManager`.
    *   **Rewrite** `/start_scan` to trigger `HardwareManager`'s analysis thread.
3.  **Port Extractors**: Move logic from `bridge.py` -> `DataFusion` -> `HardwareManager`.
4.  **Update `frontend/index.html`**:
    *   Remove `Camera` and `FaceMesh/Pose` JS objects.
    *   Replace `<video>` with `<img src="/api/v1/hardware/video-feed">`.
    *   Update polling loop to handle status messages.

## 4. Conclusion
Argument `app/AGENT2.md` is a solid plan. It correctly identifies the architecture flaw. The only meaningful omission is the conflict with the *existing* `video_feed` implementation, which must be deleted/replaced, unlikely just "added" as implied.

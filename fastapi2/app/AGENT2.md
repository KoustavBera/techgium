# Unified Architecture Migration Plan (Option 2)

## 1. Executive Summary
This document outlines the migration from the current "Split-Process" architecture (Frontend `getUserMedia` vs. Backend `bridge.py`) to a **Unified "Kiosk-Style" Architecture**.

In this new model:
*   **The Backend (`main.py`) owns all hardware sensors** through a singleton `HardwareManager`.
*   **The Frontend (`index.html`) is a passive display** that consumes a video stream (`/video_feed`) and sends commands (`start_scan`, `stop_scan`).
*   **Analysis happens on the live stream**, eliminating the need for "hand-off" or "stopping/starting" cameras.

**Goal**: "Flawless" user experience with zero-latency startup and robust hardware management.

---

## 2. Technical Design

### A. The Hardware Manager (`app/core/hardware/manager.py`)
This new class is the heart of the system. It replaces `bridge.py` and `app/core/hardware/bridge.py` (if it existed).

**Responsibilities:**
1.  **Persistent Connection**: Opens Camera(0), Radar(COM6), Thermal(COM5) *once* at startup.
2.  **Multithreaded Capture**: Runs `threading.Thread` loops to constantly buffer the latest frames/data.
3.  **Data Diffusion**:
    *   **Hot Stream**: Provides latest JPEG frame for `/video_feed`.
    *   **Analysis Buffer**: When "Screening Mode" is active, it routes frames to `PhysiologicalSystem` extractors.

**Key Definition:**
```python
class HardwareManager:
    _instance = None # Singleton
    
    def __init__(self):
        self.camera = None
        self.radar = None
        self.thermal = None
        self.running = False
        self.latest_frame_jpeg = None
        self.analysis_mode = False
        self.buffer = []

    async def startup(self):
        # Start threads for Camera, Radar, Thermal
        pass

    def get_video_stream(self):
        # Generator for Multipart MJPEG
        while True:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + self.latest_frame_jpeg + b'\r\n')
```

### B. FastAPI Integration (`app/main.py`)
We remove the `subprocess.run(["python", "bridge.py"])` logic entirely.

**Changes:**
1.  **Lifespan Event**: Initialize `HardwareManager.startup()` on app boot.
2.  **Endpoint 1 (`GET /video_feed`)**: Streams the camera view to the browser.
3.  **Endpoint 2 (`POST /start_scan`)**: Sets `HardwareManager.analysis_mode = True`. Returns immediately. Frontend polls for completion.
4.  **Endpoint 3 (`GET /scan_status`)**: Returns progress (e.g., "Face: 50%", "Pose: Waiting") and final `screening_id`.

### C. Frontend Refactor (`frontend/index.html`)
The complex MediaPipe JS logic is removed in favor of Server-Side Rendering (SSR) of overlays (optional) or just raw video.

**Changes:**
1.  **Remove**: `getUserMedia`, `faceMesh` JS, `pose` JS.
2.  **Add**: `<img src="/api/video_feed" style="width: 100%" />`.
3.  **Logic**:
    *   Click "Start" -> calls `/start_scan`.
    *   Poll `/scan_status` every 1s.
    *   When status="Complete", show report link.

---

## 3. Migration Checklist

### Phase 1: Hardware Abstraction Layer
1.  [ ] Create `app/core/hardware/` package.
2.  [ ] **Action**: Copy `CameraCapture`, `RadarReader`, `ESP32Reader` classes from `bridge.py` to `app/core/hardware/drivers.py`.
3.  [ ] **Action**: Create `app/core/hardware/manager.py` implementing the Singleton pattern and Threading logic.

### Phase 2: FastAPI Wiring
1.  [ ] **Modify** `app/main.py`:
    *   Import `HardwareManager`.
    *   Add `lifespan` handler.
    *   Add `/video_feed` endpoint using `StreamingResponse`.
    *   Add `/hardware/control` endpoints.

### Phase 3: Extraction Logic Port
1.  [ ] **Analyze**: `bridge.py` has a main loop that runs extractors (`SkinExtractor`, `CardiovascularExtractor`, etc.) sequentially.
2.  [ ] **Port**: Move this logic into `HardwareManager.run_analysis_routine()`. This method will be called in a background thread when `/start_scan` is hit.
3.  [ ] **State**: Store the results (`system_results`, `overall_risk`) in an in-memory `_screenings` dictionary (same as `main.py` currently uses, or a shared state).

### Phase 4: Frontend "dumb" Mode
1.  [ ] Update `index.html` to remove all client-side webcam logic.
2.  [ ] Point the main view to the MJPEG stream.
3.  [ ] Re-implement the UI overlay to reflect the *Backend's* state (received via polling).

---

## 4. Risk Analysis

*   **Performance**: Python `cv2` is slower than JS MediaPipe. Expected output FPS: 15-20. **Mitigation**: Resize frames to 640x480 for analysis, maybe stream 720p.
*   **Concurrency**: FastAPI is Async, OpenCV is Blocking. **Mitigation**: Must use `asyncio.to_thread` or `ThreadPoolExecutor` for all hardware interaction. The `video_feed` generator must yield clearly to avoid blocking the event loop.
*   **Hardware Locks**: If the server crashes/reloads (WatchFiles), the camera might not release. **Mitigation**: Robust `try...finally` blocks in the lifespan handler to ensure `cap.release()` is called.

## 5. Why this is "Flawless"
*   **No "Permission" Popups**: Browser just loads an image.
*   **Instant On**: Camera is already warm when user clicks "Start".
*   **Synchronization**: Radar and Thermal are perfectly aligned with Video because the same process holds the timestamps.

**Proceeding with Phase 1...**

# Camera Testing Guide

## Quick Start

Your webcam test script is ready! Run:

```bash
.\benv\Scripts\python.exe camera_test.py
```

## What It Does

1. **Opens your webcam** - 10 second recording
2. **Extracts biomarkers** using MediaPipe:
   - Cardiovascular: Heart rate (from motion)
   - CNS: Gait, balance (from pose)
   - Eyes: Blink, gaze tracking
   - Skeletal: Posture analysis
3. **Sends to API** - your screening endpoint
4. **Generates reports** - Patient & Doctor PDFs

## Requirements (Already Installed ✅)

- ✅ opencv-python: 4.13.0.90
- ✅ mediapipe: 0.10.32
- ✅ API server running on port 8000

## Usage Tips

- **Lighting**: Use good lighting for best extraction
- **Position**: Sit ~1 meter from camera, full upper body visible
- **Movement**: Breathe normally, minimal movement
- **Duration**: Default 10 seconds (can be changed in code)

## Expected Output

```
📹 Opening webcam...
🎥 Recording... Look at the camera
✓ Captured 300 frames

🔬 Extracting biomarkers...
  → Cardiovascular... ✓ Found 4 biomarkers
  → CNS... ✓ Found 3 biomarkers
  → Eyes... ✓ Found 5 biomarkers

📤 Sending to API
✅ Screening Complete!
   ID: SCR-XXXXXXXX
   Overall Risk: low

📄 Generating reports...
✓ Patient Report: PR-XXXXXXXX
✓ Doctor Report: DR-XXXXXXXX
```

## Troubleshooting

**Camera not opening:**
- Check if another app is using the camera
- Try changing camera index in code: `cv2.VideoCapture(1)`

**No biomarkers extracted:**
- Ensure good lighting
- Make sure full upper body is visible
- Check console for specific errors

**API connection error:**
- Make sure uvicorn server is running
- Check it's on port 8000

## Next Steps

Once tested with camera, add RIS hardware later by:
1. Connect RIS sensor
2. Modify script to include RIS data
3. Extractors already support RIS input!

# Hardware Architecture Design for Non-Contact Physiological Measurement System

> **Target:** Hackathon prototype system for walk-through health screening chamber  
> **Budget:** 20-30k INR (~$250-350 USD)  
> **Goal:** 70-90% accuracy match with software algorithms  
> **Architecture:** Split-USB Topology (Modular, Safe, Hackathon-Ready)

---

## System Overview

This design uses a **Split-USB Architecture** where all sensors connect to the laptop via a single USB hub. This eliminates complex UART wiring between components and provides a safer, more modular setup.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| USB Hub as central connection | Single laptop USB port, easy cable management |
| Seeed Radar Kit (standalone USB) | Built-in MCU, no UART wiring to ESP32 needed |
| ESP32 NodeMCU for thermal only | Simplified role, just I2C → USB serial bridge |
| MLX90640 55° FOV | Narrow FOV for face-focused thermal imaging |

---

## System Block Diagram

> [!IMPORTANT]
> **Split-USB Architecture:** All devices connect through a USB Hub. The laptop runs `bridge.py` which reads from **two separate COM ports** (Radar and ESP32).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WALK-THROUGH HEALTH SCREENING CHAMBER                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                     YOUR LAPTOP / PC (HOST)                               │  │
│   │                                                                           │  │
│   │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│   │   │                      bridge.py                                   │    │  │
│   │   │                                                                  │    │  │
│   │   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │  │
│   │   │  │ CameraCapture│  │ RadarReader  │  │ ESP32Reader  │           │    │  │
│   │   │  │ (OpenCV)     │  │ (Serial)     │  │ (Serial)     │           │    │  │
│   │   │  │ Webcam       │  │ COM_A        │  │ COM_B        │           │    │  │
│   │   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │    │  │
│   │   │         │                 │                 │                    │    │  │
│   │   │         └─────────────────┴─────────────────┘                    │    │  │
│   │   │                           │                                      │    │  │
│   │   │                    DataFusion                                    │    │  │
│   │   │                           │                                      │    │  │
│   │   │                           ▼                                      │    │  │
│   │   │               POST /api/v1/screening                             │    │  │
│   │   └─────────────────────────────────────────────────────────────────┘    │  │
│   │                                      ▲                                    │  │
│   │                                      │ USB                                │  │
│   │   ┌──────────────────────────────────┴────────────────────────────────┐  │  │
│   │   │                    4-PORT USB 2.0 HUB                              │  │  │
│   │   │               (Powered from Laptop USB Port)                       │  │  │
│   │   └───┬─────────────────────┬─────────────────────┬───────────────────┘  │  │
│   │       │                     │                     │                       │  │
│   └───────┼─────────────────────┼─────────────────────┼───────────────────────┘  │
│           │                     │                     │                          │
│           ▼                     ▼                     ▼                          │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐                  │
│   │    WEBCAM     │     │  SEEED RADAR  │     │ ESP32 NodeMCU │                  │
│   │  Logitech C270│     │  MR60BHA2 Kit │     │  (30-Pin)     │                  │
│   │   or Laptop   │     │               │     │  CP2102 USB   │                  │
│   │   built-in    │     │ Built-in XIAO │     │               │                  │
│   │               │     │ ESP32C6 MCU   │     │      ┌────┐   │                  │
│   │  ┌─────────┐  │     │               │     │      │I2C │   │                  │
│   │  │ Video   │  │     │ ┌───────────┐ │     │      └──┬─┘   │                  │
│   │  │ Frames  │  │     │ │ Breathing │ │     │         │     │                  │
│   │  │ @30fps  │  │     │ │ Heartbeat │ │     │         ▼     │                  │
│   │  └─────────┘  │     │ │ Data      │ │     │  ┌──────────┐ │                  │
│   │               │     │ └───────────┘ │     │  │ MLX90640 │ │                  │
│   │               │     │               │     │  │ Thermal  │ │                  │
│   │               │     │ Serial/COM_A  │     │  │ 55° FOV  │ │                  │
│   │               │     │ @115200 baud  │     │  └──────────┘ │                  │
│   └───────────────┘     └───────────────┘     │               │                  │
│                                               │ Serial/COM_B  │                  │
│                                               │ @115200 baud  │                  │
│                                               └───────────────┘                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

SUMMARY OF CONNECTIONS:
═══════════════════════
✓ USB Hub         → USB      → Laptop (single port from laptop)
✓ Webcam          → USB      → Hub Port 1 → Video to OpenCV
✓ Seeed Radar Kit → USB      → Hub Port 2 → Serial/COM_A (Breathing/Heartbeat)
✓ ESP32 NodeMCU   → USB      → Hub Port 3 → Serial/COM_B (Thermal JSON)
✓ MLX90640        → I2C      → ESP32 NodeMCU (D21 SDA, D22 SCL)
```

---

## 1. Power Supply Design

> [!NOTE]
> **USB Hub Powers Everything:** The laptop USB port powers the hub, which distributes power to all connected devices. No separate power supply needed for the hackathon prototype.

### USB Hub Powered Configuration (Recommended)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    USB HUB POWERED CONFIGURATION (Simplest)                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   YOUR LAPTOP                                                                    │
│   └── USB 3.0 Port ──► 4-Port USB 2.0 Hub (draws up to 500mA per port)          │
│                            │                                                     │
│                            ├── Port 1 ──► Webcam ─────────────── ~100mA         │
│                            │                                                     │
│                            ├── Port 2 ──► Seeed Radar Kit ────── ~300mA         │
│                            │              (Built-in XIAO ESP32C6)                │
│                            │                                                     │
│                            └── Port 3 ──► ESP32 NodeMCU ────────── ~150mA       │
│                                           │                                      │
│                                           └── 3.3V ──► MLX90640 ─── ~25mA       │
│                                                                                  │
│   Total Power Draw: ~575mA (well within USB 3.0 900mA limit)                    │
│                                                                                  │
│   If using USB 2.0 port (500mA limit), use a POWERED USB hub instead           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Power Budget

| Device | Voltage | Current (typ) | Current (max) | Power |
|--------|---------|---------------|---------------|-------|
| Webcam (Logitech C270) | 5V USB | 100mA | 200mA | 1.0W |
| Seeed Radar Kit MR60BHA2 | 5V USB | 200mA | 350mA | 1.75W |
| ESP32 NodeMCU (30-Pin) | 5V USB | 80mA | 150mA | 0.75W |
| MLX90640 (via ESP32 3.3V) | 3.3V | 18mA | 25mA | 0.08W |
| **Total** | - | ~400mA | ~725mA | **~3.6W** |

> [!TIP]
> For reliable operation, use a **powered USB hub** if your laptop has weak USB power delivery. Look for hubs with external 5V/2A power adapters.

---

## 2. Circuit Schematic - Sensor Connections

### 2.1 ESP32 NodeMCU (Thermal Only)

The ESP32 NodeMCU's sole purpose is to read the MLX90640 thermal camera via I2C and send JSON data over USB serial.

```
                          ESP32 NodeMCU 30-Pin (CP2102)
                          ┌──────────────────────────┐
                    3V3 ──┤ 3V3                  GND ├─── GND
                    RST ──┤ EN                   D23 ├─── NC
               (NC) VP ───┤ VP                   D22 ├─── I2C SCL ──► MLX90640
               (NC) VN ───┤ VN                   D1  ├─── USB TX (to PC)
               (NC) D34 ──┤ D34                  D3  ├─── USB RX (from PC)
               (NC) D35 ──┤ D35                  D21 ├─── I2C SDA ──► MLX90640
               (NC) D32 ──┤ D32                  D19 ├─── NC
               (NC) D33 ──┤ D33                  D18 ├─── NC
               (NC) D25 ──┤ D25                  D5  ├─── LED Activity (Green)
               (NC) D26 ──┤ D26                  D17 ├─── NC (was UART2 TX)
               (NC) D27 ──┤ D27                  D16 ├─── NC (was UART2 RX)
               (NC) D14 ──┤ D14                  D4  ├─── LED Status (Blue)
               (NC) D12 ──┤ D12                  D0  ├─── Boot (leave floating)
                    GND ──┤ GND                  D2  ├─── NC
               (NC) D13 ──┤ D13                  D15 ├─── NC
                    VIN ──┤ VIN (5V from USB)   GND ├─── GND
                          └──────────────────────────┘

SIMPLIFIED WIRING (Only 4 wires!):
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   MLX90640 Breakout (55° FOV)                                   │
│   ┌────────────────────────┐                                    │
│   │  VIN ─────────────────────── ESP32 3V3                      │
│   │  GND ─────────────────────── ESP32 GND                      │
│   │  SDA ─────────────────────── ESP32 D21 (GPIO21)             │
│   │  SCL ─────────────────────── ESP32 D22 (GPIO22)             │
│   └────────────────────────┘                                    │
│                                                                  │
│   Optional: 4.7kΩ pull-up resistors on SDA/SCL to 3.3V          │
│   (Most breakout boards have these built-in)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Seeed Radar Kit (MR60BHA2) - Standalone USB

The Seeed MR60BHA2 kit is a self-contained unit with a built-in XIAO ESP32C6 microcontroller. It connects directly to the USB hub and appears as a serial COM port.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Seeed MR60BHA2 60GHz mmWave Breathing & Heartbeat Kit         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────┐    ┌─────────────────┐            │   │
│   │   │ 60GHz mmWave    │◄──►│ XIAO ESP32C6   │            │   │
│   │   │ Radar Sensor    │    │ (Built-in MCU)  │            │   │
│   │   │                 │    │                 │            │   │
│   │   │ • Breathing     │    │ Processes radar │            │   │
│   │   │ • Heartbeat     │    │ data and sends  │            │   │
│   │   │ • Presence      │    │ via USB Serial  │            │   │
│   │   └─────────────────┘    └────────┬────────┘            │   │
│   │                                   │                      │   │
│   │                                   │ USB-C                │   │
│   │                                   ▼                      │   │
│   │                          ┌─────────────────┐            │   │
│   │                          │ To USB Hub      │            │   │
│   │                          │ (appears as COM │            │   │
│   │                          │  port on PC)    │            │   │
│   │                          └─────────────────┘            │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   No wiring required to ESP32 NodeMCU!                          │
│   Power: 5V from USB                                            │
│   Data: Serial @ 115200 baud (configurable)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Camera Connection

The webcam connects directly to the USB hub. No wiring to any microcontroller.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Webcam Options:                                               │
│                                                                  │
│   Option A: External USB Webcam (Logitech C270 or similar)      │
│   ┌────────────────────────┐                                    │
│   │                        │                                    │
│   │   Logitech C270        │───► USB Cable ───► USB Hub        │
│   │   720p/30fps           │                                    │
│   │                        │                                    │
│   └────────────────────────┘                                    │
│                                                                  │
│   Option B: Laptop Built-in Webcam                              │
│   • No additional hardware needed                                │
│   • Use camera index 0 in OpenCV                                │
│   • Frees up one USB hub port                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Pin Assignment Table

### ESP32 NodeMCU Pinout (Thermal Camera Only)

| ESP32 Pin | GPIO | Function | Interface | Voltage | Connected To |
|-----------|------|----------|-----------|---------|--------------|
| 3V3 | - | Power Out | Power | 3.3V | MLX90640 VIN |
| GND | - | Ground | Power | 0V | MLX90640 GND |
| D21 | 21 | I2C SDA | I2C | 3.3V | MLX90640 SDA |
| D22 | 22 | I2C SCL | I2C | 3.3V | MLX90640 SCL |
| D4 | 04 | LED Status | GPIO Out | 3.3V | Blue LED (optional) |
| D5 | 05 | LED Activity | GPIO Out | 3.3V | Green LED (optional) |
| VIN | - | Power In | USB | 5V | From USB Hub |
| D1/D3 | 01/03 | USB Serial | UART0 | 3.3V | USB-to-Serial (CP2102) |

> [!NOTE]
> **Simplified Pinout:** Only 4 GPIO pins are used (D21, D22 for I2C + optional D4, D5 for LEDs). GPIO 16/17 (UART2) are no longer used since the radar is on a separate USB connection.

---

## 4. Communication Protocol Summary

| Device | Interface | Protocol | Speed | Data | COM Port Example |
|--------|-----------|----------|-------|------|------------------|
| Webcam | USB | UVC | 30 fps | Video frames | N/A (OpenCV) |
| Seeed Radar Kit | USB Serial | UART | 115200 baud | Breathing/HR JSON | COM3 (Windows) |
| ESP32 NodeMCU | USB Serial | UART | 115200 baud | Thermal JSON | COM4 (Windows) |

### Data Format: Seeed Radar Kit → bridge.py (COM_A)

```json
{
  "timestamp": 1707050232,
  "radar": {
    "respiration_rate": 15.2,
    "heart_rate": 72,
    "breathing_depth": 0.73,
    "presence_detected": true
  }
}
```

> [!WARNING]
> **Verify Radar Protocol:** The exact JSON format from the Seeed MR60BHA2 kit may differ. Refer to [Seeed MR60BHA2 documentation](https://wiki.seeedstudio.com/mmwave_kit/) for the actual output format and adjust `bridge.py` accordingly.

### Data Format: ESP32 NodeMCU → bridge.py (COM_B)

The ESP32 thermal bridge firmware outputs detailed clinical biomarkers at **8 fps**:

```json
{
  "timestamp": 12345,
  "thermal": {
    "fever": {
      "canthus_temp": 36.42,
      "neck_temp": 36.85,
      "neck_stability": 0.45,
      "fever_risk": 0
    },
    "diabetes": {
      "canthus_temp": 36.42,
      "canthus_stability": 0.32,
      "risk_flag": 0
    },
    "cardiovascular": {
      "thermal_asymmetry": 0.285,
      "left_cheek_temp": 35.92,
      "right_cheek_temp": 36.21,
      "risk_flag": 0
    },
    "inflammation": {
      "hot_pixel_pct": 3.25,
      "face_mean_temp": 35.78,
      "detected": 0
    },
    "autonomic": {
      "nose_temp": 34.52,
      "forehead_temp": 35.85,
      "stress_gradient": 1.33,
      "stress_flag": 0
    },
    "metadata": {
      "face_detected": 1,
      "valid_rois": 7
    }
  }
}
```

### Data Flow: bridge.py → FastAPI

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW ARCHITECTURE                             │
│                                                                               │
│   ┌───────────────┐  Video    ┌─────────────────┐                            │
│   │    Webcam     │─────────►│ CameraCapture    │                            │
│   │               │          │ (rPPG, Pose)     │                            │
│   └───────────────┘          └────────┬─────────┘                            │
│                                       │                                       │
│   ┌───────────────┐  COM_A   ┌────────┼─────────┐                            │
│   │ Seeed Radar   │────────►│ RadarReader      │                            │
│   │ Kit (USB)     │          │ (Breathing Data) │                            │
│   └───────────────┘          └────────┼─────────┘                            │
│                                       │                                       │
│   ┌───────────────┐  COM_B   ┌────────┼─────────┐                            │
│   │ ESP32 NodeMCU │────────►│ ESP32Reader      │                            │
│   │ (USB)         │          │ (Thermal Data)   │                            │
│   └───────────────┘          └────────┼─────────┘                            │
│                                       │                                       │
│                                       ▼                                       │
│                         ┌─────────────────────────┐                          │
│                         │     DataFusion          │                          │
│                         │                         │                          │
│                         │ • Merge rPPG biomarkers │                          │
│                         │ • Merge radar data      │                          │
│                         │ • Merge thermal data    │                          │
│                         │ • Build API request     │                          │
│                         └───────────┬─────────────┘                          │
│                                     │                                         │
│                                     ▼                                         │
│                         ┌─────────────────────────┐                          │
│                         │  POST /api/v1/screening │                          │
│                         │     → FastAPI Server    │                          │
│                         └─────────────────────────┘                          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Bill of Materials (BOM)

> [!IMPORTANT]
> **Updated BOM with specific SKUs.** Your laptop is the brain - no Raspberry Pi needed.

| # | Component | Description | SKU | Qty | Unit Price (INR) | Total (INR) | Source |
|---|-----------|-------------|-----|-----|------------------|-------------|--------|
| 1 | ESP32 NodeMCU Dev Board | 30-Pin, CP2102 USB-Serial | TIFCC01062 | 1 | ₹450 | ₹450 | Robu.in |
| 2 | MLX90640 Thermal Camera | **55° FOV** IR Array Breakout | TIFCM0036 | 1 | ₹3,500 | ₹3,500 | Robu.in |
| 3 | Seeed MR60BHA2 Radar Kit | 60GHz mmWave Breathing & Heartbeat | R1959399 | 1 | ₹8,000 | ₹8,000 | Seeed Studio |
| 4 | Logitech C270 Webcam | HD 720p USB Camera | C270 | 1 | ₹1,500 | ₹1,500 | Amazon.in |
| 5 | 4-Port USB 2.0 Hub | Standard USB hub (powered optional) | Generic | 1 | ₹300 | ₹300 | Amazon.in |
| 6 | USB-A to USB-C Cable | For Seeed Radar Kit | Generic | 1 | ₹150 | ₹150 | Amazon.in |
| 7 | USB-A to Micro-USB Cable | For ESP32 NodeMCU | Generic | 1 | ₹100 | ₹100 | Amazon.in |
| 8 | Dupont Jumper Wires | Female-to-Female, 20cm | F-F 40pcs | 1 | ₹80 | ₹80 | Local |
| 9 | LEDs (Blue, Green) | 5mm, optional status indicators | 5mm LED | 4 | ₹5 | ₹20 | Local |
| 10 | Resistors (330Ω) | LED current limiting | 1/4W | 4 | ₹2 | ₹8 | Local |
| | | | | | **Total** | **₹14,108** | |

> [!TIP]
> **Budget-Conscious Alternative (~₹6,000):**
> - Use laptop built-in webcam (saves ₹1,500)
> - Use LD2410 radar module instead of Seeed kit (~₹800 vs ₹8,000) - simpler but less accurate
> - Total: ~₹5,358

---

## 6. Mechanical Mounting Considerations

```
                    WALK-THROUGH CHAMBER LAYOUT (Top View)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │                        ENTRY                                │
    │                          │                                  │
    │    ┌─────────────────────▼─────────────────────────────┐   │
    │    │                                                    │   │
    │    │        [SEEED RADAR KIT]                          │   │
    │    │        Height: 1.0m (chest level)                  │   │
    │    │        Distance: 0.5-1.5m from subject             │   │
    │    │        Angle: Perpendicular to chest               │   │
    │    │                                                    │   │
    │    │    ────────────── WALK PATH ────────────────       │   │
    │    │                      ↑                             │   │
    │    │               [MLX90640 THERMAL]                   │   │
    │    │               Height: 1.5m (face level)            │   │
    │    │               Distance: 0.5-1.0m                   │   │
    │    │               FOV: 55° (narrow, face-focused)      │   │
    │    │                      ↑                             │   │
    │    │              [WEBCAM / CAMERA]                     │   │
    │    │              Height: 1.5m (face) / 1.0m (body)     │   │
    │    │              Distance: 1.0-2.0m from subject       │   │
    │    │                                                    │   │
    │    │    ────────────── WALK PATH ────────────────       │   │
    │    │                                                    │   │
    │    │    [CONTROL STATION]                               │   │
    │    │    Laptop + USB Hub + ESP32                        │   │
    │    │    Side-mounted, accessible for maintenance        │   │
    │    │                                                    │   │
    │    └────────────────────────────────────────────────────┘   │
    │                          │                                  │
    │                        EXIT                                 │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Lighting Requirements:
• Diffused LED panels (3000-4000K, CRI>90)
• Avoid direct reflections on subject's face
• 200-500 lux at face level for optimal rPPG
• Consistent lighting reduces motion artifacts
```

### Sensor Mounting Specifications

| Sensor | Height | Distance | Angle | Notes |
|--------|--------|----------|-------|-------|
| Webcam (Face) | 1.5m | 0.5-1.0m | Eye level | Phase 1: Close-up face for rPPG |
| Webcam (Body) | 1.0m | 2.0-3.0m | Full body | Phase 2: Gait analysis |
| MLX90640 | 1.5m | 0.5-1.0m | Face-level | 55° FOV focused on forehead/cheeks |
| Seeed Radar | 1.0m | 0.5-1.5m | Chest-level | Perpendicular to chest wall |

---

## 7. Software Integration

### Key Files

| File | Purpose |
|------|---------|
| [bridge.py](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/bridge.py) | Main integration - reads from TWO serial ports + camera |
| [esp32_thermal_bridge.ino](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/firmware/esp32_thermal_bridge.ino) | ESP32 firmware (thermal camera only) |

### Running the Bridge

```bash
# Start the FastAPI server first
uvicorn app.main:app --reload

# Run bridge with all hardware (Windows example)
python bridge.py --radar-port COM3 --port COM4 --camera 0

# Run with simulated data (no hardware)
python bridge.py --simulate
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--radar-port` | COM3 | Serial port for Seeed Radar Kit |
| `--port` | COM4 | Serial port for ESP32 NodeMCU (thermal) |
| `--camera` | 0 | Camera index for OpenCV |
| `--api-url` | http://localhost:8000 | FastAPI server URL |
| `--simulate` | False | Use simulated sensor data |
| `--patient-id` | WALKTHROUGH_001 | Patient identifier |

---

## 8. Verification Plan

### Automated Tests

| Test | Method | Pass Criteria |
|------|--------|---------------|
| USB Hub detection | Device Manager / `lsusb` | All 3 devices visible |
| Camera detection | `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` | Returns `True` |
| Radar serial | `python -c "import serial; print(serial.Serial('COM3', 115200))"` | No exception |
| ESP32 serial | `python -c "import serial; print(serial.Serial('COM4', 115200))"` | No exception |
| Simulated run | `python bridge.py --simulate` | Screening ID returned |

### Manual Verification

1. **Camera Test:**
   - Run `python camera_test.py`
   - Verify face detection and pose landmarks
   - Check frame rate: should be 25-30 fps

2. **Thermal Camera Test:**
   - Point MLX90640 at known temperature (body ~36.5°C)
   - Verify ±1°C accuracy via ESP32 serial output

3. **Radar Test:**
   - Stand 1m from radar, breathe normally
   - Check Seeed Radar serial output for respiration_rate
   - Verify matches manual count (±1 BPM)

4. **Full Integration Test:**
   ```bash
   python bridge.py --radar-port COM3 --port COM4 --camera 0
   ```
   - Verify all 3 data sources appear in API request
   - Check PDF report generation

---

## 9. Design Notes

### Component Selection Rationale

| Component | Why Chosen | Alternatives |
|-----------|------------|--------------|
| ESP32 NodeMCU (30-Pin) | Simple, CP2102 driver works on Windows, cheap | Wemos D1 Mini, Arduino Nano |
| MLX90640 55° FOV | Narrow FOV better for face focus, I2C, radiometric | AMG8833 (8×8 too low), 110° FOV (too wide) |
| Seeed MR60BHA2 | All-in-one solution, USB plug-and-play, built-in MCU | TI IWR6843 (needs UART wiring), LD2410 (simpler) |
| 4-Port USB Hub | Central connection, single laptop USB port used | USB 3.0 Hub (overkill), direct connections (uses 3 ports) |

### Hardware Limitations & Mitigations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Camera motion blur | rPPG artifacts | Fixed mount, software stabilization |
| Ambient light variation | rPPG SNR degradation | Controlled LED lighting |
| Thermal drift | Temperature offset | Calibration every 30 min |
| USB bandwidth | Video lag with 3 devices | Use USB 3.0 hub if issues arise |
| Two serial ports | More complex software | RadarReader class in bridge.py |

### Safety Considerations

- **Electrical:** All USB-powered, low-voltage DC only (5V max)
- **EMI:** 60GHz radar FCC/CE compliant, sealed module
- **Optical:** Camera uses visible light only, no IR laser
- **Thermal:** Passive sensors, no active heating

---

## 10. Known Issues & Items for Review

> [!WARNING]
> **Seeed Radar Data Format Unknown:** The exact JSON output format from the MR60BHA2 kit needs verification. The `bridge.py` `RadarReader` class may need adjustment based on actual output. Refer to [Seeed Wiki](https://wiki.seeedstudio.com/mmwave_kit/) for protocol documentation.

> [!IMPORTANT]
> **MAX30102 Pulse Oximeter Removed:** The new Split-USB design focuses on webcam, radar, and thermal sensors. If SpO₂/pulse oximetry is still required:
> - Add MAX30102 to ESP32 I2C bus (address 0x57)
> - Update ESP32 firmware and bridge.py to include pulse_ox data
> - This adds ~₹250 to the BOM

> [!NOTE]
> **MLX90640 FOV Selection:** The 55° FOV version (TIFCM0036) is specified for face-focused thermal imaging. The 110° FOV version covers a wider area but with lower resolution per subject. Ensure you order the correct SKU.

> [!CAUTION]
> **COM Port Assignment:** Windows assigns COM port numbers dynamically. After plugging in devices:
> 1. Open Device Manager → Ports (COM & LPT)
> 2. Note which device is on which COM port
> 3. Update `bridge.py` arguments accordingly
> 4. Consider setting fixed COM ports via Device Manager → Port Settings → Advanced

---

## Summary

This **Split-USB Architecture** provides a:

- ✅ **Simpler** hardware setup (no UART wiring between components)
- ✅ **Modular** design (each sensor is independent USB device)
- ✅ **Safer** approach (no level shifters, no custom wiring)
- ✅ **Hackathon-ready** (plug-and-play USB connections)
- ✅ **Low cost** (~₹14,000 for sensors + ESP32 + hub)
- ✅ **Compatible** with existing FastAPI software extractors

The bridge.py script reads from **two serial ports** (Radar on COM_A, ESP32/Thermal on COM_B) plus the webcam, fuses the data, and sends screening requests to the FastAPI server.

---

## Changelog

### 2026-02-09
- **Fixed `parse_radar_binary` indentation** in `bridge.py` - was incorrectly at module level instead of inside `RadarReader` class
- **Updated `generate_simulated_esp32_data`** to match the HARDWARE.md thermal biomarker structure (fever, diabetes, cardiovascular, inflammation, autonomic categories)
- **Radar Binary Protocol Verified**: MR60BHA2 sends binary frames with 0x02 0x81 header, followed by respiration and heart rate as little-endian floats

### 2026-02-07
- Integrated ESP32 thermal firmware v2
- Added JSON parsing in ESP32Reader (bridge.py)
- Distributed biomarkers to appropriate physiological systems

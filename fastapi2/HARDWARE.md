# Hardware Architecture Design for Non-Contact Physiological Measurement System

> **Target:** Hackathon prototype system for walk-through health screening chamber  
> **Budget:** 20-30k INR (~$250-350 USD)  
> **Goal:** 70-90% accuracy match with software algorithms

## User Review Required

> [!IMPORTANT]
> **Sensor Selection Decision:** The design uses mmWave radar (60 GHz) as the primary respiratory sensor. This replaces RIS (Radar Impedance Spectroscopy) which is unavailable. Please confirm this substitution is acceptable.

> [!IMPORTANT]
> **Thermal Camera Option:** MLX90640 (32×24 pixels, ~₹3,500) vs AMG8833 (8×8 pixels, ~₹1,800). MLX90640 recommended for better thermal mapping. Confirm preference.

---

## Proposed Design

### System Block Diagram

> [!IMPORTANT]
> **Camera Connection:** The webcam connects DIRECTLY to your laptop/PC via USB, NOT to the ESP32. The ESP32 only handles the other sensors (thermal, radar, pulse-ox).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WALK-THROUGH HEALTH SCREENING CHAMBER                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                  YOUR LAPTOP / PC (HOST)                                  │  │
│   │  ┌─────────────────────────────────────────────────────────────────────┐ │  │
│   │  │                                                                      │ │  │
│   │  │   ┌─────────────────┐      ┌────────────────────────────────────┐   │ │  │
│   │  │   │ Logitech C270   │      │           bridge.py                │   │ │  │
│   │  │   │ / Laptop Webcam │─USB─►│  • CameraCapture (OpenCV)          │   │ │  │
│   │  │   │ 1080p/30fps     │      │  • ESP32Reader (serial)            │   │ │  │
│   │  │   └─────────────────┘      │  • DataFusion → FastAPI            │   │ │  │
│   │  │                            └───────────────┬────────────────────┘   │ │  │
│   │  │   ┌─────────────────┐                      │                        │ │  │
│   │  │   │ ESP32 DevKit    │──────USB Serial──────┘                        │ │  │
│   │  │   │ (via USB cable) │      @ 115200 baud                            │ │  │
│   │  │   └─────────────────┘                                               │ │  │
│   │  │                                                                      │ │  │
│   │  └─────────────────────────────────────────────────────────────────────┘ │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│                                      │ USB (long cable if needed)                │
│                                      ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                     ESP32 + SENSORS (Separate Station)                    │  │
│   │                                                                           │  │
│   │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │  │
│   │   │ MLX90640        │  │ 60GHz mmWave    │  │ MAX30102        │          │  │
│   │   │ Thermal Camera  │  │ Radar           │  │ Pulse Oximeter  │          │  │
│   │   │ (I2C @ 0x33)    │  │ (UART2)         │  │ (I2C @ 0x57)    │          │  │
│   │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │  │
│   │            │                    │                    │                    │  │
│   │            └────────────────────┼────────────────────┘                    │  │
│   │                                 ▼                                         │  │
│   │                    ┌────────────────────────┐                             │  │
│   │                    │   ESP32-WROOM-32D      │                             │  │
│   │                    │   (Sensor Hub)         │                             │  │
│   │                    │   Sends JSON @ 20 Hz   │                             │  │
│   │                    └────────────────────────┘                             │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

SUMMARY OF CONNECTIONS:
═══════════════════════
✓ Webcam/Camera    → USB      → Laptop (OpenCV reads directly)
✓ ESP32            → USB      → Laptop (Serial JSON data)
✓ Thermal Camera   → I2C      → ESP32
✓ mmWave Radar     → UART     → ESP32
✓ Pulse Oximeter   → I2C      → ESP32
```


---

### 1. Power Supply Design

> [!NOTE]
> **Using Your Laptop as the Brain:** Your laptop provides processing power and camera. You only need to power the ESP32 + sensors (via laptop USB or a small 5V supply).

**Option A: Laptop-Powered Setup (Recommended for Hackathon)**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LAPTOP-POWERED CONFIGURATION (Simplest)                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   YOUR LAPTOP                                                                    │
│   ├── USB Port 1 ──► Webcam (or built-in) ─── Draws ~500mA from laptop          │
│   │                                                                              │
│   └── USB Port 2 ──► ESP32 DevKit ─── Powers ESP32 + sensors via USB            │
│                          │                                                       │
│                          └── 5V USB power shared to:                             │
│                              ├── ESP32 (3.3V onboard regulator)                  │
│                              ├── MLX90640 (via 3.3V breakout)                    │
│                              └── MAX30102 (via 3.3V)                             │
│                                                                                  │
│   Note: mmWave radar may need separate 5V/1A supply if power hungry             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Option B: Standalone Power Supply (For Full Chamber Setup)**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           POWER DISTRIBUTION NETWORK                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   AC MAINS                                                                       │
│   220V/50Hz ──► [Wall Adapter 12V/2A] ──► DC 12V RAIL                           │
│                      │                                                           │
│                      ▼                                                           │
│   ┌─────────────────────────────┐  (Laptop has its own power adapter)           │
│   │ Buck Converter              │                                                │
│   │ LM2596 (12V→5V/2A)          │                                                │
│   └─────────────┬───────────────┘                                                │
│                 │                                                                │
│                 ▼                                                                │
│   ┌─────────────────────────────┐  ┌─────────────────────────────┐              │
│   │      5V Rail (Sensors)      │  │     3.3V Rail (Logic)       │              │
│   │  ┌────────────────────────┐ │  │  ┌────────────────────────┐ │              │
│   │  │ mmWave Radar      1.0A │ │  │  │ ESP32 Module     500mA │ │              │
│   │  │ ESP32 (if not USB) 0.5A│ │  │  │ MLX90640         80mA  │ │              │
│   │  └────────────────────────┘ │  │  │ MAX30102         50mA  │ │              │
│   │  Total: ~1.5A               │  │  └────────────────────────┘ │              │
│   └─────────────────────────────┘  │  Total: ~700mA              │              │
│                                    └─────────────────────────────┘              │
│   DECOUPLING:                                                                    │
│   • 10μF bulk cap at each regulator output                                      │
│   • 100nF ceramic at each IC VCC pin                                            │
│   • 10μF at ESP32 3.3V input                                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Power Budget (ESP32 + Sensors Only):**

| Component | Voltage | Current (typ) | Current (max) | Power |
|-----------|---------|---------------|---------------|-------|
| ESP32-WROOM-32 | 3.3V | 80mA | 500mA | 1.65W |
| MLX90640 | 3.3V | 18mA | 80mA | 0.26W |
| MAX30102 | 3.3V | 20mA | 50mA | 0.17W |
| mmWave Radar | 5V | 700mA | 1A | 5W |
| LEDs/Misc | 3.3V | 50mA | 100mA | 0.33W |
| **Total** | - | - | - | **~7.5W** |

> [!TIP]
> For hackathon: Just plug ESP32 into laptop USB. A 12V/1A adapter is sufficient for standalone sensor power if needed.

---

### 2. Circuit Schematic - Sensor Connections

#### 2.1 ESP32 Core Circuit

```
                                ESP32-WROOM-32D
                          ┌──────────────────────────┐
                    EN ───┤1   EN               GND  ├─── GND
              GPIO36/VP ───┤2   VP               3V3  ├─── 3.3V        
              GPIO39/VN ───┤3   VN          GPIO23   ├─── NC
              GPIO34/A6 ───┤4   34          GPIO22   ├─── I2C SCL ──►
              GPIO35/A7 ───┤5   35          GPIO01/TX├─── UART0 TX (Debug/Host)
              GPIO32/A4 ───┤6   32          GPIO03/RX├─── UART0 RX (Debug/Host)
              GPIO33/A5 ───┤7   33          GPIO21   ├─── I2C SDA ──►
              GPIO25/A18───┤8   25          GPIO19   ├─── Test Point 2
              GPIO26/A19───┤9   26          GPIO18   ├─── Test Point 1
              GPIO27/A17───┤10  27          GPIO05   ├─── LED Activity (Green)
              GPIO14/A16───┤11  14          GPIO17/TX├─── UART2 TX (Radar) ──►
              GPIO12/A15───┤12  12          GPIO16/RX├─── UART2 RX (Radar) ◄──
                    GND ───┤13  GND         GPIO04   ├─── LED Status (Blue)
              GPIO13/A14───┤14  13          GPIO00   ├─── Boot (10k pull-up)
              SD2/GPIO09───┤15  SD2         GPIO02   ├─── NC (Boot strapping)
              SD3/GPIO10───┤16  SD3         GPIO15/A13├── Test Point 3
              CMD/GPIO11───┤17  CMD          SD1/GPIO8├── NC
              CLK/GPIO06───┤18  CLK          SD0/GPIO7├── NC
              SD0/GPIO07───┤19  SD0               GND ├─── GND
                          └──────────────────────────┘

Power Filtering:
    3.3V ───┬─── 10μF ─┬─── ESP32 3V3
            │          │
            └─ 100nF ──┘
                       │
                      GND

Boot Circuit:
    GPIO0 ──┬── 10kΩ ── 3.3V
            │
            └── [SW_BOOT] ── GND

Reset Circuit:
    EN ─────┬── 10kΩ ── 3.3V
            │
            └── 10μF ── GND
            │
            └── [SW_RST] ── GND
```

#### 2.2 I2C Bus (Thermal Camera + Pulse Oximeter)

```
                         I2C BUS SCHEMATIC
            ┌────────────────────────────────────────────────────┐
            │                                                    │
            │     SDA (GPIO21)                SCL (GPIO22)       │
            │        │                           │               │
            │        ├── 4.7kΩ ── 3.3V           ├── 4.7kΩ ── 3.3V
            │        │                           │               │
            │        │     ┌─────────────────────┘               │
            │        │     │                                     │
            │    ┌───▼─────▼───┐                                 │
            │    │   MLX90640   │  Addr: 0x33                    │
            │    │  32×24 IR    │                                │
            │    │  Thermal Cam │                                │
            │    └──────┬───────┘                                │
            │           │ VDD=3.3V, GND                          │
            │           │                                        │
            │    ┌──────▼───────┐                                │
            │    │   MAX30102   │  Addr: 0x57                    │
            │    │ Pulse/SpO2   │                                │
            │    │              │                                │
            │    └──────┬───────┘                                │
            │           │ VDD=3.3V, GND                          │
            │           │                                        │
            │    ┌──────▼───────┐                                │
            │    │   OPTIONAL   │                                │
            │    │  SSD1306     │  Addr: 0x3C (OLED Display)     │
            │    │  128×64 OLED │                                │
            │    └──────────────┘                                │
            │                                                    │
            └────────────────────────────────────────────────────┘

MLX90640 Pinout (SMD breakout):
    VDD ─── 3.3V (with 10μF + 100nF)
    GND ─── GND
    SDA ─── GPIO21
    SCL ─── GPIO22

MAX30102 Pinout:
    VIN ─── 3.3V (with 10μF + 100nF)  
    GND ─── GND
    SDA ─── GPIO21
    SCL ─── GPIO22
    INT ─── GPIO35 (optional interrupt)
```

#### 2.3 mmWave Radar Interface (UART)

```
                     60 GHz RADAR MODULE CONNECTION
            ┌────────────────────────────────────────────────────┐
            │                                                    │
            │    IWR6843 / AWR1642 EVM (TI Radar Module)         │
            │    ┌─────────────────────────────────────┐         │
            │    │                                     │         │
            │    │  VCC ─────────────── 5V Rail        │         │
            │    │  GND ─────────────── GND            │         │
            │    │                                     │         │
            │    │  UART TX ─────┬───── Level Shift ───┼─── GPIO16 (ESP32 RX)
            │    │               │     (5V → 3.3V)     │         │
            │    │  UART RX ────┬┼───── Level Shift ───┼─── GPIO17 (ESP32 TX)
            │    │              ││     (3.3V → 5V)     │         │
            │    │              ││                     │         │
            │    │  RESET ──────┼┼── GPIO27 (Optional) │         │
            │    │              ││                     │         │
            │    └──────────────┼┼─────────────────────┘         │
            │                   ││                               │
            │                   ▼▼                               │
            │    ┌─────────────────────────────────────┐         │
            │    │  BSS138 Level Shifter Module        │         │
            │    │  ┌─────────────────────────────────┐│         │
            │    │  │ LV ─── 3.3V    HV ─── 5V        ││         │
            │    │  │ GND ─── GND   GND ─── GND       ││         │
            │    │  │ L1 ──── ◄───► ── H1             ││         │
            │    │  │ L2 ──── ◄───► ── H2             ││         │
            │    │  └─────────────────────────────────┘│         │
            │    └─────────────────────────────────────┘         │
            │                                                    │
            │    UART Configuration:                             │
            │      Baud Rate: 921600 (typical for TI radar)      │
            │      Data Bits: 8                                  │
            │      Stop Bits: 1                                  │
            │      Parity: None                                  │
            │                                                    │
            └────────────────────────────────────────────────────┘
```

#### 2.4 Camera and USB Connections

> [!IMPORTANT]
> The **camera connects directly to your laptop** via USB or uses the built-in laptop webcam. It does NOT connect to the ESP32 or Raspberry Pi.

```
                LAPTOP USB CONNECTIONS
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │    YOUR LAPTOP / PC                                      │
    │    ┌─────────────────┐                                   │
    │    │  USB Port 1   ──┼──► Logitech C270 / Webcam        │
    │    │                 │    (or use built-in camera)       │
    │    │                 │                                   │
    │    │  USB Port 2   ──┼──► ESP32 DevKit (USB-Micro)      │
    │    │                 │    Serial @ 115200 baud           │
    │    │                 │                                   │
    │    │  (Optional)     │                                   │
    │    │  USB Hub      ──┼──► Additional devices             │
    │    └─────────────────┘                                   │
    │                                                          │
    │    For Demo/Hackathon:                                   │
    │    • Just use laptop's built-in webcam (index 0)         │
    │    • ESP32 connected via single USB cable                │
    │    • No Raspberry Pi needed - laptop runs everything!    │
    │                                                          │
    │    Camera Settings:                                      │
    │    • 720p or 1080p @ 30fps                               │
    │    • Good lighting for rPPG                              │
    │    • Distance: face capture 0.5-1m, body 2-3m            │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```


---

### 3. Pin Assignment Table

| ESP32 Pin | GPIO | Function | Interface | Voltage | Connected To |
|-----------|------|----------|-----------|---------|--------------|
| 3V3 | - | Power | Power | 3.3V | All 3.3V devices |
| GND | - | Ground | Power | 0V | All grounds |
| GPIO21 | 21 | I2C SDA | I2C | 3.3V | MLX90640, MAX30102, OLED |
| GPIO22 | 22 | I2C SCL | I2C | 3.3V | MLX90640, MAX30102, OLED |
| GPIO16 | 16 | UART2 RX | UART | 3.3V | mmWave Radar TX (via level shifter) |
| GPIO17 | 17 | UART2 TX | UART | 3.3V | mmWave Radar RX (via level shifter) |
| GPIO01 | 01 | UART0 TX | UART | 3.3V | Host Processor / Debug |
| GPIO03 | 03 | UART0 RX | UART | 3.3V | Host Processor / Debug |
| GPIO04 | 04 | LED Status | GPIO | 3.3V | Blue LED (status) |
| GPIO05 | 05 | LED Activity | GPIO | 3.3V | Green LED (data activity) |
| GPIO18 | 18 | Test Point 1 | Debug | 3.3V | TP1 Header |
| GPIO19 | 19 | Test Point 2 | Debug | 3.3V | TP2 Header |
| GPIO15 | 15 | Test Point 3 | Debug | 3.3V | TP3 Header |
| GPIO35 | 35 | Interrupt | GPIO In | 3.3V | MAX30102 INT (optional) |
| GPIO27 | 27 | Radar Reset | GPIO Out | 3.3V | mmWave Reset (optional) |
| GPIO00 | 00 | Boot Mode | Boot | 3.3V | Boot Button + 10k pull-up |
| EN | - | Enable/Reset | Reset | 3.3V | Reset Button + RC filter |

---

### 4. Communication Protocol Summary

| Interface | Protocol | Speed | Devices | Data Flow |
|-----------|----------|-------|---------|-----------|
| I2C (GPIO21/22) | I2C @ 400kHz | 400 kbps | MLX90640, MAX30102, OLED | Bidirectional |
| UART2 (GPIO16/17) | UART | 921600 baud | mmWave Radar | Radar → ESP32 (processed frames) |
| UART0 (GPIO01/03) | UART | 115200 baud | Host Processor | ESP32 ↔ Host (JSON data) |
| USB | USB 2.0 | 480 Mbps | Logitech C270 | Camera → Host (video frames) |
| USB | USB 2.0 | 12 Mbps | ESP32 DevKit | Host ↔ ESP32 (optional) |

**Data Format (ESP32 → Host via UART):**

> [!IMPORTANT]
> The ESP32 must send data in the **exact format expected by the FastAPI `/api/v1/screening` endpoint**. See format below.

**Raw Sensor Data (ESP32 → Python Serial Reader):**
```json
{
  "timestamp": 1707050232,
  "radar": {
    "respiration_rate": 15.2,
    "breathing_depth": 0.73,
    "micro_motion": 0.12
  },
  "thermal": {
    "skin_temp_avg": 36.4,
    "skin_temp_max": 37.1,
    "thermal_asymmetry": 0.3,
    "thermal_map": [[36.1, 36.2, ...], ...]
  },
  "pulse_ox": {
    "spo2": 98,
    "heart_rate": 72
  }
}
```

**Python Bridge (Transforms ESP32 data → API ScreeningRequest):**

A Python script on the Raspberry Pi must transform the raw sensor data into the API format:

```python
# bridge.py - ESP32 to API bridge
import serial
import json
import requests

API_URL = "http://localhost:8000/api/v1/screening"

def transform_to_api_format(esp32_data: dict) -> dict:
    """Transform ESP32 sensor data to API ScreeningRequest format."""
    return {
        "patient_id": "WALK_THROUGH_PATIENT",
        "include_validation": True,
        "systems": [
            {
                "system": "cardiovascular",
                "biomarkers": [
                    {"name": "heart_rate", "value": esp32_data["pulse_ox"]["heart_rate"], 
                     "unit": "bpm", "normal_range": [60, 100]},
                    {"name": "spo2", "value": esp32_data["pulse_ox"]["spo2"], 
                     "unit": "%", "normal_range": [95, 100]},
                    {"name": "chest_micro_motion", "value": esp32_data["radar"]["micro_motion"],
                     "unit": "normalized_amplitude", "normal_range": [0.001, 0.01]}
                ]
            },
            {
                "system": "respiratory",  # Maps to pulmonary
                "biomarkers": [
                    {"name": "respiration_rate", "value": esp32_data["radar"]["respiration_rate"],
                     "unit": "breaths/min", "normal_range": [12, 20]},
                    {"name": "breathing_depth", "value": esp32_data["radar"]["breathing_depth"],
                     "unit": "normalized", "normal_range": [0.5, 1.0]}
                ]
            },
            {
                "system": "skin",
                "biomarkers": [
                    {"name": "skin_temperature", "value": esp32_data["thermal"]["skin_temp_avg"],
                     "unit": "celsius", "normal_range": [35.5, 37.5]},
                    {"name": "thermal_asymmetry", "value": esp32_data["thermal"]["thermal_asymmetry"],
                     "unit": "delta_celsius", "normal_range": [0, 0.5]}
                ]
            }
        ]
    }

# Main loop
ser = serial.Serial('/dev/ttyUSB0', 115200)
while True:
    line = ser.readline().decode().strip()
    if line:
        esp32_data = json.loads(line)
        api_payload = transform_to_api_format(esp32_data)
        response = requests.post(API_URL, json=api_payload)
        print(f"Screening ID: {response.json()['screening_id']}")
```

**Final API Request Format (what the API receives):**
```json
{
  "patient_id": "WALK_THROUGH_PATIENT",
  "include_validation": true,
  "systems": [
    {
      "system": "cardiovascular",
      "biomarkers": [
        {"name": "heart_rate", "value": 72, "unit": "bpm", "normal_range": [60, 100]},
        {"name": "spo2", "value": 98, "unit": "%", "normal_range": [95, 100]},
        {"name": "chest_micro_motion", "value": 0.12, "unit": "normalized_amplitude"}
      ]
    },
    {
      "system": "pulmonary",
      "biomarkers": [
        {"name": "respiration_rate", "value": 15.2, "unit": "breaths/min"},
        {"name": "breathing_depth", "value": 0.73, "unit": "normalized"}
      ]
    },
    {
      "system": "skin",
      "biomarkers": [
        {"name": "skin_temperature", "value": 36.4, "unit": "celsius"},
        {"name": "thermal_asymmetry", "value": 0.3, "unit": "delta_celsius"}
      ]
    }
  ]
}
```

---

### 5. Bill of Materials (BOM)

> [!IMPORTANT]
> **Your laptop is the brain!** No Raspberry Pi needed. The BOM below is only for sensors + ESP32.

| # | Component | Part Number | Qty | Unit Price (INR) | Total (INR) | Source |
|---|-----------|-------------|-----|------------------|-------------|--------|
| 1 | ESP32-WROOM-32D DevKit | ESP32-DevKitC-32D | 1 | ₹450 | ₹450 | Robu.in |
| 2 | Logitech C270 HD Webcam | C270 | 1 | ₹1,500 | ₹1,500 | Amazon.in |
| 3 | MLX90640 Thermal Camera | MLX90640ESF-BAB | 1 | ₹3,500 | ₹3,500 | Robu.in |
| 4 | MAX30102 Pulse Oximeter | MAX30102 Module | 1 | ₹250 | ₹250 | Robu.in |
| 5 | 60GHz mmWave Radar | IWR6843ISK-ODS | 1 | ₹8,000 | ₹8,000 | TI Store* |
| 6 | LM2596 Buck Converter | LM2596-ADJ | 1 | ₹100 | ₹100 | Robu.in |
| 7 | BSS138 Level Shifter | BSS138 Module | 1 | ₹50 | ₹50 | Robu.in |
| 8 | SSD1306 OLED 128×64 | 0.96" I2C OLED | 1 | ₹180 | ₹180 | Robu.in |
| 9 | LEDs (Blue, Green) | 5mm LED | 4 | ₹5 | ₹20 | Local |
| 10 | Resistors (10k, 4.7k, 330Ω) | Assorted | 20 | ₹2 | ₹40 | Local |
| 11 | Capacitors (10μF, 100nF) | Ceramic/Electrolytic | 20 | ₹3 | ₹60 | Local |
| 12 | Headers (Male/Female) | 2.54mm Pitch | 4 | ₹20 | ₹80 | Local |
| 13 | Jumper Wires | Dupont Wires | 40 | ₹2 | ₹80 | Local |
| 14 | Prototype PCB | 7×9cm Perf Board | 2 | ₹50 | ₹100 | Local |
| 15 | USB Cables | Type-A to Micro | 2 | ₹80 | ₹160 | Amazon.in |
| 16 | Enclosure/Mount | 3D Printed | 1 | ₹500 | ₹500 | Custom |
| | | | | **Total** | **₹15,070** | |

> [!NOTE]
> *TI mmWave radar modules may require IndiaMART or direct TI order. Alternative: LD2410 radar module (~₹800) for simplified respiration detection.

**Budget-Conscious Alternative (₹7,000):**
- Replace IWR6843 with LD2410 (₹800) - saves ₹7,200
- Use laptop's built-in webcam - saves ₹1,500
- Total: ~₹6,370

---

### 6. PCB Footprint Recommendations

| Component | Package | Footprint | Notes |
|-----------|---------|-----------|-------|
| ESP32-WROOM-32D | Module | 25.5×18mm | 38-pin castellated |
| MLX90640 | QFP-24 | 7×7mm | IR-transparent window required |
| MAX30102 | Module | 14×11mm | Finger clip mount |
| LM2596 | TO-263 | 10×15mm | Use adequate copper pour |
| BSS138 | SOT-23 | 3×3mm | 4-channel module preferred |
| USB Connector | Type-B | Standard | Use through-hole for durability |
| LED | 5mm THT | 5mm | 330Ω series resistor |

---

### 7. Mechanical Mounting Considerations

```
                    WALK-THROUGH CHAMBER LAYOUT (Top View)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │                        ENTRY                                │
    │                          │                                  │
    │    ┌─────────────────────▼─────────────────────────────┐   │
    │    │                                                    │   │
    │    │        [THERMAL CAM]                               │   │
    │    │        Height: 1.2m                                │   │
    │    │        Angle: Slightly downward                    │   │
    │    │                                                    │   │
    │    │    ────────────── WALK PATH ────────────────       │   │
    │    │                      ↑                             │   │
    │    │                      │                             │   │
    │    │              [LOGITECH C270]                       │   │
    │    │              Height: 1.2m (face) / 0.8m (body)     │   │
    │    │              Distance: 1.5-2.0m from subject       │   │
    │    │                                                    │   │
    │    │    [mmWAVE RADAR]                                  │   │
    │    │    Height: 1.0m (chest level)                      │   │
    │    │    Distance: 0.5-1.5m optimal range                │   │
    │    │                                                    │   │
    │    │    ────────────── WALK PATH ────────────────       │   │
    │    │                                                    │   │
    │    │    [CONTROL STATION]                               │   │
    │    │    ESP32 + LAPTOP (on table) + Power               │   │
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

**Sensor Mounting Specifications:**

| Sensor | Height | Distance | Angle | Notes |
|--------|--------|----------|-------|-------|
| Logitech C270 (Face) | 1.2m | 0.5-1.0m | Eye level | Phase 1: Close-up face |
| Logitech C270 (Body) | 0.8-1.0m | 2.0-3.0m | Full body | Phase 2: Gait analysis |
| MLX90640 | 1.2m | 0.5-1.0m | Face-level | Forehead/cheek thermal |
| mmWave Radar | 1.0m | 0.5-1.5m | Chest-level | Perpendicular to chest |
| MAX30102 | Side station | N/A | Finger clip | Calibration station |

---

### 8. EMI Shielding and Cable Guidelines

> [!WARNING]
> 60 GHz radar is extremely sensitive to EMI and reflections. Proper shielding is critical.

**Shielding Guidelines:**

1. **mmWave Radar:**
   - Use shielded twisted pair for UART lines
   - Keep radar module away from power cables (>10cm)
   - Shield radar enclosure with metal mesh (aperture <λ/10 = 0.5mm)
   - Grounded metal plate behind radar to reduce multipath

2. **Camera:**
   - Use shielded USB cables with ferrite cores
   - Keep USB cables away from radar and power
   - Max USB cable length: 3m (use active extension if longer)

3. **I2C Bus:**
   - Keep I2C traces short (<30cm)
   - Use proper pull-up resistors (4.7kΩ for 400kHz)
   - Avoid running parallel to high-speed signals

4. **Power:**
   - Separate analog and digital grounds (star topology)
   - Add ferrite beads on power input
   - Use ground plane on PCB

---

### 9. Design Notes

#### 9.1 Component Selection Rationale

| Component | Why Chosen | Alternatives |
|-----------|------------|--------------|
| ESP32-WROOM-32 | WiFi/BT, dual-core, multiple UART, low cost | Arduino Mega (no WiFi), STM32 (complex) |
| Your Laptop | Runs MediaPipe/OpenCV/FastAPI, already have it | Raspberry Pi 4B (slower), Intel NUC (expensive) |
| Logitech C270 / Laptop webcam | 720p/30fps, good low-light, USB, cheap | Logitech C920 (overkill), Intel RealSense (expensive) |
| MLX90640 | 32×24 resolution, I2C, radiometric | AMG8833 (8×8, too low), FLIR Lepton (expensive) |
| MAX30102 | Proven accuracy, I2C, low power | MAX30100 (older), pulse sensor (unreliable) |
| IWR6843 | TI's 60GHz vital signs radar, SDK available | AWR1642 (similar), LD2410 (simpler but less data) |

#### 9.2 Hardware Limitations & Mitigations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Camera motion blur | rPPG artifacts | Fixed mount, software stabilization |
| Ambient light variation | rPPG SNR | Controlled LED lighting |
| Thermal drift | Temperature offset | Calibration every 30 min |
| Radar multipath | False respiratory rate | Shielding, signal processing |
| ESP32 UART limited buffers | Data loss | DMA transfer, flow control |
| I2C clock stretching | Timing issues | Proper timeouts in firmware |

#### 9.3 Safety Considerations

- **Electrical:** Use fused power input, low-voltage DC only
- **EMI:** Radar complies with FCC Part 15, CE marked modules only
- **Optical:** Camera uses visible light only, no IR laser
- **Thermal:** Passive heat dissipation, no active cooling needed (<40°C ambient)

---

### 10. Verification Plan

#### Automated Tests

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Power rail voltages | Multimeter at test points | 5V ±0.25V, 3.3V ±0.1V |
| I2C device scan | `i2cdetect -y 1` | Detect 0x33, 0x57, 0x3C |
| ESP32 UART echo | Loopback test | 100% byte accuracy |
| Camera detection | `lsusb` / OpenCV | Device ID visible |
| Radar data stream | Serial monitor | JSON frames at 20Hz |

#### Manual Verification

1. **Camera Test:**
   - Run `python camera_test.py`
   - Verify face detection and pose landmarks
   - Check frame rate: should be 25-30 fps

2. **Thermal Camera Test:**
   - Point at known temperature reference (body temp ~36.5°C)
   - Verify ±1°C accuracy after calibration

3. **Radar Test:**
   - Stand 1m from radar, breathe normally
   - Verify respiration rate matches manual count (±1 BPM)

4. **Pulse Oximeter Test:**
   - Compare SpO₂ with clinical reference device
   - Verify ±2% accuracy

#### Data Synchronization Checklist

- [ ] All sensor streams tagged with Unix timestamp
- [ ] ESP32 NTP sync at boot (via WiFi)
- [ ] Maximum timestamp drift: <50ms over 60s session
- [ ] JSON packet includes `timestamp` field

---

### 11. Software Integration Notes

The hardware interfaces with the existing FastAPI software via the **bridge.py** script:

#### Key Files

| File | Purpose |
|------|---------|
| [bridge.py](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/bridge.py) | Main integration script |
| [esp32_health_bridge.ino](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/firmware/esp32_health_bridge.ino) | ESP32 Arduino firmware |
| [HARDWARE_QUICKSTART.md](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/HARDWARE_QUICKSTART.md) | Quick start guide |

#### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ mmWave Radar │  │ MLX90640     │  │ MAX30102     │               │
│  │ (Respiration)│  │ (Thermal)    │  │ (SpO2/HR)    │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                        │
│         └────────────────┬┴─────────────────┘                        │
│                          ▼                                           │
│              ┌────────────────────────┐                              │
│              │  ESP32 (sensor fusion) │                              │
│              │  JSON via UART @ 20 Hz │                              │
│              └───────────┬────────────┘                              │
└──────────────────────────┼───────────────────────────────────────────┘
                           │ Serial USB
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        BRIDGE LAYER (bridge.py)                      │
│                                                                      │
│  ┌──────────────┐  ┌──────────────────────┐  ┌─────────────────┐    │
│  │ ESP32Reader  │  │ CameraCapture        │  │ DataFusion      │    │
│  │ (Serial JSON)│  │ (rPPG + Pose)        │  │ (API Transform) │    │
│  └──────┬───────┘  └──────────┬───────────┘  └────────┬────────┘    │
│         │                     │                       │              │
│         └─────────────────────┴───────────────────────┘              │
│                               │                                      │
│                               ▼                                      │
│              ┌────────────────────────────┐                          │
│              │  build_screening_request() │                          │
│              │  → POST /api/v1/screening  │                          │
│              └────────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────────┘
                                │ HTTP POST
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                           │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ /api/v1/screening                                          │      │
│  │  → RiskEngine.compute_risk()                               │      │
│  │  → CompositeRiskCalculator                                 │      │
│  │  → AgentConsensus (LLM validation)                         │      │
│  └───────────────────────────────────────────────────────────┘      │
│                               │                                      │
│                               ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ /api/v1/reports/generate                                   │      │
│  │  → PatientReportGenerator.generate()                       │      │
│  │  → DoctorReportGenerator.generate()                        │      │
│  └───────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │   PDF Reports     │
                    │   reports/*.pdf   │
                    └───────────────────┘
```

#### Quick Test (No Hardware)

```bash
# Terminal 1: Start API
uvicorn app.main:app --reload

# Terminal 2: Test with simulated sensors
python bridge.py --simulate
```

#### ESP32 Firmware

The ESP32 firmware sends JSON at 20 Hz:

```c
// firmware/esp32_health_bridge.ino
void sendJsonData() {
    StaticJsonDocument<512> doc;
    doc["timestamp"] = millis() / 1000;
    
    JsonObject radar = doc.createNestedObject("radar");
    radar["respiration_rate"] = respirationRate;
    radar["breathing_depth"] = breathingDepth;
    
    JsonObject thermal = doc.createNestedObject("thermal");
    thermal["skin_temp_avg"] = skinTempAvg;
    
    JsonObject pulseOx = doc.createNestedObject("pulse_ox");
    pulseOx["heart_rate"] = heartRate;
    pulseOx["spo2"] = spO2;
    
    serializeJson(doc, Serial);
    Serial.println();
}
```

---

## Summary

This design provides a **modular, hackathon-ready hardware architecture** that:

- ✅ Uses your laptop as the brain (no Raspberry Pi needed!)
- ✅ Camera connects directly to laptop via USB or built-in
- ✅ ESP32 handles sensors: thermal, radar, pulse-ox
- ✅ Low cost: ~₹15,000 for sensors + ESP32
- ✅ Provides clear schematics, pin assignments, and BOM
- ✅ Supports the existing FastAPI software extractors
- ✅ Prioritizes workability over sophistication

The mmWave radar replaces RIS for respiratory and micro-motion sensing, which should achieve 70-90% of the accuracy target for the hackathon demonstration.

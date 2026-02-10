// esp32_thermal_biomarkers_v3_FIXED.ino
// ============================================================================
// DISCLAIMER: This code is for RESEARCH AND EDUCATIONAL PURPOSES ONLY
// NOT intended for medical diagnosis or clinical use without proper validation
// Requires FDA/CE certification and clinical trials for medical applications
// ============================================================================

#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[768]; // 32×24 pixels

struct ROI {
float temp_mean;
float temp_max;
float temp_min;
bool valid;
};

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================
// I2C Pins - Adjust for your board:
// - Standard ESP32: SDA=21, SCL=22
// - ESP32-CAM: SDA=16, SCL=0
// - Custom boards: Check your schematic
const int I2C_SDA = 21;
const int I2C_SCL = 22;

// IMPORTANT: Hardware pull-up resistors required on I2C lines
// - Recommended: 2.2kΩ - 4.7kΩ for 400kHz
// - Minimum: 1kΩ (for high-speed 1MHz operation)
// - Check if your MLX90640 breakout board has built-in pull-ups

const int LED_STATUS = 4; // Blue LED for status
const int LED_ACTIVITY = 5; // Green LED for activity

// ============================================================================
// MLX90640 CONFIGURATION
// ============================================================================
// Refresh Rate Options: MLX90640_0_5_HZ, MLX90640_1_HZ, MLX90640_2_HZ,
// MLX90640_4_HZ, MLX90640_8_HZ, MLX90640_16_HZ, MLX90640_32_HZ
// I2C Clock Speed vs Refresh Rate:
// - 100kHz: Stable, max 4Hz, ~600ms/frame
// - 400kHz: Good balance, max 8Hz, ~300ms/frame
// - 1MHz: Fast, max 32Hz, ~150ms/frame (requires strong pull-ups)
const int REFRESH_RATE_HZ = 4;
const int I2C_CLOCK_SPEED = 400000; // 400kHz - change to 1000000 for higher rates
const unsigned long EXPECTED_FRAME_TIME_MS = 600; // Realistic for 400kHz at 4Hz

// ============================================================================
// BIOMARKER THRESHOLDS (Research Values - NOT Clinically Validated)
// ============================================================================
const float FEVER_THRESHOLD = 37.35; // Neck/canthus temperature
const float DIABETES_THRESHOLD = 35.5; // Reduced canthus temperature
const float CVD_ASYMMETRY_THRESHOLD = 0.5; // Left-right cheek difference
const float INFLAMMATION_PCT_THRESHOLD = 5.0; // % pixels >1°C above baseline
const float STRESS_GRADIENT_THRESHOLD = 1.5; // Forehead-nose temperature diff

void setup() {
Serial.begin(115200);
while (!Serial && millis() < 5000); // Wait up to 5s for serial monitor

pinMode(LED_STATUS, OUTPUT);
pinMode(LED_ACTIVITY, OUTPUT);

// Startup indication
digitalWrite(LED_STATUS, HIGH);
digitalWrite(LED_ACTIVITY, HIGH);

Serial.println("\n=== ESP32 Thermal Camera v3 ===");
Serial.println("Research/Educational Use Only");

// ============================================================================
// FIX 1: Initialize I2C with explicit pins and timeout
// ============================================================================
Wire.begin(I2C_SDA, I2C_SCL);

// Set I2C timeout (ESP-IDF specific)
// Default timeout may be too short for MLX90640
Wire.setTimeOut(5000); // 5 seconds

delay(100); // Allow I2C bus to stabilize

// ============================================================================
// Test I2C communication before sensor initialization
// ============================================================================
Serial.print("Testing I2C connection to MLX90640 (0x");
Serial.print(MLX90640_I2CADDR_DEFAULT, HEX);
Serial.print(")... ");

Wire.beginTransmission(MLX90640_I2CADDR_DEFAULT);
byte i2c_error = Wire.endTransmission();

if (i2c_error != 0) {
Serial.println("FAILED");
Serial.print("{\"error\":\"I2C communication failed\",\"code\":");
Serial.print(i2c_error);
Serial.println(",\"check\":[\"wiring\",\"pull-up resistors\",\"power supply\"]}");

// Blink status LED rapidly on error
while (1) {
digitalWrite(LED_STATUS, !digitalRead(LED_STATUS));
delay(200);
}
}
Serial.println("OK");

// ============================================================================
// Initialize MLX90640 sensor
// ============================================================================
Serial.print("Initializing MLX90640... ");
if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
Serial.println("FAILED");
Serial.println("{\"error\":\"MLX90640 not found\",\"check\":[\"sensor connection\",\"I2C address\",\"power\"]}");

// Blink both LEDs on sensor error
while (1) {
digitalWrite(LED_STATUS, !digitalRead(LED_STATUS));
digitalWrite(LED_ACTIVITY, !digitalRead(LED_ACTIVITY));
delay(500);
}
}
Serial.println("OK");

// ============================================================================
// FIX 2: Set I2C clock speed AFTER mlx.begin()
// The Adafruit library resets I2C speed to 100kHz in begin()
// ============================================================================
Wire.setClock(I2C_CLOCK_SPEED);
Serial.print("I2C clock set to ");
Serial.print(I2C_CLOCK_SPEED / 1000);
Serial.println(" kHz");

// ============================================================================
// Configure sensor parameters
// ============================================================================
mlx.setMode(MLX90640_CHESS); // Chess pattern (interleaved)
mlx.setResolution(MLX90640_ADC_18BIT); // 18-bit ADC (highest resolution)
mlx.setRefreshRate(MLX90640_4_HZ); // 4Hz refresh rate

delay(500); // Let sensor stabilize after configuration

// ============================================================================
// Sensor ready
// ============================================================================
digitalWrite(LED_STATUS, LOW);
digitalWrite(LED_ACTIVITY, LOW);

Serial.println("{\"status\":\"ready\",\"version\":\"3.0_fixed\",\"fps\":" +
String(REFRESH_RATE_HZ) + ",\"i2c_clock\":" +
String(I2C_CLOCK_SPEED) + "}");
Serial.println();
}

void loop() {
unsigned long frame_start = millis();

// Activity LED on during frame capture
digitalWrite(LED_ACTIVITY, HIGH);

// ============================================================================
// FIX 3: Add retry logic with exponential backoff
// ============================================================================
int status = -1;
int retry_count = 0;
const int MAX_RETRIES = 3;

while (retry_count < MAX_RETRIES) {
status = mlx.getFrame(frame);

if (status == 0) {
break; // Success
}

// Exponential backoff: 50ms, 100ms, 200ms
retry_count++;
delay(50 * retry_count);

Serial.print("{\"warning\":\"Frame read retry\",\"attempt\":");
Serial.print(retry_count);
Serial.println("}");
}

unsigned long frame_read_time = millis() - frame_start;

if (status != 0) {
// Frame read failed after all retries
Serial.print("{\"error\":\"Frame read failed\",\"status\":");
Serial.print(status);
Serial.print(",\"retries\":");
Serial.print(retry_count);
Serial.print(",\"read_time_ms\":");
Serial.print(frame_read_time);
Serial.println("}");

digitalWrite(LED_ACTIVITY, LOW);
delay(EXPECTED_FRAME_TIME_MS);
return;
}

// ============================================================================
// Validate frame data quality
// ============================================================================
if (!isFrameValid()) {
Serial.println("{\"error\":\"Invalid frame data\",\"reason\":\"Temperature out of range or too many bad pixels\"}");
digitalWrite(LED_ACTIVITY, LOW);
delay(EXPECTED_FRAME_TIME_MS);
return;
}

// ============================================================================
// Extract biomarkers and output JSON
// ============================================================================
extractClinicalBiomarkers(frame_read_time);

digitalWrite(LED_ACTIVITY, LOW);

// ============================================================================
// FIX 4: Timing adjustment based on actual frame read time
// Maintain target FPS while accounting for processing time
// ============================================================================
unsigned long total_time = millis() - frame_start;
unsigned long target_period = 1000 / REFRESH_RATE_HZ;

if (total_time < target_period) {
delay(target_period - total_time);
}
}

// ============================================================================
// Frame validation - ensure data is realistic
// ============================================================================
bool isFrameValid() {
float min_temp = 999.0, max_temp = -999.0;
int valid_pixels = 0;
int nan_pixels = 0;

for (int i = 0; i < 768; i++) {
float temp = frame[i];

// Count NaN pixels
if (isnan(temp)) {
nan_pixels++;
continue;
}

// Realistic human/environment temperature range
if (temp >= 10.0 && temp <= 50.0) {
if (temp < min_temp) min_temp = temp;
if (temp > max_temp) max_temp = temp;
valid_pixels++;
}
}

// Validation criteria
bool has_enough_pixels = (valid_pixels > 690); // At least 90% valid (0.9 * 768 = 691)
bool not_too_many_nans = (nan_pixels < 77); // Less than 10% NaN
bool temp_in_range = (min_temp >= 15.0 && max_temp <= 45.0);
bool has_variation = (max_temp - min_temp > 0.5);

return has_enough_pixels && not_too_many_nans && temp_in_range && has_variation;
}

// ============================================================================
// Clinical biomarker extraction
// ============================================================================
void extractClinicalBiomarkers(unsigned long frame_read_time) {
// Extract ROIs (Region of Interest) - assumes centered face
// NOTE: These coordinates are NOT adaptive - face must be centered
ROI left_canthus = getROIStats(10, 14, 12, 14); // Inner eye corner (left)
ROI right_canthus = getROIStats(10, 14, 18, 20); // Inner eye corner (right)
ROI neck = getROIStats(22, 24, 10, 22); // Neck/supraclavicular
ROI left_cheek = getROIStats(10, 18, 6, 14); // Left cheek
ROI right_cheek = getROIStats(10, 18, 18, 26); // Right cheek
ROI nose_tip = getROIStats(16, 20, 14, 18); // Nose tip
ROI forehead = getROIStats(2, 8, 10, 22); // Forehead

// Validate critical ROIs
if (!left_canthus.valid || !right_canthus.valid || !neck.valid) {
Serial.println("{\"error\":\"Critical ROI extraction failed\",\"reason\":\"Face not detected or misaligned\"}");
return;
}

// Calculate biomarkers
float canthus_temp = (left_canthus.temp_mean + right_canthus.temp_mean) / 2.0;
float thermal_asymmetry = abs(left_cheek.temp_mean - right_cheek.temp_mean);
float face_mean_temp = calculateFaceMeanTemp();
float hot_pixel_pct = calculateHotPixelPercentage(face_mean_temp, 1.0);
float stress_gradient = forehead.valid ? (forehead.temp_mean - nose_tip.temp_mean) : 0.0;

// Calculate stability metrics (temperature range within ROIs)
float canthus_stability = (left_canthus.temp_max - left_canthus.temp_min +
right_canthus.temp_max - right_canthus.temp_min) / 2.0;
float neck_stability = neck.temp_max - neck.temp_min;

// ============================================================================
// Output JSON (STRICT schema compliance)
// ============================================================================
String json = "{";
json += "\"thermal\":{";

// Fever biomarkers
json += "\"fever\":{";
json += "\"canthus_temp\":" + String(canthus_temp, 2) + ",";
json += "\"neck_temp\":" + String(neck.temp_mean, 2) + ",";
json += "\"neck_stability\":" + String(neck_stability, 2) + ",";
json += "\"fever_risk\":" + String((neck.temp_mean > FEVER_THRESHOLD) ? 1 : 0);
json += "},";

// Diabetes biomarkers
json += "\"diabetes\":{";
json += "\"canthus_temp\":" + String(canthus_temp, 2) + ",";
json += "\"canthus_stability\":" + String(canthus_stability, 2) + ",";
json += "\"risk_flag\":" + String((canthus_temp < DIABETES_THRESHOLD) ? 1 : 0);
json += "},";

// Cardiovascular biomarkers
json += "\"cardiovascular\":{";
json += "\"thermal_asymmetry\":" + String(thermal_asymmetry, 3) + ",";
json += "\"left_cheek_temp\":" + String(left_cheek.temp_mean, 2) + ",";
json += "\"right_cheek_temp\":" + String(right_cheek.temp_mean, 2) + ",";
json += "\"risk_flag\":" + String((thermal_asymmetry > CVD_ASYMMETRY_THRESHOLD) ? 1 : 0);
json += "},";

// Inflammation biomarkers
json += "\"inflammation\":{";
json += "\"hot_pixel_pct\":" + String(hot_pixel_pct, 2) + ",";
json += "\"face_mean_temp\":" + String(face_mean_temp, 2) + ",";
json += "\"detected\":" + String((hot_pixel_pct > INFLAMMATION_PCT_THRESHOLD) ? 1 : 0);
json += "},";

// Autonomic / Stress biomarkers
json += "\"autonomic\":{";
json += "\"stress_gradient\":" + String(stress_gradient, 2) + ",";
json += "\"nose_temp\":" + String(nose_tip.temp_mean, 2) + ",";
json += "\"forehead_temp\":" + String(forehead.valid ? forehead.temp_mean : 0.0, 2) + ",";
json += "\"stress_flag\":" + String((stress_gradient > STRESS_GRADIENT_THRESHOLD) ? 1 : 0);
json += "}";

json += "}}";

// Emit single-line JSON
Serial.println(json);
}

// ============================================================================
// ROI extraction with defensive bounds checking
// ============================================================================
ROI getROIStats(int row_start, int row_end, int col_start, int col_end) {
// Bounds validation
row_start = constrain(row_start, 0, 23);
row_end = constrain(row_end, 1, 24);
col_start = constrain(col_start, 0, 31);
col_end = constrain(col_end, 1, 32);

ROI result;
result.temp_mean = 0.0;
result.temp_max = -999.0;
result.temp_min = 999.0;
result.valid = false;

// Check for invalid range
if (row_start >= row_end || col_start >= col_end) {
return result;
}

float sum = 0.0;
int count = 0;

for (int r = row_start; r < row_end; r++) {
for (int c = col_start; c < col_end; c++) {
int index = r * 32 + c;

// Paranoid bounds check
if (index < 0 || index >= 768) continue;

float temp = frame[index];

// Skip invalid values
if (isnan(temp) || temp < 10.0 || temp > 65.0) continue;

sum += temp;
if (temp > result.temp_max) result.temp_max = temp;
if (temp < result.temp_min) result.temp_min = temp;
count++;
}
}

if (count > 0) {
result.temp_mean = sum / count;
result.valid = true;
}

return result;
}

// ============================================================================
// Helper functions for biomarker calculation
// ============================================================================
float calculateFaceMeanTemp() {
float sum = 0.0;
int count = 0;

// Face region (exclude edges which may be background)
for (int r = 2; r < 22; r++) {
for (int c = 6; c < 26; c++) {
int index = r * 32 + c;
float temp = frame[index];

if (!isnan(temp) && temp > 20.0 && temp < 50.0) {
sum += temp;
count++;
}
}
}

return (count > 0) ? sum / count : 0.0;
}

float calculateHotPixelPercentage(float baseline_temp, float threshold) {
if (baseline_temp < 20.0) return 0.0; // Invalid baseline

int hot_pixels = 0;
int valid_pixels = 0;

for (int i = 0; i < 768; i++) {
float temp = frame[i];

if (isnan(temp) || temp < 20.0) continue;

valid_pixels++;
if (temp - baseline_temp > threshold) {
hot_pixels++;
}
}

return (valid_pixels > 0) ? (float)hot_pixels / valid_pixels * 100.0 : 0.0;
}

float getMinTemp() {
float min_temp = 999.0;
for (int i = 0; i < 768; i++) {
if (!isnan(frame[i]) && frame[i] < min_temp) {
min_temp = frame[i];
}
}
return min_temp;
}

float getMaxTemp() {
float max_temp = -999.0;
for (int i = 0; i < 768; i++) {
if (!isnan(frame[i]) && frame[i] > max_temp) {
max_temp = frame[i];
}
}
return max_temp;
}

int countValidROIs(ROI r1, ROI r2, ROI r3, ROI r4, ROI r5, ROI r6, ROI r7) {
int count = 0;
if (r1.valid) count++;
if (r2.valid) count++;
if (r3.valid) count++;
if (r4.valid) count++;
if (r5.valid) count++;
if (r6.valid) count++;
if (r7.valid) count++;
return count;
}
// esp32_thermal_biomarkers_safe_labels_fixed.ino
// Thermal feature extraction (non-diagnostic, safe labeling)
// BUG-FIXED VERSION (canthus_range issue resolved)

#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[768];  // 32×24 pixels

struct ROI {
  float temp_mean;
  float temp_max;
  float temp_min;
  bool valid;
};

// ---- CONFIG ----
#define SDA_PIN 21
#define SCL_PIN 22

void setup() {
  Serial.begin(115200);
  delay(1500);

  // Explicit ESP32 I2C pins
  Wire.begin(SDA_PIN, SCL_PIN);

  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("{\"error\":\"MLX90640_not_detected\"}");
    while (1) delay(1000);
  }

  // SAFE sensor configuration
  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_4_HZ);

  Serial.println("{\"status\":\"thermal_ready\",\"fps\":4}");
}

void loop() {
  if (mlx.getFrame(frame) != 0) {
    Serial.println("{\"error\":\"frame_read_failed\"}");
    delay(250);
    return;
  }

  if (!isFrameValid()) {
    Serial.println("{\"error\":\"invalid_frame\"}");
    delay(250);
    return;
  }

  extractThermalFeatures();
  delay(250);  // ~4 FPS
}

// ---------- FRAME VALIDATION ----------
bool isFrameValid() {
  int valid = 0;
  float minT = 999, maxT = -999;

  for (int i = 0; i < 768; i++) {
    float t = frame[i];
    if (isnan(t) || t < 15 || t > 60) continue;

    valid++;
    minT = min(minT, t);
    maxT = max(maxT, t);
  }

  return (valid > 700) && (maxT - minT > 0.5);
}

// ---------- ROI EXTRACTION ----------
ROI getROIStats(int r1, int r2, int c1, int c2) {
  ROI roi;
  roi.temp_mean = 0.0;
  roi.temp_min  = 999.0;
  roi.temp_max  = -999.0;
  roi.valid     = false;

  int count = 0;

  for (int r = r1; r < r2; r++) {
    for (int c = c1; c < c2; c++) {
      int idx = r * 32 + c;
      float t = frame[idx];

      if (isnan(t) || t < 20.0 || t > 50.0) continue;

      roi.temp_mean += t;
      roi.temp_min = min(roi.temp_min, t);
      roi.temp_max = max(roi.temp_max, t);
      count++;
    }
  }

  if (count > 0) {
    roi.temp_mean /= count;
    roi.valid = true;
  }

  return roi;
}

// ---------- FEATURE EXTRACTION ----------
void extractThermalFeatures() {
  ROI left_canthus  = getROIStats(10, 14, 12, 14);
  ROI right_canthus = getROIStats(10, 14, 18, 20);
  ROI neck          = getROIStats(22, 24, 10, 22);
  ROI left_cheek    = getROIStats(10, 18, 6, 14);
  ROI right_cheek   = getROIStats(10, 18, 18, 26);
  ROI nose          = getROIStats(16, 20, 14, 18);
  ROI forehead      = getROIStats(2, 8, 10, 22);

  if (!left_canthus.valid || !right_canthus.valid || !neck.valid) {
    Serial.println("{\"error\":\"core_roi_missing\"}");
    return;
  }

  float canthus_mean =
    (left_canthus.temp_mean + right_canthus.temp_mean) / 2.0;

  float cheek_asymmetry = 0.0;
  if (left_cheek.valid && right_cheek.valid) {
    cheek_asymmetry = abs(left_cheek.temp_mean - right_cheek.temp_mean);
  }

  float forehead_nose_gradient = 0.0;
  if (forehead.valid && nose.valid) {
    forehead_nose_gradient = forehead.temp_mean - nose.temp_mean;
  }

  // ✅ FIXED canthus_range calculation
  float canthus_range = 0.0;
  if (left_canthus.valid && right_canthus.valid) {
    canthus_range =
      ((left_canthus.temp_max - left_canthus.temp_min) +
       (right_canthus.temp_max - right_canthus.temp_min)) / 2.0;
  }

  // ---------- JSON OUTPUT ----------
  String json = "{";
  json += "\"timestamp\":" + String(millis()) + ",";
  json += "\"thermal\":{";

  json += "\"core_regions\":{";
  json += "\"canthus_mean\":" + String(canthus_mean, 2) + ",";
  json += "\"neck_mean\":" + String(neck.temp_mean, 2);
  json += "},";

  json += "\"stability_metrics\":{";
  json += "\"canthus_range\":" + String(canthus_range, 2);
  json += "},";

  json += "\"symmetry\":{";
  json += "\"cheek_asymmetry\":" + String(cheek_asymmetry, 3);
  json += "},";

  json += "\"gradients\":{";
  json += "\"forehead_nose_gradient\":" + String(forehead_nose_gradient, 2);
  json += "}";

  json += "}}";

  Serial.println(json);
}
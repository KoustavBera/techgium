#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Seed_Arduino_mmWave.h>

//Replace with your credentials
const char* ssid = "Your_SSID";
const char* password = "Your_Password";


//Static IP configuration
IPAddress local_IP(192,168,1,150);
IPAddress gateway(192,168,1,1);
IPAddress subnet(255,255,255,0);

//Create a web server on port 80
WebServer server(80);

#ifdef ESP32
Har
//Create a mmWave object
Seed_Arduino_mmWave mmWave;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  //Initialize mmWave sensor
  mmWave.begin();

  //Set up web server routes
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/plain", "mmWave sensor is running!");
  });

  server.on("/data", HTTP_GET, []() {
    //Read data from mmWave sensor
    mmWave.read();
    
    //Get heart rate and respiration rate
    float heartRate = mmWave.getHeartRate();
    float respirationRate = mmWave.getRespirationRate();
    
    //Send data as JSON
    String json = "{\"heartRate\": " + String(heartRate) + ", \"respirationRate\": " + String(respirationRate) + "}";
    server.send(200, "application/json", json);
  });

  server.begin();
  Serial.println("Web server started");
}

void loop() {
  server.handleClient();
}
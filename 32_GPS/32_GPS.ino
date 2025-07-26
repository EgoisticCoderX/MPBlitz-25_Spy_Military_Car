// Filename: esp32_gps_locator.ino
#include <WiFi.h>
#include <WebServer.h>
#include <TinyGPS++.h>
#include <HardwareSerial.h>

const char* ssid = "simple123";
const char* password = "simple123";

#define GPS_RX_PIN 16
#define GPS_TX_PIN 17

WebServer server(80);
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

void handleGetLocation() {
    String responseJson = "{\"status\":\"error\", \"message\":\"No valid GPS fix.\"}";
    if (gps.location.isValid()) {
        responseJson = "{\"status\":\"success\",\"lat\":" + String(gps.location.lat(), 6) + ",\"lon\":" + String(gps.location.lng(), 6) + "}";
        Serial.println("GPS location sent: " + responseJson);
    } else {
        Serial.println("Location requested, but no valid GPS fix.");
    }
    server.send(200, "application/json", responseJson);
}

void connectToWiFi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.printf("\nWiFi Connected! IP: %s\n", WiFi.localIP().toString().c_str());
}

void setup() {
    Serial.begin(115200);
    Serial.println("\n\nBooting ESP32 GPS Locator...");
    gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    Serial.println("GPS Serial Port Initialized.");
    connectToWiFi();
    server.on("/get-location", HTTP_GET, handleGetLocation);
    server.begin();
    Serial.println("GPS Locator online and ready.");
}

void loop() {
    while (gpsSerial.available() > 0) gps.encode(gpsSerial.read());
    server.handleClient();
    if (WiFi.status() != WL_CONNECTED) { Serial.println("WiFi lost! Reconnecting..."); connectToWiFi(); }
}
/*
 * ESP32_Gun_LED.ino
 * ArduinoJson 6.x
 * LED = ON only when POST {"gun_detected":true}
 */

#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

const char* ssid     = "simple123";
const char* password = "simple123";

const int LED = 2;               // onboard LED
AsyncWebServer server(80);

void setup() {
  Serial.begin(115200);
  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);        // start OFF

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nESP32 ready @ http://%s\n", WiFi.localIP().toString().c_str());

  // GET /led   → returns current state
  server.on("/led", HTTP_GET, [](AsyncWebServerRequest *req) {
    StaticJsonDocument<128> doc;
    doc["gun_detected"] = digitalRead(LED) == HIGH;
    String body;
    serializeJson(doc, body);
    req->send(200, "application/json", body);
  });

  // POST /led  → {"gun_detected":true/false}
  server.on("/led", HTTP_POST,
            [](AsyncWebServerRequest *req) {},              // no params
            NULL,                                           // no file upload
            [](AsyncWebServerRequest *req, uint8_t *data, size_t len,
               size_t index, size_t total) {
              StaticJsonDocument<128> doc;
              DeserializationError err = deserializeJson(doc, (const char*)data, len);
              if (!err && doc.containsKey("gun_detected")) {
                bool on = doc["gun_detected"].as<bool>();
                digitalWrite(LED, on ? HIGH : LOW);
                Serial.printf("Gun %s\n", on ? "detected → LED ON" : "gone → LED OFF");
                req->send(200, "application/json", "{\"status\":\"ok\"}");
              } else {
                req->send(400, "application/json", "{\"status\":\"bad request\"}");
              }
            });

  server.begin();
}

void loop() {
  /* nothing */
}
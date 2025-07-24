// Code3: Modified ESP32 Arduino Sketch (.ino)

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// --- WiFi and Camera Pin configuration (Unchanged) ---
const char* ssid = "AsishJIOBroadband";
const char* password = "asish123";
   
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_GPIO_NUM       4
#define LEDC_CHANNEL_1     1

WebServer server(80);
WebServer streamServer(81);


bool initCamera() {
    // This function is unchanged
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM; config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM; config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM; config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM; config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
    if(psramFound()){ config.frame_size = FRAMESIZE_VGA; config.jpeg_quality = 12; config.fb_count = 2; }
    else { config.frame_size = FRAMESIZE_CIF; config.jpeg_quality = 15; config.fb_count = 1; }
    if(esp_camera_init(&config) != ESP_OK) { Serial.println("Camera init failed"); return false; }
    return true;
}

// The snapshot function is unchanged.
void handleCapture() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { server.send(500, "text/plain", "Capture failed"); return; }
    server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
}

void handleStream() {
    // **KEY FIX:** Added more verbose logging.
    WiFiClient client = streamServer.client();
    if (client) {
        Serial.println("Stream client connected!");
        String response = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        client.print(response);

        while (client.connected()) {
            camera_fb_t *fb = esp_camera_fb_get();
            if (!fb) {
                Serial.println("!!! Camera capture FAILED");
                break;
            }
            client.print("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + String(fb->len) + "\r\n\r\n");
            client.write(fb->buf, fb->len);
            client.print("\r\n");
            esp_camera_fb_return(fb);
        }
        Serial.println("Stream client disconnected.");
        client.stop();
    }
}

// Flashlight control function is unchanged.
void handleLedControl() {
    if (server.hasArg("level")) {
        ledcWrite(LEDC_CHANNEL_1, server.arg("level").toInt());
        server.send(200, "text/plain", "OK");
    } else {
        server.send(400, "text/plain", "Missing 'level'");
    }
}

void setup() {
    Serial.begin(115200);
    Serial.println("\n\nESP32 System Booting...");

    if (!initCamera()) { Serial.println("FATAL: Camera init failed! Restarting..."); ESP.restart(); }

    ledcSetup(LEDC_CHANNEL_1, 5000, 8);
    ledcAttachPin(LED_GPIO_NUM, LEDC_CHANNEL_1);
    ledcWrite(LEDC_CHANNEL_1, 0);

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi ");
    int wifi_tries = 0;
    while (WiFi.status() != WL_CONNECTED && wifi_tries < 20) {
        delay(500);
        Serial.print(".");
        wifi_tries++;
    }
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\nFATAL: WiFi connection failed! Restarting...");
        delay(1000);
        ESP.restart();
    }
    
    Serial.printf("\nWiFi Connected! IP Address: %s\n", WiFi.localIP().toString().c_str());

    server.on("/capture", HTTP_GET, handleCapture);
    server.on("/led", HTTP_GET, handleLedControl);
    server.begin();
    Serial.println("HTTP server started on port 80.");

    streamServer.on("/stream", HTTP_GET, handleStream);
    streamServer.begin();
    Serial.println("Stream server started on port 81.");
    Serial.println("====================");
    Serial.println("Setup complete!");
}

void loop() {
    server.handleClient();
    streamServer.handleClient();
    delay(1); // Yield to other tasks
}
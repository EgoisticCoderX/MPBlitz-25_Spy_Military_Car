// Filename: esp32_unified_controller.ino
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include <ESP32Servo.h>

// --- CONFIGURATION ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// --- PINOUT ---
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
#define LED_GPIO_NUM       4  // Onboard flash
#define PAN_SERVO_PIN     12
#define TILT_SERVO_PIN    13

// --- DEDICATED SERVERS ---
WebServer controlServer(80);
WebServer streamServer(81);
WebServer flashServer(82);

Servo panServo;
Servo tiltServo;

#define LEDC_FLASH_CHANNEL 2 // Safe, non-conflicting timer channel

// --- Functions ---
bool initCamera() {
    camera_config_t config;

    // --- ** THIS IS THE LINE THAT WAS FIXED ** ---
    // Correctly assign a CHANNEL to ledc_channel, not a timer.
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    // The rest of the configuration remains correct.
    config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM; config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM; config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_SVGA; config.jpeg_quality = 10; config.fb_count = 2;
    
    // Initialize the camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return false;
    }
    return true;
}

void handleStream() {
    WiFiClient client = streamServer.client();
    if (!client) return;
    client.print("HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n");
    while (client.connected()) {
        if (camera_fb_t *fb = esp_camera_fb_get()) {
            client.print("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + String(fb->len) + "\r\n\r\n");
            client.write(fb->buf, fb->len);
            client.print("\r\n");
            esp_camera_fb_return(fb);
        }
    }
}

void handleLedControl() {
    if (flashServer.hasArg("level")) {
        int level = flashServer.arg("level").toInt();
        ledcWrite(LEDC_FLASH_CHANNEL, constrain(level, 0, 255));
        Serial.printf("Flashlight (Port 82): Level %d\n", level);
        flashServer.send(200, "text/plain", "OK");
    }
}

void handleServoControl() {
    if (controlServer.hasArg("axis") && controlServer.hasArg("value")) {
        String axis = controlServer.arg("axis");
        int value = controlServer.arg("value").toInt();
        if (axis == "pan") panServo.write(value);
        else if (axis == "tilt") tiltServo.write(value);
        Serial.printf("Servo (Port 80): %s -> %d\n", axis.c_str(), value);
        controlServer.send(200, "text/plain", "OK");
    }
}

void setup() {
    Serial.begin(115200);
    ledcSetup(LEDC_FLASH_CHANNEL, 5000, 8);
    ledcAttachPin(LED_GPIO_NUM, LEDC_FLASH_CHANNEL);
    ledcWrite(LEDC_FLASH_CHANNEL, 0);

    if (!initCamera()) { 
      Serial.println("Camera Init Failed! Restarting...");
      delay(3000);
      ESP.restart(); 
    }
    
    panServo.attach(PAN_SERVO_PIN);
    tiltServo.attach(TILT_SERVO_PIN);
    panServo.write(90); tiltServo.write(90);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) { delay(500); }
    String ip = WiFi.localIP().toString();

    controlServer.on("/set-servo", HTTP_GET, handleServoControl);
    controlServer.begin();
    
    streamServer.on("/stream", HTTP_GET, handleStream);
    streamServer.begin();
    
    flashServer.on("/led", HTTP_GET, handleLedControl);
    flashServer.begin();
    
    Serial.printf("\nTriple Server Online at IP: %s\n", ip.c_str());
}

void loop() {
    controlServer.handleClient();
    streamServer.handleClient();
    flashServer.handleClient();
} 
#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ===================
// WiFi Configuration
// ===================
const char* ssid = "AsishJIOBroadband"; // Replace with your WiFi SSID
const char* password = "asish123";      // Replace with your WiFi password

// ===================
// Camera Pin Definition for ESP32-CAM AI-Thinker
// ===================
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

// ===================
// NEW: Flashlight LED Configuration
// ===================
#define LED_GPIO_NUM 4       // GPIO4 is the onboard LED/Flashlight
#define LEDC_CHANNEL_1 1     // Use a different channel for the LED to avoid conflicts
#define LEDC_TIMER_1_BIT 8   // 8-bit resolution (0-255) for brightness

// ===================
// Web Server Setup
// ===================
WebServer server(80);
WebServer streamServer(81);

// ===================
// Camera Configuration Function
// ===================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Settings for better performance and reduced lag
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 12; // Lower value is higher quality, higher value is smaller size
    config.fb_count = 2; // Use 2 frame buffers for smoother capture
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_CIF; // 352x288
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return false;
  }

  // Optimize camera sensor settings for speed
  sensor_t * s = esp_camera_sensor_get();
  s->set_vflip(s, 1);        // Flip image if needed
  s->set_hmirror(s, 1);      // Mirror image if needed
  s->set_whitebal(s, 1);     // Auto white balance
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1); // Auto exposure
  s->set_aec2(s, 1);
  s->set_gain_ctrl(s, 1);    // Auto gain
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)2);

  Serial.println("Camera initialized successfully");
  return true;
}

// ===================
// NEW: Flashlight Control Function
// ===================
void handleLedControl() {
  int level = 0;
  if (server.hasArg("level")) {
    level = server.arg("level").toInt();
    level = constrain(level, 0, 255); // Ensure value is within 0-255
    ledcWrite(LEDC_CHANNEL_1, level); // Set brightness
    Serial.printf("LED brightness set to: %d\n", level);
    String response = "LED brightness set to " + String(level);
    server.send(200, "text/plain", response);
  } else {
    server.send(400, "text/plain", "Missing 'level' parameter");
  }
}


// ===================
// WiFi Connection Function
// ===================
void connectToWiFi() {
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());
}


// ===================
// Snapshot and Stream Handlers
// ===================
// This is your snapshot function, I have left it as is.
void handleCapture() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleStream() {
  WiFiClient client = streamServer.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      break;
    }

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: " + String(fb->len) + "\r\n\r\n");
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);
    
    // A small delay can help prevent overwhelming the network or client
    delay(10);
  }
  client.stop();
}


// ===================
// Setup Function
// ===================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\nESP32-CAM System Starting...");

  if (!initCamera()) {
    Serial.println("Camera init failed! Restarting...");
    ESP.restart();
  }

  // NEW: Initialize the LEDC for the flashlight
  ledcSetup(LEDC_CHANNEL_1, 5000, LEDC_TIMER_1_BIT); // Setup PWM channel
  ledcAttachPin(LED_GPIO_NUM, LEDC_CHANNEL_1);     // Attach GPIO4 to channel
  ledcWrite(LEDC_CHANNEL_1, 0); // Start with LED off

  connectToWiFi();

  // CORS headers (good practice for APIs)
  server.onNotFound([]() {
    if (server.method() == HTTP_OPTIONS) {
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.sendHeader("Access-Control-Max-Age", "10000");
        server.sendHeader("Access-Control-Allow-Methods", "POST,GET,OPTIONS");
        server.sendHeader("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
        server.send(204);
    } else {
        server.send(404, "text/plain", "Not Found");
    }
  });
  
  // Server routes on Port 80
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/led", HTTP_GET, handleLedControl); // NEW: Route for LED control
  
  server.begin();
  Serial.println("HTTP server started on port 80");

  // Stream server on Port 81
  streamServer.on("/stream", HTTP_GET, handleStream);
  streamServer.begin();
  Serial.println("Stream server started on port 81");
  Serial.println("===================");
}

// ===================
// Main Loop
// ===================
void loop() {
  server.handleClient();
  streamServer.handleClient();
  
  // Add a small delay to yield to other tasks and prevent watchdog timeouts
  delay(1);
}
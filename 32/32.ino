// Filename: esp32_controller.ino
// ROLE: Dedicated Pan-Tilt Servo and GPS Controller

#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <HardwareSerial.h> // For secondary serial port
#include <TinyGPS++.h>

// --- YOUR WIFI CREDENTIALS ---
const char* ssid = "AsishJIOBroadband";
const char* password = "asish123";

// --- PIN DEFINITIONS (for a standard ESP32 Dev Board) ---
// GPS Module (like L89)
#define GPS_RX_PIN 16 // Connect to TX of GPS module
#define GPS_TX_PIN 17 // Connect to RX of GPS module

// Pan-Tilt Servos
#define PAN_SERVO_PIN  12
#define TILT_SERVO_PIN 13

// --- GLOBAL OBJECTS ---
WebServer server(80);
HardwareSerial gpsSerial(2); // Use hardware serial port 2
TinyGPSPlus gps;
Servo panServo;
Servo tiltServo;

// Complete HTML and JavaScript for the standalone control panel
const char* html_page = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>ESP32 Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: system-ui, sans-serif; background-color: #222; color: #eee; padding: 20px; }
        .container { max-width: 500px; margin: auto; background-color: #333; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
        h1, h2 { color: #4CAF50; border-bottom: 1px solid #555; padding-bottom: 5px;}
        .slider-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="range"] { width: 100%; cursor: pointer; }
        #coords { font-weight: bold; font-size: 1.1em; color: #ffeb3b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ESP32 Controller</h1>
        <h2>GPS Status</h2>
        <p id="coords">Lat: --, Lon: --</p>
        <h2>Pan/Tilt Control</h2>
        <div class="slider-group">
            <label for="pan">Pan</label>
            <input type="range" id="pan" min="0" max="180" value="90">
        </div>
        <div class="slider-group">
            <label for="tilt">Tilt</label>
            <input type="range" id="tilt" min="0" max="180" value="90">
        </div>
    </div>
<script>
    // Send a command to a servo without waiting for a reply
    function sendServo(axis, value) {
        fetch(`/set-servo?axis=${axis}&value=${value}`);
    }

    // Get the latest location and update the text on the page
    function updateLocation() {
        fetch('/get-location')
            .then(response => response.json())
            .then(data => {
                const el = document.getElementById('coords');
                if (data.status === 'success') {
                    el.innerText = `Lat: ${data.lat.toFixed(5)}, Lon: ${data.lon.toFixed(5)}`;
                } else {
                    el.innerText = 'No valid GPS fix.';
                }
            }).catch(error => console.error('Error fetching location:', error));
    }

    // Attach listeners to the sliders
    document.getElementById('pan').addEventListener('input', (e) => sendServo('pan', e.target.value));
    document.getElementById('tilt').addEventListener('input', (e) => sendServo('tilt', e.target.value));

    // Update location on page load, and then every 3 seconds
    document.addEventListener('DOMContentLoaded', updateLocation);
    setInterval(updateLocation, 3000);
</script>
</body>
</html>
)rawliteral";

// Handles showing the HTML page
void handleRoot() {
  server.send(200, "text/html", html_page);
}

// API endpoint for servo control
void handleServoControl() {
    if (server.hasArg("axis") && server.hasArg("value")) {
        String axis = server.arg("axis");
        int value = server.arg("value").toInt();
        if (axis == "pan") panServo.write(value);
        else if (axis == "tilt") tiltServo.write(value);
        server.send(200, "text/plain", "OK");
    } else {
        server.send(400, "text/plain", "Missing 'axis' or 'value'");
    }
}

// API endpoint to get location data as JSON
void handleGetLocation() {
    String jsonResponse;
    if (gps.location.isValid() && gps.location.age() < 10000) { // Check for valid and recent fix
        jsonResponse = "{\"status\":\"success\",\"lat\":" + String(gps.location.lat(), 6) + ",\"lon\":" + String(gps.location.lng(), 6) + "}";
    } else {
        jsonResponse = "{\"status\":\"error\", \"message\":\"No valid GPS fix.\"}";
    }
    server.send(200, "application/json", jsonResponse);
}

void connectToWiFi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.println("\n----------------------------------------------");
    Serial.println(">>> ESP32 CONTROLLER (GPS & Servos) ONLINE <<<");
    Serial.printf("IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.println("==> Use this IP for ESP32_CONTROLLER_URL in your .env file!");
    Serial.println("----------------------------------------------");
}

void setup() {
    Serial.begin(115200);
    
    // Attach servos to pins and center them
    panServo.attach(PAN_SERVO_PIN);
    tiltServo.attach(TILT_SERVO_PIN);
    panServo.write(90);
    tiltServo.write(90);

    // Initialize the serial port for the GPS module
    gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

    connectToWiFi();
    
    // Define all web server endpoints
    server.on("/", HTTP_GET, handleRoot);           // Serves the HTML control panel
    server.on("/set-servo", HTTP_GET, handleServoControl);
    server.on("/get-location", HTTP_GET, handleGetLocation);
    server.begin();

    Serial.println("ESP32 Controller is fully operational.");
}

void loop() {
    server.handleClient(); // Handle incoming web requests

    // Continuously read and parse GPS data in the background
    while (gpsSerial.available() > 0) {
        gps.encode(gpsSerial.read());
    }
}
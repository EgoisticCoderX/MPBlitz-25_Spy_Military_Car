// =======================================================================
//   AUTONOMOUS GPS BACKTRACKING BOT WITH MILITARY DETECTION - ESP32 FIRMWARE
// =======================================================================

// --- Core Libraries ---
#include <WiFi.h>
#include <AsyncTCP.h>
#include <WebServer.h>
#include <ElegantOTA.h>  // UPDATED: Use AsyncElegantOTA
#include <ArduinoJson.h>
#include <vector>

// --- Sensor Libraries ---
#include <TinyGPSPlus.h>
#include <HardwareSerial.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_HMC5883_U.h>

// --- WiFi Credentials ---
const char* ssid = "AndroidAP56C4"; // Your WiFi SSID
const char* password = "priya123"; // Your WiFi Password

// --- GPS Configuration ---
static const int RXPin = 16, TXPin = 17;
static const uint32_t GPSBaud = 9600;
TinyGPSPlus gps;
HardwareSerial ss(2);

// --- Motor Driver (L298N) Pin Configuration ---
#define ENA 25 // Left motor speed
#define IN1 26
#define IN2 27
#define IN3 14
#define IN4 12
#define ENB 33 // Right motor speed
const int motorSpeed = 150; // Speed from 0-255

// --- Compass Sensor ---
Adafruit_HMC5883_Unified mag = Adafruit_HMC5883_Unified(12345);

// --- Navigation & State Management ---
enum BotState { IDLE, EXPLORING, BACKTRACKING, EMERGENCY_STOP };
BotState currentState = IDLE;

struct GPSCoordinate {
  double lat;
  double lon;
};

struct MilitaryDetection {
  double lat;
  double lon;
  String detectionType;
  String className;
  float confidence;
  unsigned long timestamp;
};

std::vector<GPSCoordinate> loggedPath;
std::vector<MilitaryDetection> militaryDetections;
int backtrackingWaypointIndex = -1;
const double WAYPOINT_RADIUS_METERS = 2.0; // How close to get to a waypoint
const double LOG_DISTANCE_METERS = 3.0;   // How far to travel before logging a new point

// --- Web Server & WebSocket ---
WebServer server(80);
AsyncWebSocket ws("/ws");

// --- HTML Webpage ---
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>Military Detection Bot - Command & Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body, html { 
            margin: 0; 
            padding: 0; 
            height: 100%; 
            width: 100%; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
            overflow: hidden; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        .grid-container { 
            display: grid; 
            grid-template-columns: 320px 1fr; 
            grid-template-rows: 100vh; 
            height: 100%; 
        }
        #map { 
            height: 100%; 
            width: 100%; 
            background-color: #1a1a1a; 
            border-left: 2px solid #333;
        }
        .panel { 
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
            padding: 15px; 
            display: flex; 
            flex-direction: column; 
            color: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.3);
        }
        .panel h2 { 
            margin: 0 0 20px 0; 
            text-align: center; 
            font-size: 20px;
            color: #ecf0f1;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .controls button { 
            display: block; 
            width: 100%; 
            padding: 15px; 
            margin-bottom: 10px; 
            font-size: 16px; 
            border: none; 
            border-radius: 8px; 
            color: white; 
            cursor: pointer; 
            transition: all 0.3s ease;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        #btn-explore { 
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
        }
        #btn-explore:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(39, 174, 96, 0.4);
        }
        #btn-return { 
            background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
            box-shadow: 0 4px 15px rgba(41, 128, 185, 0.3);
        }
        #btn-return:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(41, 128, 185, 0.4);
        }
        #btn-stop { 
            background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
            box-shadow: 0 4px 15px rgba(192, 57, 43, 0.3);
        }
        #btn-stop:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(192, 57, 43, 0.4);
        }
        .status-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 10px; 
            margin-top: 20px; 
        }
        .info-box { 
            background: rgba(255, 255, 255, 0.1); 
            padding: 12px; 
            border-radius: 8px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .info-box .label { 
            font-size: 11px; 
            color: #bdc3c7; 
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .info-box .value { 
            font-size: 16px; 
            font-weight: bold; 
            text-align: center; 
            margin-top: 4px;
        }
        #state { color: #f39c12; }
        .detections-section {
            margin-top: 20px;
        }
        .detections-section h3 {
            color: #e74c3c;
            font-size: 16px;
            margin-bottom: 10px;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .detection-item {
            background: rgba(231, 76, 60, 0.2);
            border: 1px solid #e74c3c;
            border-radius: 6px;
            padding: 8px;
            margin-bottom: 8px;
            font-size: 12px;
        }
        .detection-type {
            font-weight: bold;
            color: #e74c3c;
        }
        .detection-coords {
            color: #bdc3c7;
            margin-top: 2px;
        }
        .detection-confidence {
            color: #f39c12;
            font-size: 11px;
        }
        .alert-banner {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
            border-radius: 6px;
            display: none;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="grid-container">
        <div class="panel">
            <h2>🎯 Military Detection Bot</h2>
            <div class="alert-banner" id="alert-banner">
                ⚠️ MILITARY TARGET DETECTED ⚠️
            </div>
            <div class="controls">
                <button id="btn-explore" onclick="sendCommand('explore')">🚀 Start Mission</button>
                <button id="btn-return" onclick="sendCommand('return')">🏠 Return to Base</button>
                <button id="btn-stop" onclick="sendCommand('stop')">🛑 Emergency Stop</button>
            </div>
            <div class="status-grid">
                <div class="info-box"><div class="label">Status</div><div class="value" id="state">CONNECTING...</div></div>
                <div class="info-box"><div class="label">GPS Lock</div><div class="value" id="gps-status">NO FIX</div></div>
                <div class="info-box"><div class="label">Satellites</div><div class="value" id="sats">--</div></div>
                <div class="info-box"><div class="label">Speed (km/h)</div><div class="value" id="speed">--</div></div>
                <div class="info-box"><div class="label">Path Points</div><div class="value" id="points">--</div></div>
                <div class="info-box"><div class="label">Detections</div><div class="value" id="detection-count">0</div></div>
            </div>
            <div class="info-box" style="margin-top: 10px; grid-column: span 2;">
                <div class="label">Current Position</div>
                <div class="value" id="current-coords">N/A</div>
            </div>
            <div class="info-box" style="margin-top: 10px;">
                <div class="label">Base Location</div>
                <div class="value" id="home-coords">N/A</div>
            </div>
            
            <div class="detections-section">
                <h3>🎯 Military Detections</h3>
                <div id="detections-list">
                    <div style="text-align: center; color: #7f8c8d; font-size: 12px;">No detections yet</div>
                </div>
            </div>
        </div>
        <div id="map"></div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Initialize map with dark theme
        const map = L.map('map', {
            zoomControl: true,
            attributionControl: false
        }).setView([20, 0], 2);
        
        // Dark tile layer
        L.tileLayer('https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png', {
            attribution: '© CartoDB'
        }).addTo(map);

        let botMarker = null, homeMarker = null;
        let pathPolyline = L.polyline([], {color: '#3498db', weight: 4, opacity: 0.8}).addTo(map);
        let detectionMarkers = [];

        // Custom icons
        const militaryIcon = L.divIcon({
            className: 'military-marker',
            html: '<div style="background: #e74c3c; border-radius: 50%; width: 20px; height: 20px; border: 3px solid white; box-shadow: 0 0 10px rgba(231, 76, 60, 0.8);"></div>',
            iconSize: [26, 26],
            iconAnchor: [13, 13]
        });

        const botIcon = L.divIcon({
            className: 'bot-marker',
            html: '<div style="background: #2ecc71; border-radius: 50%; width: 16px; height: 16px; border: 2px solid white; box-shadow: 0 0 8px rgba(46, 204, 113, 0.8);"></div>',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });

        const homeIcon = L.divIcon({
            className: 'home-marker',
            html: '<div style="background: #f39c12; border-radius: 50%; width: 18px; height: 18px; border: 2px solid white; box-shadow: 0 0 8px rgba(243, 156, 18, 0.8);"></div>',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
        });

        function sendCommand(cmd) {
            ws.send(JSON.stringify({ "command": cmd }));
        }

        function addMilitaryDetection(detection) {
            const marker = L.marker([detection.lat, detection.lon], {icon: militaryIcon})
                .addTo(map)
                .bindPopup(`
                    <div style="font-family: Arial; font-size: 12px;">
                        <b style="color: #e74c3c;">⚠️ ${detection.detectionType}</b><br>
                        <b>Class:</b> ${detection.className}<br>
                        <b>Confidence:</b> ${(detection.confidence * 100).toFixed(1)}%<br>
                        <b>Coordinates:</b><br>
                        ${detection.lat.toFixed(6)}, ${detection.lon.toFixed(6)}<br>
                        <b>Time:</b> ${new Date(detection.timestamp).toLocaleTimeString()}
                    </div>
                `);
            
            detectionMarkers.push(marker);
            
            // Show alert banner
            const banner = document.getElementById('alert-banner');
            banner.style.display = 'block';
            setTimeout(() => {
                banner.style.display = 'none';
            }, 5000);
        }

        function updateDetectionsList(detections) {
            const listElement = document.getElementById('detections-list');
            if (detections.length === 0) {
                listElement.innerHTML = '<div style="text-align: center; color: #7f8c8d; font-size: 12px;">No detections yet</div>';
                return;
            }

            let html = '';
            detections.forEach((detection, index) => {
                html += `
                    <div class="detection-item">
                        <div class="detection-type">${detection.detectionType} - ${detection.className}</div>
                        <div class="detection-coords">📍 ${detection.lat.toFixed(5)}, ${detection.lon.toFixed(5)}</div>
                        <div class="detection-confidence">Confidence: ${(detection.confidence * 100).toFixed(1)}%</div>
                    </div>
                `;
            });
            listElement.innerHTML = html;
        }

        const ws = new WebSocket(`ws://${location.host}/ws`);
        let lastDetectionCount = 0;
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            // Update status displays
            document.getElementById('state').textContent = data.state || 'N/A';
            document.getElementById('gps-status').textContent = data.valid ? '🟢 LOCKED' : '🔴 NO FIX';
            document.getElementById('sats').textContent = data.sats || '--';
            document.getElementById('speed').textContent = data.speed ? data.speed.toFixed(1) : '--';
            document.getElementById('points').textContent = data.points || '0';
            
            // Update bot position
            if (data.valid) {
                const latLng = [data.lat, data.lon];
                document.getElementById('current-coords').textContent = `${data.lat.toFixed(5)}, ${data.lon.toFixed(5)}`;
                
                if (!botMarker) {
                    botMarker = L.marker(latLng, {icon: botIcon}).addTo(map);
                    map.setView(latLng, 16);
                } else {
                    botMarker.setLatLng(latLng);
                    map.panTo(latLng);
                }
            }
            
            // Update path
            if (data.path && data.path.length > 0) {
                const pathCoords = data.path.map(p => [p.lat, p.lon]);
                pathPolyline.setLatLngs(pathCoords);
                
                const homeCoords = data.path[0];
                document.getElementById('home-coords').textContent = `${homeCoords.lat.toFixed(5)}, ${homeCoords.lon.toFixed(5)}`;
                if(!homeMarker) {
                   homeMarker = L.marker([homeCoords.lat, homeCoords.lon], {icon: homeIcon})
                       .addTo(map)
                       .bindPopup("<b>🏠 Base Location</b>");
                }
            }

            // Handle military detections
            if (data.detections) {
                document.getElementById('detection-count').textContent = data.detections.length;
                
                // Add new detection markers
                if (data.detections.length > lastDetectionCount) {
                    for (let i = lastDetectionCount; i < data.detections.length; i++) {
                        addMilitaryDetection(data.detections[i]);
                    }
                    lastDetectionCount = data.detections.length;
                }
                
                updateDetectionsList(data.detections);
            }
        };

        ws.onopen = () => {
            console.log('WebSocket connected');
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            setTimeout(() => {
                location.reload();
            }, 3000);
        };
    </script>
</body>
</html>
)rawliteral";

// =======================================================================
// MOTOR CONTROL FUNCTIONS
// =======================================================================
void moveForward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
}

void turnRight() {
  digitalWrite(IN1, HIGH); // Left wheels forward
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); // Right wheels backward
  digitalWrite(IN4, LOW);
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
}

void turnLeft() {
  digitalWrite(IN1, LOW); // Left wheels backward
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); // Right wheels forward
  digitalWrite(IN4, HIGH);
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}

// =======================================================================
// MILITARY DETECTION FUNCTIONS
// =======================================================================
void addMilitaryDetection(String detectionType, String className, float confidence) {
  if (gps.location.isValid()) {
    MilitaryDetection detection;
    detection.lat = gps.location.lat();
    detection.lon = gps.location.lng();
    detection.detectionType = detectionType;
    detection.className = className;
    detection.confidence = confidence;
    detection.timestamp = millis();
    
    militaryDetections.push_back(detection);
    
    Serial.println("MILITARY DETECTION ADDED:");
    Serial.println("Type: " + detectionType);
    Serial.println("Class: " + className);
    Serial.println("Confidence: " + String(confidence));
    Serial.println("Location: " + String(detection.lat, 6) + ", " + String(detection.lon, 6));
  }
}

// =======================================================================
// NAVIGATION LOGIC
// =======================================================================
void navigateToWaypoint() {
    if (backtrackingWaypointIndex < 0 || !gps.location.isValid()) {
        stopMotors();
        currentState = IDLE;
        return;
    }

    GPSCoordinate target = loggedPath[backtrackingWaypointIndex];
    double distanceToTarget = TinyGPSPlus::distanceBetween(gps.location.lat(), gps.location.lng(), target.lat, target.lon);

    // Check if we have arrived at the waypoint
    if (distanceToTarget < WAYPOINT_RADIUS_METERS) {
        backtrackingWaypointIndex--; // Target the next point in the path
        if (backtrackingWaypointIndex < 0) { // We have arrived home
            currentState = IDLE;
            stopMotors();
            return;
        }
    }

    // --- Navigation Calculation ---
    double targetBearing = TinyGPSPlus::courseTo(gps.location.lat(), gps.location.lng(), target.lat, target.lon);

    sensors_event_t event;
    mag.getEvent(&event);
    float currentHeading = atan2(event.magnetic.y, event.magnetic.x) * (180 / PI);
    if (currentHeading < 0) currentHeading += 360;

    double headingError = targetBearing - currentHeading;
    // Normalize the error to be between -180 and 180
    if (headingError > 180) headingError -= 360;
    if (headingError < -180) headingError += 360;

    // --- Motor Commands based on Heading Error ---
    if (abs(headingError) < 15) { // If we are pointing in the right direction
        moveForward();
    } else if (headingError > 0) {
        turnRight();
    } else {
        turnLeft();
    }
}

// =======================================================================
// WEBSOCKET AND DATA HANDLING
// =======================================================================
void broadcastData() {
    String jsonString;
    DynamicJsonDocument doc(2048); // Increased size for detections data

    doc["lat"] = gps.location.lat();
    doc["lon"] = gps.location.lng();
    doc["sats"] = gps.satellites.value();
    doc["speed"] = gps.speed.kmph();
    doc["valid"] = gps.location.isValid();
    
    // Convert enum state to string
    switch(currentState) {
        case IDLE: doc["state"] = "IDLE"; break;
        case EXPLORING: doc["state"] = "EXPLORING"; break;
        case BACKTRACKING: doc["state"] = "BACKTRACKING"; break;
        case EMERGENCY_STOP: doc["state"] = "STOPPED"; break;
    }

    doc["points"] = loggedPath.size();

    // Send the full path
    if (loggedPath.size() > 0) {
        JsonArray path = doc.createNestedArray("path");
        for(const auto& p : loggedPath) {
            JsonObject point = path.createNestedObject();
            point["lat"] = p.lat;
            point["lon"] = p.lon;
        }
    }

    // Send military detections
    if (militaryDetections.size() > 0) {
        JsonArray detections = doc.createNestedArray("detections");
        for(const auto& d : militaryDetections) {
            JsonObject detection = detections.createNestedObject();
            detection["lat"] = d.lat;
            detection["lon"] = d.lon;
            detection["detectionType"] = d.detectionType;
            detection["className"] = d.className;
            detection["confidence"] = d.confidence;
            detection["timestamp"] = d.timestamp;
        }
    }

    serializeJson(doc, jsonString);
    ws.textAll(jsonString);
}

void onWebSocketEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
    if (type == WS_EVT_CONNECT) {
        Serial.printf("Client #%u connected\n", client->id());
        broadcastData(); // Send initial full data dump
    } else if (type == WS_EVT_DISCONNECT) {
        Serial.printf("Client #%u disconnected\n", client->id());
    } else if (type == WS_EVT_DATA) {
        AwsFrameInfo *info = (AwsFrameInfo*)arg;
        if (info->final && info->index == 0 && info->len == len && info->opcode == WS_TEXT) {
            data[len] = 0;
            DynamicJsonDocument doc(512);
            if (deserializeJson(doc, (char*)data) == DeserializationError::Ok) {
                const char* command = doc["command"];
                if (strcmp(command, "explore") == 0) {
                    currentState = EXPLORING;
                } else if (strcmp(command, "return") == 0) {
                    if (!loggedPath.empty()) {
                        backtrackingWaypointIndex = loggedPath.size() - 1; // Start from the last point
                        currentState = BACKTRACKING;
                    }
                } else if (strcmp(command, "stop") == 0) {
                    currentState = EMERGENCY_STOP;
                } else if (strcmp(command, "detection") == 0) {
                    // Handle detection from Python script
                    String detectionType = doc["type"];
                    String className = doc["class"];
                    float confidence = doc["confidence"];
                    addMilitaryDetection(detectionType, className, confidence);
                }
            }
        }
    }
}

// =======================================================================
// HTTP ENDPOINTS FOR PYTHON INTEGRATION
// =======================================================================
void handleDetectionEndpoint(AsyncWebServerRequest *request) {
    if (request->hasParam("type", true) && request->hasParam("class", true) && request->hasParam("confidence", true)) {
        String detectionType = request->getParam("type", true)->value();
        String className = request->getParam("class", true)->value();
        float confidence = request->getParam("confidence", true)->value().toFloat();
        
        addMilitaryDetection(detectionType, className, confidence);
        request->send(200, "application/json", "{\"status\":\"detection_added\",\"location\":{\"lat\":" + String(gps.location.lat(), 6) + ",\"lon\":" + String(gps.location.lng(), 6) + "}}");
    } else {
        request->send(400, "application/json", "{\"error\":\"missing_parameters\"}");
    }
}

void handleStatusEndpoint(AsyncWebServerRequest *request) {
    DynamicJsonDocument doc(1024);
    doc["lat"] = gps.location.lat();
    doc["lon"] = gps.location.lng();
    doc["valid"] = gps.location.isValid();
    doc["detections"] = militaryDetections.size();
    
    String response;
    serializeJson(doc, response);
    request->send(200, "application/json", response);
}

// =======================================================================
// SETUP AND LOOP
// =======================================================================
void setup() {
    Serial.begin(115200);
    ss.begin(GPSBaud, SERIAL_8N1, RXPin, TXPin);
    Wire.begin();

    // Initialize Motor Pins
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);
    pinMode(IN3, OUTPUT);
    pinMode(IN4, OUTPUT);
    pinMode(ENA, OUTPUT);
    pinMode(ENB, OUTPUT);
    stopMotors();

    // Initialize Compass
    if(!mag.begin()){
      Serial.println("Could not find a valid HMC5883L sensor, check wiring!");
      while(1);
    }

    // Initialize WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
    }
    Serial.println("\nWiFi connected! IP: " + WiFi.localIP().toString());

    // Initialize Web Server
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
        request->send_P(200, "text/html", index_html);
    });
    
    // API endpoints for Python integration
    server.on("/api/detection", HTTP_POST, handleDetectionEndpoint);
    server.on("/api/status", HTTP_GET, handleStatusEndpoint);
    
    // CORS headers for API calls
    DefaultHeaders::Instance().addHeader("Access-Control-Allow-Origin", "*");
    DefaultHeaders::Instance().addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    DefaultHeaders::Instance().addHeader("Access-Control-Allow-Headers", "Content-Type");
    
    ws.onEvent(onWebSocketEvent);
    server.addHandler(&ws);
    ElegantOTA.begin(&server);  // UPDATED: Use AsyncElegantOTA
    server.begin();
    
    Serial.println("Military Detection Bot initialized!");
    Serial.println("Web interface: http://" + WiFi.localIP().toString());
    Serial.println("Detection API: http://" + WiFi.localIP().toString() + "/api/detection");
}

unsigned long lastBroadcastTime = 0;

void loop() {
    while (ss.available() > 0) {
        gps.encode(ss.read());
    }

    // Main State Machine Logic
    switch(currentState) {
        case EXPLORING:
            moveForward();
            if (gps.location.isValid()) {
                if (loggedPath.empty()) {
                    // Log the very first point (home)
                    loggedPath.push_back({gps.location.lat(), gps.location.lng()});
                } else {
                    double distFromLast = TinyGPSPlus::distanceBetween(
                        gps.location.lat(), gps.location.lng(),
                        loggedPath.back().lat, loggedPath.back().lon
                    );
                    if (distFromLast > LOG_DISTANCE_METERS) {
                        loggedPath.push_back({gps.location.lat(), gps.location.lng()});
                    }
                }
            }
            break;

        case BACKTRACKING:
            navigateToWaypoint();
            break;
            
        case IDLE:
        case EMERGENCY_STOP:
            stopMotors();
            break;
    }

    // Broadcast data periodically
    if (millis() - lastBroadcastTime > 1000) {
        lastBroadcastTime = millis();
        broadcastData();
    }
    ElegantOTA.loop();  // UPDATED: Use AsyncElegantOTA
}
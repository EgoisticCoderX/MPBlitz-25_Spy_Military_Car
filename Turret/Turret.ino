#include <WiFi.h>
#include <WebServer.h>
#include <Servo.h>

// WiFi credentials
const char* ssid = "ESP32_Turret";
const char* password = "12345678";

// Motor pins


const int motorA1 = 26;
const int motorA2 = 27;
const int motorB1 = 14;
const int motorB2 = 12;

// Servo
Servo turretServo;
const int servoPin = 13;
int angle = 90;  // Mid position

// Web server
WebServer server(80);

void stopMotors() {
  digitalWrite(motorA1, LOW);
  digitalWrite(motorA2, LOW);
  digitalWrite(motorB1, LOW);
  digitalWrite(motorB2, LOW);
}

void setup() {
  Serial.begin(115200);

  // Setup pins
  pinMode(motorA1, OUTPUT);
  pinMode(motorA2, OUTPUT);
  pinMode(motorB1, OUTPUT);
  pinMode(motorB2, OUTPUT);

  stopMotors();

  turretServo.attach(servoPin);
  turretServo.write(angle);

  // WiFi setup as Access Point
  WiFi.softAP(ssid, password);
  Serial.println("Access Point Started");
  Serial.println(WiFi.softAPIP());

  // HTML control page
  server.on("/", []() {
    server.send(200, "text/html", R"rawliteral(
      <html><body>
      <h2>ESP32 Turret Control</h2>
      <button onclick="sendCmd('forward')">⬆ Forward</button><br><br>
      <button onclick="sendCmd('left')">⬅ Left</button>
      <button onclick="sendCmd('stop')">⏹ Stop</button>
      <button onclick="sendCmd('right')">➡ Right</button><br><br>
      <button onclick="sendCmd('backward')">⬇ Backward</button><br><br>
      <button onclick="sendCmd('left_servo')">🔄 Servo Left</button>
      <button onclick="sendCmd('right_servo')">🔁 Servo Right</button>

      <script>
        function sendCmd(cmd) {
          fetch('/cmd?move=' + cmd);
        }
      </script>
      </body></html>
    )rawliteral");
  });

  server.on("/cmd", []() {
    String move = server.arg("move");

    if (move == "forward") {
      digitalWrite(motorA1, HIGH); digitalWrite(motorA2, LOW);
      digitalWrite(motorB1, HIGH); digitalWrite(motorB2, LOW);
    }
    else if (move == "backward") {
      digitalWrite(motorA1, LOW); digitalWrite(motorA2, HIGH);
      digitalWrite(motorB1, LOW); digitalWrite(motorB2, HIGH);
    }
    else if (move == "left") {
      digitalWrite(motorA1, LOW); digitalWrite(motorA2, HIGH);
      digitalWrite(motorB1, HIGH); digitalWrite(motorB2, LOW);
    }
    else if (move == "right") {
      digitalWrite(motorA1, HIGH); digitalWrite(motorA2, LOW);
      digitalWrite(motorB1, LOW); digitalWrite(motorB2, HIGH);
    }
    else if (move == "stop") {
      stopMotors();
    }
    else if (move == "left_servo") {
      angle = constrain(angle - 10, 0, 180);
      turretServo.write(angle);
    }
    else if (move == "right_servo") {
      angle = constrain(angle + 10, 0, 180);
      turretServo.write(angle);
    }

    server.send(200, "text/plain", "OK");
  });

  server.begin();
}

void loop() {
  server.handleClient();
}

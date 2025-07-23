# Code1: Modified Flask Application (app.py)

from flask import Flask, render_template, Response, request, jsonify
import os
import cv2
import numpy as np
import requests
import base64
from dotenv import load_dotenv
import threading
import time
from collections import deque

# Load API key from .env file
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# --- Default ESP32 Camera Configuration ---
# This will be updated by the UI
ESP32_CAM_URL = "http://192.168.29.177"
STREAM_URL = f"{ESP32_CAM_URL}:81/stream"
SNAPSHOT_URL = f"{ESP32_CAM_URL}/capture"

# --- Roboflow Model Definitions ---
# Your model configuration remains unchanged
DETECTION_MODELS = {
    "soldier_detection": {
        "name": "Soldier Detection",
        "models": [
            {"id": "civil-soldier/1", "weight": 0.3},
            {"id": "millitaryobjectdetection/6", "weight": 0.25},
            {"id": "hiit/9", "weight": 0.25},
            {"id": "soldier-ijybv-wnxqu/1", "weight": 0.2}
        ],
        "conf": 0.35,
        "color": (0, 255, 0)
    },
    "landmine_detection": {
        "name": "Landmine",
        "models": [{"id": "landmine-k5eze-ylmos/1", "weight": 1}],
        "conf": 0.45,
        "color": (0, 0, 255)
    },
    "aircraft_detection": {
        "name": "Aircraft",
        "models": [{"id": "drone-uav-detection/3", "weight": 0.5},
                   {"id": "fighter-jet-detection/1", "weight": 0.5}],
        "conf": 0.35,
        "color": (255, 0, 0)
    },
    "tank_detection": {
        "name": "Tank",
        "models": [{"id": "tank-sl17s/1", "weight": 1}],
        "conf": 0.45,
        "color": (0, 255, 255)
    },
    "military_equipment": {
        "name": "Equipment",
        "models": [{"id": "military-f5tbj/1", "weight": 0.5},
                   {"id": "weapon-detection-ssvfk/1", "weight": 0.5}],
        "conf": 0.35,
        "color": (255, 0, 255)
    },
    "gun_detection": {
        "name": "Gun",
        "models": [{"id": "weapon-detection-ssvfk/1", "weight": 0.5},
                   {"id": "gun-d8mga/2", "weight": 0.5}],
        "conf": 0.35,
        "color": (128, 0, 128)
    }
}


app = Flask(__name__)

# --- REVAMPED ESP32CAMSTREAM CLASS ---
class ESP32CamStream:
    def __init__(self, esp32_url=ESP32_CAM_URL):
        self.esp32_url = esp32_url
        self.stream_url = f"{self.esp32_url}:81/stream"
        
        self.current_frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self.running = False
        self.last_detection_time = 0
        
        # --- Performance Tuning settings ---
        self.detection_interval = 1.0  # Time between detections
        self.frame_skip = 3            # Process 1 of every N frames for detection

    def start_stream(self):
        """Start the camera stream and detection threads"""
        if self.running:
            print("Stream is already running.")
            return

        self.running = True
        # Start the main stream loop in a separate thread
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        # Start the detection loop in a separate thread
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

    def _stream_loop(self):
        """
        **MAJOR FIX**: Connects to the MJPEG stream from the ESP32.
        This is much more efficient than taking snapshots and solves the timeout errors.
        """
        print(f"Connecting to ESP32 stream at: {self.stream_url}")
        try:
            stream_response = requests.get(self.stream_url, stream=True, timeout=10)
            stream_response.raise_for_status() # Raise an exception for bad status codes
            
            jpeg_bytes = bytes()
            for chunk in stream_response.iter_content(chunk_size=4096):
                if not self.running:
                    break # Stop if requested
                
                jpeg_bytes += chunk
                # Find start and end of a JPEG image
                a = jpeg_bytes.find(b'\xff\xd8') # JPEG start marker
                b = jpeg_bytes.find(b'\xff\xd9') # JPEG end marker
                
                if a != -1 and b != -1:
                    jpg = jpeg_bytes[a:b+2]
                    jpeg_bytes = jpeg_bytes[b+2:]
                    
                    # Decode the image
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Resize for performance if necessary
                        height, width = frame.shape[:2]
                        if width > 640:
                            scale = 640 / width
                            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                        
                        with self.frame_lock:
                            self.current_frame = frame.copy()
            
        except requests.exceptions.RequestException as e:
            print(f"CRITICAL: Failed to connect to ESP32 stream: {e}")
            # Optionally, place a "Connection Lost" image on the frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "ESP32 Connection Lost", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.processed_frame = error_frame

        finally:
            print("Stream loop stopped.")
            self.running = False


    def _detection_loop(self):
        """Separate thread for running ML detection to avoid blocking the stream."""
        frame_count = 0
        while self.running:
            current_time = time.time()
            if not self.running:
                break

            # Run detection at specified intervals
            if (current_time - self.last_detection_time >= self.detection_interval):
                frame_to_process = None
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_to_process = self.current_frame.copy()

                if frame_to_process is not None:
                    # Frame skipping logic
                    frame_count += 1
                    if frame_count % self.frame_skip == 0:
                        processed = self.process_frame_with_detection(frame_to_process)
                        with self.detection_lock:
                            self.processed_frame = processed
                    else:
                        # If skipping, just update the frame without detection
                        with self.detection_lock:
                           self.processed_frame = frame_to_process

                    self.last_detection_time = current_time

            time.sleep(0.05) # Small delay to prevent busy-waiting

    def process_frame_with_detection(self, frame):
        """Process a single frame with Roboflow models and draw detections."""
        if frame is None:
            return None
        
        # The rectangle drawing issue was due to frames not being received.
        # With the new streaming method, this function will now work correctly.
        for detection_key, detection_info in DETECTION_MODELS.items():
            all_predictions = []
            weights = []
            
            for model_info in detection_info['models']:
                predictions = call_roboflow_api(frame, model_info['id'], detection_info['conf'])
                if predictions: # Only add if we got a result
                    all_predictions.append(predictions)
                    weights.append(model_info['weight'])
            
            if not all_predictions:
                continue

            final_predictions = ensemble_predictions(all_predictions, weights, detection_info['conf'])
            
            # Draw detections on frame
            color = detection_info['color']
            for pred in final_predictions:
                x, y, w, h = pred.get('x'), pred.get('y'), pred.get('width'), pred.get('height')
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                
                class_name = pred.get('class', 'Unknown')
                confidence = pred.get('confidence', 0)
                
                label = f"{detection_info['name']}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def get_latest_frame(self):
        """Get the latest processed frame for display."""
        with self.detection_lock:
            if self.processed_frame is not None:
                return self.processed_frame.copy()
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def stop(self):
        """Stop the stream and detection threads."""
        self.running = False
        print("Stop signal sent. Waiting for threads to terminate...")
        # Add join to ensure threads have cleaned up
        if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        print("Threads terminated.")


# --- Global Camera Instance ---
cam_stream = ESP32CamStream()


def ensemble_predictions(predictions_list, weights, confidence_threshold):
    # This function remains unchanged but will now receive valid data
    if not predictions_list or not weights:
        return []
    final_predictions = []
    for predictions, weight in zip(predictions_list, weights):
        for pred in predictions:
            weighted_conf = pred.get('confidence', 0) * weight
            if weighted_conf >= confidence_threshold:
                pred_copy = pred.copy()
                pred_copy['confidence'] = weighted_conf
                final_predictions.append(pred_copy)
    return final_predictions


def call_roboflow_api(image, model_id, confidence_threshold):
    """
    Call Roboflow API. Increased timeout for more robustness.
    """
    url = f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}"
    params = {"confidence": confidence_threshold, "overlap": 30, "format": "json"}
    
    try:
        # Image resizing for faster API calls
        height, width = image.shape[:2]
        if width > 416:
            scale = 416 / width
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Increased timeout to 5 seconds to prevent read timeouts
        response = requests.post(url, params=params, data=img_base64,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"},
                                 timeout=5)
        
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json().get('predictions', [])

    except requests.exceptions.RequestException as e:
        print(f"API call failed for {model_id}: {e}")
        return []
    except Exception as e:
        print(f"Error calling API for {model_id}: {str(e)}")
        return []


def generate_frames():
    """Generate frames for video streaming to the browser."""
    while True:
        frame = cam_stream.get_latest_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/30) # Limit to ~30 FPS

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('live_stream.html', models=DETECTION_MODELS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream_route():
    global cam_stream
    if cam_stream.running:
        cam_stream.stop()
        time.sleep(1) # Give it a moment to shut down
    
    data = request.get_json()
    if data and 'esp32_url' in data:
        cam_stream = ESP32CamStream(esp32_url=data['esp32_url'])
        cam_stream.start_stream()
        return jsonify({"status": "success", "message": "Stream started"})
    return jsonify({"status": "error", "message": "Invalid request"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream_route():
    cam_stream.stop()
    return jsonify({"status": "success", "message": "Stream stopped"})

@app.route('/update_detection_settings', methods=['POST'])
def update_detection_settings():
    data = request.get_json()
    if 'detection_interval' in data:
        cam_stream.detection_interval = float(data['detection_interval'])
    if 'frame_skip' in data:
        cam_stream.frame_skip = int(data['frame_skip'])
    return jsonify({"status": "success", "message": "Settings updated"})

# --- NEW: FLASHLIGHT CONTROL ROUTE ---
@app.route('/set_brightness', methods=['POST'])
def set_brightness():
    """NEW: This route forwards the brightness command to the ESP32."""
    try:
        data = request.get_json()
        esp32_url = data.get('esp32_url')
        level = data.get('level')

        if esp32_url is None or level is None:
            return jsonify({"status": "error", "message": "Missing URL or brightness level."})
        
        # The ESP32 will have an endpoint like /led?level=...
        led_url = f"{esp32_url}/led?level={level}"
        
        # Send the command to the ESP32
        response = requests.get(led_url, timeout=2)
        response.raise_for_status()

        return jsonify({"status": "success", "message": f"Brightness set to {level}"})

    except requests.exceptions.RequestException as e:
        print(f"Failed to set brightness on ESP32: {e}")
        return jsonify({"status": "error", "message": "Failed to communicate with ESP32."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    print("Starting ESP32 Military Detection System...")
    print(f"Default ESP32 cam IP: {ESP32_CAM_URL} (change in browser UI)")
    print("Navigate to http://localhost:5001 to view the live stream")
    # Using threaded=True is important for handling multiple requests
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
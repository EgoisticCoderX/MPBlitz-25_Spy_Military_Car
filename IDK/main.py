"""
Flask Object Detection with Image Upload and Roboflow Models
Modified to use image upload for detection with Flask web interface
Enhanced with ensemble model predictions
"""

import cv2
import numpy as np
import requests
import json
import base64
import time
from flask import Flask, render_template, request, jsonify, Response
import threading
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load API key from .env file
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Enhanced Model Configuration with ensemble support
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

# Flask app
app = Flask(__name__)

# Global variables
detector = None
current_detection_type = "soldier_detection"
detection_enabled = True
latest_detections = []

def ensemble_predictions(predictions_list, weights, confidence_threshold):
    """Combine predictions from multiple models using weighted ensemble"""
    if not predictions_list:
        return []
    
    final_predictions = []
    
    for i, (predictions, weight) in enumerate(zip(predictions_list, weights)):
        for pred in predictions:
            # Weight the confidence
            weighted_conf = pred.get('confidence', 0) * weight
            if weighted_conf >= confidence_threshold:
                pred_copy = pred.copy()
                pred_copy['confidence'] = weighted_conf
                pred_copy['model_weight'] = weight
                final_predictions.append(pred_copy)
    
    return final_predictions

def call_roboflow_api(image, model_id, confidence_threshold):
    """Call Roboflow API for a single model"""
    try:
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        url = f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}"
        params = {
            "confidence": confidence_threshold,
            "overlap": 30,
            "format": "json"
        }
        
        response = requests.post(
            url,
            params=params,
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            return response.json().get('predictions', [])
        else:
            print(f"API call failed for {model_id}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error calling API for {model_id}: {str(e)}")
        return []

class RoboflowDetector:
    """
    Enhanced Roboflow object detection class with ensemble support
    """
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.current_detection_type = current_detection_type
        
    def detect_objects(self, image):
        """
        Perform object detection using ensemble of models for current detection type
        """
        try:
            detection_info = self.model_config[self.current_detection_type]
            
            # Get predictions from all models for this detection type
            all_predictions = []
            weights = []
            
            for model_info in detection_info['models']:
                model_id = model_info['id']
                weight = model_info['weight']
                
                predictions = call_roboflow_api(image, model_id, detection_info['conf'])
                all_predictions.append(predictions)
                weights.append(weight)
            
            # Combine predictions using ensemble
            final_predictions = ensemble_predictions(all_predictions, weights, detection_info['conf'])
            
            return self.process_predictions(final_predictions, image, detection_info)
                
        except Exception as e:
            print(f"Detection error: {e}")
            return image, []
    
    def process_predictions(self, predictions, image, detection_info):
        """
        Process ensemble predictions and draw bounding boxes
        """
        detections = []
        color = detection_info['color']
        
        for prediction in predictions:
            # Extract detection information
            class_name = prediction.get('class', 'Unknown')
            confidence = prediction.get('confidence', 0)
            
            # Get bounding box coordinates
            x_center = prediction.get('x', 0)
            y_center = prediction.get('y', 0)
            width = prediction.get('width', 0)
            height = prediction.get('height', 0)
            
            # Convert to OpenCV format (top-left corner)
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Store detection info
            detection_info_obj = {
                'class': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'detection_type': detection_info['name']
            }
            detections.append(detection_info_obj)
            
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{detection_info['name']}: {class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image, detections
    
    def switch_detection_type(self, detection_key):
        """
        Switch to a different detection type
        """
        if detection_key in self.model_config:
            self.current_detection_type = detection_key
            detection_info = self.model_config[detection_key]
            print(f"Switched to detection type: {detection_key}")
            print(f"Name: {detection_info['name']}")
            print(f"Models: {len(detection_info['models'])}")
            return True
        else:
            print(f"Detection type {detection_key} not found in configuration")
            return False
    
    def detect_objects_with_type(self, image, detection_key):
        """
        Detect objects using a specific detection type
        """
        try:
            if detection_key not in self.model_config:
                return image, []
                
            detection_info = self.model_config[detection_key]
            
            # Get predictions from all models for this detection type
            all_predictions = []
            weights = []
            
            for model_info in detection_info['models']:
                model_id = model_info['id']
                weight = model_info['weight']
                
                predictions = call_roboflow_api(image, model_id, detection_info['conf'])
                all_predictions.append(predictions)
                weights.append(weight)
            
            # Combine predictions using ensemble
            final_predictions = ensemble_predictions(all_predictions, weights, detection_info['conf'])
            
            return self.process_predictions(final_predictions, image, detection_info)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return image, []

class CameraManager:
    """
    Manages laptop camera capture
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.initialize_camera()
        
    def initialize_camera(self):
        """
        Initialize the laptop camera
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Camera {self.camera_index} initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
            
    def get_frame(self):
        """
        Get a frame from the camera
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None
            
    def release(self):
        """
        Release the camera
        """
        if self.cap is not None:
            self.cap.release()

def generate_frames():
    """
    Generate video frames for Flask streaming
    """
    global camera, detector, detection_enabled, latest_detections, current_detection_type
    
    while True:
        if camera is None:
            break
            
        frame = camera.get_frame()
        if frame is None:
            continue
            
        detections = []
        detected_frame = frame.copy()
        
        # Perform object detection if enabled
        if detection_enabled and detector is not None:
            detected_frame, detections = detector.detect_objects(frame.copy())
            latest_detections = detections
            
        # Add detection type info to frame
        detection_info = DETECTION_MODELS.get(current_detection_type, {"name": "Detection"})
        info_text = f"Detection: {detection_info['name']} | Models: {len(detection_info.get('models', []))}"
        cv2.putText(detected_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main page for image upload and detection results
    """
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', models=DETECTION_MODELS, error='No image uploaded')
            
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', models=DETECTION_MODELS, error='No image selected')
            
        # Read image as numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return render_template('index.html', models=DETECTION_MODELS, error='Invalid image file')
            
        # Run detection for all detection types
        detector = RoboflowDetector(DETECTION_MODELS)
        results = {}
        
        for detection_key in DETECTION_MODELS.keys():
            detected_img, detections = detector.detect_objects_with_type(image.copy(), detection_key)
            results[detection_key] = detections
            
        return render_template('index.html', models=DETECTION_MODELS, results=results, image_uploaded=True)
        
    return render_template('index.html', models=DETECTION_MODELS)

@app.route('/switch_detection', methods=['POST'])
def switch_detection():
    """
    Switch between different detection types
    """
    global current_detection_type, detector
    
    data = request.get_json()
    detection_key = data.get('detection_key')
    
    if detection_key in DETECTION_MODELS:
        current_detection_type = detection_key
        detector.switch_detection_type(detection_key)
        return jsonify({'status': 'success', 'detection_type': detection_key})
    else:
        return jsonify({'status': 'error', 'message': 'Detection type not found'})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """
    Toggle object detection on/off
    """
    global detection_enabled
    
    detection_enabled = not detection_enabled
    return jsonify({'status': 'success', 'enabled': detection_enabled})

@app.route('/get_detections')
def get_detections():
    """
    Get current detection results
    """
    return jsonify({'detections': latest_detections})

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    """
    Update confidence threshold for current detection type
    """
    global detector
    
    data = request.get_json()
    confidence = data.get('confidence', 0.5)
    
    if 0 <= confidence <= 1:
        DETECTION_MODELS[current_detection_type]['conf'] = confidence
        return jsonify({'status': 'success', 'confidence': confidence})
    else:
        return jsonify({'status': 'error', 'message': 'Confidence must be between 0 and 1'})

if __name__ == '__main__':
    detector = RoboflowDetector(DETECTION_MODELS)
    camera = None
    
    # Try multiple camera indices for webcam compatibility
    for cam_index in [0, 1, 2]:
        test_camera = CameraManager(camera_index=cam_index)
        if test_camera.cap is not None and test_camera.cap.isOpened():
            camera = test_camera
            print(f"Camera {cam_index} initialized and opened.")
            break
        else:
            print(f"Camera {cam_index} not available.")
            
    if camera is None:
        print("No available camera found. Please check your webcam connection or try a different index.")
        exit(1)
        
    print("Starting Flask Object Detection Server...")
    print("Camera initialized, starting web server...")
    print("Access the application at: http://localhost:5000")
    print(f"Current detection type: {current_detection_type}")
    print(f"Detection name: {DETECTION_MODELS[current_detection_type]['name']}")
    
    try:
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        if camera:
            camera.release()
        print("Server shutting down")

"""
ENHANCED HTML TEMPLATE NEEDED:
Create a folder called 'templates' in the same directory as this script
and create a file called 'index.html' with the following content:

<!DOCTYPE html>
<html>
<head>
    <title>Military Object Detection with Ensemble Models</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .video-container { text-align: center; margin-bottom: 30px; }
        .video-stream { border: 3px solid #333; border-radius: 10px; max-width: 100%; height: auto; }
        .controls { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 20px; }
        .control-group { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .detection-buttons { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
        .detection-btn { padding: 15px; border: none; border-radius: 5px; cursor: pointer; transition: all 0.3s; text-align: left; }
        .detection-btn.active { background-color: #4CAF50; color: white; }
        .detection-btn:not(.active) { background-color: #e0e0e0; }
        .detection-btn:hover { opacity: 0.8; }
        .detection-btn .title { font-weight: bold; font-size: 14px; }
        .detection-btn .models { font-size: 12px; opacity: 0.8; margin-top: 5px; }
        .toggle-btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .toggle-btn.enabled { background-color: #4CAF50; color: white; }
        .toggle-btn.disabled { background-color: #f44336; color: white; }
        .confidence-slider { width: 200px; margin: 10px; }
        .detection-info { background: white; padding: 15px; border-radius: 10px; margin-top: 20px; }
        .detection-list { max-height: 200px; overflow-y: auto; }
        .upload-section { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .file-input { margin: 10px 0; }
        .upload-btn { background-color: #008CBA; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Military Object Detection with Ensemble Models</h1>
            <p>Advanced detection using multiple AI models per detection type</p>
        </div>
        
        <div class="upload-section">
            <h3>Upload Image for Detection</h3>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" class="file-input" required>
                <button type="submit" class="upload-btn">Analyze Image</button>
            </form>
            
            {% if error %}
            <div style="color: red; margin-top: 10px;">{{ error }}</div>
            {% endif %}
            
            {% if results %}
            <div style="margin-top: 20px;">
                <h4>Detection Results:</h4>
                {% for detection_key, detections in results.items() %}
                <div style="margin: 10px 0; padding: 10px; border-left: 3px solid #4CAF50;">
                    <strong>{{ models[detection_key].name }}:</strong>
                    {% if detections %}
                        {% for det in detections %}
                        <div style="margin-left: 20px;">
                            {{ det.class }}: {{ (det.confidence * 100)|round(1) }}%
                        </div>
                        {% endfor %}
                    {% else %}
                        <span style="color: #666;">No detections</span>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Camera Feed">
        </div>
        
        <div class="controls">
            <div class="control-group">
                <h3>Detection Type Selection</h3>
                <div class="detection-buttons">
                    {% for detection_key, detection_info in models.items() %}
                    <button class="detection-btn {% if detection_key == current_detection_type %}active{% endif %}" 
                            onclick="switchDetection('{{ detection_key }}')">
                        <div class="title">{{ detection_info.name }}</div>
                        <div class="models">{{ detection_info.models|length }} model(s) | Conf: {{ detection_info.conf }}</div>
                    </button>
                    {% endfor %}
                </div>
            </div>
            
            <div class="control-group">
                <h3>Detection Control</h3>
                <button id="toggle-detection" class="toggle-btn enabled" onclick="toggleDetection()">
                    Detection ON
                </button>
                <br><br>
                <label>Confidence Threshold: <span id="confidence-value">0.35</span></label><br>
                <input type="range" class="confidence-slider" min="0" max="1" step="0.05" value="0.35" 
                       onchange="updateConfidence(this.value)">
            </div>
        </div>
        
        <div class="detection-info">
            <h3>Live Detection Results</h3>
            <div id="detection-list" class="detection-list">
                No detections yet...
            </div>
        </div>
    </div>

    <script>
        let detectionEnabled = true;
        let currentDetectionType = '{{ current_detection_type }}';
        
        function switchDetection(detectionKey) {
            fetch('/switch_detection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ detection_key: detectionKey })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentDetectionType = detectionKey;
                    document.querySelectorAll('.detection-btn').forEach(btn => btn.classList.remove('active'));
                    event.target.closest('.detection-btn').classList.add('active');
                }
            });
        }
        
        function toggleDetection() {
            fetch('/toggle_detection', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                detectionEnabled = data.enabled;
                const btn = document.getElementById('toggle-detection');
                btn.textContent = detectionEnabled ? 'Detection ON' : 'Detection OFF';
                btn.className = detectionEnabled ? 'toggle-btn enabled' : 'toggle-btn disabled';
            });
        }
        
        function updateConfidence(value) {
            document.getElementById('confidence-value').textContent = value;
            fetch('/update_confidence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ confidence: parseFloat(value) })
            });
        }
        
        function updateDetections() {
            fetch('/get_detections')
            .then(response => response.json())
            .then(data => {
                const detectionList = document.getElementById('detection-list');
                if (data.detections.length === 0) {
                    detectionList.innerHTML = 'No detections...';
                } else {
                    detectionList.innerHTML = data.detections
                        .map(det => `<div><strong>${det.detection_type}:</strong> ${det.class} (${(det.confidence * 100).toFixed(1)}%)</div>`)
                        .join('');
                }
            });
        }
        
        // Update detections every second
        setInterval(updateDetections, 1000);
    </script>
</body>
</html>

ENHANCED FEATURES:
==================

1. ENSEMBLE MODEL SUPPORT:
   - Multiple models per detection type
   - Weighted predictions combining
   - Improved accuracy through model diversity

2. ENHANCED DETECTION TYPES:
   - Soldier Detection (4 models)
   - Landmine Detection (1 model)
   - Aircraft Detection (2 models)
   - Tank Detection (1 model)
   - Military Equipment (2 models)
   - Gun Detection (2 models)

3. IMPROVED VISUALIZATION:
   - Color-coded detection types
   - Detection type names in labels
   - Model count information
   - Enhanced UI with better organization

4. ADVANCED FEATURES:
   - Confidence threshold per detection type
   - Real-time camera feed with live detection
   - Image upload for batch analysis
   - Ensemble prediction weighting
   - Error handling for API calls

SETUP INSTRUCTIONS:
==================

1. INSTALL DEPENDENCIES:
   pip install flask opencv-python requests numpy python-dotenv

2. CREATE FOLDER STRUCTURE:
   your_project/
   ├── app.py (this file)
   ├── .env (create this file to store your API key)
   └── templates/
       └── index.html (enhanced HTML template above)

3. SET UP .ENV FILE:
   Create a file named '.env' in the project root and add your Roboflow API key:
   ROBOFLOW_API_KEY=your_api_key_here

4. RUN THE APPLICATION:
   python app.py

5. ACCESS THE WEB INTERFACE:
   Open your browser and go to: http://localhost:5000
"""
from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import requests
import base64
from dotenv import load_dotenv
import io
from flask import send_file
from collections import defaultdict
import threading
import time

# Load API key from .env file
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# ESP32 Bot Configuration
ESP32_IP = "192.168.1.100"  # Replace with your ESP32's IP address
ESP32_PORT = 80
ESP32_DETECTION_URL = f"http://{ESP32_IP}:{ESP32_PORT}/api/detection"
ESP32_STATUS_URL = f"http://{ESP32_IP}:{ESP32_PORT}/api/status"

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

class ESP32Communicator:
    def __init__(self):
        self.esp32_connected = False
        self.last_status_check = 0
        self.status_check_interval = 10  # seconds
    
    def check_esp32_connection(self):
        """Check if ESP32 is accessible"""
        try:
            response = requests.get(ESP32_STATUS_URL, timeout=3)
            if response.status_code == 200:
                self.esp32_connected = True
                print(f"✅ ESP32 Bot connected at {ESP32_IP}")
                return True
        except requests.exceptions.RequestException as e:
            self.esp32_connected = False
            print(f"❌ ESP32 Bot not reachable: {e}")
        return False
    
    def send_detection_to_esp32(self, detection_type, class_name, confidence):
        """Send military detection to ESP32 bot"""
        if not self.esp32_connected:
            if not self.check_esp32_connection():
                return False
        
        try:
            payload = {
                'type': detection_type,
                'class': class_name,
                'confidence': confidence
            }
            
            response = requests.post(ESP32_DETECTION_URL, data=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"🎯 Detection sent to ESP32: {detection_type} - {class_name}")
                if 'location' in result:
                    print(f"📍 Bot Location: {result['location']['lat']:.6f}, {result['location']['lon']:.6f}")
                return True
            else:
                print(f"⚠️ ESP32 responded with status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to send detection to ESP32: {e}")
            self.esp32_connected = False
        
        return False
    
    def get_esp32_status(self):
        """Get current status from ESP32 bot"""
        try:
            response = requests.get(ESP32_STATUS_URL, timeout=3)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

# Initialize ESP32 communicator
esp32_comm = ESP32Communicator()

def ensemble_predictions(predictions_list, weights, confidence_threshold):
    """Combine predictions from multiple models using weighted ensemble"""
    if not predictions_list:
        return []
    
    # Group predictions by spatial proximity (simple NMS-like approach)
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
    url = f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}"
    params = {
        "confidence": confidence_threshold,
        "overlap": 30,
        "format": "json"
    }
    
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    try:
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

@app.route('/')
def index():
    return render_template('upload.html', models=DETECTION_MODELS, esp32_status=esp32_comm.get_esp32_status())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # For preview, keep a copy
    preview_img = image.copy()
    results = {}
    total_detections = 0
    
    # Process each detection type
    for detection_key, detection_info in DETECTION_MODELS.items():
        print(f"Processing {detection_info['name']}...")
        
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
        
        # Process final predictions for this detection type
        results[detection_key] = []
        color = detection_info['color']
        
        for pred in final_predictions:
            x_center = pred.get('x', 0)
            y_center = pred.get('y', 0)
            width = pred.get('width', 0)
            height = pred.get('height', 0)
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            class_name = pred.get('class', 'Unknown')
            confidence = pred.get('confidence', 0)
            
            results[detection_key].append({
                'class': class_name, 
                'confidence': confidence,
                'detection_type': detection_info['name']
            })
            
            # Send detection to ESP32 bot
            esp32_comm.send_detection_to_esp32(
                detection_info['name'], 
                class_name, 
                confidence
            )
            total_detections += 1
            
            # Draw rectangle and label on preview_img
            cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, 2)
            label = f"{detection_info['name']}: {class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(preview_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(preview_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Encode preview_img for display
    _, preview_buffer = cv2.imencode('.jpg', preview_img)
    preview_base64 = base64.b64encode(preview_buffer).decode('utf-8')
    
    return jsonify({
        'results': results,
        'preview_img': preview_base64,
        'total_detections': total_detections,
        'esp32_status': esp32_comm.get_esp32_status()
    })

@app.route('/esp32_status')
def get_esp32_status():
    """Get current ESP32 bot status"""
    return jsonify({
        'connected': esp32_comm.esp32_connected,
        'status': esp32_comm.get_esp32_status()
    })

@app.route('/test_detection')
def test_detection():
    """Test endpoint to send a fake detection to ESP32"""
    success = esp32_comm.send_detection_to_esp32("Test Detection", "Test Target", 0.85)
    return jsonify({
        'success': success,
        'message': 'Test detection sent' if success else 'Failed to send test detection'
    })

# Background task to periodically check ESP32 connection
def background_esp32_monitor():
    while True:
        esp32_comm.check_esp32_connection()
        time.sleep(esp32_comm.status_check_interval)

if __name__ == "__main__":
    # Start background ESP32 monitor
    monitor_thread = threading.Thread(target=background_esp32_monitor, daemon=True)
    monitor_thread.start()
    
    # Check initial connection
    esp32_comm.check_esp32_connection()
    
    print(f"🚀 Military Detection System starting...")
    print(f"🔗 ESP32 Bot URL: {ESP32_DETECTION_URL}")
    print(f"🌐 Web interface will be available at: http://localhost:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
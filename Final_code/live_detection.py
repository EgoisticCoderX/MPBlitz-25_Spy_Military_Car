"""
Live Military Object Detection – Port 5000
Black-screen free camera, rectangles on feed, working checkboxes
"""

import os, cv2, base64, time, threading, requests, numpy as np
from flask import Flask, render_template, Response, request, jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY missing in .env")

# ------------- MODEL CONFIG -------------
MODELS = {
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
executor = ThreadPoolExecutor(max_workers=4)

active_groups = list(MODELS.keys())
detection_enabled = True

# ------------- CAMERA -------------
class Camera:
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest = None
        self.lock = threading.Lock()
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.latest = frame
            time.sleep(0.03)

    def get(self):
        with self.lock:
            return self.latest.copy() if self.latest is not None else None

    def release(self):
        self.cap.release()

cam = Camera()

# ------------- HELPERS -------------
def b64(img):
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf).decode()

def infer(img, model_id, conf):
    url = f"https://detect.roboflow.com/{model_id}?api_key={API_KEY}"
    try:
        r = requests.post(url, params={"confidence": conf, "overlap": 30, "format": "json"},
                          data=b64(img), headers={"Content-Type": "application/x-www-form-urlencoded"},
                          timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def detect(img):
    all_dets = []
    for group in active_groups:
        cfg = MODELS[group]
        for m in cfg["models"]:
            res = infer(img, m["id"], cfg["conf"])
            if res and "predictions" in res:
                for p in res["predictions"]:
                    all_dets.append({
                        "class": p["class"],
                        "confidence": p["confidence"] * m["weight"],
                        "x": p["x"], "y": p["y"],
                        "width": p["width"], "height": p["height"],
                        "group": group
                    })
    return all_dets

def draw(img, dets):
    for d in dets:
        color = MODELS[d["group"]]["color"]
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["width"]), int(d["height"])
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 18), (x1 + 140, y1), color, -1)
        cv2.putText(img, f"{d['class']} {d['confidence']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

# ------------- STREAM -------------
def gen():
    last = 0
    while True:
        frame = cam.get()
        if frame is None:
            continue
        if detection_enabled and (time.time() - last) > 0.15:
            frame = draw(frame, detect(frame))
            last = time.time()
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# ------------- ROUTES -------------
@app.route('/')
def index():
    return render_template('live_detection.html', models=MODELS, active=active_groups)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_model', methods=['POST'])
def toggle_model():
    global active_groups
    key = request.json["model_key"]
    if key in MODELS:
        active_groups.remove(key) if key in active_groups else active_groups.append(key)
        return jsonify({"status": "ok", "active": active_groups})
    return jsonify({"status": "bad"}), 400

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({"status": "ok", "enabled": detection_enabled})

if __name__ == '__main__':
    print("Live server: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
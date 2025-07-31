import os, cv2, numpy as np, requests, base64, threading, time, json
from flask import Flask, render_template, Response, request, jsonify, url_for
from dotenv import load_dotenv
import google.generativeai as genai

# --- Setup: Load all keys and URLs ---
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ESP32_CONTROL_URL = os.getenv("ESP32_CONTROL_URL")
ESP32_STREAM_URL = os.getenv("ESP32_STREAM_URL")
ESP32_FLASHLIGHT_URL = os.getenv("ESP32_FLASHLIGHT_URL")

if not all([ROBOFLOW_API_KEY, GEMINI_API_KEY, ESP32_CONTROL_URL, ESP32_STREAM_URL, ESP32_FLASHLIGHT_URL]):
    raise ValueError("FATAL ERROR: Please set ALL FIVE required variables in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
os.makedirs("static/captures", exist_ok=True)
app = Flask(__name__)

# --- Model Config (Same as before) ---
DETECTION_MODELS = [{"id": "pistol-fire-and-gun/1", "name": "Pistol/Gun Fire", "conf": 0.50}, {"id": "gun-and-weapon-detection/1", "name": "Weapon Detection v1", "conf": 0.80}, {"id": "knife-and-gun-modelv2/2", "name": "Knife/Gun v2", "conf": 0.50}, {"id": "military-f5tbj/1", "name": "Military Equipment", "conf": 0.50}, {"id": "weapon-detection-ssvfk/1", "name": "Weapon Detection v2", "conf": 0.99}, {"id": "gun-d8mga/2", "name": "Gun Model v2", "conf": 0.69}]
for model in DETECTION_MODELS: model['color'] = tuple(np.random.randint(100, 255, size=3).tolist())

# --- Helper Functions ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def verify_with_gemini(image_bytes):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        safety_settings = {k: 'block_none' for k in ['HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH', 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT']}
        response = model.generate_content([ "Does this image contain a real weapon (gun, rifle, knife)? Respond only 'yes' or 'no'.", {"mime_type": "image/jpeg", "data": image_bytes}], safety_settings=safety_settings)
        response.resolve()
        print(f"🧠 Gemini Verification Response: '{response.text.strip().lower()}'")
        return "yes" in response.text.strip().lower()
    except Exception as e: print(f"❌ ERROR: Gemini verification failed: {e}"); return False

# --- Core Detection and Control Class ---
class DetectionStreamer:
    def __init__(self):
        self.stream_url = ESP32_STREAM_URL; self.last_frame = None; self.running = False
        self.frame_lock = threading.Lock(); self.state_lock = threading.Lock()
        self.state = "STREAMING"; self.last_detection_info = None; self.processed_frame = None

    def start(self):
        if self.running: return
        self.running = True; threading.Thread(target=self._stream_loop, daemon=True).start()
        threading.Thread(target=self._detection_pipeline, daemon=True).start()

    def _stream_loop(self):
        try:
            r = requests.get(self.stream_url, stream=True, timeout=10); r.raise_for_status()
            buffer = bytes()
            for chunk in r.iter_content(chunk_size=4096):
                if not self.running: break
                buffer += chunk; start, end = buffer.find(b'\xff\xd8'), buffer.find(b'\xff\d9')
                if start != -1 and end != -1:
                    with self.frame_lock: self.last_frame = cv2.imdecode(np.frombuffer(buffer[start:end+2], dtype=np.uint8), cv2.IMREAD_COLOR)
                    buffer = buffer[end+2:]
        except requests.exceptions.RequestException as e: print(f"❌ Stream connection error: {e}")
        finally: self.running = False

    def _detection_pipeline(self):
        while self.running:
            with self.state_lock:
                if self.state != "STREAMING": time.sleep(0.5); continue
            with self.frame_lock:
                if self.last_frame is None: time.sleep(0.1); continue
                frame = self.last_frame.copy()

            all_preds = []; GLOBAL_MIN_CONFIDENCE = 0.60
            for model in DETECTION_MODELS:
                preds = self._call_roboflow(frame, model['id'], max(model['conf'], GLOBAL_MIN_CONFIDENCE))
                for p in preds: p.update({'model_name': model['name'], 'color': model['color']}); all_preds.append(p)

            if verified_candidates := self._heuristic_filter(all_preds):
                with self.state_lock: self.state = "VERIFYING"
                if any(verify_with_gemini(cv2.imencode('.jpg', frame[det['box'][1]:det['box'][3], det['box'][0]:det['box'][2]])[1].tobytes()) for det in verified_candidates):
                    self._handle_confirmed_threat(frame, verified_candidates)
                else:
                    with self.state_lock: self.state = "STREAMING"
            
            self.processed_frame = self._annotate_frame(frame, all_preds, [])
            time.sleep(1/5) # 5 FPS detection rate

    def _heuristic_filter(self, predictions):
        if len(predictions) < 2: return []
        boxes = [[int(p['x']-p['width']/2), int(p['y']-p['height']/2), int(p['x']+p['width']/2), int(p['y']+p['height']/2)] for p in predictions]
        
        clusters = []
        for i, boxA in enumerate(boxes):
            if predictions[i].get('visited'): continue
            current_cluster = {'boxes': [boxA], 'models': {predictions[i]['model_name']}}
            predictions[i]['visited'] = True
            for j, boxB in enumerate(boxes):
                if i == j or predictions[j].get('visited'): continue
                if calculate_iou(boxA, boxB) > 0.6: # 60% overlap threshold
                    current_cluster['boxes'].append(boxB)
                    current_cluster['models'].add(predictions[j]['model_name'])
                    predictions[j]['visited'] = True
            if len(current_cluster['models']) >= 2: # At least 2 different models agree
                x_coords = [b[0] for b in current_cluster['boxes']] + [b[2] for b in current_cluster['boxes']]
                y_coords = [b[1] for b in current_cluster['boxes']] + [b[3] for b in current_cluster['boxes']]
                avg_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                clusters.append({'box': avg_box, 'models': list(current_cluster['models'])})
        return clusters
    
    def _handle_confirmed_threat(self, frame, detections):
        with self.state_lock:
            self.state = "PAUSED"
            annotated_frame = self._annotate_frame(frame.copy(), [], detections)
            location = self._get_gps_location()
            threat_models = list(set(model for det in detections for model in det['models']))
            self.last_detection_info = {"threats": threat_models, "location": location}
            
            ts = time.strftime('%Y%m%d_%H%M%S')
            img_filepath = os.path.join(app.static_folder, "captures", f"capture_{ts}.jpg")
            try:
                if cv2.imwrite(img_filepath, annotated_frame):
                    meta_filepath = os.path.join(app.static_folder, "captures", f"capture_{ts}.json")
                    with open(meta_filepath, 'w') as f: json.dump(self.last_detection_info, f, indent=2)
                else: self.state = "STREAMING" # Resume if we can't save
            except: self.state = "STREAMING"
            self.processed_frame = annotated_frame
    
    def send_control_command(self, action, endpoint, params):
        base_url = None
        if action == "flashlight": base_url = ESP32_FLASHLIGHT_URL
        elif action in ["pan", "tilt"]: base_url = ESP32_CONTROL_URL
        if not base_url or not self.running: return
        try: requests.get(f"{base_url}/{endpoint}", params=params, timeout=1.5)
        except requests.RequestException: pass

    # --- Other unchanged helper functions ---
    def _call_roboflow(self, i, m, c): r = requests.post(f"https://detect.roboflow.com/{m}?api_key={ROBOFLOW_API_KEY}",data=base64.b64encode(cv2.imencode('.jpg', i)[1]),headers={'Content-Type':'application/x-www-form-urlencoded'},params={"confidence":c},timeout=3);r.raise_for_status();return r.json().get('predictions',[])
    def _get_gps_location(self): pass
    def _annotate_frame(self, f, r, c): # Same as before, logic correct
        if c: 
            for d in c: cv2.rectangle(f,(d['box'][0],d['box'][1]),(d['box'][2],d['box'][3]),(0,0,255),3);cv2.putText(f,"CONFIRMED",(d['box'][0],d['box'][1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        else: 
            for p in r: b=[int(p['x']-p['width']/2),int(p['y']-p['height']/2),int(p['x']+p['width']/2),int(p['y']+p['height']/2)];cv2.rectangle(f,(b[0],b[1]),(b[2],b[3]),p['color'],2);cv2.putText(f,f"{p['model_name']} {p['confidence']:.0%}",(b[0],b[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,p['color'],1)
        return f
    def get_frame_bytes(self): # Logic is fine
        with self.frame_lock: f = self.processed_frame if self.processed_frame is not None else self.last_frame
        if f is None: f=np.zeros((600,800,3),np.uint8);cv2.putText(f,"OFFLINE",(300,300),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
        ret, buf = cv2.imencode('.jpg',f); return buf.tobytes() if ret else None
    def resume(self): self.state = "STREAMING"; self.last_detection_info=None; self.processed_frame=None
    def stop(self): self.running=False; time.sleep(0.5)

# --- Global Instance & Routes ---
streamer = None
def generate_frames(): #... unchanged
    while True:
        if streamer and streamer.running and (b := streamer.get_frame_bytes()): yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b + b'\r\n'); time.sleep(1/30)
        else:
            with open("static/offline.jpg","rb") as f: yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+f.read()+b'\r\n')
            time.sleep(1)

@app.route('/')
def index(): return render_template('live_stream.html')
@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/start_stream', methods=['POST'])
def start(): global streamer; streamer=DetectionStreamer(); streamer.start(); return jsonify({})
@app.route('/stop_stream', methods=['POST'])
def stop(): global streamer; if streamer: streamer.stop(); streamer=None; return jsonify({})
@app.route('/resume_stream', methods=['POST'])
def resume(): streamer.resume(); return jsonify({})
@app.route('/get_status', methods=['GET'])
def status(): 
    if not (streamer and streamer.running): return jsonify(state="OFFLINE")
    with streamer.state_lock: s={"state":streamer.state}; s.update(streamer.last_detection_info or {}); return jsonify(s)
@app.route('/control', methods=['POST'])
def control(): # Same as simple version, sends data to streamer method
    if streamer and streamer.running: d=request.json; streamer.send_control_command(d['action'],d['endpoint'],d['params']); return jsonify({})
    return jsonify(status="error"), 400
@app.route('/captures')
def captures(): # Unchanged logic, reads from captures folder
    captures_data = []
    captures_dir = os.path.join(app.static_folder, "captures")
    if os.path.exists(captures_dir):
        for img_file in sorted([f for f in os.listdir(captures_dir) if f.endswith('.jpg')],reverse=True):
            item={"image_url":url_for('static',filename=f'captures/{img_file}'),"threats":"N/A","location":None}
            if os.path.exists(meta_path:=os.path.join(captures_dir,os.path.splitext(img_file)[0]+'.json')):
                with open(meta_path) as f: meta=json.load(f)
                item.update({'location':meta.get('location'),'threats':", ".join(meta.get('threats',[]))})
            captures_data.append(item)
    return render_template('captures.html', captures=captures_data)

if __name__ == '__main__':
    if not os.path.exists("static/offline.jpg"): cv2.imwrite("static/offline.jpg", np.zeros((600,800,3),np.uint8))
    app.run(host='0.0.0.0', port=5001, debug=False)
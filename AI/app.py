# # Code1: Modified Flask Application (app.py) - BUG FIXES APPLIED

# from flask import Flask, render_template, Response, request, jsonify
# import os, cv2, numpy as np, requests, base64, threading, time
# from dotenv import load_dotenv

# # --- All initial setup is the same ---
# load_dotenv()
# ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# DETECTION_MODELS = {
#     # Models are unchanged
#     "soldier_detection": {"name": "Soldier Detection", "models": [{"id": "civil-soldier/1", "weight": 0.3}, {"id": "millitaryobjectdetection/6", "weight": 0.25}, {"id": "hiit/9", "weight": 0.25}, {"id": "soldier-ijybv-wnxqu/1", "weight": 0.2}], "conf": 0.35, "color": (0, 255, 0)},
#     "landmine_detection": {"name": "Landmine", "models": [{"id": "landmine-k5eze-ylmos/1", "weight": 1}], "conf": 0.45, "color": (0, 0, 255)},
#     "aircraft_detection": {"name": "Aircraft", "models": [{"id": "drone-uav-detection/3", "weight": 0.5}, {"id": "fighter-jet-detection/1", "weight": 0.5}], "conf": 0.35, "color": (255, 0, 0)},
#     "tank_detection": {"name": "Tank", "models": [{"id": "tank-sl17s/1", "weight": 1}], "conf": 0.45, "color": (0, 255, 255)},
#     "military_equipment": {"name": "Equipment", "models": [{"id": "military-f5tbj/1", "weight": 0.5}, {"id": "weapon-detection-ssvfk/1", "weight": 0.5}], "conf": 0.35, "color": (255, 0, 255)},
#     "gun_detection": {"name": "Gun", "models": [{"id": "weapon-detection-ssvfk/1", "weight": 0.5}, {"id": "gun-d8mga/2", "weight": 0.5}], "conf": 0.35, "color": (128, 0, 128)}
# }

# app = Flask(__name__)

# # The Camera Stream class and its core methods are correct. No changes needed there.
# class ESP32CamStream:
#     def __init__(self, esp32_url):
#         self.esp32_url=esp32_url;self.stream_url=f"{self.esp32_url}:81/stream"
#         self.current_frame=None;self.processed_frame=None;self.frame_lock=threading.Lock();self.detection_lock=threading.Lock()
#         self.running=False;self.stream_active=False;self.detection_interval=1.0;self.active_model="all"
#     def start_stream(self):
#         if self.running: return
#         self.running=True;self.stream_thread=threading.Thread(target=self._stream_loop,daemon=True);self.stream_thread.start()
#         self.detection_thread=threading.Thread(target=self._detection_loop,daemon=True);self.detection_thread.start()
#     def _stream_loop(self):
#         try:
#             r=requests.get(self.stream_url,stream=True,timeout=10);r.raise_for_status();self.stream_active=True;b=bytes()
#             for c in r.iter_content(chunk_size=4096):
#                 if not self.running: break
#                 b+=c; a,d=b.find(b'\xff\xd8'),b.find(b'\xff\xd9')
#                 if a!=-1 and d!=-1:f=cv2.imdecode(np.frombuffer(b[a:d+2],dtype=np.uint8),cv2.IMREAD_COLOR);b=b[d+2:];[f is not None and setattr(self, 'current_frame', f) for _ in [1] if self.frame_lock.acquire()] and self.frame_lock.release()
#         finally: self.stream_active=False;self.running=False
#     def _detection_loop(self):
#         while self.running:
#             if not self.stream_active: time.sleep(0.5); continue
#             with self.frame_lock:f = self.current_frame.copy() if self.current_frame is not None else None
#             if f is not None:
#                 p=self.process_frame_with_detection(f)
#                 with self.detection_lock:self.processed_frame=p
#             time.sleep(self.detection_interval)
#     def get_latest_frame(self):
#         with self.detection_lock:
#             if self.processed_frame is not None: return self.processed_frame.copy()
#         with self.frame_lock:
#             if self.current_frame is not None: return self.current_frame.copy()
#         return None
#     def stop(self):
#         self.running=False
#         if hasattr(self,'stream_thread')and self.stream_thread.is_alive(): self.stream_thread.join(2)
#         if hasattr(self,'detection_thread')and self.detection_thread.is_alive():self.detection_thread.join(2)
#     def process_frame_with_detection(self, frame):
#         display_frame=frame.copy();models_to_run={};detected_objects_summary=[]
#         if self.active_model=="all":models_to_run=DETECTION_MODELS
#         elif self.active_model in DETECTION_MODELS:models_to_run={self.active_model:DETECTION_MODELS[self.active_model]}
#         for key,detection_info in models_to_run.items():
#             all_preds,weights=[],[]
#             for model_info in detection_info['models']:
#                 preds=call_roboflow_api(frame,model_info['id'],detection_info['conf'])
#                 if preds:all_preds.append(preds);weights.append(model_info['weight'])
#             if not all_preds:continue
#             final_preds=ensemble_predictions(all_preds,weights,detection_info['conf']) # This will now work
#             if final_preds:
#                 detected_objects_summary.append(detection_info['name'])
#                 for pred in final_preds:
#                     if not all(k in pred for k in['x','y','width','height','class','confidence']):continue
#                     x,y,w,h=int(pred['x']),int(pred['y']),int(pred['width']),int(pred['height'])
#                     confidence,class_name=pred['confidence'],pred['class']
#                     x1,y1=x-w//2,y-h//2;x2,y2=x+w//2,y+h//2
#                     label=f"{class_name}:{confidence:.2f}";cv2.rectangle(display_frame,(x1,y1),(x2,y2),detection_info['color'],2)
#                     cv2.putText(display_frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,detection_info['color'],2)
#         if detected_objects_summary:print(f"\n{'='*25}> DETECTION CONFIRMED <{'='*25}\n>>> Objects Found: [ {', '.join(set(detected_objects_summary))} ] <<<\n{'='*75}\n")
#         return display_frame

# # --- GLOBAL AND HELPER FUNCTIONS ---
# cam_stream=None

# # ===============================================================
# # === CRITICAL BUG FIX in ensemble_predictions FUNCTION         ===
# # ===============================================================
# def ensemble_predictions(predictions_list, weights, confidence_threshold):
#     """
#     This function has been re-written to be clear and correct,
#     fixing the 'NameError' crash.
#     """
#     final_predictions = []
#     # Use standard, readable for-loops to avoid scoping bugs
#     for predictions, weight in zip(predictions_list, weights):
#         for pred in predictions:
#             weighted_conf = pred.get('confidence', 0) * weight
#             if weighted_conf >= confidence_threshold:
#                 pred_copy = pred.copy()
#                 pred_copy['confidence'] = weighted_conf
#                 final_predictions.append(pred_copy)
#     return final_predictions
# # ===============================================================

# def call_roboflow_api(image, model_id, confidence):
#     url=f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}"
#     try:_,buffer=cv2.imencode('.jpg',image);img_b64=base64.b64encode(buffer);r=requests.post(url,data=img_b64,headers={'Content-Type':'application/x-www-form-urlencoded'},params={"confidence":confidence,"format":"json"},timeout=3);r.raise_for_status();return r.json().get('predictions',[])
#     except Exception: return []

# def generate_frames():
#     while True:
#         if cam_stream and cam_stream.running and cam_stream.get_latest_frame() is not None:
#             ret,buffer=cv2.imencode('.jpg',cam_stream.get_latest_frame())
#             if ret:yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n');continue
#         error_frame=np.zeros((480,640,3),np.uint8);msg="STREAM STOPPED"
#         if cam_stream and not cam_stream.stream_active:msg="CONNECTING..."
#         cv2.putText(error_frame,msg,(50,240),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#         ret,buffer=cv2.imencode('.jpg',error_frame);yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n');time.sleep(1)

# # --- FLASK ROUTES ---
# @app.route('/')
# def index(): return render_template('live_stream.html', models=DETECTION_MODELS)
# @app.route('/video_feed')
# def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# @app.route('/start_stream', methods=['POST'])
# def start_stream_route():global cam_stream;d=request.json;cam_stream=ESP32CamStream(d.get('esp32_url'));cam_stream.start_stream();return jsonify(status="success")
# @app.route('/stop_stream', methods=['POST'])
# def stop_stream_route():
#     if cam_stream:cam_stream.stop()
#     return jsonify(status="success")
# @app.route('/set_active_model', methods=['POST'])
# def set_active_model_route():
#     if cam_stream:cam_stream.active_model=request.json.get('model_key')
#     return jsonify(status="success")

# # ===============================================================
# # === ROBUST FLASHLIGHT ROUTE to prevent crashing             ===
# # ===============================================================
# @app.route('/set_brightness', methods=['POST'])
# def set_brightness_route():
#     data = request.json
#     esp32_url, level = data.get('esp32_url'), data.get('level')
#     if not all([esp32_url, level is not None]):
#         return jsonify(status="error", message="Invalid request")
    
#     try:
#         led_url = f"{esp32_url}/led?level={level}"
#         requests.get(led_url, timeout=1) # Using a short timeout
#         return jsonify(status="success")
#     except requests.exceptions.RequestException as e:
#         # This will now print a clean one-line warning instead of a huge error
#         print(f"!!! WARNING: Flashlight command failed. ESP32 at {esp32_url} is not responding.")
#         return jsonify(status="error", message="ESP32 Timeout")
# # ===============================================================

# if __name__ == '__main__':
#     print("Starting Flask Server...")
#     app.run(host='0.0.0.0', port=5001, debug=False)

# Code1: Modified Flask Application (app.py) - INTERACTIVE PAUSE/RESUME

# Code1: Modified Flask Application (app.py) - INDENTATION FIX

from flask import Flask, render_template, Response, request, jsonify
import os, cv2, numpy as np, requests, base64, threading, time
from dotenv import load_dotenv

# --- All initial setup is the same ---
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

DETECTION_MODELS = {
    "soldier_detection": {"name": "Soldier Detection", "models": [{"id": "civil-soldier/1", "weight": 0.3}, {"id": "millitaryobjectdetection/6", "weight": 0.25}, {"id": "hiit/9", "weight": 0.25}, {"id": "soldier-ijybv-wnxqu/1", "weight": 0.2}], "conf": 0.35, "color": (0, 255, 0)},
    "landmine_detection": {"name": "Landmine", "models": [{"id": "landmine-k5eze-ylmos/1", "weight": 1}], "conf": 0.45, "color": (0, 0, 255)},
    "aircraft_detection": {"name": "Aircraft", "models": [{"id": "drone-uav-detection/3", "weight": 0.5}, {"id": "fighter-jet-detection/1", "weight": 0.5}], "conf": 0.35, "color": (255, 0, 0)},
    "tank_detection": {"name": "Tank", "models": [{"id": "tank-sl17s/1", "weight": 1}], "conf": 0.45, "color": (0, 255, 255)},
    "military_equipment": {"name": "Equipment", "models": [{"id": "military-f5tbj/1", "weight": 0.5}, {"id": "weapon-detection-ssvfk/1", "weight": 0.5}], "conf": 0.35, "color": (255, 0, 255)},
    "gun_detection": {"name": "Gun", "models": [{"id": "weapon-detection-ssvfk/1", "weight": 0.5}, {"id": "gun-d8mga/2", "weight": 0.5}], "conf": 0.35, "color": (128, 0, 128)}
}

app = Flask(__name__)

class ESP32CamStream:
    def __init__(self, esp32_url):
        self.esp32_url=esp32_url
        self.stream_url=f"{self.esp32_url}:81/stream"
        self.current_frame=None
        self.frame_lock=threading.Lock()
        self.state = "STREAMING_LIVE"
        self.paused_frame = None
        self.state_lock = threading.Lock()
        self.running=False
        self.stream_active=False
        self.detection_interval=2.0
        self.active_model="all"

    def start_stream(self):
        if self.running: return
        self.running=True
        self.stream_thread=threading.Thread(target=self._stream_loop,daemon=True)
        self.stream_thread.start()
        self.detection_thread=threading.Thread(target=self._detection_loop,daemon=True)
        self.detection_thread.start()

    def _stream_loop(self):
        try:
            r=requests.get(self.stream_url,stream=True,timeout=10)
            r.raise_for_status()
            self.stream_active=True
            b=bytes()
            for c in r.iter_content(chunk_size=4096):
                if not self.running: break
                b+=c
                a,d=b.find(b'\xff\xd8'),b.find(b'\xff\xd9')
                if a!=-1 and d!=-1:
                    f=cv2.imdecode(np.frombuffer(b[a:d+2],dtype=np.uint8),cv2.IMREAD_COLOR)
                    b=b[d+2:]
                    if f is not None:
                        with self.frame_lock: self.current_frame=f
        finally:
            self.stream_active=False
            self.running=False

    # ===============================================================
    # === CRITICAL BUG FIX: _detection_loop INDENTATION REPAIRED  ===
    # ===============================================================
    def _detection_loop(self):
        """ This thread now controls the state based on detections. """
        while self.running:
            # Check the state first
            with self.state_lock:
                current_state = self.state

            # If paused, this thread just waits and loops again.
            if current_state == "PAUSED_FOR_CONFIRMATION":
                time.sleep(0.5)
                continue
            
            # If not paused, try to get a frame for processing.
            frame_to_process = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()
            
            # THIS IS THE BLOCK WHERE THE INDENTATION ERROR OCCURRED.
            # It is now correctly structured.
            if frame_to_process is not None:
                # Perform detection logic...
                detected_boxes, frame_with_boxes = self.get_detections_on_frame(frame_to_process)
                
                # --- STATE CHANGE LOGIC ---
                if detected_boxes:
                    print("!!! DETECTION FOUND! Pausing stream for confirmation. !!!")
                    with self.state_lock:
                        self.paused_frame = frame_with_boxes.copy()
                        self.state = "PAUSED_FOR_CONFIRMATION"
            
            time.sleep(self.detection_interval)
    # ===============================================================

    def get_detections_on_frame(self, frame): # This function is correct
        frame_with_boxes=frame.copy();models_to_run={};found_boxes_data=[]
        if self.active_model=="all":models_to_run=DETECTION_MODELS
        elif self.active_model in DETECTION_MODELS:models_to_run={self.active_model:DETECTION_MODELS[self.active_model]}
        for key,detection_info in models_to_run.items():
            all_preds,weights=[],[]
            for model_info in detection_info['models']:
                preds=call_roboflow_api(frame,model_info['id'],detection_info['conf'])
                if preds:all_preds.append(preds);weights.append(model_info['weight'])
            if not all_preds:continue
            final_preds=ensemble_predictions(all_preds,weights,detection_info['conf'])
            if final_preds:
                found_boxes_data.extend(final_preds)
                for pred in final_preds:
                    if not all(k in pred for k in['x','y','width','height']):continue
                    x,y,w,h=int(pred['x']),int(pred['y']),int(pred['width']),int(pred['height'])
                    label=f"{pred.get('class','Obj')}:{pred.get('confidence',0):.2f}"
                    cv2.rectangle(frame_with_boxes,(x-w//2,y-h//2),(x+w//2,y+h//2),detection_info['color'],2)
                    cv2.putText(frame_with_boxes,label,(x-w//2,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,detection_info['color'],2)
        return found_boxes_data,frame_with_boxes
    
    def resume_streaming(self): # This function is correct
        with self.state_lock:self.state="STREAMING_LIVE";self.paused_frame=None
        print("--- Confirmation received. Resuming live stream. ---")

    def get_current_frame(self):
        with self.frame_lock:return self.current_frame.copy()if self.current_frame is not None else None
    def stop(self):
        self.running=False
        for t in[getattr(self,'stream_thread',None),getattr(self,'detection_thread',None)]:
            if t and t.is_alive():t.join(2)

# --- HELPER FUNCTIONS & FLASK ROUTES ---
cam_stream=None
def ensemble_predictions(p,w,c):f=[];[[f.append({**pred,'confidence':conf})for pred in preds if(conf:=pred.get('confidence',0)*wt)>=c]for preds,wt in zip(p,w)];return f
def call_roboflow_api(i,m,c):
    try:_,b=cv2.imencode('.jpg',i);d=base64.b64encode(b);r=requests.post(f"https://detect.roboflow.com/{m}?api_key={ROBOFLOW_API_KEY}",data=d,headers={'Content-Type':'application/x-www-form-urlencoded'},params={"confidence":c,"format":"json"},timeout=3);r.raise_for_status();return r.json().get('predictions',[])
    except:return[]

def generate_frames(): # This function is correct
    while True:
        if not(cam_stream and cam_stream.running):
            err_frame=np.zeros((480,640,3),np.uint8);cv2.putText(err_frame,"STREAM STOPPED",(50,240),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);r,b=cv2.imencode('.jpg',err_frame);yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n'+b.tobytes()+b'\r\n');time.sleep(1);continue
        with cam_stream.state_lock:
            if cam_stream.state=="PAUSED_FOR_CONFIRMATION" and cam_stream.paused_frame is not None:
                r,b=cv2.imencode('.jpg',cam_stream.paused_frame);yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n'+b.tobytes()+b'\r\n');time.sleep(0.1);continue
        live_frame=cam_stream.get_current_frame()
        if live_frame is not None:
            r,b=cv2.imencode('.jpg',live_frame);yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n'+b.tobytes()+b'\r\n')

@app.route('/')
def index():return render_template('live_stream.html',models=DETECTION_MODELS)
@app.route('/video_feed')
def video_feed():return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/start_stream', methods=['POST'])
def start_stream_route():
    global cam_stream
    cam_stream=ESP32CamStream(request.json.get('esp32_url'))
    cam_stream.start_stream()
    return jsonify(status="success")
@app.route('/stop_stream', methods=['POST'])
def stop_stream_route():
    if cam_stream:cam_stream.stop()
    return jsonify(status="success")
@app.route('/set_active_model', methods=['POST'])
def set_active_model_route():
    if cam_stream:cam_stream.active_model=request.json.get('model_key')
    return jsonify(status="success")
@app.route('/set_brightness', methods=['POST'])
def set_brightness_route():
    data=request.json;url,lvl=data.get('esp32_url'),data.get('level')
    try:requests.get(f"{url}/led?level={lvl}",timeout=1)
    except:print(f"WARN: Flashlight command to {url} failed.")
    return jsonify(status="success")
@app.route('/get_status', methods=['GET'])
def get_status_route():
    if not cam_stream or not cam_stream.running:return jsonify(state="STOPPED")
    return jsonify(state=cam_stream.state)
@app.route('/resume_stream', methods=['POST'])
def resume_stream_route():
    if cam_stream:cam_stream.resume_streaming()
    return jsonify(status="success")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)